"""
An LSTM model for writing fake histories that tries to optimize distance in
embedded space between model output and target word.

@author: rileypsmith
Created: 5/4/2023
"""

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

import utils

class MLP(layers.Layer):
    """Simple dense network for head of LSTM network"""
    def __init__(self, num_layers, output_dim, hidden_dim=256,
                 activation='relu', **kwargs):
        super().__init__(**kwargs)
        dense_layers = [
            layers.Dense(hidden_dim, activation=activation, use_bias=False) for _ in range(num_layers)
        ]
        dense_layers.append(layers.Dense(output_dim, activation='softmax', use_bias=False))
        self.body = Sequential(dense_layers)
    def call(self, x):
        return self.body(x)

class LSTMSublayer(layers.Layer):
    """A sublayer to implement LSTM followed by small dense layer"""
    def __init__(self, units, bidirectional=False, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.lstm = layers.LSTM(units, return_sequences=return_sequences)
        if bidirectional:
            self.lstm = layers.Bidirectional(self.lstm)
            self.dense = layers.Dense(units, activation='sigmoid')
        else:
            self.dense = layers.Lambda(lambda x: x)
    def call(self, x, mask=None):
        return self.dense(self.lstm(x, mask=mask))

class EmbeddedLSTMModel(Model):
    """A super simple LSTM model for history text generation"""
    def __init__(self, vocab_size, embedding_dim=64, lstm_units=128, num_layers=5,
                 bidirectional=False, num_dense_layers=2, hidden_dim=256, 
                 return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        
        self.embedding_dim = embedding_dim
        
        # Embedding from vocab into embedding dimension
        self.embed = layers.Embedding(vocab_size + 1, embedding_dim)
        
        # LSTM layers
        lstm_layers = []
        for _ in range(num_layers - 1):
            local_layer = LSTMSublayer(lstm_units, bidirectional, return_sequences=True)
            lstm_layers.append(local_layer)
        # Add final layer
        final_layer = LSTMSublayer(lstm_units, bidirectional, return_sequences)
        lstm_layers.append(final_layer)
        self.lstm_layers = lstm_layers

        # Small MLP at the end
        self.head = MLP(num_dense_layers, embedding_dim, hidden_dim)
    
    def call(self, x, labels):
        """
        Parameters
        ----------
        x : tf.Tensor
            Tensor of shape (bs, seq_length). The integer vectorization of the
            input text.
        labels
            Not used. Added as an argument because it is needed for Transformer
            (this way one training pipeline works for both).
        """
        # Make a mask for this input
        mask = utils.make_padding_mask(x, invert=False, transformer=False)
        mask = tf.cast(mask, tf.bool)
        # Put it through the LSTM
        x = self.embed(x)
        for lstm in self.lstm_layers:
            x = lstm(x, mask=mask)
        # Project back into embedding dimension
        x = self.head(x)
        # Now embed the label
        y = self.embed(labels)
        return x, y
    
    def sim_score(self, x, y):
        tmpx = x.numpy()
        tmpy = y.numpy()
        return tmpx.dot(tmpy) / (np.linalg.norm(tmpx) * np.linalg.norm(tmpy))
    
    def distance_regularizer(self, var_alpha=0.005):
        """Add regularization based on Mahalanobis distance"""
        # Variance regularizer will try to maximize variance between rows of
        # embedding matrix
        mat = self.layers[0].weights[0]
        var = tf.math.reduce_variance(mat, axis=0)
        var_loss = 1 / tf.clip_by_value(var, 1e-100, 1e100)
        var_loss = tf.math.reduce_mean(var_loss)
        # Characterize prior for distribution as multivariate normal
        dist = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(mat), axis=1))
        dist_loss = tf.math.reduce_mean(dist)
        return (var_loss * var_alpha), dist_loss
        # return (var_loss * var_alpha) + dist_loss
    
    def predict(self, x):
        """Take the input and predict the next word"""
        # Make a mask for this input
        mask = tf.cast(utils.make_padding_mask(x, invert=False, transformer=False), tf.bool)
        # Put it through the LSTM
        x = self.embed(x)
        for lstm in self.lstm_layers:
            x = lstm(x, mask=mask)
        # Project back into embedding dimension
        x = self.head(x)
        # Now we have an embedded vector, we have to convert it to a word
        # mat = self.layers[0].weights[0]
        # sim_scores = [self.sim_score(x, mat[i]) for i in range(mat.shape[0])]
        # return np.array(sim_scores).argmax()
        
        
        inverse = tf.linalg.matmul(x, self.inv_embed)
        inverse = tf.nn.softmax(inverse)
        return inverse.numpy().argmax()
    
    def finish_sentence(self, sentence_start, end_tokens, seq_length=128,
                        max_seq_length=128):
        """
        Finish the rest of the sentence (until next punctuation character).
        
        Parameters
        ----------
        sentence_start : ndarray
            Array of integers. The vectorized start to a sentence.
        end_tokens : listlike
            A list of punctuation tokens that end a sentence.
        """
        # Make inverse embedding matrix
        self.inv_embed = tf.linalg.pinv(self.layers[0].weights[0], rcond=1e-6)
        # Make a mask for missing items
        last_token = 0
        words_predicted = 0
        sentence = sentence_start.copy()
        while not last_token in end_tokens:
            if len(sentence) < seq_length:
                in_sequence = np.pad(sentence, (0, seq_length - len(sentence)))
            else:
                excess = len(sentence) - seq_length
                in_sequence = sentence[excess:]
            in_sequence = tf.expand_dims(tf.convert_to_tensor(in_sequence), axis=0)
            # Use LSTM to predict the next word
            last_token = self.predict(in_sequence)
            sentence = np.append(sentence, last_token)
            words_predicted += 1
            # Stop after so many iterations (early on in training it can run
            # for awhile before hitting an end token)
            if words_predicted >= max_seq_length:
                break
        return sentence
    
    def get_loss(self, *args):
        """Return a loss function suitable for training this model"""
        return tf.keras.losses.CosineSimilarity()
        
    def train_step(self, data, labels):
        """Do one training step and update weights"""
        with tf.GradientTape() as tape:
            # Forward pass
            pred, true = self(data, labels)
            # Compute loss
            var_loss, dist_loss = self.distance_regularizer()
            sim_loss = self.loss(true, pred) + 1
            loss = sim_loss + var_loss + dist_loss
            # loss = self.loss(true, pred) + self.distance_regularizer()
            # Compute gradients
            grad = tape.gradient(loss, self.trainable_variables)
            # Backpropogate
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return sim_loss, var_loss, dist_loss
    
    def val_step(self, data, labels):
        """Do one step of evaluation (just don't do backpropogation)"""
        pred, true = self(data, labels)
        sim_loss = self.loss(true, pred) + 1
        var_loss, dist_loss = self.distance_regularizer()
        return sim_loss, var_loss, dist_loss
        # loss = self.loss(true, pred) + var_loss + dist_loss
        # return loss
