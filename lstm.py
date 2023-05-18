"""
An LSTM model for writing fake histories.

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
            layers.Dense(hidden_dim) for _ in range(num_layers)
        ]
        dense_layers.append(layers.Dense(output_dim, activation='softmax'))
        self.body = Sequential(dense_layers)
    def call(self, x):
        return self.body(x)

class LSTMModel(Model):
    """A super simple LSTM model for history text generation"""
    def __init__(self, vocab_size, embedding_dim=64, lstm_units=128, num_layers=5,
                 bidirectional=False, num_dense_layers=2, hidden_dim=256, 
                 return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        
        # Embedding from vocab into embedding dimension
        self.embed = layers.Embedding(vocab_size + 1, embedding_dim)
        
        # LSTM layers
        lstm_layers = []
        for _ in range(num_layers - 1):
            local_layer = layers.LSTM(lstm_units, return_sequences=True)
            if bidirectional:
                local_layer = layers.Bidirectional(local_layer)
            lstm_layers.append(local_layer)
        # Add final layer
        final_layer = layers.LSTM(lstm_units, return_sequences=return_sequences)
        if bidirectional:
            final_layer = layers.Bidirectional(final_layer)
        lstm_layers.append(final_layer)
        self.lstm_layers = lstm_layers

        # Small MLP at the end
        self.head = MLP(num_dense_layers, vocab_size, hidden_dim)
        
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
            mask = tf.equal(in_sequence, 0)
            # Use LSTM to predict the next word
            preds = self(in_sequence)
            last_token = preds[0,-1].numpy().argmax() + 1
            sentence = np.append(sentence, last_token)
            words_predicted += 1
            print(f'predicted {words_predicted} words')
            
            # Stop after so many iterations (early on in training it can run
            # for awhile before hitting an end token)
            if words_predicted >= max_seq_length:
                break
        return sentence
    
    def call(self, x, labels=None):
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
        # return mask
        mask = tf.cast(mask, tf.bool)
        x = self.embed(x)
        for lstm in self.lstm_layers:
            x = lstm(x, mask=mask)
        return self.head(x)
        