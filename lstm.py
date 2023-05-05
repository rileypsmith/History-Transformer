"""
An LSTM model for writing fake histories.

@author: rileypsmith
Created: 5/4/2023
"""

from tensorflow.keras import layers, Model, Sequential

class MLP(layers.Layer):
    """Simple dense network for head of LSTM network"""
    def __init__(self, num_layers, output_dim, hidden_dim=256,
                 activation='relu', **kwargs):
        super().__init__(**kwargs)
        dense_layers = [
            layers.Dense(hidden_dim) for _ in range(num_layers)
        ]
        dense_layers.append(Dense(output_dim, activation='softmax'))
        self.body = Sequential(dense_layers)
    def call(self, x):
        return self.body(x)

class LSTMModel(Model):
    """A super simple LSTM model for history text generation"""
    def __init__(self, vocab_size, embedding_dim, lstm_units=128, 
                 bidirectional=True, num_dense_layers=2, hidden_dim=256, 
                 **kwargs):
        super().__init__(**kwargs)
        
        # Embedding from vocab into embedding dimension
        self.embed = layers.Embedding(vocab_size, embedding_dim)
        
        # LSTM component
        lstm_layer = layers.LSTM(lstm_units)
        if bidirectional:
            lstm_layer = layers.Bidirectional(lstm_layer)
        self.lstm = lstm_layer

        # Small MLP at the end
        self.head = MLP(num_dense_layers, vocab_size, hidden_dim)
    
    def call(self, x):
        x = self.embed(x)
        x = self.lstm(x)
        return self.head(x)
        
        