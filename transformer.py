"""
An implementation of the original transformer from the paper "Attention is all
you need" in Tensorflow.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

def make_position_mask(shape):
    """
    Mask for positions in input sequence so that decoder cannot attend to future
    items in sequence.
    """
    return 1 - tf.linalg.band_part(tf.ones((shape, shape), dtype=tf.float32), -1, 0)

def make_padding_mask(x):
    """
    Mask for postions in input tensor that are 0 (padding tokens that should be
    ignored by encoder and decoder).
    """
    return tf.cast(tf.math.equal(x, 0), tf.float32)

class ScaledDotProductAttention(layers.Layer):
    def __init__(self, scaling_constant, **kwargs):
        super().__init__(**kwargs)
        # Set scaling constant (square root of dimensionality)
        self.scaling_constant = scaling_constant
        
        # Softmax activation
        self.softmax = layers.Softmax()
    
    def call(self, Q, K, V, mask=None):
        """
        Compute the scaled dot product attention of the input queries, keys,
        and values. At this point, they should have shape:
            Q,K : (bs, heads, seq_length, d_k)
            V   : (bs, heads, seq_length, d_v)
        """
        attn = tf.linalg.matmul(Q, K, transpose_b=True)
        # Apply mask if given
        if mask is not None:
            attn += (mask * tf.math.pow(-2., 32.))
        attn = self.softmax(attn * self.scaling_constant)
        attn = tf.linalg.matmul(attn, V)
        return attn

class ReshapeHeads(layers.Layer):
    """A class to handle reshaping with the proper heads dimension"""
    def __init__(self, heads=8, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
    def call(self, x, undo=False):
        """If undo, will fold the head dimension back into the other dimensions"""
        if undo:
            x = tf.transpose(x, perm=(0,2,1,3))
            bs, n, heads, dim = x.shape
            x = tf.reshape(x, (bs, n, heads*dim))
        else:
            bs, n, dim = x.shape
            x = tf.reshape(x, (bs, n, self.heads, -1))
            x = tf.transpose(x, perm=(0,2,1,3))
        return x

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_k, d_v=None, heads=8, **kwargs):
        super().__init__(**kwargs)
        
        d_v = d_k if d_v is None else d_v

        # Projections for queries, keys, and values
        self.W_q = layers.Dense(d_k * heads)
        self.W_k = layers.Dense(d_k * heads)
        self.W_v = layers.Dense(d_v * heads)
        
        # Layer to handle reshaping with the heads dimension
        self.reshape_heads = ReshapeHeads(heads) 
        
        # Layer that actually computes the attention
        scaling_constant = 1. / tf.sqrt(tf.cast(d_k, tf.float32))
        self.attention = ScaledDotProductAttention(scaling_constant)
        
    def build(self, input_shape):
        # Read dimensionality of each element in sequence
        dim = input_shape[-1]
        # Set linear output layer accordingly
        self.linear_head = layers.Dense(dim)
        
    def call(self, queries, keys, values, mask=None):
        # Project input into new dimensionality
        Q = self.W_q(queries)
        K = self.W_k(keys)
        V = self.W_v(values)
        
        # Roll heads dimension into its own thing
        Q = self.reshape_heads(Q)
        K = self.reshape_heads(K)
        V = self.reshape_heads(V)
        
        # Compute scaled dot product attention
        attn = self.attention(Q, K, V, mask)
        
        # Roll head dimension back into final dimension
        attn = self.reshape_heads(attn, undo=True)
        
        # Apply output projection
        return self.linear_head(attn)
    
class FeedForward(layers.Layer):
    """The feed forward layer from the original Transformer"""
    def __init__(self, hidden_dim=2048, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
    def build(self, input_shape):
        # Setup second dense layer to project back to output dimensionality
        in_dim = input_shape[-1]
        self.dense2 = layers.Dense(in_dim)
    def call(self, x):
        return self.dense2(self.dense1(x))
    
class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, max_seq_length, embedding_dim=64, **kwargs):
        super().__init__(**kwargs)
        # Setup learned embedding for vocabulary and position
        self.vocab_embedding = layers.Embedding(vocab_size, embedding_dim)
        self.position_embedding = layers.Embedding(max_seq_length, embedding_dim)
    def call(self, x):
        return self.vocab_embedding(x) + self.position_embedding(x)
    
class EncoderSublayer(layers.Layer):
    """One sublayer of the Transformer encoder"""
    def __init__(self, d_k, d_v=None, heads=8, hidden_dim=2048, **kwargs):
        super().__init__(**kwargs)
        
        # Sublayer consists of multi-head attention and feed forward network,
        # with LayerNorm in between
        self.attention = MultiHeadAttention(d_k, d_v, heads)
        self.layer_norm1 = layers.LayerNormalization()
        self.feed_forward = FeedForward(hidden_dim)
        self.layer_norm2 = layers.LayerNormalization()
        
    def call(self, x, padding_mask, training=True):
        a = self.attention(x, x, x, padding_mask)
        x = self.layer_norm1(x + a, training=training)
        a = self.feed_forward(x)
        return self.layer_norm2(x + a, training=training)
        
class Encoder(layers.Layer):
    """The Transformer encoder"""
    def __init__(self, vocab_size, max_seq_length, n_layers=6, d_k=512, d_v=None,
                 heads=8, hidden_dim=2048, embedding_dim=64, **kwargs):
        super().__init__(**kwargs)
        
        # Positional embedding layer
        self.embedding = PositionalEmbedding(vocab_size, max_seq_length, 
                                             embedding_dim)
        
        # Multiple Encoder Sublayers
        self.encoder_body = []
        for i in range(n_layers):
            self.encoder_body.append(
                EncoderSublayer(d_k, d_v, heads, hidden_dim, name=f'EncoderSublayer_{i:02}')
            )
        
    def call(self, x, padding_mask, training=True):
        x = self.embedding(x)
        # Apply sublayers
        for sublayer in self.encoder_body:
            x = sublayer(x, padding_mask)
        return x
    
class DecoderSublayer(layers.Layer):
    """One sublayer of the Transformer Decoder"""
    def __init__(self, d_k, d_v=None, heads=8, hidden_dim=2048, **kwargs):
        super().__init__(**kwargs)
        
        # Initial self-attention
        self.attention1 = MultiHeadAttention(d_k, d_v, heads)
        self.layer_norm1 = layers.LayerNormalization()
        
        # Attention which attends to output of encoder
        self.attention2 = MultiHeadAttention(d_k, d_v, heads)
        self.layer_norm2 = layers.LayerNormalization()
        
        # Feed forward portion
        self.feed_forward = FeedForward(hidden_dim)
        self.layer_norm3 = layers.LayerNormalization()
        
    def call(self, x, encoder_out, position_mask, padding_mask, training=False):
        # First do self attention on inputs, preventing attending to unseen positions
        a = self.attention1(x, x, x, position_mask)
        x = self.layer_norm1(x + a, training=training)
        
        # Now attend to encoder outputs
        a = self.attention2(x, encoder_out, encoder_out, padding_mask)
        x = self.layer_norm2(x + a, training=training)
        
        # Finally, the feed-forward network
        a = self.feed_forward(x)
        return self.layer_norm3(x + a, training=training)
    
class Decoder(layers.Layer):
    """The Transformer decoder"""
    def __init__(self, vocab_size, max_seq_length, n_layers=6, d_k=512, d_v=None,
                 heads=8, hidden_dim=2048, embedding_dim=64, **kwargs):
        super().__init__(**kwargs)
        
        # Positional embedding
        self.embedding = PositionalEmbedding(vocab_size, max_seq_length,
                                             embedding_dim)
        
        # Multiple decoder sublayers
        self.decoder_body = []
        for i in range(n_layers):
            self.decoder_body.append(
                DecoderSublayer(d_k, d_v, heads, hidden_dim, name=f'DecoderSublayer_{i:02}')
            )
        
    def call(self, x, encoder_out, position_mask, padding_mask, training=True):
        # Apply embedding
        x = self.embedding(x)
        
        # Now Decoder body
        for sublayer in self.decoder_body:
            x = sublayer(x, encoder_out, position_mask, padding_mask, training=training)
        return x
            
class Transformer(Model):
    """
    The main transformer class. Imitates the original paper, except embeddings
    for position and word are learned instead of fixed.
    """
    def __init__(self, vocab_size, max_seq_length, embedding_dim=64, 
                 d_k=128, d_v=128, heads=8, hidden_dim=2048, 
                 n_encoder_layers=6, n_decoder_layers=6, **kwargs):
        super().__init__(**kwargs)
        
        # Build encoder and decoder
        self.encoder = Encoder(vocab_size, max_seq_length, n_encoder_layers,
                               d_k, d_v, heads, hidden_dim, embedding_dim)
        
        self.decoder = Decoder(vocab_size, max_seq_length, n_decoder_layers, 
                               d_k, d_v, heads, hidden_dim, embedding_dim)
        
        # Build linear head
        self.head = layers.Dense(vocab_size, activation='softmax')
        
        # Set positional mask, which depends only on the sequence length
        self.position_mask = make_position_mask(max_seq_length)
        
    def call(self, encoder_input, decoder_input, training=False):
        # Make padding mask for items shorter than max sequence length
        encoder_padding_mask = make_padding_mask(encoder_input)
        
        # Combine position and padding mask for decoder to prevent attending
        # to subsequent positions
        decoder_padding_mask = make_padding_mask(decoder_input)
        position_mask = tf.math.maximum(decoder_padding_mask, self.position_mask)
        
        # Encoder the inputs
        encoder_output = self.encoder(encoder_input, encoder_padding_mask,
                                      training=training)
        
        # Send decoder inputs and encoder outputs to the decoder
        decoder_output = self.decoder(decoder_input, encoder_output, position_mask,
                                      encoder_padding_mask, training=training)
        
        # Finally, send the outputs through the head to get predictions over vocabulary
        return self.head(decoder_output)
        
        
        
        
        
        
        
    
    