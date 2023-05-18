"""
Utility functions and classes for training the HistoryTransformer.

@author: rileypsmith
Created: 5/3/2023
"""
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, callbacks, metrics

from data import load_vectorizer

class LabelSmoothingSCC(losses.Loss):
    """
    Simple implementation of sparse categorical crossentropy loss, but with
    label smoothing.
    """
    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.loss = losses.CategoricalCrossentropy(label_smoothing=alpha)
    
    def call(self, y_true, y_pred):
        # Encode true labels to one-hot vectors
        y_true = tf.cast(tf.one_hot(y_true - 1, y_pred.shape[-1]), tf.float32)
        # return y_true
        # Now apply Tensorflow's categorical crossentropy with label smoothing
        return self.loss(y_true, y_pred)
    
class CustomCSVLogger(callbacks.Callback):
    """
    A simple adaptation of the CSVLogger from Tensorflow. This one logs out
    intermediate results (not just every epoch).
    """
    def __init__(self, filepath, batch_interval=100, **kwargs):
        super().__init__(**kwargs)
        
        self.filepath = filepath
        self.batch_interval = batch_interval
        
    def on_train_begin(self, logs=None):
        # Create an open file object to write to
        self.csv_file = tf.io.gfile.GFile(self.filepath, 'w')
        # Keep track of keys that have been seen
        self.keys = []
        self.epoch = 0
        
    def on_train_batch_end(self, batch, logs=None):
        # Log information received
        if not (batch + 1) % self.batch_interval == 0:
            return
        if not self.keys:
            train_keys = sorted(list(logs.keys()))
            val_keys = [f'val_{k}' for k in train_keys]
            self.keys = ['epoch', 'batch', 'valid_flag'] + train_keys + val_keys
            # Set CSV writer
            self.writer = csv.DictWriter(self.csv_file, fieldnames=self.keys)
            self.writer.writeheader()
        # Write the received at this batch
        write_data = {'epoch': self.epoch + 1, 'batch': batch + 1, 
                      'valid_flag': 0, **logs}
        self.writer.writerow(write_data)
        self.csv_file.flush()
    
    def on_epoch_end(self, epoch, logs=None):
        # Write the data received at this epoch
        write_data = {'epoch': epoch + 1, 'valid_flag': 1, **logs}
        self.writer.writerow(write_data)
        self.csv_file.flush()
        self.epoch += 1
        
    def on_train_end(self, logs=None):
        # Close out the file
        self.csv_file.close()

def setup_output_dir(output_dir):
    """Build given output directory, or append numbers if it already exists"""
    i = 0
    output_dir = Path(output_dir)
    parent = output_dir.absolute().parent
    started = False
    while output_dir.exists():
        stem = output_dir.stem[:-4] if started else output_dir.stem
        output_dir = Path(parent, stem + f'_{i:03}')
        started = True
        i += 1
    output_dir.mkdir()
    return str(output_dir)

class CosineAnnealingLR():
    """Simple cosine annealing learning rate scheduler"""
    def __init__(self, initial_lr=1e-3, eta_min=1e-8, num_epochs=100, 
                 period=10, **kwargs):
        # super().__init__(**kwargs)
        self.initial_lr = initial_lr
        self.eta_min = eta_min
        self.num_epochs = num_epochs
        self.period = period
        
    def __call__(self, epoch, logs=None):
        if epoch >= self.num_epochs:
            return 0
        decay_factor = 1 - (epoch / self.num_epochs)
        cos_factor = 1 + np.cos((epoch / self.period) * np.pi)
        return (self.eta_min + ((self.initial_lr - self.eta_min) / 2) * cos_factor) * decay_factor
    
class FadingMemoryMean(metrics.Metric):
    """Simple mean but with fading memory"""
    def __init__(self, memory_constant=0.95, **kwargs):
        super().__init__(**kwargs)
        self.memory_constant = memory_constant
        self.value = None
    def update_state(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = (self.value * self.memory_constant) + (value * (1 - self.memory_constant))
    def result(self):
        return self.value
    def reset_states(self):
        self.value = None
        
class SampleHistoryWriter(callbacks.Callback):
    """Every time validation is called, write a sample paragraph."""
    def __init__(self, vectorizer_file=None, outdir=None, 
                 seq_length=128, num_sentences=3, max_seq_length=64, 
                 train_batch_interval=5000, **kwargs):
        super().__init__(**kwargs)
        
        self.num_sentences = num_sentences
        self.seq_length = seq_length
        self.max_seq_length = max_seq_length
        # Load the vectorizer
        self.vect = load_vectorizer(vectorizer_file)
        # And load the end tokens
        self.end_tokens = self.vect(['<.>', '<p>'])
        self.outdir = setup_output_dir(outdir)
        
        self.epoch = 0
        self.train_batch_interval = train_batch_interval
        
    def write_history(self, context={}):
        # Sentence start (comes from pre-trained vectorizer). If vectorizer
        # is re-fit, this will need to be modified
        # Corresponds to '<country> is a country located in'
        paragraph = np.array([11, 15, 8, 42, 398, 6])
        # Complete this sentence and predict however many more you want
        for i in range(self.num_sentences):
            print(f'generating sequence {i}')
            paragraph = self.model.finish_sentence(paragraph, self.end_tokens, 
                                                   self.seq_length, self.max_seq_length)
        
        # Convert output back to text
        text_output = self.vect.to_text(paragraph)
        text_output = text_output.replace(' <p> ', '\n\n')
        
        # Write the output sentence
        batch = context.get('batch', None)
        outname = f'EPOCH{self.epoch:03}_BATCH{batch:05}.txt' if batch else f'EPOCH{self.epoch:03}.txt'
        with open(str(Path(self.outdir, outname)), 'w+') as fp:
            fp.write(text_output)
        
    def on_epoch_end(self, epoch, logs=None):
        self.write_history({'batch': None})
        self.epoch += 1
    
    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % self.train_batch_interval == 0:
            self.write_history({'batch': batch + 1})
            
def make_padding_mask(x, invert=False, transformer=True):
    """
    Mask for postions in input tensor that are 0 (padding tokens that should be
    ignored by encoder and decoder).
    
    Parameters
    ----------
    x : tf.Tensor
        A Tensor of shape (bs, seq_length).
    """
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # Repeat mask along new dimension
    if transformer:
        mask = tf.stack([mask] * mask.shape[1], axis=1)
    if invert:
        return mask
    return 1 - mask