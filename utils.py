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
        y_true = tf.cast(tf.one_hot(y_true, y_pred.shape[-1]), tf.float32)
        # Now apply Tensorflow's categorical crossentropy with label smoothing
        return self.loss(y_true, y_pred)
    
class CustomCSVLogger(callbacks.Callback):
    """
    A simple adaptation of the CSVLogger from Tensorflow. This one logs out
    intermediate results (not just every epoch).
    """
    def __init__(self, filepath, batch_interval=20, **kwargs):
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