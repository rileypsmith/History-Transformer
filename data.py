"""
Data handling for training a history transformer.

@author: rileypsmith
Created: 4/26/2023
"""
from pathlib import Path
import pickle

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

def load_vectorizer(filepath):
    """Load a TextVectorization layer from a previously saved path"""
    return joblib.load(filepath)

def load_all_text(input_dirs):
    """
    From a list of input directories, load all text into memory. Maintain
    paragraph distinction.
    """
    all_text = []
    for top_level_dir in input_dirs:
        subdirs = sorted(list(Path(top_level_dir).iterdir()))
        for country_dir in subdirs:
            # Loop over paragraphs for this country
            for textfile in sorted(list(country_dir.glob('*.txt'))):
                with open(textfile, 'r') as fp:
                    # Read the text and convert to ndarray
                    local_text = fp.read().strip().split(' ')
                    all_text.append(np.array(local_text, dtype=str))
    # all_text is now a list of arrays, each one containing text for a paragraph
    return all_text

class TextVectorization():
    """A knockoff of the Tensorflow class since I am using TF 2.4"""
    def __init__(self, max_vocab_size=10_000):
        self.mapping = {}
        self.max_vocab_size = max_vocab_size
        
    def adapt(self, data):
        unique, counts = np.unique(data, return_counts=True)
        sort_idx = np.argsort(counts).ravel()[::-1]
        # return sort_idx
        unique = unique[sort_idx][:self.max_vocab_size - 1]
        for i, item in enumerate(unique):
            self.mapping[item] = i + 1
            
    def __call__(self, data, seq_length=None):
        """
        Convert the data to a sequence of integers. Data may be either a string
        or a Numpy array. If seq_length is specified, this will truncate the
        end of the string to make it fit or pad with zeros if too short.
        """
        if isinstance(data, str):
            data = np.array([word for word in data.split(' ')])
        indices = np.array([self.mapping.get(word, self.max_vocab_size) for word in data])
        
        # Handle sequence length if given
        if seq_length is not None:
            if len(indices) > seq_length:
                indices = indices[:seq_length]
            else:
                pad_dims = [(0, seq_length - len(indices))]
                indices = np.pad(indices, pad_dims)
        
        return indices

def fit_vectorizer(input_dirs, outpath, **kwargs):
    """
    Fit a text vectorization to the entire history dataset, then save its
    weights to the given outpath so that it can be loaded later and train and
    evaluation time.
    
    Parameters
    ----------
    input_dirs : list
        A list of directories containing text.
    outpath : str
        Path to save the text vectorizer to. Should be a .joblib file.
    """
    # First, load all the text into memory
    all_text = load_all_text(input_dirs)
    
    # Concatenate all the text into a single giant ndarray
    all_text = np.concatenate(all_text, axis=0)
    
    # Build TextVectorization layer
    vectorizer = TextVectorization(**kwargs)
    
    # Adapt it to the dataset
    vectorizer.adapt(all_text)
    
    # Save it
    joblib.dump(vectorizer, outpath)
    
    return vectorizer

def make_dataset(data, batch_size=16, shuffle=True):
    """Turn the incoming data (ndarray) into a Tensorflow Dataset object"""
    ds = tf.data.Dataset.from_tensor_slices(data)
    # Apply function to get training example and label, offset by one position
    data_ds = ds.map(lambda x: x[:-1])
    label_ds = ds.map(lambda x: x[1:])
    ds = tf.data.Dataset.zip((data_ds, label_ds))
    ds = ds.batch(batch_size)
    if shuffle:
        ds = ds.shuffle(1000, reshuffle_each_iteration=True)
    ds = ds.prefetch(1000)
    return ds

def load_datasets(input_dirs, vectorizer_file, seq_length=128, spacing=10,
                  quiet=False, random_seed=1234, **kwargs):
    """
    Load all the available text data into memory and turn it into a Tensorflow
    Dataset object for training on.
    """
    # Load all the text into memory
    all_text = load_all_text(input_dirs)
    
    # Load text vectorizer
    vect = load_vectorizer(vectorizer_file)
    
    # Container for all data examples
    data_container = []
    
    # Turn text into usable data
    iterator = all_text if quiet else tqdm(all_text)
    for paragraph in iterator:
        for start_idx in range(0, len(paragraph) - (seq_length + 1), spacing):
            # Extract a sequence of text and vectorize it
            text_vector = vect(paragraph[start_idx : start_idx + seq_length + 1],
                               seq_length=(seq_length + 1))
            data_container.append(text_vector)
    data_container = np.stack(data_container, axis=0)
            
    # Randomly separate into train and validation data
    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(data_container))
    rng.shuffle(indices)
    num_train = int(0.8 * len(indices))
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    train_data = data_container[train_indices]
    val_data = data_container[val_indices]
    
    # Build training and validation datasets
    train_ds = make_dataset(train_data, **kwargs)
    val_ds = make_dataset(val_data, **kwargs)
    
    return train_ds, val_ds
    
    
    
    