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

def fix_file(filepath, country_name):
    with open(filepath, 'r') as fp:
        fixed = re.sub(r' +', ' ', fp.read())
        # fixed = fp.read().lower().replace('  ', '')
    with open(filepath, 'w+') as fp:
        fp.write(fixed)

from tqdm import tqdm
def fix_all(base_dir):
    for country_dir in tqdm(sorted(list(Path(base_dir).iterdir()))):
        country_name = country_dir.stem
        for f in country_dir.glob('*.txt'):
            fix_file(str(f), country_name)

# def save_vectorizer(vectorizer, outpath):
#     """Save the config for a TextVectorization layer so it can be re-loaded later"""
#     with open('outpath', 'wb+') as fp:
#         pickle.dump({'config': vectorizer.get_config(),
#                     'weights': vectorizer.get_weights()}, fp)

# def load_vectorizer(filepath):
#     """Load a TextVectorization layer from a previously saved path"""
#     specs = pickle.load(filepath)
#     vectorizer = layers.TextVectorization.from_config(specs['config'])
#     vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["dummy"]))
#     vectorizer.set_weights(specs['weights'])
#     return vectorizer
def load_vectorizer(filepath):
    """Load a TextVectorization layer from a previously saved path"""
    return joblib.load(filepath)

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
            
    def save(self, file):
        joblib.dump(self, file)
            
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
    
    # Concatenate all the text into a single giant ndarray
    all_text = np.concatenate(all_text, axis=0)
    
    # Build TextVectorization layer
    vectorizer = TextVectorization(**kwargs)
    
    # Adapt it to the dataset
    vectorizer.adapt(all_text)
    
    # Save it
    joblib.dump(vectorizer, outpath)
    
    return vectorizer