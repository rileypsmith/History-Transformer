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
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tqdm import tqdm

from scraper import preprocess

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
    def __init__(self, max_vocab_size=10_000, mapping={}, counts={}):
        self.mapping = mapping
        self.counts = counts
        self.max_vocab_size = max_vocab_size
        
    def sort_counts(self):
        """Sort the dictionary of counts"""
        tmp = sorted([(k, v) for k, v in self.counts.items()], 
                     key=lambda x: x[1], reverse=True)
        self.counts = {k: v for k, v in tmp}
        
    def make_mapping(self):
        """Turn the dictionary of counts into a mapping and inverse mapping"""
        self.sort_counts()
        mapping = {}
        for i, word in enumerate(self.counts):
            if i >= self.max_vocab_size:
                break
            mapping[word] = i
        self.mapping = mapping
        
    def invert_mapping(self):
        """Invert the mapping so we can go from network outputs back to text"""
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.inverse_mapping[self.max_vocab_size] = '<unk>'
        
    def adapt(self, data):
        unique, counts = np.unique(data, return_counts=True)
        sort_idx = np.argsort(counts).ravel()[::-1]
        # return sort_idx
        unique = unique[sort_idx][:self.max_vocab_size - 1]
        counts = counts[sort_idx][:self.max_vocab_size - 1]
        for i, item in enumerate(unique):
            if item in self.counts:
                self.counts[item] += counts[i]
            else:
                self.counts[item] = counts[i]
                
        # Convert counts to mapping
        self.make_mapping()
        self.invert_mapping()
            
    def to_text(self, data):
        """
        Convert the incoming data (sequence of integers) back into text.
        """
        out = []
        for i in data:
            out.append(self.inverse_mapping[i])
        return ' '.join(out)
            
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

def fit_vectorizer_history(input_dirs, vocab_size=10_000, outpath=None):
    """
    Fit a text vectorization to the entire history dataset, then save its
    weights to the given outpath so that it can be loaded later and train and
    evaluation time.
    
    Parameters
    ----------
    input_dirs : list
        A list of directories containing text.
    vocab_size : int
        The maximum vocabulary size to use for the text vectorizer.
    outpath : str
        Optional path to save the vectorizer to.
    """
    # First, load all the text into memory
    all_text = load_all_text(input_dirs)
    # Concatenate all the text into a single giant ndarray
    all_text = np.concatenate(all_text, axis=0)
    # Build TextVectorization layer
    vectorizer = TextVectorization(vocab_size)
    # Adapt it to the dataset
    vectorizer.adapt(all_text)
    # Optionally save it
    if outpath is not None:
        joblib.dump(vectorizer, outpath)
    return vectorizer

def fit_vectorizer_lm1b(vocab_size=10_000, outpath=None):
    """
    Fit a TextVectorizer class to the contents of the lm1b dataset.
    
    Parameters
    ----------
    vocab_size : int
        The maximum vocabulary size to use for the text vectorizer.
    outpath : str
        Optional path to save the vectorizer to.
    """
    # Load the dataset (from tensorflow datasets)
    ds = tfds.load('lm1b', split='train')
    # Build a vectorizer
    vectorizer = TextVectorization(vocab_size)
    # Use the first 20,000 examples to fit the vectorizer
    for i, example in tqdm(enumerate(ds), total=20_000):
        if i >= 20_000:
            break
        # Preprocess the example
        text = preprocess(example['text'].numpy().decode('utf-8'))
        text = np.array(text.split(' '))
        # Adapt vectorizer
        vectorizer.adapt(text)
    # Optionally save it
    if outpath is not None:
        joblib.dump(vectorizer, outpath)
    return vectorizer
        
def fit_combined_vectorizer(input_dirs, history_vocab_size=1000, 
                            total_vocab_size=2000, outpath=None):
    """
    Fit a text vectorization instance to the dataset and save it to a file.
    """
    # Fit to history dataset
    history_vect = fit_vectorizer_history(input_dirs, history_vocab_size)
    # Fit to lm1b dataset
    lm1b_vect = fit_vectorizer_lm1b(total_vocab_size)
    # Borrow words from lm1b dataset to fill vocab size
    n_lm1b = total_vocab_size - history_vocab_size
    tmp = sorted([k for k in lm1b_vect.mapping.keys()],
                 key=lambda x: lm1b_vect.counts[x], reverse=True)
    lm1b_words = []
    for word in tmp:
        if word not in history_vect.mapping:
            lm1b_words.append(word)
        if len(lm1b_words) >= n_lm1b:
            break

    # Make combined mapping
    all_keys = list(history_vect.mapping.keys()) + lm1b_words
    mapping = {word: i + 1 for i, word in enumerate(all_keys)}
    vectorizer = TextVectorization(total_vocab_size, mapping)
    vectorizer.invert_mapping()
    
    if outpath is not None:
        joblib.dump(vectorizer, outpath)
    return vect

def split_sequence(sequence, seq_length, min_seq_length=8):
    """
    Take a sequence of words and split it into many padded sequences. This
    way it can be fed into an LSTM.
    """
    sequences = []
    labels = []
    for i in range(min_seq_length, len(sequence) - 1):
        subsequence = sequence[:i+1]
        pad_dims = (0, seq_length - (i + 1))
        sequences.append(np.pad(np.array(subsequence), pad_dims))
        labels.append(sequence[i + 1])
    return np.stack(sequences, axis=0), np.array(labels)

def make_transformer_dataset(data, batch_size=16, shuffle=True):
    """Turn the incoming data (ndarray) into a Tensorflow Dataset object"""
    ds = tf.data.Dataset.from_tensor_slices(data)
    # Apply function to get training example and label, offset by one position
    data_ds = ds.map(lambda x: x[:-1])
    label_ds = ds.map(lambda x: x[1:])
    ds = tf.data.Dataset.zip((data_ds, label_ds))
    ds = ds.batch(batch_size)
    if shuffle:
        ds = ds.shuffle(2000, reshuffle_each_iteration=True)
    ds = ds.prefetch(1000)
    return ds

def make_lstm_dataset(data, seq_length, batch_size=16, shuffle=True, quiet=False):
    """Turn the incoming data (ndarray) into a Tensorflow Dataset object"""
    # Convert each sequence into a split sequence
    print('Making LSTM dataset')
    iterator = data if quiet else tqdm(data)
    all_sequences = [split_sequence(sequence, seq_length) for sequence in iterator]
    print('\t -> Concatenating')
    data = np.concatenate([s[0] for s in all_sequences], axis=0)
    labels = np.concatenate([np.array(s[1]) for s in all_sequences], axis=0)
    # Do an initial shuffle
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx]
    labels = labels[idx]
    print('\t -> Turning into dataset')
    data_ds = tf.data.Dataset.from_tensor_slices(data)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((data_ds, label_ds))
    print('\t -> Batching, shuffling, and prefetching')
    ds = ds.batch(batch_size)
    if shuffle:
        ds = ds.shuffle(1000, reshuffle_each_iteration=True)
    ds = ds.prefetch(1000)
    return ds

def load_datasets(input_dirs, vectorizer_file, seq_length=128, spacing=64,
                  quiet=False, random_seed=1234, lstm=False, **kwargs):
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
    
    # If fetching transformer data, add 1 to seq length so that you can shift by 1 when training
    if not lstm:
        seq_length += 1

    # Turn text into usable data
    iterator = all_text if quiet else tqdm(all_text)
    for paragraph in iterator:
        for start_idx in range(0, len(paragraph) - seq_length, spacing):
            # Extract a sequence of text and vectorize it
            text_vector = vect(paragraph[start_idx : start_idx + seq_length],
                               seq_length=seq_length)
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
    if lstm:
        train_ds = make_lstm_dataset(train_data, seq_length, quiet=quiet, **kwargs)
        val_ds = make_lstm_dataset(val_data, seq_length, quiet=quiet, **kwargs)
    else:
        train_ds = make_transformer_dataset(train_data, **kwargs)
        val_ds = make_transformer_dataset(val_data, **kwargs)
    
    return train_ds, val_ds
    
def get_end_tokens(vectorizer_file):
    """Return the indices for end tokens, <.> and <p>"""
    vect = load_vectorizer(vectorizer_file)
    return vect(['<.>', '<p>'])

def prep_lm1b(item, vect, seq_length):
    return vect(preprocess(x['text'].numpy().decode('utf-8')))

def make_lm1b_dataset(vectorizer_file, seq_length=64, batch_size=16):
    vect = load_vectorizer(vectorizer_file)
    ds = tfds.load('lm1b', split='train')
    ds = ds.map(lambda x: tf.py_function(prep_lm1b, inp=[x, vect, seq_length], Tout=tf.int32))
    # ds = ds.map(lambda x: preprocess(x['text'].numpy().decode('utf-8')))
    # ds = ds.map(lambda x: vect(x, seq_length))
    ds = ds.batch(batch_size)
    ds = ds.shuffle(1000)
    ds = ds.prefetch()
    return ds
    
class LM1BDataset():
    def __init__(self, vectorizer_file, seq_length=48, batch_size=16, 
                 shuffle=True):
        # Load vectorizer
        self.vect = load_vectorizer(vectorizer_file)
        self.seq_length = seq_length
        
        # Load lm1b dataset
        ds = tfds.load('lm1b', split='train')
        ds = ds.batch(batch_size)
        if shuffle:
            ds = ds.shuffle(1000)
        self.ds = iter(ds.prefetch(1000))
    
    def preprocess(self, sentence):
        text = preprocess(sentence.numpy().decode('utf-8'))
        return self.vect(text, self.seq_length)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batch = next(self.ds)
        preprocessed = [self.preprocess(sentence) for sentence in batch['text']]
        return tf.stack(preprocessed, axis=0)
        


    