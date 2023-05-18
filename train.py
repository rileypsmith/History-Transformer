"""
A training pipeline for the history transformer.

@author: rileypsmith
Created: 5/2/2023
"""
from pathlib import Path
import time

import tensorflow as tf
from tensorflow.keras import callbacks

from data import load_datasets, TextVectorization, get_end_tokens
from transformer import Transformer
from lstm import LSTMModel
import utils
    
def train_step(model, opt, data, labels, loss_fn):
    """Do one step of gradient descent"""
    with tf.GradientTape() as tape:
        # Forward pass
        preds = model(data, labels)
        # Compute loss
        loss = loss_fn(labels, preds)
        # Compute gradients
        grad = tape.gradient(loss, model.trainable_variables)
        # Backpropogate
        opt.apply_gradients(zip(grad, model.trainable_variables))
    return loss

def val_step(model, data, labels, loss_fn):
    """Do one step of evaluation (just don't do backpropogation)"""
    preds = model(data, labels)
    loss = loss_fn(labels, preds)
    return loss

def train(
    input_dirs,
    output_dir,
    vectorizer_file,
    epochs=100,
    batches_per_epoch=-1,
    batch_size=16,
    vocab_size=10_000,
    seq_length=128,
    verbose=True,
    model_type=LSTMModel,
    lstm=False,
    optimizer='adam',
    initial_lr=1e-3,
    label_smoothing=0.1,
    **model_kwargs
):
    """Main training function"""
    
    # Setup output directory
    output_dir = utils.setup_output_dir(output_dir)
    logfile = str(Path(output_dir, 'training_log.csv'))
    modelfile = str(Path(output_dir, 'weights_{epoch:02d}.hdf5'))
    sample_outdir = str(Path(output_dir, 'fake_history_samples'))
    
    # Load data
    train_ds, val_ds = load_datasets(input_dirs, vectorizer_file, seq_length=seq_length, 
                                     lstm=lstm, batch_size=batch_size, quiet=False)
    
    # Define loss function
    loss_fn = utils.LabelSmoothingSCC(label_smoothing)
    
    # Setup optimizer
    opt_config = {'class_name': optimizer, 'config': {'learning_rate': initial_lr}}
    opt = tf.keras.optimizers.get(opt_config)
    
    # Build model
    model = model_type(vocab_size + 1, **model_kwargs)
    model.compile(optimizer=opt)
    
    # Setup callbacks
    lr_scheduler = callbacks.LearningRateScheduler(
        utils.CosineAnnealingLR()
    )
    lr_scheduler.set_model(model)
    model_checkpoint = callbacks.ModelCheckpoint(modelfile, save_best_only=True,
                                                 save_weights_only=True)
    model_checkpoint.set_model(model)
    sample_writer = utils.SampleHistoryWriter(vectorizer_file, sample_outdir,
                                              seq_length)
    sample_writer.set_model(model)
    my_callbacks = [
        utils.CustomCSVLogger(logfile),
        sample_writer,
        model_checkpoint,
        lr_scheduler
    ]
    if verbose:
        progbar = callbacks.ProgbarLogger('steps')
        progbar.set_params({'steps': len(train_ds), 'verbose': 1, 'epochs': epochs})
        my_callbacks.append(progbar)
    callback = callbacks.CallbackList(my_callbacks)
    
    # Setup metrics
    loss_tracker = utils.FadingMemoryMean()
    
    # Main training loop
    callback.on_train_begin()
    start = time.time()
    for epoch in range(epochs):
        print(f'EPOCH {epoch+1}')
        
        # Update callbacks
        callback.on_epoch_begin(epoch)
        
        for batch_num, (data, labels) in enumerate(train_ds):
            # Optionally end the epoch early (to speed up getting validation results)
            if (batches_per_epoch > 0) and (batch_num >= batches_per_epoch):
                break
            
            # Start callbacks
            callback.on_train_batch_begin(batch_num)
            
            # Do one step of gradient descent
            loss = train_step(model, opt, data, labels, loss_fn)
            loss_tracker.update_state(loss)
            
            # Pass logs to callbacks
            elapsed = round((time.time() - start) / 60, 2)
            logs = {'loss': loss_tracker.result(), 'time': elapsed}
            callback.on_train_batch_end(batch_num, logs)
        
        # Run validation
        loss_tracker.reset_states()
        for batch_num, (data, labels) in enumerate(val_ds):
            if (batches_per_epoch > 0) and (batch_num >= batches_per_epoch):
                break
            # Get validation loss for this batch
            val_loss = val_step(model, data, labels, loss_fn)
            loss_tracker.update_state(val_loss)
        
        # Store epoch level metrics
        elapsed = round((time.time() - start) / 60, 2)
        logs = {'val_loss': loss_tracker.result(), 'time': elapsed}
        callback.on_epoch_end(epoch, logs)
        loss_tracker.reset_states()

if __name__ == '__main__':
    # Where data is located
    input_dirs = [
        '/home/rpsmith/NLP/History-Transformer/data/country_data/',
        '/home/rpsmith/NLP/History-Transformer/data/history_data/'
    ]
    
    # Pre-fit text vectorizer
    vectorizer_file = '/home/rpsmith/NLP/History-Transformer/trained_models/text_vectorizer_1000.joblib'
    
    # Where to write outputs
    output_dir = '/home/rpsmith/NLP/History-Transformer/output/20230517_LSTM'
    # output_dir = '/home/rpsmith/NLP/History-Transformer/output/TEST'
    
    # Training call for Transformer
    # train(input_dirs, output_dir, vectorizer_file, batch_size=2, verbose=False,
    #       d_k=768, d_v=768, heads=12, hidden_dim=768, n_encoder_layers=12, 
    #       n_decoder_layers=12, model_type=Transformer)
    
    # Model parameters
    model_kwargs = {
        'num_layers': 5,
        'num_dense_layers': 5,
        'lstm_units': 64,
        'embedding_dim': 128,
        'bidirectional': True
    }
    
    # Training call for LSTM
    train(input_dirs, output_dir, vectorizer_file, batch_size=16, verbose=False,
          model_type=LSTMModel, vocab_size=1000, lstm=True, seq_length=128, 
          label_smoothing=0.02, optimizer='rmsprop', initial_lr=0.01, **model_kwargs)
            
            
            
    
    
        

