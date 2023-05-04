"""
A training pipeline for the history transformer.

@author: rileypsmith
Created: 5/2/2023
"""
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import callbacks

from data import load_datasets, TextVectorization
from transformer import Transformer
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
    alpha=0.1,
    epochs=100,
    batch_size=16,
    vocab_size=10_000,
    seq_length=128,
    verbose=True
):
    """Main training function"""
    
    # Setup output directory
    output_dir = utils.setup_output_dir(output_dir)
    logfile = str(Path(output_dir, 'training_log.csv'))
    modelfile = str(Path(output_dir, 'weights_{epoch:02d}.hdf5'))
    
    # Load data
    train_ds, val_ds = load_datasets(input_dirs, vectorizer_file, batch_size=batch_size)
    
    # Define loss function
    loss_fn = utils.LabelSmoothingSCC(alpha)
    
    # Setup optimizer
    opt = tf.keras.optimizers.Adam()
    
    # Build model
    model = Transformer(vocab_size + 1, seq_length)
    model.compile(optimizer=opt)
    
    # Setup callbacks
    lr_scheduler = callbacks.LearningRateScheduler(
        utils.CosineAnnealingLR()
    )
    lr_scheduler.set_model(model)
    model_checkpoint = callbacks.ModelCheckpoint(modelfile, save_best_only=True,
                                                 save_weights_only=True)
    model_checkpoint.set_model(model)
    my_callbacks = [
        utils.CustomCSVLogger(logfile),
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
    for epoch in range(epochs):
        print(f'EPOCH {epoch+1}')
        
        # Update callbacks
        callback.on_epoch_begin(epoch)
        
        for batch_num, (data, labels) in enumerate(train_ds):
            callback.on_train_batch_begin(batch_num)
            
            # Do one step of gradient descent
            loss = train_step(model, opt, data, labels, loss_fn)
            loss_tracker.update_state(loss)
            
            # Pass logs to callbacks
            logs = {'loss': loss_tracker.result()}
            callback.on_train_batch_end(batch_num, logs)
        
        # Run validation
        loss_tracker.reset_states()
        for batch_num, (data, labels) in enumerate(val_ds):
            # Get validation loss for this batch
            val_loss = val_step(model, data, labels, loss_fn)
            loss_tracker.update_state(val_loss)
        
        # Store epoch level metrics
        logs = {'val_loss': loss_tracker.result()}
        callback.on_epoch_end(epoch, logs)
        loss_tracker.reset_states()

if __name__ == '__main__':
    input_dirs = [
        '/home/rpsmith/NLP/History-Transformer/data/country_data/',
        '/home/rpsmith/NLP/History-Transformer/data/history_data/'
    ]
    output_dir = '/home/rpsmith/NLP/History-Transformer/output/20230503'
    vectorizer_file = '/home/rpsmith/NLP/History-Transformer/trained_models/text_vectorizer.joblib'
    train(input_dirs, output_dir, vectorizer_file, batch_size=32, verbose=False)
    
            
            
            
    
    
        

