
# Import modules
import os
import sys
import glob
import random
import shutil
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# Import DAE scripts
import DAE


def retraining(parfile):
    """
    Main function to read parameters from file to start retraining.
    
    Parameters
    ----------
    parfile : str
        Full path to the retraining parameter file
    
    Returns
    -------
    None
    """
    
    # Read in the parameter file
    parameters, _ = DAE.read_parfile(parfile, retraining=True)
    
    # Set up the training data
    signal_files, noise_files, parameters = DAE.set_up_training_data(parameters)
    
    # Create callbacks
    callbacks = DAE.set_up_callbacks(patience=parameters['patience'], 
                                     filename=parameters['filename'],
                                     pathname=parameters["checkpoints_output"])
    
    # Create optimizer
    # This could be different to the original to utilise a better learning rate
    optimizer = tf.keras.optimizers.Adam(
                                learning_rate=float(parameters['learning_rate']),
                                beta_1=0.9, beta_2=0.999, 
                                epsilon=1e-7, amsgrad=False,
                                name='Adam')

    # Load config file to initialize tensorflow model
    config = DAE.load_obj(parameters['config'])

    # Initialize TF model
    channels = config['channels'] if 'channels' in config.keys() else 2
    m = DAE.Model(name=parameters["filename"],
                  channels=channels,
                  ts_length=config['ts_length'], 
                  dt=config['dt'],
                  optimizer=optimizer,
                  callbacks=callbacks,
                  stft_parameters=config['stft_parameters'])

    # Load previoulsy trained model
    m.model = load_model(parameters['model'], compile=False)
    m.model.summary()        
    
    # Start the training process
    m.train_model(signal_files=signal_files,
                  noise_files=noise_files,
                  traces_per_file=parameters['traces_per_file'],
                  noise_level=parameters['noise_level'],
                  noise_scale=parameters['noise_scale'],
                  signal_scale=parameters['signal_scale'],
                  epochs=parameters["epochs"],
                  batch_size=parameters['batch_size'],
                  verbose=parameters['verbose'],
                  validation_split=parameters['validation_split'],
                  freeze_encoder=parameters['freeze_encoder'],
                  retraining=True)

    # Save model and plot its history
    m.save_all(pathname_model=parameters["model_output"],
               pathname_config=parameters["config_output"],
               pathname_history=parameters["history_output"],
               pathname_figures=parameters["figure_output"],
               plot_history=False)

    print(f"Training is complete, enjoy the rest of your day")


if __name__ == "__main__":
    
    # Assign the parameter file
    if len(sys.argv) <= 1:
        parfile = "./retrain_parfile"
    elif len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) is False:
        msg = (f"The given paramter file {sys.argv[1]} does not exist.")
        raise FileNotFoundError(msg)
    else:
        parfile = sys.argv[1]
    
    # Start to run training with parameters from parfile
    retraining(parfile=parfile)
