
# Import modules
import os
import sys
import glob
import random
import warnings
import numpy as np
import tensorflow as tf

# Import DAE scripts
import DAE

def training(parfile):
    """
    Main script to start training of the denoising autoencoder.

    Parameters
    ----------
    parfile : str
        Full path to the parameter file.
    
    Returns
    -------
    None
    """
    
    # Read in the parameter file
    parameters, stft_parameters = DAE.read_parfile(parfile)
    
    # Set up the training data
    signal_files, noise_files, parameters = DAE.set_up_training_data(parameters)
    
    # Create callbacks
    callbacks = DAE.set_up_callbacks(patience=parameters['patience'], 
                                     filename=parameters['filename'],
                                     pathname=parameters["checkpoints_output"])
    
    # Create optimizer 
    optimizer = tf.keras.optimizers.Adam(
                            learning_rate=parameters['learning_rate'],
                            beta_1=0.9, beta_2=0.999, 
                            epsilon=1e-7, amsgrad=False,
                            name='Adam')
     
    # 2. Create Model
    m = DAE.Model(name=parameters["filename"],
                  ts_length=parameters['ts_length'],
                  optimizer=optimizer,
                  dt=parameters['dt'],
                  callbacks=callbacks,
                  stft_parameters=stft_parameters,
                  filter_root=parameters['filter_root'], 
                  depth=parameters['depth'],
                  strides=parameters['strides'],
                  drop_rate=parameters['drop_rate'],
                  kernel_regularizer=tf.keras.regularizers.L2(0))
    
    # 3. Build model
    m.build_model()
    
    # 4. Start training
    m.train_model(signal_files=signal_files,
                  noise_files=noise_files,
                  traces_per_file=parameters['traces_per_file'],
                  noise_level=parameters['noise_level'],
                  noise_scale=parameters['noise_scale'],
                  signal_scale=parameters['signal_scale'],
                  batch_size=parameters['batch_size'],
                  epochs=parameters["epochs"],
                  validation_split=parameters['validation_split'],
                  verbose=parameters['verbose'])
    
    # 5. Save model and plot its history
    m.save_all(pathname_model=parameters["model_output"],
               pathname_config=parameters["config_output"],
               pathname_history=parameters["history_output"],
               pathname_figures=parameters["figure_output"],
               plot_history=False)
    
    print(f"Training is complete, enjoy the rest of your day")



if __name__ == "__main__":
    
    # Assign the parameter file
    if len(sys.argv) <= 1:
        parfile = "./parfile"
    elif len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) is False:
        msg = (f"The given paramter file {sys.argv[1]} does not exist.")
        raise FileNotFoundError(msg)
    else:
        parfile = sys.argv[1]
    
    # Start to run training with parameters from parfile
    training(parfile=parfile)
