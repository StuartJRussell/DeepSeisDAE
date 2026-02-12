
# Import modules
import numpy as np
import tensorflow as tf
from keras.models import load_model
from scipy.signal import stft, istft

from .utils import load_obj


def denoise(model_filename, config_filename, data_list):
    """
    Function to predict data in data_list.
    
    The input data to denoise should have the same length and sampling rate 
    that was set during  the training process. The function makes no assumptions
    about how traces of incorrect length should be handled, and instead an error
    is raised.
    
    Parameters
    ----------
    model_filename : str
        Full path to the .h5 model file.
    config_filename : str
        Full path to trhe corresponding .config file.
    data_list : list
        List of traces (most usually in numpy.array format) to denoise.
    
    Returns
    -------
    recovered : numpy.array
        The recovered signal and noise, where recovered[:,:,0] is the signal and 
        recovered [:,:,1] is the noise.
    
    Raises
    ------
    ValueError
        When the input data has no length.
    ValueError
        When a trace does not have the correct input length.
    """
    
    # Check that the list to denoise is not length zero
    if len(data_list) == 0:
        msg = "There are no data available for prediction."
        raise ValueError(msg)

    # Read in config file
    config = load_obj(config_filename)
    fs = 1. / config['dt']
    stft_parameters = config['stft_parameters']
    channels = config['channels'] if 'channels' in config.keys() else 2

    # Read in trained model
    model_dae = load_model(model_filename, compile=False)
    input_shape = (model_dae.input_shape[1], model_dae.input_shape[2])

    # Allocate empty arrays for data
    X = np.empty(shape=(len(data_list), *input_shape, channels), dtype="float")
    transform_list = np.empty(shape=(len(data_list), *input_shape, 3), dtype="complex") 

    # Loop over each data array in data_list and transform to time-frequency 
    # domain for denoising
    mean_values = []; norm_factors = []
    for i, array in enumerate(data_list):
    
        # Check size of signal and compare with input shape for prediction
        if len(array) != config['ts_length']:
            msg = (f"The shape of your input signal is not correct. " +
                   f"{config['ts_length']} samples are required " +
                   f"instead of {len(array)}.")
            raise ValueError(msg)
        
        # Take the mean of the trace
        mean = np.mean(array)
        mean_values.append(mean)
        
        # Demean the trace
        signal = array - mean
        
        # Normalise the trace
        # If trace is all zeros then keep as is
        norm = np.max(np.abs(signal))
        if norm == 0:
            norm = 1
        signal = signal / norm
        norm_factors.append(norm)
        
        # Take STFT of the trace
        freqs, _, cns = stft(signal, fs=fs, **stft_parameters)
        
        # Add transform to transform_list
        transform_list[i, :, :, 0] = cns
        
        # Write the real and imaginary parts of the data to empty arrays
        X[i, :, :, 0] = cns.real / np.max(np.abs(cns.real))
        X[i, :, :, 1] = cns.imag / np.max(np.abs(cns.imag))
        
    # Denoise data by prediction with model
    X_pred = model_dae.predict(X, verbose=0)

    # Clear keras session to limit memory usage
    tf.keras.backend.clear_session()
    
    # Loop over each element in predicted data and estimate denoised data
    recovered = np.empty(shape=(len(data_list), len(data_list[0]), 2), dtype="float")
    for i in range(X_pred.shape[0]):
        
        # Apply masks to get to recovered signal and noise
        transform_list[i, :, :, 1] = (transform_list[i, :, :, 0] * 
                            X_pred[i, :, :, 0])
        transform_list[i, :, :, 2] = (transform_list[i, :, :, 0] * 
                            X_pred[i, :, :, 1])
        
        # Transform back into time domain
        t, rec_signal = istft(transform_list[i, :, :, 1], 
                                        fs=fs, **stft_parameters)
        t, rec_noise = istft(transform_list[i, :, :, 2], 
                                        fs=fs, **stft_parameters)
        
        # Multiply denoised trace by normalization factor to recover amplitude
        # prior normalization
        rec_signal = np.real(rec_signal * norm_factors[i])
        rec_noise = np.real(rec_noise * norm_factors[i])
        
        # Add mean back on to both signal and noise
        rec_signal += mean_values[i]
        rec_noise += mean_values[i]
        
        # Append denoised traces to list
        recovered[i, :, 0] = rec_signal
        recovered[i, :, 1] = rec_noise
    
    return recovered



