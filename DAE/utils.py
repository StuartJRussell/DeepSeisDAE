
# Import modules
import os
import glob
import pickle
import warnings
import numpy as np
import tensorflow as tf
from zipfile import ZipFile
from scipy.signal.windows import tukey



def save_obj(dictionary, filename):
    """
    Function saves dictionary object as pickle file.
    
    Parameters
    ----------
    dictionary : dict
        The dictionary to save.
    filename : str
        Full filename to write to.
    
    Returns
    -------
    None
    """
    
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f, protocol=0)



def load_obj(filename):
    """
    Function loads dictionary object from pickle file.
    
    Parameters
    ----------
    filename : str
        Full filename to read.
    
    Returns
    -------
    dict
        Data from file.
    """
    
    with open(filename, 'rb') as f:
        return pickle.load(f)



def str2bool(string):
    """
    Function returns a bool from a string of "True" or "False".
        
    Parameters
    ----------
    string : str
        Ideally either 'True' or 'False'.
    
    Returns
    -------
    bool
        True or False.
    """
    
    # Raise error if not going to work...
    if string.lower() not in ["true", "false"]:
        msg = "string is not True or False"
        raise ValueError(msg)
    
    # Otherwise let's go!
    else:
        if string.lower() == "true":
            return True
        elif string.lower() == "false":
            return False



def read_parfile(parfile, retraining=False):
    """
    Function reads in the parameter file, ensures everything that is required is 
    present, and returns all necessary values as dictionaries.
    
    Parameters
    ----------
    parfile : str
        Full path to the parameter file.
    retraining : bool, optional
        Whether this is a retraining parfile or not, defaults to False.
    
    Returns
    -------
    parameters : dict
        Dictionary of parameters.
    stft_parameters : dict
        Parameters for the STFT.
    """
    
    # Read in the parameter file
    print(f"Reading data input arguments from {parfile}.")
    parameters = readtxt(parfile)
    
    # Check that the parfile is good to go
    parameters = check_parfile(parameters, retraining=retraining)
    
    # Get the parameters for the STFT
    if not retraining:
        stft_parameters = dict(nfft=parameters['nfft'],
                               nperseg=parameters['nperseg'],
                               noverlap=None)
    else:
        stft_parameters = None
    
    return parameters, stft_parameters



def readtxt(filename):
    """
    Reads a file containing parameters. "#" are comments and will be ignored. 
    The parameter name and its value are seperated by "=". First mention the name 
    and afterwards the value. Example for a short file:

        # Here are some examples
        have_fun = True             # Whether to have fun or not
        output  = fun_record.npz    # Where to output the fun time series

    Parameters
    ----------
    filename : str
        Full path to the file to read.
    
    Returns
    -------
    parameters : dict
        Dictionary of parameters.
    """
    
    # Open file and creating empty dictionary
    parameters = {}
    with open(filename, "r") as fopen:
        
        # Read through each line in the file and output the parameters and values
        line = fopen.readline()
        while line:
            
            # Lines containing "#" in first values are ignored as these are comments
            if "=" in line and line.strip()[0] != '#':
                
                # Split between value name and rest
                param = line.split("=")
                # Name of the paramter
                name = param[0].strip()
                # Value of the paramter and strip following comments
                value = param[1].split("#")[0].strip()
                
                # Try to convert value into number
                # If there is a '.' then assume a float
                if '.' in value:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                # Otherwise try for an integer
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                
                # Otherwise value is of type string
                if isinstance(value, str) is True:
                    value = value.split("\n")[0]
                    
                    # Test for None
                    if value.lower() == 'none':
                        value = None
                    
                    # Try to convert value into bool
                    try:
                        value = str2bool(value)
                    except ValueError:
                        pass
                    
                    # Test for tuple
                    try:
                        value = tuple(map(int, value.split(', ')))
                    except:
                        pass
                        
                # Put the parameter into the output dictionary
                parameters.update({name: value})
                
            # And on we go...
            line = fopen.readline()
            
    return parameters



def check_parfile(parameters, retraining=False):
    """
    Function checks that the parameter file has all the required information.
    - Where these are not given and are needed then errors are raised.
    - Where these are not given and are optional then warnings are given and 
      values assumed. You know what assuming did...
        
    Parameters
    ----------
    parameters : dict
        Parameters read from the parfile.
    
    Returns
    -------
    parameters : dict
        Revised dictionary of parameters.
    
    Raises
    ------
    ValueError
        When required parameters are not found.
    """
    
    # Check for required parameters, these are hard-coded and are different if retraining
    if not retraining:
        required_keys = ['signal_pathname', 'noise_pathname', 'dt']
    else:
        required_keys = ['model', 'config', 'signal_pathname', 'noise_pathname']
    for key in required_keys:
        if key not in parameters.keys():
            msg = (f"Parameter `{key}' is a required parameter and is not" +
                    "currently specified in the parameter file")
            raise ValueError(msg)
    
    # Further check for keys required for STFT
    # Not required for retraining
    if not retraining:
        stft_keys = ['nfft', 'nperseg']
        for key in stft_keys:
            if key not in parameters.keys():
                msg = (f"Parameter `{key}' is a required parameter " +
                        "and is not currently specified in the parameter file")
                raise ValueError(msg)
    
    # Now we will go through the rest of the parameters and where they are not set in the 
    # paramter file we will assume some values.
    # Set the default values here, these are hard-coded and again are 
    # different if retraining
    default_values = {'model_output': './models/',
                      'config_output': './configs/',
                      'history_output': './histories/',
                      'figure_output': './figures/',
                      'checkpoints_output': './checkpoints/',
                      'num_signals': 'All',
                      'depth': 6,
                      'filter_root': 8,
                      'drop_rate': 0.1,
                      'strides': (2, 2),
                      'validation_split': 0.2,
                      'epochs': 200,
                      'learning_rate': 0.000001,
                      'batch_size': 32,
                      'patience': 10,
                      'noise_level': 1.0,
                      'noise_scale': 3.0,
                      'signal_scale': 3.0,
                      'verbose': 1,
                      'freeze_encoder':False}
    
    # If retraining then remove unnecessary values
    if retraining:
        superfluous_keys = ['drop_rate', 'filter_root', 'depth', 'strides']
        for key in superfluous_keys:
            default_values.pop(key, None)
            
    # If not retraining then remove unnecessary values
    if not retraining:
        superfluous_keys = ['freeze_encoder']
        for key in superfluous_keys:
            default_values.pop(key, None)
    
    # Now iterate through and set the values
    # Make sure to warn the user each time
    for key in default_values.keys():
        if key not in parameters.keys():
            msg = (f"Parameter `{key}' has not been set, assuming a " +
                   f"value of {default_values[key]}." +
                    " To change this, please specify a value in the parameter file.")
            warnings.warn(msg)
            parameters[key] = default_values[key]
    
    return parameters



def set_up_training_data(parameters):
    """
    Function sets up two lists of signal and noise files based on the paths
    in the parameters dictionary.
        
    Parameters
    ----------
    parameters : dict
        Parameters read from the parfile.
    
    Returns
    -------
    signal_files : list
        List of signal filepaths.
    noise_files : list
        List of noise filepaths.
    parameters : dict
        Parameters dictionary with any extra info added.
    """
    
    # Find the signal and noise files
    signal_files = glob.glob(f"{parameters['signal_pathname']}*.npz")
    noise_files = glob.glob(f"{parameters['noise_pathname']}*.npz")
    len_signal = len(signal_files)
    len_noise = len(noise_files)
    
    # If no files are found then raise errors
    if len_signal == 0:
        msg = f"Found no .npz for signal in path {parameters['signal_pathname']}."
        raise ValueError(msg)
    if len_noise == 0:
        msg = f"Found no .npz for noise in path {parameters['noise_pathname']}."
        raise ValueError(msg)
    
    # Get the size of the files and time series
    # Hopfully these are defined, if not then read a file.
    # Assumes all files are the same size - this will also be verified when 
    # each file is read during training
    if 'traces_per_file' in parameters.keys() and 'ts_length' in parameters.keys():
        traces_per_file = parameters['traces_per_file']
        ts_length = parameters['ts_length']
    else:
        traces_per_file, ts_length = npz_size(noise_files[0])
        parameters['traces_per_file'] = traces_per_file
        parameters['ts_length'] = ts_length
    
    # Put some rubbish through the logger
    print(f"  There are {len_signal} signal files, " +
              f"which is {len_signal * traces_per_file} signal traces.")
    print(f"  There are {len_noise} noise files, " +
              f"which is {len_noise * traces_per_file} noise traces.")
    
    # Shuffle the files
    np.random.shuffle(signal_files)
    np.random.shuffle(noise_files)
    
    # The total length of training data is length of the noise file list
    # If needed then we can repeat the synthetics as these are typically 
    # all rather similar
    len_training = len_noise
    while len_signal < len_noise:
        signal_files = np.append(signal_files, signal_files)
        len_signal = len(signal_files)
    
    # But the user may request fewer data if they want...
    if parameters['num_signals'] == 'All':
        number_of_files = len_training
    else:
        number_of_files = int(np.floor(parameters['num_signals'] / traces_per_file))
    
    # Take only the number of files that are needed/requested
    signal_files = signal_files[:number_of_files]
    noise_files = noise_files[:number_of_files]
    print(f"  Training will use {number_of_files} files, " +
            f"which is {number_of_files * traces_per_file} traces, each for both " +
            f"noise and signal.")
    
    # Check that the batch size is a factor of traces per file
    # At present this is a requirement, may change in the future
    # TODO think about doing this
    if traces_per_file % parameters['batch_size'] != 0:
        msg = (f"The batch size must be a factor of number of traces per file. " +
               f"The current values of {batch_size} and {traces_per_file} do " +
               f"not satisfy this.")
        raise ValueError(msg)
    
    return signal_files, noise_files, parameters



def npz_size(npzfile):
    """
    Function takes a .npz file and gets the size using zipfile. 
    
    This is faster than fully reading each file with np.load.
        
    Parameters
    ----------
    npzfile : str
        Full path to the .npz file to open.
    
    Returns
    -------
    shape : tuple
        Shape of the data in the .npz file.
    """
    
    with ZipFile(npzfile) as openzip:
        npy = openzip.open('data.npy')
        version = np.lib.format.read_magic(npy)
        shape, _, _ = np.lib.format._read_array_header(npy, version)
        return shape



def set_up_callbacks(patience, filename, pathname="./checkpoints"):
    """
    Function sets up the callbacks.
    
    Parameters
    ----------
    patience : int
        The patience during training, i.e. how many epochs to wait before
        ending training early.
    filename : str
        What to call the directory that contains the checkpoints.
    pathname : str, optional
        Directory where checkpoints are stored. Defaults to "./checkpoints".
    
    Returns
    -------
    callbacks : list of keras.callbacks
        Keras callbacks objects that can be passed to the model.
    """
    
    # Create directory for checkpoints
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    
    # Create callbacks
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                        patience=patience, verbose=1),
                 tf.keras.callbacks.ModelCheckpoint(
                        filepath=f"{pathname}/{filename}/latest_checkpoint.ckpt",
                        save_weights_only=True,
                        save_best_only=True,
                        savefreq="epoch", verbose=1)]
    
    return callbacks



def rms(x):
    """
    Function calculates the root-mean-square of array x.
        
    Parameters
    ----------
    x : numpy.array
        The array to calculate the RMS of.
    
    Returns
    -------
    float
        The calculated RMS of x.
    """
    
    # Remove mean
    x = x - np.mean(x)
    
    # Output RMS
    return np.sqrt(np.sum(x ** 2.) / x.shape[0])



def signal_to_noise_ratio(signal, noise, decibel=True):
    """
    Function calculates the SNR of two time series. Output in decibels is optional.

    Parameters
    ----------
    signal : numpy.array
        The signal time series.
    noise : numpy.array
        The noise time series.
    decibel : bool, optional
        Output in decibels or not, defaults to True.
    
    Returns
    -------
    float
        The calculated SNR.
    """
    
    if decibel is True:
        return 20. * np.log10(rms(signal) / rms(noise))
    else:
        return rms(signal) / rms(noise)



def taper_array(array, alpha=0.05):
    """
    Function applies an tukey (cosine) taper to an array.
        
    Parameters
    ----------
    array : numpy.array
        The array to apply the taper to.
    alpha : float, optional
        Shape parameter of the Tukey window, defaults to 0.05.
    
    Returns
    -------
    tapered_array : numpy.array
        The tapered array.
    """
    
    taper = tukey(len(array), alpha=alpha)
    tapered_array = array * taper
    
    return tapered_array



def shift_array(array, taper=True):
    """
    Function shifts numpy array randomly to the left or right. The array is 
    zero-padded appropriately to keep it the same length.
    
    The shift can be the entire length of the array, in which case most
    of the signal will have value zero and the DAE is given essentially pure
    noise.
    
    Parameters
    ----------
    array : numpy.array
        The array to shift,
    taper : bool, optional
        Whether to taper or not, defaults to False.
    
    Returns
    -------
    result : numpy.array
        The shifted and zero-padded array.
    """
    
    # Generate a random shift in samples
    shift = np.random.randint(-int(len(array) / 2), int(len(array) / 2))
    result = np.empty_like(array)
    
    # If shift is posivitve then shift to the right
    if shift > 0:
        result[:shift] = 0
        shifted_array = array[:-shift]
        if taper is True:
            result[shift:] = taper_array(shifted_array)
        else:
            result[shift:] = shifted_array
    
    # Otherwise shift to the left
    elif shift < 0:
        result[shift:] = 0
        shifted_array = array[-shift:]
        if taper is True:
            result[:shift] = taper_array(shifted_array)
        else:
            result[:shift] = shifted_array
    
    # And if neither then do not shift!
    else:
        result[:] = array

    return result



def cropping_layer(needed_shape, is_shape):
    diff1 = is_shape[0] - needed_shape[0]
    if diff1 % 2 == 0 and diff1 > 0:
        shape1 = (diff1//2, diff1//2)
    elif diff1 % 2 == 1 and diff1 > 0:
        shape1 = (diff1 // 2, (is_shape[0] - needed_shape[0]) - diff1 // 2)
    elif diff1 == 0:
        shape1 = (0, 0)
    
    diff2 = is_shape[1] - needed_shape[1]
    if diff2 % 2 == 0 and diff2 > 0:
        shape2 = (diff2//2, diff2//2)
    elif diff2 % 2 == 1 and diff2 > 0:
        shape2 = (diff2//2, (is_shape[1] - needed_shape[1]) - diff2 // 2)
    elif diff2 == 0:
        shape2 = (0, 0)
    
    return shape1, shape2
