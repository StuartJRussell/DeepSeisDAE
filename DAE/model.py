
# Import modules
import os
import subprocess
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.signal import stft, istft
from tensorflow.keras.utils import Sequence
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model as TFmodel
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, \
                                    Dropout, Conv2DTranspose, Cropping2D, \
                                    Softmax, Add, LeakyReLU

from .losses import BCE2
from .utils import cropping_layer, save_obj, shift_array

# Set the global precision policy to float32
# This means that the resulting models can be used efficiently on a
# local machine. Depending on your training processor it may be better
# to use float16.
mixed_precision.set_global_policy('float32')



class Model:
    """
    Class that contains all of the necessary metadata, parameters and methods
    to train the DAE.
    """
    def __init__(self, name, ts_length, dt, stft_parameters, optimizer,
                 callbacks=None, filter_root=8, depth=6, 
                 kernel_size=(3, 3), strides=(2, 2), drop_rate=0.1, 
                 **kwargs):
        
        # Assign inputs as attributes to Model
        self.modelname = name
        self.channels = 2
        
        # Time series parameters
        self.dt = dt
        self.ts_length = ts_length
        
        # STFT parameters
        self.stft_parameters = stft_parameters
        
        # Training parameters
        self.optimizer = optimizer
        self.callbacks = callbacks
        
        # Set the loss function
        self.loss = BCE2
        
        # Model building parameters
        self.filter_root = filter_root
        self.depth = depth
        self.kernel_size = kernel_size
        self.strides = strides
        self.drop_rate = drop_rate
        self.build_kwargs = kwargs
        
        # Date and time
        self.now = datetime.now()
        self.now_str = self.now.strftime("%d-%m-%Y_%H-%M-%S")
        
        # Get the shape of the input and output layers
        self.shape = self.calculate_shape()



    def build_model(self):
        """
        Method builds and compiles the DAE.
        
        Uses the atrributes as passed to __init__ to build the model. Output is
        a compiled model stored as an attribute.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """

        #################
        ##### Input #####
        #################
        
        # Define Input layer
        input_layer = Input((self.shape[0], self.shape[1], self.channels))
        
        # Empty dict to save shape for each layer
        layer_shapes = dict()
        convs = []
        
        ###################
        ##### Encoder #####
        ###################
        
        # First Layer
        h = Conv2D(self.filter_root, self.kernel_size,
                   padding='same', **self.build_kwargs)(input_layer)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.1)(h)
        h = Dropout(rate=self.drop_rate)(h)
        
        # More Layers
        for i in range(self.depth):
            h = Conv2D(int(2 ** i * self.filter_root), self.kernel_size,
                       padding='same', **self.build_kwargs)(h)
            h = BatchNormalization()(h)
            h = LeakyReLU(alpha=0.1)(h)
            h = Dropout(rate=self.drop_rate)(h)
            
            layer_shapes.update({i: (h.shape[1], h.shape[2])})
            convs.append(h)
            
            if i < self.depth - 1:
                h = Conv2D(int(2 ** i * self.filter_root), self.kernel_size,
                           padding='same', strides=self.strides, 
                           **self.build_kwargs)(h)

                h = BatchNormalization()(h)
                h = LeakyReLU(alpha=0.1)(h)
                h = Dropout(rate=self.drop_rate)(h)

        ####################
        ##### Dencoder #####
        ####################
        
        for i in range(self.depth - 2, -1, -1):
            needed_shape = layer_shapes[i]
            
            h = Conv2DTranspose(int(2 ** i * self.filter_root), self.kernel_size,
                                padding='same', strides=self.strides,
                                **self.build_kwargs)(h)
            h = BatchNormalization()(h)
            h = LeakyReLU(alpha=0.1)(h)
            h = Dropout(rate=self.drop_rate)(h)
            
            # Crop network and add skip connections
            crop = cropping_layer(needed_shape, is_shape=(h.shape[1], h.shape[2]))
            h = Cropping2D(cropping=(crop[0], crop[1]))(h)
            h = Add()([convs[i], h])
            
            h = Conv2D(int(2 ** i * self.filter_root), self.kernel_size,
                       padding='same', **self.build_kwargs)(h)
            h = BatchNormalization()(h)
            h = LeakyReLU(alpha=0.1)(h)
            h = Dropout(rate=self.drop_rate)(h)

        ##################
        ##### Output #####
        ##################
        
        # Output layers
        h = Conv2D(filters=self.channels, kernel_size=(1, 1), 
                   padding="same", **self.build_kwargs)(h)
        
        # Softmax generates the mask functions
        h = Softmax(axis=-1, dtype='float32')(h)
        
        # Build and compile model
        self.model = TFmodel(input_layer, h)
        self.model.compile(optimizer=self.optimizer, loss=self.loss,
                                     metrics=['accuracy'], run_eagerly=True)



    def freeze_encoder(self):
        """
        Method freezes the weights of the encoder, leaving only the final bottleneck
        layer and the decoder to evolve during retraining.
        
        This is useful if you want to preserve to the information in the encoder
        learned from a previous (probably significantly larger) training dataset.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        # Get where the encoder ends
        i = [i for i, l in enumerate(self.model.layers) 
                                if 'conv2d_transpose' in l.name][0] - 4
        
        # Set layers to non-trainable
        for layer in self.model.layers[:i]:
            layer.trainable = False
        
        # Must compile the model for changes to take effect
        self.model.compile(optimizer=self.optimizer, loss=self.loss,
                                         metrics=['accuracy'])



    def calculate_shape(self):
        """
        Method calculates the shape of the input and output layers, and 
        checks that the inverse STFT is the correct sdhape.
        
        The easiest to do this by calculating the STFT of a dummy trace, 
        and then the iSTFT to compare to said dummy trace.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If the iSTFT output is not the same shape as the original input.
        """
        
        # Make dummy trace
        dummy = np.random.rand(self.ts_length)
       
        # Calculate STFT and take the shape
        _, _, dummystft = stft(dummy.data, fs=1./self.dt, **self.stft_parameters)
        
        # Also check that out inverse is the correct shape
        _, dummytrace = istft(dummystft, fs=1./self.dt, **self.stft_parameters)
        if not dummytrace.shape == dummy.data.shape:
            raise ValueError('STFT parameters do not invert to original shape: original '
                        + f'length was {dummy.data.shape} but recovered shape was '
                        + f'{dummytrace.shape}.')
        
        return dummystft.shape



    def plot_history(self, filename, pathname="./figures"):
        """
        Method plots figures showing the training history.
        
        Parameters
        ----------
        filename : str
            The filename of the figures.
        pathname : str, optional
            The directory name to save the figures into, defaults to "./figures".
        
        Returns
        -------
        None
        """
        
        # Accuracy history
        fig_acc = plt.figure()
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f"{pathname}/{filename}_accuracy.png", bbox_inches="tight")
        plt.close()
        
        # Loss history
        fig_loss = plt.figure()
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f"{pathname}/{filename}_loss.png", bbox_inches="tight")
        plt.close()
        print(f"  History plots saved in {pathname}")



    def save_history(self, filename, pathname="./histories"):
        """
        Method saves the training history into a readable file.
        
        Parameters
        ----------
        filename : str
            The filename of the history file.
        pathname : str, optional
            The directory name to save the history file into, defaults to "./histories".
        
        Returns
        -------
        None
        """
        
        # The full filepath
        filepath = f"{pathname}/{filename}.hist"
        
        # Save the history as a dictionary
        save_obj(dictionary=self.history.history, filename=filepath)
        print(f"  Saved history file as {filepath}")



    def save_config(self, filename, pathname="./configs"):
        """
        Method saves the config details into a readable file.
        
        Parameters
        ----------
        filename : str
            The filename of the config file.
        pathname : str, optional
            The directory name to save the config file into, defaults to "./configs".
        
        Returns
        -------
        None
        """
        
        # The full filepath
        filepath = f"{pathname}/{filename}.config"
        
        # Get the name of the optimizer
        if isinstance(self.optimizer, str) is True:
            optimizer_name = self.optimizer
        else:
            try:
                optimizer_name = self.optimizer.name
            except AttributeError:
                optimizer_name = self.optimizer._name
        
        # Write the config file with what is needed for denoising
        config_dict = dict(channels=self.channels,
                           shape=self.shape, 
                           ts_length=self.ts_length, 
                           dt=self.dt,
                           optimizer=optimizer_name,
                           stft_parameters=self.stft_parameters)
        save_obj(dictionary=config_dict, filename=filepath)
        print(f"  Saved config file as {filepath}")



    def save_model(self, filename, pathname="./models"):
        """
        Method saves the model into a .h5 file.
        
        Parameters
        ----------
        filename : str
            The filename of the model file.
        pathname : str, optional
            The directory name to save the model file into, defaults to "./models".
        
        Returns
        -------
        None
        """
        
        # The full filepath
        filepath = f"{pathname}/{filename}.h5"
        
        # Save fully trained model
        # If checkpoints are available, the model is saved from the latest 
        # checkpoint to prevent overfitting
        if self.callbacks:
            for callback_index, callback_val in enumerate(self.callbacks):
                if type(callback_val) == tf.keras.callbacks.ModelCheckpoint:
                    self.model.load_weights(self.callbacks[callback_index].filepath)
                    print("  Model is saved from latest checkpoints.")
                    break
        
        # Save model as .h5
        self.model.save(filepath, overwrite=True)
        print(f"  Saved Model as {filepath}")



    def save_all(self, pathname_model="./models", pathname_config="./configs",
                       pathname_history="./histories", pathname_figures="./figures",
                       plot_history=False):
        """
        Method saves all important model outputs.
        
        Parameters
        ----------
        pathname_model : str, optional
            The directory name to save the model file into, defaults to "./models".
        pathname_config : str, optional
            The directory name to save the config file into, defaults to "./configs".
        pathname_history : str, optional
            The directory name to save the history file into, defaults to "./histories".
        pathname_figures : str, optional
            The directory name to save the figures into, defaults to "./figures".
        plot_history : bool, optional
            Whether to plot the figures or not, defaults to False.
        
        Returns
        -------
        None
        """
        
        # The filename will be the modelname
        filename = self.modelname
        
        # Check whether pathnames exists, otherwise create new directories
        # Don't create figure directory if we are not plotting
        if plot_history:
            pathnames = [pathname_model, pathname_config, pathname_history,
                                             pathname_figures]
        else:
            pathnames = [pathname_model, pathname_config, pathname_history]
        for pathname in pathnames:
            if not os.path.exists(pathname):
                os.makedirs(pathname)
        
        # Plot the training history
        if plot_history:
            self.plot_history(filename=filename, pathname=pathname_figures)
        
        # Save the training history
        self.save_history(filename=filename, pathname=pathname_history)
        
        # Save config file
        self.save_config(filename=filename, pathname=pathname_config)
        
        # Save the model file
        self.save_model(filename=filename, pathname=pathname_model)



    def train_model(self, signal_files, noise_files, traces_per_file,
                          noise_level=1., noise_scale=3., signal_scale=3.,
                          zeros_chance=0.001, epochs=50, batch_size=32,
                          repeats=1, validation_split=0.2, verbose=1, 
                          retraining=False, freeze_encoder=False):
        """
        Method trains the model.
        
        Parameters
        ----------
        signal_files : list
            List of pathnames to the signal .npz files.
        noise_files : list
            List of pathnames to the noise .npz files.
        traces_per_file : int
            How many traces are in each .npz file, must be the same for all files.
        noise_level : float, optional
            Weight towards more noise or less noise, greater value is more noise,
            defaults to 1.0, i.e. equal chance of noise or signal being higher in
            amplitude in the summed traces for training.
        noise_scale : float, optional
            Maxmimum amplitude of the noise in the summed trace relative to the
            signal, defaults to 3.0.
        signal_scale : float, optional
            Maxmimum amplitude of the signal in the summed trace relative to the
            noise, defaults to 3.0.
        zeros_chance: float, optional
            This is the chance of any particular signal time series being set to
            all zeros. This is useful to help the DAE identify pure noise.
        epochs : int, optional
            Number of epochs to train for, defaults to 50
        batch_size : int, optional
            Batch size for the training process, i.e. the number of traces passed
            through the network in each iteratin of the training process, defaults
            to 32.
        repeats : int, optional
            How many times to repeat the training data per epoch. Due to data
            augmentation (random pairing, scaling, time shifting) there can be
            a slight improvement in performance from repeating data as slightly
            different combinations will be shown each time, but this is seemingly 
            usually minimal when the training dataset is already a decent size. 
            Defaults to 1.
        validation_split : float, optional
            What split to have for training data vs validation data between 0.0
            and 1.0, defaults to 0.2 (i.e. 20% of the data is used for validation,
            80% for training).
        verbose : int, optional
            How verbose to make the training process, defaults to 1. Can be either
            0, 1, or 2.: 0 = silent, 1 = progress bar, 2 = one line per epoch.
        retraining : bool, optional
            Whether this is retraining or not, defaults to False.
        freeze_encoder : bool, optional
            Whether to freeze the encoder or not, defaults to False and is only
            applied when `retraining` is True.
        
        Returns
        -------
        None
        """
        
        # Split value to split data into training and validation datasets
        split = int(len(signal_files) * (1 - validation_split))
        
        # Create DataGenerators that control data flow during training
        generator_train = DataGenerator(signal_list=signal_files[:split], 
                                        noise_list=noise_files[:split],
                                        repeats=repeats,
                                        traces_per_file=traces_per_file,
                                        noise_level=noise_level,
                                        noise_scale=noise_scale,
                                        signal_scale=signal_scale,
                                        zeros_chance=zeros_chance,
                                        channels=self.channels,
                                        batch_size=batch_size,
                                        shape=self.shape, 
                                        dt=self.dt,
                                        ts_length=self.ts_length,
                                        stft_parameters=self.stft_parameters)
        generator_validate = DataGenerator(signal_list=signal_files[split:],
                                        noise_list=noise_files[split:],
                                        traces_per_file=traces_per_file,
                                        noise_level=noise_level,
                                        noise_scale=noise_scale,
                                        signal_scale=signal_scale,
                                        zeros_chance=zeros_chance,
                                        channels=self.channels,
                                        batch_size=batch_size,
                                        shape=self.shape, 
                                        dt=self.dt,
                                        ts_length=self.ts_length,
                                        stft_parameters=self.stft_parameters)
        
        # Freeze the encoder if needed
        # Only do if retraing and explicitly asked for
        if retraining and freeze_encoder:
            self.freeze_encoder()
        
        # Do the training!
        self.history = self.model.fit(x=generator_train, epochs=epochs,
                                      verbose=verbose,
                                      validation_data=generator_validate,
                                      callbacks=self.callbacks)



class DataGenerator(Sequence):
    """
    Class for generating batches of training data for the DAE.
    """
    def __init__(self, signal_list, noise_list, traces_per_file,
                 stft_parameters, channels=2, batch_size=32, shape=(90, 6001),
                 dt=1.0, ts_length=6001, noise_level=1., noise_scale=3., 
                 signal_scale=3., zeros_chance=0.001, repeats=1):
        
        # Add inputs as attributes to DataGenerator
        self.shape = shape
        self.batch_size = batch_size
        self.traces_per_file = traces_per_file
        self.channels = channels
        self.dt = dt
        self.ts_length = ts_length
        self.noise_level = noise_level
        self.noise_scale = noise_scale
        self.signal_scale = signal_scale
        self.zeros_chance = zeros_chance
        self.flipping = True
        self.shuffle = True
        self.shift = True
        self.repeats = repeats
        self.stft_parameters = stft_parameters
        
        # As we apply various augmentations (time shifting, flipping, zeroing) during
        # batch generation there is some advantage to simply repeating input data as
        # each time the signal - noise pairing and the augmentations will change
        self.signal_list = np.repeat(signal_list, self.repeats)
        self.noise_list = np.repeat(noise_list, self.repeats)
        
        # Every file of training data is the same size, so the indices
        # to access are the same for every file
        self.access_indices = np.append(
                    np.arange(0, self.traces_per_file, self.batch_size), 
                                        self.traces_per_file)
        self.access_indices_list = [[self.access_indices[c], self.access_indices[c+1]]
                                        for c in range(len(self.access_indices)-1)]
        
        # Pair up the files with the access indices so that each batch has a
        # unique set of traces
        self.signal_access = sum([[[f, i] 
                    for f in self.signal_list] for i in self.access_indices_list], [])
        self.noise_access = sum([[[f, i] 
                    for f in self.noise_list] for i in self.access_indices_list], [])
        
        # Shuffle
        if self.shuffle:
            np.random.shuffle(self.signal_access)
            np.random.shuffle(self.noise_access)
        
        # Raise error if there are no noise files
        if len(self.noise_list) == 0:
            msg = f"Could not load noise files from {self.noise_list}"
            raise ValueError(msg)
        
        # Raise error if there are no signal files
        if len(self.signal_list) == 0:
            msg = f"Could not load signal files from {self.signal_list}"
            raise ValueError(msg)



    def __len__(self):
        """
        Method controls how many batches to have per epoch.
        """
        return int(np.floor(len(self.signal_list) * self.traces_per_file /
                             self.batch_size))



    def __getitem__(self, index):
        """
        Method gets a batch of data.
        """
        
        return self.data_generation(index)



    def on_epoch_end(self):
        """
        Method shuffles training data at the end of each epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.signal_access)
            np.random.shuffle(self.noise_access)



    def data_generation(self, index):
        """
        Method generates a batch of training data.
        """
        
        # Empty input data
        X = np.empty(shape=(self.batch_size, *self.shape, self.channels),
                                         dtype=np.float16)
        # Empty target data
        Y = np.empty(shape=(self.batch_size, *self.shape, self.channels),
                                         dtype=np.float16)
        
        # Signal and noise traces to use for this batch
        signal_filename = self.signal_access[index][0]
        signal_indices = np.arange(self.signal_access[index][1][0],
                                   self.signal_access[index][1][1])
        noise_filename =  self.noise_access[index][0]
        noise_indices  = np.arange(self.noise_access[index][1][0],
                                   self.noise_access[index][1][1])
        
        # Read in the noise and signal
        all_signal = np.load(signal_filename)['data'][signal_indices,:self.ts_length]
        all_noise = np.load(noise_filename)['data'][noise_indices,:self.ts_length]
        
        # Loop through pairs of signal and noise
        for i, (signal, noise) in enumerate(zip(all_signal, all_noise)):
            
            # To help the algorithm to identify pure noise, sometimes we set
            # some signal time series to pure zeros
            zeros = np.random.choice([True, False], 
                                   p=[self.zeros_chance, 1-self.zeros_chance])
            if zeros:
                signal = np.zeros(signal.shape)
            
            # Shift signal array to the left or right. Zeros are instead
            if self.shift:
                signal = shift_array(array=signal)
            
            # Demean the time series
            noise = noise - np.mean(noise)
            signal = signal - np.mean(signal)
            
            # Normalise noise
            if np.max(np.abs(noise)) > 0:
                noise = noise / np.max(np.abs(noise))
            
            # Normalise signal
            if np.max(np.abs(signal)) > 0:
                signal = signal / np.max(np.abs(signal))
            
            # If augmenting by flipping then do this
            if self.flipping:
                
                # There are three possible flips
                # Do not flip signal LR as there is time dependence
                flip_signal_ud = np.random.choice([True, False])
                flip_noise_ud = np.random.choice([True, False])
                flip_noise_lr = np.random.choice([True, False])
                
                # Do the flips
                if flip_signal_ud:
                    signal = -signal
                if flip_noise_ud:
                    noise = -noise
                if flip_noise_lr:
                    noise = np.flip(noise)
                
            # We need to add the noise and signal together but they can 
            # have different relative amplitudes. We may wish to weight
            # towards a greater number of training data with larger
            # amplitude signal or noise.
            options = ['signal bigger', 'noise bigger']
            weights = np.array([1., self.noise_level])
            option = np.random.choice(options, p=(weights / np.sum(weights)))
            
            # Deal with case where noise is bigger
            if option == 'noise bigger':
                scale = np.random.uniform(1, self.noise_scale)
                noise *= scale
            
            # Deal with case where signal is bigger
            if option == 'signal bigger':
                scale = np.random.uniform(1, self.signal_scale)
                signal *= scale
            
            # Adding signal and noise
            noisy_signal = signal + noise
            
            # Normalise noisy signal
            norm = np.max(np.abs(noisy_signal))
            noisy_signal = noisy_signal / norm
            signal = signal / norm
            noise = noise / norm
            
            # Calculate STFTs
            _, _, cns = stft(noisy_signal, fs=1./self.dt, **self.stft_parameters)
            _, _, cs = stft(signal, fs=1./self.dt, **self.stft_parameters)
            _, _, cn = stft(noise, fs=1./self.dt, **self.stft_parameters)
            
            # Ignore Runtime Warnings for division by zero
            np.seterr(divide='ignore', invalid='ignore')
            
            # X contains the input data
            # The first two images are the normalised real and imaginary parts of 
            # the noisy signal
            X[i, :, :, 0] = cns.real / np.max(np.abs(cns.real))
            X[i, :, :, 1] = cns.imag / np.max(np.abs(cns.imag))
            
            # Y contains the target data
            # The first two images are the two masks of Zhu et al (2019)
            Y[i, :, :, 0] = 1. / (1. + np.abs(cn) / np.abs(cs))
            Y[i, :, :, 1] = 1. - Y[i, :, :, 0]
            
        # Replace nan and inf values
        Y = np.nan_to_num(Y)
        
        return X, Y



