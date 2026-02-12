
# Import modules
import os
import time
import obspy
import warnings
import instaseis
import numpy as np
import progressbar
import scipy.signal as sig
from obspy.taup import TauPyModel
from obspy.taup.utils import get_phase_names
from obspy.core.utcdatetime import UTCDateTime

# Import class to handle generation of moment tensors
from .cmt import CMT



class SyntheticsDownloader:
    """
    Class containing attributes and methods facilitating the downloading of synthetic
    datasets from Instaseis databases for training the DAE.
    """
    def __init__(self, traces_per_file, length_of_trace_in_samples,
                       sampling_rate_in_Hz, components, phase_list=None,
                       minimum_distance=0., maximum_distance=180.,
                       minimum_source_depth=0., maximum_source_depth=500.,
                       minimum_period=None, maximum_period=100.,
                       database='prem_a_2s', taup_model=None,
                       output_directory=None):
        """
        Parameters
        ----------
        traces_per_file : int
            Number of traces to put into each .npz file.
        length_of_trace_in_samples : int
            Number of samples per trace.
        sampling_rate_in_Hz : float
            The sampling rate of the traces in Hertz.
        components : str
            Which components to download, for teleseismic phases usual choices
            would 'Z', 'R', or 'T'. Several components can be joined by commas,
            e.g. 'R,T'.
        phase_list : list of str, optional
            What seismic phases to download, these will be at the centre of each
            trace. TauP naming conventions are used. Defaults to None, in which case
            obspy TauP "ttbasic" is used: https://docs.obspy.org/packages/obspy.taup.html
        minimum_distance : float, optional
            Minimum epicentral distance to download in degrees, defaults to 0.
        maximum_distance : float, optional
            Maximum epicentral distance to download in degrees, defaults to 180.
        minimum_source_depth : float, optional
            Minimum source depth in kilometres, defaults to 0.
        maximum_source_depth : float, optional
            Maximum source depth in kilometres, defaults to 500.
        minimum_period : float, optional
            Minimum period to filter at, defaults to None in which case the minimum
            period of the database is used. If set to less than the minimum period
            of the database then it will be increased to match this.
        maximum_period : float, optional
            Maximum period to filter at, defaults to 100 s, which is the maximum
            valid period of Instaseis databases due to the use of the Cowling 
            approximation in AxiSEM.
        database : str, optional
            The name of the Instaseis database that will be passed to open_db(). Can
            be either a syngine database or a path to a local database.
            Defaults to syngine database "prem_a_2s".
        taup_model : str, optional
            Name of the TauP model that will be passed to TauPyModel for calculating
            travel times. Can be an inbuilt model in obspy TauP, or a path to an
            appropriate .npz file. Defaults to None, in which case a guess is 
            made of what is appropriate for the Instaseis database.
        output_directory : str, optional
            The path to the directory of where to save the downloaded synthetics.
            Defaults to None, in which case a sensible choice is made.
        """
        
        # Assign inputs to class attributes
        # File and time series parameters
        self.traces_per_file = traces_per_file
        self.length_of_trace_in_samples = length_of_trace_in_samples
        self.length_of_trace_in_s = (length_of_trace_in_samples /
                                                     sampling_rate_in_Hz)
        self.sampling_rate_in_Hz = sampling_rate_in_Hz
        
        # Initialise a database
        self.given_database_name = database
        self.database_name = self._check_database_name()
        self.db = instaseis.open_db(self.database_name)
        
        # Initialse TauP model, usually first part of database name
        if taup_model is None:
            self.taup_model = TauPyModel(self.database_name.split('//')[1].split('_')[0])
        else:
            self.taup_model = TauPyModel(taup_model)
        
        # What components to dowload
        self.channel = components
        
        # Filtering parameters
        self.minimum_period = (self.db.info['period'] if minimum_period is None 
                                                            else minimum_period)
        self.maximum_period = maximum_period
        
        # The phases to use
        if phase_list is None:
            phase_list = get_phase_names('ttbasic')
        self.phase_list = phase_list
        
        # Constrains on source-receiver geometry
        self.minimum_distance = minimum_distance
        self.maximum_distance = maximum_distance
        self.minimum_source_depth = minimum_source_depth
        self.maximum_source_depth = maximum_source_depth
        
        # Create the output directory if it doesn't exist
        if output_directory is None:
            output_directory = '_'.join(['./Data/Synthetics', self.channel,
                 f'{self.minimum_period:.1f}_{self.maximum_period:.1f}']) + '/'
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        print(f'Outputting traces to {self.output_directory}')
        
        # Do a quick check of some parameters
        self._check_inputs()



    def build_dataset(self, number_of_traces, startcount=0):
        """
        Method downloads and saves synthetic traces.
        
        Parameters
        ----------
        number_of_traces : int
            Total number of traces to download.
        startcount : int, optional
            What file number to save from. Useful if files already exist in the output
            directory and should not be overwritten. Defaults to 0.
        
        Returns
        -------
        None
        """
        
        # How many files to download
        number_of_files = int(number_of_traces / self.traces_per_file)
        
        # Monitor time
        t0 = time.time()
        
        # Intialise a progress bar for download progress
        b = progressbar.ProgressBar(maxval=number_of_traces)
        print('Downloader progress:')
        b.start()
        
        # Keep going until we have all of the stations that we want
        tcount = 0; scount = 0; fcount = startcount
        signal = np.full((self.traces_per_file, self.length_of_trace_in_samples), np.nan)
        while tcount < number_of_traces:
            
            # First we need to get a random trace
            trace = self._get_trace()
            
            # Output the trace to the signal output
            signal[scount,:] = trace
            
            # Update the counts
            tcount += 1
            scount += 1
            b.update(tcount)
            
            # If we have enough for a full file then write the file
            if scount == self.traces_per_file:
                fname = f'{self.output_directory}/{fcount}.npz'
                np.savez(data=signal, file=fname)
                fcount += 1; scount = 0
            
        # Finish the progress bar
        b.finish()
        t1 = time.time()
        print(f'Downloading {number_of_traces} traces took {t1 - t0:.0f} s')
        print(f'That is {(t1 - t0) / number_of_traces:.2f} s per trace')



    def _get_trace(self):
        """
        Method sets up and downloads a synthetic trace, including selecting 
        a random arrival, setting up the source and receiver, getting the seismogram
        from the database, and doing some processing.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        data : numpy.array
            The seismic data from the trace.
        """
        
        # Select an arrival to download
        arr, ch = self._select_arrival()
        
        # Setup the source
        source = self._setup_source(arr.source_depth)
        
        # Setup the receiver
        receiver = instaseis.Receiver(latitude=0., longitude=arr.distance)
        
        # Download the trace
        trace = self.db.get_seismograms(source, receiver, components=ch, 
                               kind='velocity', dt=(1./self.sampling_rate_in_Hz))[0]
        
        # Process and trim the trace
        data = self._process_trace(trace, self.sampling_rate_in_Hz, arr.time)
        
        return data



    def _select_arrival(self):
        """
        Method selects a random arrival to try and download.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        arrival : obspy.taup.helper_classes.Arrival
            TauP arrival object for the arrival to download.
        channel : str
            The channel string, taken from self.channel.
        """
    
        # Sometimes there may be no arrivals, so keep going until we have one
        arrival = None
        while arrival is None:
    
            # Choose a channel
            channel = np.random.choice(self.channel.split(','))
            
            # Choose a random distance
            distance = np.random.uniform(self.minimum_distance, self.maximum_distance)
            
            # Choose a random source depth
            source_depth = np.random.uniform(self.minimum_source_depth,
                                             self.maximum_source_depth)
            
            # What phases occur at this distance and source depth
            arrivals = self.taup_model.get_travel_times(
                                            source_depth_in_km=source_depth,
                                            distance_in_degree=distance,
                                            phase_list=self.phase_list)
        
            # Select one of them. If length is zero then keep going
            if len(arrivals) > 0:
                arrival = np.random.choice(arrivals)
                
                # If this arrives within the database then accept it, otherwise force it
                # to keep going.
                if arrival.time > (self.db.info['length'] - 
                                            self.length_of_trace_in_s * 0.6):
                    arrival = None
            
        return arrival, channel



    def _setup_source(self, source_depth=None):
        """
        Method sets up the Instaseis source object. 
        
        Uses the CMT class to get a random moment tensor.
        
        Parameters
        ----------
        source_depth : float, optional
            Source depth in km. Defaults to None. If None then is generated by CMT.
        
        Returns
        -------
        source : instaseis.source.Source
            Instaseis source object for this source.
        """
        
        # Get a moment tensor
        cmt = CMT()
        
        # Assertain the source depth
        if source_depth is None:
            source_depth = cmt.dep
        
        # Create source for instaseis
        # Location is fixed at (0N, 0E), with stations along the equator
        # The model is 1D so this is convenient
        source = instaseis.Source(latitude=0., longitude=0., 
                                  depth_in_m=source_depth*1e3, m_rr=cmt.mrr, 
                                  m_tt=cmt.mtt, m_pp=cmt.mpp, m_rt=cmt.mrt, 
                                  m_rp=cmt.mrp, m_tp=cmt.mtp)
        
        return source



    def _process_trace(self, trace, sampling_rate_in_Hz, arrival_time):
        """
        Method processes a downloaded trace. 
        
        Trace is tapered, filtered, and cut to length.
        
        Parameters
        ----------
        trace: obspy.core.trace.Trace
            The trace to process.
        sampling_rate_in_Hz : float
            The sampling rate in Hertz.
        arrival_time : float
            The arrival time of the phase, this will be the centre of the window.
        
        Returns
        -------
        data : numpy.array
            The seismic trace.
        """
        
        # Taper ends
        trace.taper(max_percentage=0.025)
        
        # Must filter the synthetics to the valid periods
        trace.filter('bandpass', freqmin=(1./self.maximum_period), 
                     freqmax=(1./self.minimum_period), zerophase=True, corners=2)
        
        # From here on only deal with the time window we want itself
        i = np.abs(trace.times() - arrival_time).argmin()
        i0 = i - int(np.floor(self.length_of_trace_in_samples / 2.))
        i1 = i0 + self.length_of_trace_in_samples
        data = trace[i0:i1]
        
        return data



    def _check_database_name(self):
        """
        Method decides whether a database name is likely to be local or syngine,
        and returns a guess at the correct input to open_db().
        
        Parameters
        ----------
        None
        
        Returns
        -------
        database : str
            The guess at the input to open_db().
        
        Raises
        ------
        ValueError
            If database name is unknown.
        """
        
        # Define a list of possible syngine databases
        possible_databases = ['ak135f_1s', 'ak135f_2s', 'ak135f_5s', 'iasp91_2s',
                              'prem_i_2s', 'prem_a_2s', 'prem_e_2s', 'prem_a_5s',
                              'prem_a_10s', 'prem_a_20s']
        
        # If 'syngine' is in the name, then this is clearly a syngine database
        if self.given_database_name[:10] == "syngine://":
            print('Database is a syngine database')
            return self.given_database_name
            
        # Otherwise try to find the database
        elif os.path.exists(self.given_database_name):
            print('Database is a local database')
            return self.given_database_name
            
        # Otherwise see if this is a known syngine database
        elif self.given_database_name in possible_databases:
            print('Database is a syngine database')
            return "syngine://" + self.given_database_name
        
        # Otherweise return an error
        msg = ('Unable to decide database type, please be more explicit with '
               'the name of the your database. Include "syngine://" if required,' +
               'or else that the input is the correct path to a local database')
        raise ValueError(msg)


    def _check_inputs(self):
        """
        Method checks that the numerical inputs are sensible. 
        
        If they are not sensible then it either raises errors (for distance source 
        depth) or warnings (for periods). In the latter case these are changed to be
        sensible.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If inputs are incorrect.
        """
        
        if self.minimum_distance < 0. or self.maximum_distance < 0.:
            msg = 'Neither minimum_distance nor maximum_distance can be negative'
            raise ValueError(msg)
        
        if self.minimum_distance >= self.maximum_distance:
            msg = 'Minimum_distance must be less than maximum_distance'
            raise ValueError(msg)

        if self.minimum_source_depth < 0. or self.maximum_source_depth < 0.:
            msg = 'Neither minimum_source_depth nor maximum_source_depth can be negative'
            raise ValueError(msg)

        if self.minimum_source_depth >= self.maximum_source_depth:
            msg = 'Minimum_source_depth must be less than maximum_source_depth'
            raise ValueError(msg)
        
        if self.minimum_period < 0. or self.maximum_period < 0.:
            msg = 'Neither minimum_period nor maximum_period can be negative'
            raise ValueError(msg)

        if self.minimum_period >= self.maximum_period:
            msg = 'Minimum_period must be less than maximum_period'
            raise ValueError(msg)

        if self.maximum_period > 100.:
            msg = ('Periods above 100 s are not valid for Instaseis synthetics. ' +
                   'Setting to 100 s.')
            warnings.warn(msg)
            self.maximum_period = 100.
        
        p = self.db.info['period']
        if self.minimum_period < p:
            msg = ('minimum_period is less than is valid for the database. '+
                  f'Setting to the minimum period of the database ({p:.1f} s)')
            warnings.warn(msg)
            self.minimum_period = p


