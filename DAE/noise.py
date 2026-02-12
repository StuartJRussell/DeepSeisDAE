
# Import modules
import os
import time
import obspy
import numpy as np
import progressbar
import scipy.signal as sig
from obspy.clients.fdsn import Client
from obspy.core.utcdatetime import UTCDateTime
from obspy.signal.trigger import classic_sta_lta_py as stalta



class NoiseDownloader:
    """
    Class containing attributes and methods facilitating the downloading of noise
    datasets for training the DAE.
    """
    def __init__(self, traces_per_file, length_of_trace_in_samples, sampling_rate_in_Hz,
                       components, mintime=None, maxtime=None,
                       STALTA=False, STALTA_limit=5., STALTA_nsta=100, STALTA_nlta=1000, 
                       client="IRIS", output_directory='./Data/Noise/'):
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
            Which components to download, either "horizontal" or "vertical".
        mintime : obspy.core.utcdatetime.UTCDateTime, optional
            The start of the time window from which downloads will be made. Defaults
            to None, in whcih case the start date of each individual station will
            be used.
        maxtime : obspy.core.utcdatetime.UTCDateTime, optional
            The end of the time window from which downloads will be made. Defaults
            to None, in whcih case the end date of each individual station will
            be used.
        STALTA : bool ,optional
            Whether to reject traces based on an STA/LTA procedure. Defaults to False.
            If False then all traces will be saved regardless of if they contain
            signal.
        STALTA_limit : float, optional
            The value of the STA/LTA function over which traces are rejected. Defaults
            to 5.
        STALTA_nsta : int, optional
            The length of the short-term average for the STA/LTA procedure in samples.
            Defaults to 100.
        STALTA_nlta : int, optional
            The length of the long-term average for the STA/LTA procedure in samples.
            Defaults to 1000.
        client : str, optional
            Which client to use for downloads, defaults to "IRIS".
            List of possible clients here: https://docs.obspy.org/packages/obspy.clients.fdsn.html
        output_directory : str, optional
            Path to the directory where noise traces will be saved. Defaults to
            "./Data/Noise/".
        """
        
        # Assign inputs to class attributes
        # File and time series parameters
        self.traces_per_file = traces_per_file
        self.length_of_trace_in_samples = length_of_trace_in_samples
        self.length_of_trace_length_in_s = (length_of_trace_in_samples /
                                                     sampling_rate_in_Hz)
        self.sampling_rate_in_Hz = sampling_rate_in_Hz
        self.STALTA = STALTA
        self.STALTA_params = {'limit':STALTA_limit,
                              'nsta':STALTA_nsta,
                              'nlta':STALTA_nlta}
        self.mintime = mintime
        self.maxtime = maxtime
        self.rejection_count_STALTA = 0
        self.rejection_count_nans = 0
        self.rejection_count_infs = 0
        
        # Decide on the components - we are interested in broadband and will deal
        # with either horizontal (both N and E) or vertical (Z)
        if components.lower() not in ['horizontal', 'vertical']:
            msg = "Components mush be either 'horizontal' or 'vertical'"
            raise ValueError(msg)
        self.components = components
        if self.components == 'horizontal':
            self.channel = 'BHE,BHN'
        else:
            self.channel = 'BHZ'
        
        # Initialise the download client
        self.Client = Client(client)
        
        # Create the output directory if it doesn't exist
        # If it is not given then call it something sensible
        if output_directory is None:
            output_directory = '_'.join(['./Data/Noise', self.components]) + '/'
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        print(f'Outputting traces to {self.output_directory}')



    def get_stations(self, starttime=None, endtime=None, network=None, station=None,
                     location=None, minlatitude=None, maxlatitude=None,
                     minlongitude=None, maxlongitude=None, 
                     latitude=None, longitude=None, 
                     minradius=None, maxradius=None, **kwargs):
        """
        Method gets an inventory of stations that will be used for downloading
        the dataset and calculates weights for each network, such that every station
        in the inventory has equal chance of being selected for download.
        
        Parameters are as for obspy.clients.fdsn.client.Client.get_stations. The
        following is more or less taken from that function. See the documentation
        for that function for more details and examples.
        
        Parameters
        ----------
        starttime : obspy.core.utcdatetime.UTCDateTime, optional
            Limit to metadata epochs starting on or after the specified start time,
            defaults to None.
        endtime : obspy.core.utcdatetime.UTCDateTime
            Limit to metadata epochs starting before the specified start time,
            defaults to None.
        network : str, optional
            Select one or more network codes. Can be SEED network codes or data
            center defined codes. Multiple codes are comma-separated. Defaults to
            None.
        station : str, optional
            Select one or more SEED station codes. Multiple codes are comma-separated
            (e.g. ``"ANMO,PFO"``). Defaults to None.
        location : str, optional
            Select one or more SEED location identifiers. Multiple identifiers are
             comma-separated (e.g. ``"00,01"``). Defaults to None.
        channel : str, optional
            Select one or more SEED channel codes. Multiple codes are comma-separated
            (e.g. ``"BHZ,HHZ"``). Defaults to None.
        minlatitude : float, optional
            Limit to stations with a latitude larger than the specified minimum.
            Defaults to None.
        maxlatitude : float, optional
            Limit to stations with a latitude smaller than the specified maximum.
            Defaults to None.
        minlongitude : float, optional
            Limit to stations with a longitude larger than the specified minimum.
            Defaults to None.
        maxlongitude : float, optional
            Limit to stations with a longitude smaller than the specified maximum.
            Defaults to None.
        latitude : float, optional
            Specify the latitude to be used for a radius search. Defaults to None.
        longitude : float, optional
            Specify the longitude to be used for a radius search. Defaults to None.
        minradius : float, optional
            Limit results to stations within the specified minimum number of degrees
            from the geographic point defined by the `latitude` and `longitude`
            parameters. Defaults to None.
        maxradius : float, optional
            Limit results to stations within the specified maximum number of degrees
            from the geographic point defined by the `latitude` and `longitude`
            parameters. Defaults to None.
        """
        # Use ObsPy to get an inventory of all stations
        inventory = self.Client.get_stations(
                             starttime=starttime, endtime=endtime,
                             network=network, station=station,
                             location=location, channel=self.channel,
                             minlatitude=minlatitude, maxlatitude=maxlatitude,
                             minlongitude=minlongitude, maxlongitude=maxlongitude,
                             latitude=latitude, longitude=longitude,
                             minradius=minradius, maxradius=maxradius, **kwargs)
        self.inventory = inventory
        
        # Calculate the weights for downloading
        self.weights = self._calculate_network_weights()

    
    
    def estimate_coverage(self, number_of_traces):
        """
        Method provides an estimate of the answer to:
            If the sampling of all time windows and stations in the dataset 
            is uniform, then what percentage of time-station points would be 
            sampled based on the inputs to the class and the station inventory?
        
        Parameters
        ----------
        None
        
        Returns
        -------
        percentage : float
            The answer to the above question in %
        """
        
        # How many samples will the dataset have in total
        total_samples = number_of_traces * self.length_of_trace_in_samples
        
        # How many possible unique time samples in the inventory
        possible_samples = 0
        for nw in self.inventory:
            for st in nw:
                starttime = st.start_date
                endtime = UTCDateTime.now() if st.end_date is None else st.end_date
                length = endtime - starttime
                possible_samples += int(length * self.sampling_rate_in_Hz)
       
        # So what percentage is sampled?
        return (100 * total_samples / possible_samples)



    def build_dataset(self, number_of_traces, startcount=0):
        """
        Method downloads and saves noise traces.
        
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
        tcount = 0; ncount = 0; fcount = startcount
        noise = np.full((self.traces_per_file, self.length_of_trace_in_samples), np.nan)
        while tcount < number_of_traces:
            
            # First we need to get a random trace
            trace = self._get_trace()
            
            # Output the trace to the noise output
            noise[ncount,:] = trace
            
            # Update the counts
            tcount += 1
            ncount += 1
            b.update(tcount)
            
            # If we have enough for a full file then write the file
            if ncount == self.traces_per_file:
                fname = f'{self.output_directory}/{fcount}.npz'
                np.savez(data=noise, file=fname)
                fcount += 1; ncount = 0
        
        # Finish the progress bar
        b.finish()
        t1 = time.time()
        print(f'Downloading {number_of_traces} traces took {t1 - t0:.0f} s')
        print(f'That is {(t1 - t0) / number_of_traces:.2f} s per trace')



    def _get_trace(self):
        """
        Method to get a trace. Attempts to download a trace and then performs
        some very basic processing and checks if it passed some tests.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        data : numpy.array
            The noise trace.
        """
        
        satisfactory = False
        while not satisfactory:
        
            # Download a trace
            trace, params = self._download_trace()
                
            # Process the segment - this is just some basic initial stuff
            data = self._process_trace(trace, self.sampling_rate_in_Hz)

            # Check the trace
            satisfactory = self._check_trace(data)
        
        return data



    def _calculate_network_weights(self):
        """
        Method calculates the weights for each network in the inventory. We
        ultimately want every station across all networks to have equal chance
        of being selected.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        weights : numpy.array
            The normalised weights for each network ready to be passsed
            to np.random.choice.
        """
        
        # Get the number of stations per network, these form the weights for the
        # selection as we want each station, not each network, to have equal chance
        # of being selected
        number_of_stations_per_network = [nw.total_number_of_stations 
                                        for nw in self.inventory.networks]
        total_number_of_stations = sum(number_of_stations_per_network)
        weights = [n / total_number_of_stations for n in number_of_stations_per_network]
        
        return weights



    def _select_station(self):
        """
        Method randomly selects a network, station, and channel.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        nw : obspy.core.inventory.network.Network
            A network object taken from within self.inventory.
        st : obspy.core.inventory.station.Station
            A station object taken from within nw.
        ch : str
            A channel string taken from within self.channel.
        """
    
        # Choose a network to have a go at
        # Weightings are already allocated based on the number of stations
        # in each network.
        nw_i = np.random.choice(range(len(self.inventory)), p=self.weights)
        nw = self.inventory[nw_i]
        
        # Choose a station in this network, each with equal weighting
        st_i = np.random.randint(len(nw))
        st = nw[st_i]
        
        # Choose a channel
        ch = np.random.choice(self.channel.split(','))
        
        return nw, st, ch



    def _select_time_window(self, st, length_of_trace_length_in_s, 
                            mintime=None, maxtime=None):
        """
        Method selects a random time window within the operational time
        of a given station.
        
        Parameters
        ----------
        st : obspy.core.inventory.station.Station
            The station object, for which to get a time window.
        length_of_trace_length_in_s : float
            The length of the time window in seconds.
        mintime : obspy.core.utcdatetime.UTCDateTime, optional
            If you would like to further restrict the available time window,
            then this is the minimum possible time. Defaults to None. If None
            then the start date of the deployment is taken.
        maxtime : obspy.core.utcdatetime.UTCDateTime, optional
            If you would like to further restrict the available time window,
            then this is the maximum possible time. Defaults to None. If None
            then the end date of the deployment is taken, or if the deployment
            is still going then it takes 1 month ago.
        
        Returns
        -------
        starttime : obspy.core.utcdatetime.UTCDateTime
            The time of the start of the time window.
        endtime : obspy.core.utcdatetime.UTCDateTime
            The time of the end of the time window.
        """
        
        # Use the min and max time if defined, else take the start and 
        # end date of the station
        # If the station has no end date then it is still in operation, but
        # NoneType is unhelpful so set arbritrarily to 1 month ago
        if st.end_date is None:
            st.end_date = UTCDateTime.now() - 2678400
        if mintime is None or st.start_date > mintime:
            mintime = st.start_date
        if maxtime is None or st.end_date < maxtime:
            maxtime = st.end_date
        
        # The segment that we will take is longer that what we actually need as
        # this gives us leeway during processing, so adjust min and max times
        mintime = mintime + length_of_trace_length_in_s
        mmaxtime = maxtime - length_of_trace_length_in_s
        
        # Get a random time window
        total_seconds = maxtime - mintime
        starttime = mintime + np.random.uniform(low=0., high=total_seconds)
        endtime = starttime + length_of_trace_length_in_s
        
        # The time window we will download is actually longer
        return (starttime - length_of_trace_length_in_s, 
                endtime + length_of_trace_length_in_s)



    def _download_trace(self):
        """
        Method downloads a trace. Not all time windows and stations will
        have available date, so keep going until a trace is found. In principle
        this could get stuck...
        
        Parameters
        ----------
        None
        
        Returns
        -------
        trace : obspy.core.trace.Trace
            The downloaded trace.
        params : list
            A list of parameters that define the time window, which are:
                [network code (str), station code (str), window start time 
                 (obspy.core.utcdatetime.UTCDateTime), window end time
                 (obspy.core.utcdatetime.UTCDateTime)]
        """
    
        # Try and get a trace, keep going until we have one
        trace = None
        while trace is None:
        
            # Choose a station to have a go at
            nw, st, ch = self._select_station()
            
            # Get a time segment for this station
            starttime, endtime = self._select_time_window(st,
                                             self.length_of_trace_length_in_s,
                                             self.mintime, self.maxtime)
            params = [nw.code, st.code, ch, starttime, endtime]
            
            # Try to download the segment, if fails then we will try another one
            try:
                trace = self._attempt_download(nwcode=nw.code, stcode=st.code, 
                                    location='*', channel=ch,
                                    starttime=starttime, endtime=endtime)
            except Exception as e:
                trace = None
        
        return trace, params
    
    
    
    def _attempt_download(self, nwcode, stcode, location, channel, starttime, endtime):
        """
        Method attempts to download a trace.
        
        Parameters
        ----------
        nwcode : str
            The network code.
        stcode : str
            The station code.
        location : str
            The location code.
        channel : str
            The channel code.
        starttime : obspy.core.utcdatetime.UTCDateTime
            The start time of the trace.
        endtime : obspy.core.utcdatetime.UTCDateTime
            The end time of the trace.
        
        Returns
        -------
        trace : obspy.core.trace.Trace
            The downloaded trace.
        
        Raises
        ------
        ValueError
            If the trace is too short, i.e. the whole requested time window
            has not been downloaded.
        """
        
        # Download the trace
        traces = self.Client.get_waveforms(
                        network=nwcode, station=stcode, 
                        location=location, channel=channel,
                        starttime=starttime, endtime=endtime)
        
        # Only take the first location
        traces.select(location=traces[0].stats.location)
        trace = traces[0]
        
        # Ensure that the trace is long enough
        desired_length = endtime - starttime
        actual_length = trace.stats.endtime - trace.stats.starttime
        if actual_length < 0.95 * desired_length:
            raise ValueError('Downloaded trace is too short, try again')
        
        return trace



    def _process_trace(self, trace, sampling_rate_in_Hz):
        """
        Method processes a downloaded trace. Trace is detrended, tapered, resampled,
        filtered, and cut to length.
        
        Parameters
        ----------
        trace : obspy.core.trace.Trace
            The trace to process.
        sampling_rate_in_Hz : float
            The sampling rate in Hertz.
        
        Returns
        -------
        data : numpy.array
            The seismic trace.
        """
        # Detrend with 3rd order polynomial
        trace.detrend('polynomial', order=3)
        
        # Taper ends
        trace.taper(max_percentage=0.025)
        
        # As we are using with Instaseis synthetics filter from 1 - 100 s as
        # standard
        trace.filter('bandpass', freqmin=(1./100.), freqmax=(1./1.), zerophase=True,
                     corners=2)
        
        # Resample trace
        trace.resample(sampling_rate_in_Hz)
        
        # From here on only deal with the time window we want itself
        i0 = self.length_of_trace_in_samples
        i1 = i0 + self.length_of_trace_in_samples
        data = trace[i0:i1]
        
        return data



    def _check_trace(self, data):
        """
        Method checks whether a trace meets the criteria. A trace fails if it 
        fails the STA/LTA test, contains nans, or contains infs.
        
        Parameters
        ----------
        data : numpy.array
            The trace to check.
        
        Returns
        -------
        result : bool
            True is pass, False is fail.
        """
        
        # Check the STALTA if asked for
        if self.STALTA:
            STALTA_pass = self._check_STALTA(data)
        else:
            STALTA_pass = True
        if not STALTA_pass:
            self.rejection_count_STALTA += 1
            return False
        
        # Check for nans
        if np.any(np.isnan(data)):
            self.STALTA_rejection_nans += 1
            return False
        
        # Check for infs
        if np.any(np.isinf(data)):
            self.STALTA_rejection_infs += 1
            return False
        
        # If we have made it this far then well done
        return True



    def _check_STALTA(self, data):
        """
        Method checks whether a trace passes the STA/LTA test.
        
        Parameters
        ----------
        data : numpy.array
            The trace to check.
        
        Returns
        -------
        result : bool
            True is pass, False is fail.
        """
        
        # Do on the Hilbert transform
        envelope = np.abs(sig.hilbert(data))
        
        # Take STALTA
        x = stalta(envelope, nsta=self.STALTA_params['nsta'],
                             nlta=self.STALTA_params['nlta'])
        
        return np.max(x) <= self.STALTA_params['limit']



