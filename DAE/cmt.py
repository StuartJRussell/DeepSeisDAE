
# Import modules
import numpy as np
from scipy.special import erf

class CMT:
    """
    Class to generate and handle random moment tensors for building
    synthetic seismic datasets for ML.
    
    This class produces moment tensors that are more semi-realistic. They will be
    more representitive of true earthquakes than say moment tensors with truly
    randomly generated components. But its parameterisation is crude and not based 
    on any rigorous academic analysis by the author. The parameters in this class 
    are based on fitting various elements of the CMT catalogue: 
        https://www.globalcmt.org/
    
    Attributes
    ----------
    exp : float
        The exponent for parameters in the moment tensor.
    rmt : numpy.array
        Reduced moment tensor, the six independent components of the moment tensor in NM.
    mt : numpy.array
        Full 3x3 moment tensor in NM.
    M0 : float
        Seismic moment of the moment tensor in Nm.
    Mw : float
        Moment magnitude of the moment tensor.
    mrr : float
    mtt : float
    mpp : float
    mrt : float
    mrp : float
    mtp : float
    ts : float
        Time shift in seconds.
    hd : float
        Half duration in seconds.
    lat : float
        Source latitude in degrees north.
    lon : float
        Source latitude in degrees east.
    dep : float
        Source depth in km.
    location : list
        [lat, lon, dep]
    """
    
    def __init__(self, **kwargs):
        """
        Generally speaking just call CMT() to generate a random CMTSOLUTION.
        
        If wished, then various parameters can be set by the user via **kwargs.
        Attributes that can be set directly are:
            Mw, ts, hd, lat, lon, dep.
        All others are calculated.
        """
        
        # Select the exponent for moment tensor
        # Either random, or approximate a value based on the user's Mw input
        user_Mw = kwargs.get('Mw', None)
        self.exp = self.select_exponent(user_Mw)
        # Get moment tensor elements, units are Nm
        self.rmt = self.random_moment_tensor()
        
        # Full moment tensor in Nm
        self.mt = np.array([[self.rmt[0], self.rmt[3], self.rmt[4]],
                            [self.rmt[3], self.rmt[1], self.rmt[5]],
                            [self.rmt[4], self.rmt[5], self.rmt[2]]])
        # Seismic moment in Nm
        self.M0 = (1./np.sqrt(2.)) * np.linalg.norm(self.mt)
        # Moment magnitude
        self.Mw = (2./3.) * np.log10(self.M0 * 1e7) - 10.7
        
        # If the user asked for a specific Mw, an exponent will have been approximated
        # but due to randomness this is unlikely to give the exact requsted Mw.
        # In this case, normalise the moment tensor to the requested Mw.
        if user_Mw is not None and abs(self.Mw - user_Mw) >= 1e-5:
            self.normalise_mt(user_Mw)
        
        # Elements of the moment tensor in Nm
        self.mrr = self.rmt[0]
        self.mtt = self.rmt[1]
        self.mpp = self.rmt[2]
        self.mrt = self.rmt[3]
        self.mrp = self.rmt[4]
        self.mtp = self.rmt[5]
        
        # Half shift and time duration, both in seconds
        self.ts = kwargs.get('ts', self.random_time_shift())
        self.hd = kwargs.get('hd', self.calc_half_duration())
        
        # Source location
        location = self.random_location()
        self.lat = kwargs.get('lat', location[0])                   # Degrees N
        self.lon = kwargs.get('lon', location[1])                   # Degrees E
        self.dep = kwargs.get('dep', self.random_depth())           # km
        self.location = np.array([self.lat, self.lon, self.dep])



    def write(self, filepath='CMTSOLUTION'):
        """
        Method to write text file in CMTSOLUTION format.
            
        Parameters
        ----------
        filepath : str, optional
            Full file path of output file, defaults to 'CMTSOLUTION'
        
        Returns
        -------
        None
        """
        
        # The contents of the file
        string = (" ABCD 0000 00 00 00 00 00.00  0.0000 " +
                  "000.0000 0.0 0.0 0.0 EFGHIJKLMNOPQ\n" +
                  f"event name:     000000000000X\n" +
                  f"time shift:   {self.ts:>9.4f}\n" +
                  f"half duration:{self.hd:>9.4f}\n" + 
                  f"latitude:     {self.lat:>9.4f}\n" +
                  f"longitude:    {self.lon:>9.4f}\n" + 
                  f"depth:        {self.dep:>9.4f}\n" +
                  f"Mrr:      {self.mrr * 1e7:>13.6E}\n" +
                  f"Mtt:      {self.mtt * 1e7:>13.6E}\n" +
                  f"Mpp:      {self.mpp * 1e7:>13.6E}\n" +
                  f"Mrt:      {self.mrt * 1e7:>13.6E}\n" + 
                  f"Mrp:      {self.mrp * 1e7:>13.6E}\n" +
                  f"Mtp:      {self.mtp * 1e7:>13.6E}")
        
        # Write file
        with open(filepath, 'w') as f:
            f.write(string)



    def random_moment_tensor(self):
        """
        Method generates random moment tensor components.
        
        Some key points:
            - mtt, mpp, mrt, mrp, mtp are treated as independent Gaussians, 
              the parameters of which depend on the exponent.
            - The sum of mrr, mtt and mpp must equal zero for a source with no
              volume change, so these are generated and then demeaned together.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        mt : numpy.array
            The reduced moment tensor: [mrr, mtt, mpp, mrt, mrp, mtp]
        """
        
        # Treat mtt, mpp, mrt, mrp and mtp as independent Gaussians
        x = np.linspace(-10., 10., 1001)
        params = [self.get_Gaussian_parameters(m, self.exp) for m in 
                        ['mrr', 'mtt', 'mpp', 'mrt', 'mrp', 'mtp']]
        pdfs = [self.Gaussian(x, *p) for p in params]
        ms = np.array([np.random.choice(x, p=pdf / np.sum(pdf)) for pdf in pdfs])
        
        # Demean mrr, mtt and mpp
        # This ensures that they sum to zero
        ms = np.append(ms[:3] - np.mean(ms[:3]), ms[3:])
        
        return np.array(ms) * 10. ** self.exp



    def random_location(self):
        """
        Method generates a random source latitude and longitude.
        
        Latitude and longitude are independent of the moment tensor.
        
        Latitude and longitude are treated as uniform. This isn't true for the 
        real Earth, but the synthetics do not have 3D variations anyway.

        Parameters
        ----------
        None
        
        Returns
        -------
        latitude : float
        longitude : float
        """
        
        # Random latitude and longitude
        latitude = np.random.uniform(-90, 90)
        longitude = np.random.uniform(-180, 180)
        
        return latitude, longitude



    def random_depth(self):
        """
        Method generates a random source depth.
        
        Depth is independent of the moment tensor.
        
        For geological reasons the depth distribution of earthquakes is complex
        and is not well fit by standard distributions. This is the distribution is from
        all events in the GCMT Catalogue from 2004 - 2020 in 100 equal width bins.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        depth : float
        """
        
        # Depth bins
        x = np.array([  0.,   7.,  14.,  21.,  28.,  35.,  42.,  49.,  56.,  63.,  70.,
                       77.,  84.,  91.,  98., 105., 112., 119., 126., 133., 140., 147.,
                      154., 161., 168., 175., 182., 189., 196., 203., 210., 217., 224.,
                      231., 238., 245., 252., 259., 266., 273., 280., 287., 294., 301.,
                      308., 315., 322., 329., 336., 343., 350., 357., 364., 371., 378.,
                      385., 392., 399., 406., 413., 420., 427., 434., 441., 448., 455.,
                      462., 469., 476., 483., 490., 497., 504., 511., 518., 525., 532.,
                      539., 546., 553., 560., 567., 574., 581., 588., 595., 602., 609.,
                      616., 623., 630., 637., 644., 651., 658., 665., 672., 679., 686.,
                      693.])
        
        # Bin occurence
        y = np.array([7.6300e+02, 1.5951e+04, 2.2520e+03, 1.9620e+03, 2.0620e+03,
                      3.0920e+03, 1.3180e+03, 9.4700e+02, 7.1900e+02, 5.3500e+02,
                      3.8800e+02, 3.1600e+02, 3.0200e+02, 2.8800e+02, 3.3600e+02,
                      3.3900e+02, 3.0600e+02, 2.7300e+02, 2.2700e+02, 1.8300e+02,
                      1.7700e+02, 1.5400e+02, 1.4300e+02, 1.2200e+02, 9.8000e+01,
                      8.3000e+01, 8.5000e+01, 9.3000e+01, 8.0000e+01, 7.4000e+01,
                      7.8000e+01, 6.1000e+01, 5.3000e+01, 3.7000e+01, 2.8000e+01,
                      2.6000e+01, 3.0000e+01, 2.1000e+01, 2.0000e+01, 1.1000e+01,
                      1.7000e+01, 1.7000e+01, 1.0000e+01, 9.0000e+00, 1.3000e+01,
                      1.2000e+01, 8.0000e+00, 6.0000e+00, 1.0000e+01, 7.0000e+00,
                      1.7000e+01, 1.1000e+01, 2.2000e+01, 1.1000e+01, 1.5000e+01,
                      2.4000e+01, 1.5000e+01, 2.4000e+01, 1.8000e+01, 1.0000e+01,
                      1.2000e+01, 8.0000e+00, 1.4000e+01, 1.0000e+01, 6.0000e+00,
                      1.0000e+01, 1.2000e+01, 1.1000e+01, 1.5000e+01, 1.5000e+01,
                      2.1000e+01, 2.1000e+01, 3.2000e+01, 2.2000e+01, 2.3000e+01,
                      3.8000e+01, 4.1000e+01, 3.8000e+01, 4.6000e+01, 4.0000e+01,
                      3.9000e+01, 4.5000e+01, 4.7000e+01, 5.1000e+01, 2.8000e+01,
                      4.5000e+01, 4.6000e+01, 2.8000e+01, 2.8000e+01, 1.9000e+01,
                      1.3000e+01, 1.2000e+01, 6.0000e+00, 7.0000e+00, 3.0000e+00,
                      4.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 0.0000e+00])
        
        # Take random value from the depth distribution
        # Add some scatter that is between the sampling intervals
        pdf = y / np.sum(y)
        depth = np.random.choice(x, p=pdf) + np.random.uniform(0, x[1] - x[0])
        
        return depth



    def random_time_shift(self):
        """
        Method generates a random time shift.
        
        Time shifts are well represented by a Gaussian and are essentially independent
        of the moment tensor.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        time_shift : float
        """
        
        min_ts = -5.
        max_ts = 15.
        params = np.array([2.7852479526842586, 2.9826431791841683])
        x = np.arange(min_ts, max_ts, 0.1)
        pdf = self.Gaussian(x, *params)
        
        return np.random.choice(x, p=pdf / np.sum(pdf))



    def calc_half_duration(self):
        """
        Method calculates the half duration of the source.
        
        Half duration is fixed prior to the GCMT inversion. For more details, see:
          Ekström, G., M. Nettles, and A. M. Dziewonski, The global CMT project 
          2004-2010: Centroid-moment tensors for 13,017 earthquakes, 
          Phys. Earth Planet. Inter., 200-201, 1-9, 2012. 
          doi:10.1016/j.pepi.2012.04.002 
         
        Parameters
        ----------
        None
        
        Returns
        -------
        half_duration : float
        """
        
        half_duration = 1.05e-8 * ((self.M0 * 1e7)**(1./3.))
        
        return half_duration
    
    
    
    def select_exponent(self, Mw):
        """
        Method selects an exponent based on whether there is a user input Mw or not.
        Either generates a random one, or estimates an exponent based on the Mw.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        exponent : float
        """
        
        if Mw is None:
            return self.random_exponent()
        else:
            return np.log10(np.sqrt(2.)/3. * 10. ** (1.5 *(Mw + 10.7))) - 7.



    def random_exponent(self):
        """
        Method generates an exponent for the moment tensor components.
        
        The exponent is drawn from a skewed Gaussian fitted to the GCMT catalogue.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        exponent : float
        """
        
        min_exp = 14.
        max_exp = 25.
        params = np.array([4.039460314907856, 15.646139738204411, 0.9880063620183555])
        x = np.arange(min_exp, max_exp, 0.01)
        pdf = self.skewed_Gaussian(x, *params)
        
        return np.random.choice(x, p=pdf / np.sum(pdf))



    def normalise_mt(self, Mw):
        """
        If the user choses an Mw then an exponent will have been estimated, but due
        to randomness the resulting moment tensor is not going to be the exactly the 
        correct magnitude.
        
        This Method normalises the generated moment tensor to ensure that it is the 
        magnitude that the user requested. This is edited in place.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            self.mt, self.M0, and self.Mw are modified in place.
        """
        
        # Target norm based on requested Mw
        M = (np.sqrt(2.) * 10. ** (1.5 *(Mw + 10.7))) * 1e-7
        
        # Normalise the moment tensor and assign new attributes
        self.mt  = self.mt * (M / np.linalg.norm(self.mt))
        self.M0 = (1./np.sqrt(2.)) * np.linalg.norm(self.mt)
        self.Mw = (2./3.) * np.log10(self.M0 * 1e7) - 10.7



    def get_Gaussian_parameters(self, component, exp):
        """
        Method gives the parameters of a Gaussian that fits each moment tensor
        component for a given exponent.
        
        These are calculated by fitting the distribution of moment tensor component
        coefficient in the GCMT catalogue 2004 - 2020.
        
        Parameters
        ----------
        component: str, required
            Which component of the moment tensor to get parameters for.
            Must be in ['mrr', 'mtt', 'mpp', 'mrt', 'mrp', mtp'].
        
        Returns
        -------
        mu : float
            The mean of the relevant Gaussian
        sigma : float
            The standard deviation of the relevant Gaussian
            
        """
        
        # Massive dictionary of coefficients.
        # These are calculated by fitting the GCMT catalogue
        coefficients = {
            'mrr': {15.0: (-0.43036055776892435, 2.047965493077154),
                    16.0: (0.09265761392460423, 2.6083540015857767),
                    17.0: (0.45267795301643315, 1.963168220214603),
                    18.0: (0.5465493439817456, 1.9382676729883297),
                    19.0: (0.388461126005362, 2.0983292720027293),
                    20.0: (0.44690322580645164, 2.0977319757890087)},
            'mtt': {15.0: (0.10004581673306773, 1.495010619654893),
                    16.0: (-0.056428408991515785, 2.081198026109688),
                    17.0: (-0.15642770486379176, 1.6882339242827726),
                    18.0: (-0.2264164289788933, 1.6974702680458096),
                    19.0: (-0.31248793565683647, 1.5174109486621832),
                    20.0: (-0.33756989247311825, 1.4329387308623367)},
            'mpp': {15.0: (0.3301812749003984, 1.8938101114039247),
                    16.0: (-0.0362171346103385, 2.3814216247670243),
                    17.0: (-0.2961677511856182, 1.8238043879636214),
                    18.0: (-0.3202036508841985, 1.8893751434193298),
                    19.0: (-0.07594638069705097, 1.8832770202300506),
                    20.0: (-0.10950537634408607, 1.8799522278513765)},
            'mrt': {15.0: (0.0758316733067729, 0.8350498175329074),
                    16.0: (0.20024289337881573, 1.4166599041256909),
                    17.0: (0.23098665490239326, 1.3932074348740047),
                    18.0: (0.2777124928693668, 1.426775792099824),
                    19.0: (0.2929008042895442, 1.4747510003212119),
                    20.0: (0.3620752688172043, 1.7878906008999036)},
            'mrp': {15.0: (-0.0021444223107569813, 0.9586989846069616),
                    16.0: (0.024497463482900377, 1.6357820295692629),
                    17.0: (-0.06210995919267675, 1.6115783129601329),
                    18.0: (-0.20911066742726755, 1.7337734851287832),
                    19.0: (-0.14389812332439678, 1.6338511226429493),
                    20.0: (-0.37447311827956997, 1.7742799817329982)},
            'mtp': {15.0: (0.31768625498007974, 1.5385789105257681),
                    16.0: (0.1318500393597481, 2.042439185123313),
                    17.0: (0.08599051505459358, 1.7882680591856823),
                    18.0: (0.08137706788362807, 1.6678459892972448),
                    19.0: (0.02289008042895441, 1.4979308807733835),
                    20.0: (0.2165698924731183, 1.176648433236378)}}

        # Which exponent is closest to requested
        exps = np.fromiter(coefficients['mrr'].keys(), dtype=float)
        e = exps[np.abs(exps - exp).argmin()]
        
        return coefficients[component][e]
    
    
    
    def moment_rate_triangle(self, half_n=20, amplitude=1.):
        """
        Method produces a triangular moment rate function (MRF).
        This was the assumed moment rate function used by GCMT after 2004.
        
        The triangle is an isosceles triangle with half width equal to the source 
        half duration. The amplitude and number of points per half duration can be 
        set by the user.
        
        Parameters
        ----------
        half_n : int, optional
            The number of samples per half duration. Total number of samples is
            2 * half_n + 1. Defaults to 20.
        amplitude : float, optional
            The peak amplitude of the triangle. Defaults to 1.
        
        Returns
        -------
        t : numpy.array
            Time array for the MRF
        mrf : numpy.array
            The moment rate function
        """
        
        # Gradient of triangle edge
        m = amplitude / self.hd
        
        # Moment rate function
        half = np.linspace(0., self.hd, half_n + 1) * m
        mrf = np.append(half, np.flip(half)[1:])
        
        # Time array
        t = np.linspace(0., 2. * self.hd, 2 * half_n + 1)
        
        return t, mrf



    def moment_rate_boxcar(self, half_n=20, amplitude=1.):
        """
        Method to produce a boxcar moment rate function.
        This was the assumed moment rate function used by GCMT prior to 2004.
        
        The boxcar has half width equal to the source half duration.
        The amplitude and number of points per half duration can be set by the user.
        
        Parameters
        ----------
        half_n : int, optional
            The number of samples per half duration. Total number of samples is
            2 * half_n + 1. Defaults to 20.
        amplitude : float, optional
            The peak amplitude of the boxcar. Defaults to 1.
        
        Returns
        -------
        t : numpy.array
            Time array for the MRF
        mrf : numpy.array
            The moment rate function
        """
        
        # Make boxcar
        half = np.append(0., np.full(half_n, amplitude))
        mrf = np.append(half, np.flip(half)[1:])
        
        # Time array
        t = np.linspace(0., 2. * self.hd, 2 * half_n + 1)
        
        return t, mrf



    def moment_rate_Gaussian(self, half_n=20, amplitude=1.):
        """
        Method to produce a Gaussian moment rate function.
        
        The Gaussian has 3 sigma equal to the source half duration.
        The amplitude and number of points per half duration can be set by the user.
        
        Parameters
        ----------
        half_n : int, optional
            The number of samples per half duration. Total number of samples is
            2 * half_n + 1. Defaults to 20.
        amplitude : float, optional
            The peak amplitude of the triangle. Defaults to 1.
        
        Returns
        -------
        t: numpy.array
            Time array for the MRF
        mrf : numpy.array
            The moment rate function
        """
        
        x = np.linspace(-1. * self.hd, 0., half_n + 1)
        sigma = (self.hd / 3.)
        half = self.Gaussian(x, 0., sigma)
        mrf = np.append(half, np.flip(half)[1:])
        
        # Time array
        t = np.linspace(0., 2. * self.hd, 2 * half_n + 1)
        
        return t, mrf



    def Gaussian(self, x, A, B):
        """
        Method returns a Gaussian with mean A and sigma B.
        
        Parameters
        ----------
        x : numpy.array
            The time array, i.e. the x values
        A : float
            The mean of the Gaussian
        B : float
            The sigma of the Gaussian
        
        Returns
        -------
        y : numpy.array
            The Gaussian, i.e. the y values
        """
        
        return np.exp(-0.5 * ((x - A) / B) ** 2)



    def skewed_Gaussian(self, x, A, B, C):
        """
        Method returns a skewed Gaussian with skew factor A, mean B and sigma C.
        
        Parameters
        ----------
        x : numpy.array
            The time array, i.e. the x values
        A : float
            The skew factor of the Gaussian
        B : float
            The mean of the Gaussian
        C : float
            The sigma of the Gaussian
        
        Returns
        -------
        y : numpy.array
            The Gaussian, i.e. the y values
        """
        
        return self.Gaussian(x, B, C) * (erf(A * (x - B) / (C * np.sqrt(2.))) + 1)
