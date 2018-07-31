import numpy as np
from scipy.signal import correlate
from numpy.fft import ifft, fft, fftfreq, rfft, rfftfreq
from numpy.random import choice
from collections import Counter
from SinglePhotonIO import getChannels

import sys
pathtowriter = "/nervascratch/samwatkins/scdmsPyTools_temp/scdmsPyTools/scdmsPyTools/BatTools"
if pathtowriter not in sys.path:
    sys.path.append(pathtowriter)
    
import rawdata_writer as writer
from math import log10, floor

def round_sig(x, sig=2):
    if x == 0:
        return 0
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)

def inrange(vals, bounds):
    return np.array([bounds[0] <= val <= bounds[1] for val in vals])

def foldpsd(psd, fs):
    """
    Return the one-sided version of the inputted two-sided psd.
    
    Parameters
    ----------
        psd : ndarray
            A two-sided psd to be converted to one-sided
        fs : float
            The sample rate used for the psd
            
    Returns
    -------
        f : ndarray
            The frequencies corresponding to the outputted one-sided psd
        psd_folded : ndarray
            The one-sided (folded over) psd corresponding to the inputted two-sided psd
    """
    
    psd_folded = np.copy(psd[:len(psd)//2+1])
    psd_folded[1:len(psd)//2+(len(psd))%2] *= 2.0
    f = rfftfreq(len(psd),d=1.0/fs)
    
    return f, psd_folded

def calc_psd(x, fs=1.0, folded_over=True):
    """
    Return the PSD of an n-dimensional array, assuming that we want the PSD of the last axis. By default,
    will return the one-sided (folded over) PSD.
    
    Parameters
    ----------
        x : ndarray
            Array to calculate PSD of
        fs : float, optional
            Sample rate of the data being taken, assumed to be in units of Hz
        folded_over : bool, optional
            Boolean value specifying whether or not the PSD should be folded over. 
            If True, then the symmetric values of the PSD are multiplied by two, and
            we keep only the positive frequencies. If False, then the entire PSD is 
            saved, including positive and negative frequencies.
            
    Returns
    -------
        f : ndarray
            Array of sample frequencies
        psd : ndarray
            Power spectral density of x
        
    """
    
    # calculate normalization for correct units
    norm = fs * x.shape[-1]
    
    if folded_over:
        # if folded_over = True, we calculate the Fourier Transform for only the positive frequencies
        if len(x.shape)==1:
            psd = (np.abs(rfft(x))**2.0)/norm
        else:
            psd = np.mean(np.abs(rfft(x))**2.0, axis=0)/norm
            
        # multiply the necessary frequencies by two (zeroth frequency should be the same, as
        # should the Nyquist frequency when x.shape[-1] is even)
        psd[1:x.shape[-1]//2+1 - (x.shape[-1]+1)%2] *= 2.0
        f = rfftfreq(x.shape[-1], d=1.0/fs)
    else:
        # if folded_over = False, we calculate the Fourier Transform for all frequencies
        if len(x.shape)==1:
            psd = (np.abs(fft(x))**2.0)/norm
        else:
            psd = np.mean(np.abs(fft(x))**2.0, axis=0)/norm
            
        f = fftfreq(x.shape[-1], d=1.0/fs)
    
    return f, psd

def rand_sections(x, n, l, t=None, fs=1.0):
    """
    Return random, non-overlapping sections of a 1 or 2 dimensional array.
    For 2-dimensional arrays, the function treats each row as independent from the other rows.
    
    Parameters
    ----------
        x : ndarray
            n dimensional array to choose sections from
        n : int
            Number of sections to choose
        l : int
            Length in bins of sections
        t : array_like or float, optional
            Start times (in s) associated with x
        fs : float, optional
            Sample rate of data (in Hz)
            
    Returns
    -------
        evttimes : ndarray
            Array of the corresponding event times for each section
        res : ndarray
            Array of the n sections of x, each with length l
        
    """
    
    if len(x.shape)==1:
        if len(x)-l*n<0:
            raise ValueError("Either n or l is too large, trying to find more random sections than are possible.")
        
        if t is None:
            t = 0.0
        elif not np.isscalar(t):
            raise ValueError("x is 1-dimensional, t should be a scalar value")

        res = np.zeros((n, l))
        evttimes = np.zeros(n)
        j=0
        offset = 0
        inds = np.arange(len(x) - (l-1)*n)

        for ind in sorted(choice(inds, size=n, replace=False)):
            ind += offset
            res[j] = x[ind:ind + l]
            evttimes[j] = t + (ind+l//2)/fs
            j += 1
            offset += l - 1

    else:
        if t is None:
            t = np.arange(x.shape[0])*x.shape[-1]
        elif np.isscalar(t):
            raise ValueError(f"x is {len(x.shape)}-dimensional, t should be an array")
        elif len(x) != len(t):
            raise ValueError("x and t have different lengths")
            
        tup = ((n,),x.shape[1:-1],(l,))
        sz = sum(tup,())
        
        res = np.zeros(sz)
        evttimes = np.zeros(n)
        j=0
        
        nmax = int(x.shape[-1]/l)
        
        if x.shape[0]*nmax<n:
            raise ValueError("Either n or l is too large, trying to find more random sections than are possible.")
        
        choicelist = list(range(len(x))) * nmax
        np.random.shuffle(choicelist)
        rows = np.array(choicelist[:n])
        counts = Counter(rows)

        for key in counts.keys():
            offset = 0
            ncounts = counts[key]
            inds = np.arange(x.shape[-1] - (l-1)*ncounts)
            
            for ind in sorted(choice(inds, size=ncounts, replace=False)):
                ind += offset
                res[j] = x[key, ..., ind:ind + l]
                evttimes[j] = t[key] + (ind+l//2)/fs
                j += 1
                offset += l - 1
    
    return evttimes, res

def rand_sections_wrapper(filelist, n, l, nmax=10, iotype="getChannels", saveevents=False, savepath=None, savename=None,
                          dumpnum=1, maxevts=1000):
    """
    Wrapper for the rand_sections function for getting random sections from many different files. This allows 
    the user to input a list of files that the random sections should be pulled from.
    
    Parameters
    ----------
        filelist : list of strings
            List of files to be opened to take random sections from (should be full paths)
        n : int
            Number of sections to choose
        l : int
            Length in bins of sections
        nmax : int, optional
            Max number of rows in each file. Default is 10.
        iotype : string, optional
            Type of file to open, uses a different IO function. Default is "getChannels".
                "getChannels" : Use SinglePhotonIO.getChannels to open the files
        saveevents : boolean, optional
            Boolean flag for whether or not to save the events to raw MIDAS files for use with CDMS bats
        savepath : NoneType, str, optional
            Path to save the events to, if saveevents is True. If this is left as None, then the events will not be saved.
        savename : NoneType, str, optional
            Filename to save the events as. Should follow CDMS format, which is 
            "[code][lasttwodigitsofyear][month][day]_[24hourclocktime]".  If this is left as None, then 
            the events will not be saved.
        dumpnum : int, optional
            The dump number that the file should start saving from and the event number should be determined by when saving
        maxevts : int, optional
            The maximum number of events that should be stored in each dump when saving
        
                
    Returns
    -------
        evttimes : ndarray
            Array of the corresponding event times for each section
        res : ndarray
            Array of the n sections of x, each with length l
        
    """
    
    choicelist = list(range(len(filelist))) * nmax
    np.random.shuffle(choicelist)
    rows = np.array(choicelist[:n])
    counts = Counter(rows)

    evttimes_list = []
    res_list = []

    for key in counts.keys():

        if iotype=="getChannels":
            data = getChannels(filelist[key])
            ncounts = counts[key]
            t = data["time"]
            traces = np.stack((data["A"], data["B"]), axis=1)/1024
            fs = data["prop"]["sample_rate"][0][0][0][0]
        else:
            raise ValueError("Unrecognized iotype inputted.")
            
        et, r = rand_sections(traces, ncounts, l, t=t, fs=fs)
        evttimes_list.append(et)
        res_list.append(r)
    
    evttimes = np.concatenate(evttimes_list)
    res = np.vstack(res_list)
    
    if saveevents:
        if savepath is None or savename is None:
            print("savepath or savename has not been set, cannot save events.")
        else:
            saverandoms_cdmsbats(evttimes, res, savename, savepath, data["prop"], dumpnum=dumpnum, maxevts=maxevts)
    
    return evttimes, res

def saverandoms_cdmsbats(evttimes, traces, filename, filepath, pulses_prop, dumpnum=1, maxevts=1000):
    """
    Function for writing randoms events to MIDAS files for CDMS bats. Only works with the
    getChannels iotype for the rand_sections_wrapper.
    
    Parameters
    ----------
        evttimes : ndarray
            Array of the trigger times to be saved for each event
        traces : ndarray
            Array of the traces to be saved for each events
        filename : str
            String to save the filename. Should follow CDMS format, which is 
            "[code][lasttwodigitsofyear][month][day]_[24hourclocktime]"
        filepath : str
            Full path to save the file to.
        pulses_prop : dict
            The dictionary from the getChannels function that stores all of the 
            experiment properties, i.e. the "prop" field of the output of getChannels
        dumpnum : int, optional
            The dump number that the file should start saving from and the event number should be determined by
        maxevts : int, optional
            The maximum number of events that should be stored in each dump
    
    """
    
    
    l = traces.shape[-1]
    
    srs = pulses_prop["SRS"][0][0][0][0].astype(float)
    rfb = pulses_prop["Rfb"][0][0][0][0].astype(float)
    turnratio = pulses_prop["turn_ratio"][0][0][0][0].astype(float)
    gain = srs*rfb*turnratio

    daqrange = np.diff(pulses_prop["daqrange"][0][0][0])[0]

    fs = pulses_prop["sample_rate"][0][0][0][0]
    
    mywriter = writer.DataWriter()

    settings = {'Z1': {'detectorType': 511}}
    settings['Z1']['phononTraceLength']=int(l)
    settings['Z1']['phononPreTriggerLength']=int(l//2)
    settings['Z1']['phononSampleRate']=int(fs)
    settings['Z1']['PA'] = {'driverGain': srs}
    settings['Z1']['PB'] = {'driverGain': srs}

    events = list()
    
    ii=0
    for evttime, evttrace in zip(evttimes, traces):

        # event admin information
        event_dict = {'SeriesNumber': int(filename.replace("_","")),
                      'TriggerType' : 3, 
                      'EventNumber' : dumpnum*(10000)+ii, 
                      'EventTime'   : int(evttime),
                      'SimAvgX'     : 0,
                      'SimAvgY'     : 0,
                      'SimAvgZ'     : 0}

        # event trigger information
        trigger_dict = {'TriggerUnixTime1'  : 0,
                        'TriggerTime1'      : int(evttime), 
                        'TriggerTimeFrac1'  : int((evttime-int(evttime))/100e-9), 
                        'TriggerDetNum1'    : 1, 
                        'TriggerAmplitude1' : 0,
                        'TriggerStatus1'    : 3,
                        'TriggerUnixTime2'  : 0,
                        'TriggerTime2'      : 0, 
                        'TriggerTimeFrac2'  : 0, 
                        'TriggerDetNum2'    : 0, 
                        'TriggerAmplitude2' : 0,
                        'TriggerStatus2'    : 1,
                        'TriggerUnixTime3'  : 0,
                        'TriggerTime3'      : 0, 
                        'TriggerTimeFrac3'  : 0, 
                        'TriggerDetNum3'    : 0, 
                        'TriggerAmplitude3' : 0,
                        'TriggerStatus3'    : 8}

        # Pulses
        channels_dict = {'PA' : (evttrace[0]*gain*1024*4/daqrange+2048).astype(np.int32),
                         'PB' : (evttrace[1]*gain*1024*4/daqrange+2048).astype(np.int32)}
        # Fill events
        events.insert(ii, {'event'  : event_dict,
                          'trigger' : trigger_dict, 
                          'Z1'      : channels_dict})

        ii+=1

        if ii==maxevts:
            # save events
            filename_out = f"{filename}_F{dumpnum:04}.mid"
            mywriter.open_file(filename_out, filepath)
            mywriter.write_settings_from_dict(settings)
            mywriter.write_events(events)
            mywriter.close_file()

            # start next dump
            dumpnum+=1
            ii=0

    if ii!=0:
        # save events
        filename_out = f"{filename}_F{dumpnum:04}.mid"
        mywriter.open_file(filename_out, filepath)
        mywriter.write_settings_from_dict(settings)
        mywriter.write_events(events)
        mywriter.close_file()

def getchangeslessthanthresh(x, threshold):
    """
    Helper function that returns a list of the start and ending indices of the ranges of inputted 
    values that change by less than the specified threshold value
       
    Parameters
    ----------
        x : ndarray
            1-dimensional of values.
        threshold : int
            Value to detect the different ranges of vals that change by less than this threshold value.
        
    Returns
    -------
        ranges : ndarray
            List of tuples that each store the start and ending index of each range.
            For example, vals[ranges[0][0]:ranges[0][1]] gives the first section of values that change by less than 
            the specified threshold.
        vals : ndarray
            The corresponding starting and ending values for each range in x.
    
    """
    
    diff = x[1:]-x[:-1]
    a = diff>threshold
    inds = np.where(a)[0]+1

    start_inds = np.zeros(len(inds)+1, dtype = int)
    start_inds[1:] = inds

    end_inds = np.zeros(len(inds)+1, dtype = int)
    end_inds[-1] = len(x)
    end_inds[:-1] = inds

    ranges = np.array(list(zip(start_inds,end_inds)))

    if len(x)!=0:
        vals = np.array([(x[st], x[end-1]) for (st, end) in ranges])
    else:
        vals = np.array([])

    return ranges, vals

class OptimumFilt(object):
    """
    Class for applying a time-domain optimum filter to a long trace, which can be thought of as an FIR filter.
    
    Attributes
    ----------
        phi : ndarray 
            The optimum filter in time-domain, equal to the inverse FT of (FT of the template/power 
            spectral density of noise)
        norm : float
            The normalization of the optimal amplitude.
        tracelength : int
            The desired trace length (in bins) to be saved when triggering on events.
        fs : float
            The sample rate of the data (Hz).
        pulse_range : int
            If detected events are this far away from one another (in bins), 
            then they are to be treated as the same event.
        traces : ndarray
            All of the traces to be filtered, assumed to be an ndarray of 
            shape = (# of traces, # of channels, # of trace bins). Should be in units of Amps.
        template : ndarray
            The template that will be used for the Optimum Filter.
        noisepsd : ndarray
            The two-sided noise PSD that will be used to create the Optimum Filter.
        filts : ndarray 
            The result of the FIR filter on each of the traces.
        resolution : float
            The expected energy resolution in Amps given by the template and the noisepsd, calculated
            from the Optimum Filter.
        times : ndarray
            The absolute start time of each trace (in s), should be a 1-dimensional ndarray.
        pulsetimes : ndarray
            If we triggered on a pulse, the time of the pulse trigger in seconds. Otherwise this is zero.
        pulseamps : 
            If we triggered on a pulse, the optimum amplitude at the pulse trigger time. Otherwise this is zero.
        trigtimes : ndarray
            If we triggered due to ttl, the time of the ttl trigger in seconds. Otherwise this is zero.
        pulseamps : 
            If we triggered due to ttl, the optimum amplitude at the ttl trigger time. Otherwise this is zero.
        traces : ndarray
            The corresponding trace for each detected event.
        trigtypes: ndarray
            Array of boolean vectors each of length 3. The first value indicates if the trace is a random or not.
            The second value indicates if we had a pulse trigger. The third value indicates if we had a ttl trigger.
            
    """

    def __init__(self, fs, template, noisepsd, tracelength, trigtemplate=None):
        """
        Initialization of the FIR filter.
        
        Parameters
        ----------
            fs : float
                The sample rate of the data (Hz)
            template : ndarray
                The pulse template to be used when creating the optimum filter (assumed to be normalized)
            noisepsd : ndarray
                The two-sided power spectral density in units of A^2/Hz
            tracelength : int
                The desired trace length (in bins) to be saved when triggering on events.
            trigtemplate : NoneType, ndarray, optional
                The template for the trigger channel pulse. If left as None, then the trigger channel will not
                be analyzed.
        
        """
        
        self.tracelength = tracelength
        self.fs = fs
        self.template = template
        self.noisepsd = noisepsd
        
        # calculate the time-domain optimum filter
        self.phi = ifft(fft(self.template)/self.noisepsd).real
        # calculate the normalization of the optimum filter
        self.norm = np.dot(self.phi, self.template)
        
        # calculate the expected energy resolution
        self.resolution = 1/(np.dot(self.phi, self.template)/self.fs)**0.5
        
        # calculate pulse_range as the distance (in bins) between the max of the template and 
        # the next value that is half of the max value
        tmax_ind = np.argmax(self.template)
        half_pulse_ind = np.argmin(abs(self.template[tmax_ind:]- self.template[tmax_ind]/2))+tmax_ind
        self.pulse_range = half_pulse_ind-tmax_ind
        
        # set the trigger ttl template value
        self.trigtemplate = trigtemplate
        
        # calculate the normalization of the trigger optimum filter
        if trigtemplate is not None:
            self.trignorm = np.dot(trigtemplate, trigtemplate)
        else:
            self.trignorm = None
            
        # set these attributes to None, as they are not known yet
        self.traces = None
        self.filts = None
        self.times = None
        self.trig = None
        self.trigfilts = None
        
        self.pulsetimes = None
        self.pulseamps = None
        self.trigtimes = None
        self.trigamps = None
        self.evttraces = None
        self.trigtypes = None


    def filtertraces(self, traces, times, trig=None):
        """
        Method to apply the FIR filter the inputted traces with specified times.
        
        Parameters
        ----------
            traces : ndarray
                All of the traces to be filtered, assumed to be an ndarray of 
                shape = (# of traces, # of channels, # of trace bins). Should be in units of Amps.
            times : ndarray
                The absolute start time of each trace (in s), should be a 1-dimensional ndarray.
            trig : NoneType, ndarray, optional
                The trigger channel traces to be filtered using the trigtemplate (if it exists). If
                left as None, then only the traces are analyzed. If the trigtemplate attribute
                has not been set, but this was set, then an error is raised.
        
        """
        # update the traces, times, and ttl attributes
        self.traces = traces
        self.times = times
        self.trig = trig
        
        # calculate the total pulse by summing across channels for each trace
        pulsestot = np.sum(traces, axis=1)
        
        # apply the FIR filter to each trace
        self.filts = np.array([correlate(trace, self.phi, mode="same")/self.norm for trace in pulsestot])
        
        # set the filtered values to zero near the edges, so as not to use the padded values in the analysis
        # also so that the traces that will be saved will be equal to the tracelength
        cut_len = np.max([len(self.phi),self.tracelength])

        self.filts[:, :cut_len//2] = 0.0
        self.filts[:, -(cut_len//2) + (cut_len+1)%2:] = 0.0
        
        if self.trigtemplate is None and trig is not None:
            raise ValueError("trig values have been inputted, but trigtemplate attribute has not been set, cannot filter the trig values")
        elif trig is not None:
            # apply the FIR filter to each trace
            self.trigfilts = np.array([np.correlate(trace, self.trigtemplate, mode="same")/self.trignorm for trace in trig])

            # set the filtered values to zero near the edges, so as not to use the padded values in the analysis
            # also so that the traces that will be saved will be equal to the tracelength
            self.trigfilts[:, :cut_len//2] = 0.0
            self.trigfilts[:, -(cut_len//2) + (cut_len+1)%2:] = 0.0

    def eventtrigger(self, thresh, trigthresh=None, positivepulses=True):
        """
        Method to detect events in the traces with an optimum amplitude greater than the specified threshold.
        Note that this may return duplicate events, so care should be taken in post-processing to get rid of 
        such events.
           
        Parameters
        ----------
            thresh : float
                The number of standard deviations of the energy resolution to use as the threshold for which events
                will be detected as a pulse.
            trigthresh : NoneType, float, optional
                The threshold value (in units of the trigger channel) such that any amplitudes higher than this will be 
                detected as ttl trigger event. If left as None, then only the pulses are analyzed.
            positivepulses : boolean, optional
                Boolean flag for which direction the pulses go in the traces. If they go in the positive direction, 
                then this should be set to True. If they go in the negative direction, then this should be set to False.
                Default is True.
        
        """
        
        # initialize the lists that we will save
        pulseamps_list = []
        pulsetimes_list = []
        trigamps_list = []
        trigtimes_list = []
        traces_list = []
        trigtypes_list = []
        
        # go through each filtered trace and get the events
        for ii,filt in enumerate(self.filts):
            
            if self.trigfilts is None or trigthresh is None:
                    
                # find where the filtered trace has an optimum amplitude greater than the specified amplitude
                if positivepulses:
                    evts_mask = filt>thresh*self.resolution
                else:
                    evts_mask = filt<-thresh*self.resolution
                    
                evts = np.where(evts_mask)[0]
                
                # check if any left over detected events are within the specified pulse_range from each other
                ranges = getchangeslessthanthresh(evts, self.pulse_range)[0]
                
                # set the trigger type to pulses
                trigtypes = np.zeros((len(ranges), 3), dtype=bool)
                trigtypes[:,1] = True
                
            elif trigthresh is not None:
                # find where the filtered trace has an optimum amplitude greater than the specified threshold
                if positivepulses:
                    pulseevts_mask = filt>thresh*self.resolution
                else:
                    pulseevts_mask = filt<-thresh*self.resolution
                    
                pulseevts = np.where(pulseevts_mask)[0]
                
                # check if any left over detected events are within the specified pulse_range from each other
                pulseranges, pulsevals = getchangeslessthanthresh(pulseevts, self.pulse_range)
                
                # make a boolean mask of the ranges of the events in the trace from the pulse triggering
                pulse_mask = np.zeros(self.filts[ii].shape, dtype=bool)
                for evt_range in pulseranges:
                    if evt_range[1]>evt_range[0]:
                        evt_inds = pulseevts[evt_range[0]:evt_range[1]]
                        pulse_mask[evt_inds] = True
                        
                # find where the ttl trigger has an optimum amplitude greater than the specified threshold
                trigevts_mask = self.trigfilts[ii]>trigthresh
                trigevts = np.where(trigevts_mask)[0]
                # find the ranges of the ttl trigger events
                trigranges, trigvals = getchangeslessthanthresh(trigevts, 1)
                
                # get the mask of the total events, taking the or of the pulse and ttl trigger events
                tot_mask = np.logical_or(trigevts_mask, pulse_mask)
                evts = np.where(tot_mask)[0]
                ranges, totvals = getchangeslessthanthresh(evts, self.pulse_range)
                
                # given the ranges, determine the trigger type based on if the total ranges overlap with
                # the pulse events and/or the ttl trigger events
                trigtypes = np.zeros((len(ranges), 3), dtype=bool)
                for ival, vals in enumerate(totvals):
                    for v in pulsevals:
                        if np.any(inrange(v, vals)):
                            trigtypes[ival, 1] = True
                    for v in trigvals:
                        if np.any(inrange(v, vals)):
                            trigtypes[ival, 2] = True
            
            # initialize more lists
            pulseamps = []
            pulsetimes = []
            trigamps = []
            trigtimes = []
            traces = []
            
            # for each range with changes less than the pulse_range, keep only the bin with the largest amplitude
            for irange, evt_range in enumerate(ranges):
                if evt_range[1]>evt_range[0]:
                    
                    evt_inds = evts[evt_range[0]:evt_range[1]]
                    
                    if trigtypes[irange][2]:
                        # both are triggered, use ttl as primary trigger
                        evt_ind = evt_inds[np.argmax(self.trigfilts[ii][evt_inds])]
                    else:
                        # only pulse was triggered
                        if positivepulses:
                            evt_ind = evt_inds[np.argmax(filt[evt_inds])]
                        else:
                            evt_ind = evt_inds[np.argmin(filt[evt_inds])]
                    
                    if trigtypes[irange][1] and trigtypes[irange][2]:
                        # both are triggered
                        if positivepulses:
                            pulse_ind = evt_inds[np.argmax(filt[evt_inds])]
                        else:
                            pulse_ind = evt_inds[np.argmin(filt[evt_inds])]
                        # save trigger times and amplitudes
                        pulsetimes.append(pulse_ind/self.fs + self.times[ii])
                        pulseamps.append(filt[pulse_ind])
                        trigtimes.append(evt_ind/self.fs + self.times[ii])
                        trigamps.append(filt[evt_ind])
                    elif trigtypes[irange][2]:
                        # only ttl was triggered, save trigger time and amplitudes
                        pulsetimes.append(0.0)
                        pulseamps.append(0.0)
                        trigtimes.append(evt_ind/self.fs + self.times[ii])
                        trigamps.append(filt[evt_ind])
                    else:
                        # only pulse was triggered, save trigger time and amplitudes
                        pulsetimes.append(evt_ind/self.fs + self.times[ii])
                        pulseamps.append(filt[evt_ind])
                        trigtimes.append(0.0)
                        trigamps.append(0.0)
                        
                    # save the traces that correspond to the detected event, including all channels, also with lengths
                    # specified by the attribute tracelength
                    traces.append(self.traces[ii, ..., 
                                              evt_ind - self.tracelength//2:evt_ind + self.tracelength//2 \
                                              + (self.tracelength)%2])
            
            # convert the values to ndarrays
            pulsetimes = np.array(pulsetimes)
            pulseamps = np.array(pulseamps)
            trigtimes = np.array(trigtimes)
            trigamps = np.array(trigamps)
            traces = np.array(traces)
            
            if np.any(trigtypes):
                trigtypes = np.vstack([r for r in trigtypes if np.any(r)])
            else:
                trigtypes = np.array([])
            
            # save the detected event information to the list for this trace
            pulsetimes_list.append(pulsetimes)
            pulseamps_list.append(pulseamps)
            trigtimes_list.append(trigtimes)
            trigamps_list.append(trigamps)
            traces_list.append(traces)
            trigtypes_list.append(trigtypes)
            
            
        self.pulsetimes = np.concatenate(pulsetimes_list)
        self.pulseamps = np.concatenate(pulseamps_list)
        self.trigtimes = np.concatenate(trigtimes_list)
        self.trigamps = np.concatenate(trigamps_list)
        
        if len(self.pulseamps)==0:
            self.evttraces = np.array([])
            self.trigtypes = np.array([])
        else:
            self.evttraces = np.vstack([t for t in traces_list if len(t)>0])
            self.trigtypes = np.vstack([t for t in trigtypes_list if len(t)>0])

    
def optimumfilt_wrapper(filelist, template, noisepsd, tracelength, thresh, trigtemplate=None, 
                        trigthresh=None, positivepulses=True, iotype="getChannels", saveevents=False, 
                        savepath=None, savename=None, dumpnum=1, maxevts=1000):
    """
    Wrapper function for the OptimumFilt class for running the continuous trigger on many different files. This allows 
    the user to input a list of files that should be analyzed.
    
    Parameters
    ----------
        filelist : list of strings
            List of files to be opened to take random sections from (should be full paths)
        template : ndarray
            The pulse template to be used when creating the optimum filter (assumed to be normalized)
        noisepsd : ndarray
            The two-sided power spectral density in units of A^2/Hz
        tracelength : int
            The desired trace length (in bins) to be saved when triggering on events.
        thresh : float
            The number of standard deviations of the energy resolution to use as the threshold for which events
            will be detected as a pulse.
        trigtemplate : NoneType, ndarray, optional
            The template for the trigger channel pulse. If left as None, then the trigger channel will not
            be analyzed.
        trigthresh : NoneType, float, optional
            The threshold value (in units of the trigger channel) such that any amplitudes higher than this will be 
            detected as ttl trigger event. If left as None, then only the pulses are analyzed.
        positivepulses : boolean, optional
            Boolean flag for which direction the pulses go in the traces. If they go in the positive direction, 
            then this should be set to True. If they go in the negative direction, then this should be set to False.
            Default is True.
        iotype : string, optional
            Type of file to open, uses a different IO function. Default is "getChannels".
                "getChannels" : Use SinglePhotonIO.getChannels to open the files
        saveevents : boolean, optional
            Boolean flag for whether or not to save the events to raw MIDAS files for use with CDMS bats
        savepath : NoneType, str, optional
            Path to save the events to, if saveevents is True. If this is left as None, then the events will not be saved.
        savename : NoneType, str, optional
            Filename to save the events as. Should follow CDMS format, which is 
            "[code][lasttwodigitsofyear][month][day]_[24hourclocktime]".  If this is left as None, then 
            the events will not be saved.
        dumpnum : int, optional
            The dump number that the file should start saving from and the event number should be determined by when saving
        maxevts : int, optional
            The maximum number of events that should be stored in each dump when saving
        
                
    Returns
    -------
        pulsetimes : ndarray
            If we triggered on a pulse, the time of the pulse trigger in seconds. Otherwise this is zero.
        pulseamps : 
            If we triggered on a pulse, the optimum amplitude at the pulse trigger time. Otherwise this is zero.
        trigtimes : ndarray
            If we triggered due to ttl, the time of the ttl trigger in seconds. Otherwise this is zero.
        trigamps : 
            If we triggered due to ttl, the optimum amplitude at the ttl trigger time. Otherwise this is zero.
        traces : ndarray
            The corresponding trace for each detected event.
        trigtypes: ndarray
            Array of boolean vectors each of length 3. The first value indicates if the trace is a random or not.
            The second value indicates if we had a pulse trigger. The third value indicates if we had a ttl trigger.
    """
    
    
    if type(filelist)==str:
        filelist=[filelist]
    
    pulsetimes_list = []
    pulseamps_list = []
    trigtimes_list = []
    trigamps_list = []
    traces_list = []
    trigtypes_list = []
    
    for f in filelist:
        
        if iotype=="getChannels":
            data = getChannels(f)
            fs = data["prop"]["sample_rate"][0][0][0][0]
            times = data["time"]
            traces = np.stack((data["A"], data["B"]), axis=1)/1024
            if trigtemplate is not None:
                try:
                    trig = data["T"] * 1e6/1024
                except:
                    print("There is no trigger channel in the data, only analyzing pulses...")
                    trig = None
            else:
                trig = None
        else:
            raise ValueError("Unrecognized iotype inputted.")
            
        filt = OptimumFilt(fs, template, noisepsd, tracelength, trigtemplate=trigtemplate)
        filt.filtertraces(traces, times, trig=trig)
        filt.eventtrigger(thresh, trigthresh=trigthresh)
        
        pulsetimes_list.append(filt.pulsetimes)
        pulseamps_list.append(filt.pulseamps)
        trigtimes_list.append(filt.trigtimes)
        trigamps_list.append(filt.trigamps)
        traces_list.append(filt.evttraces)
        trigtypes_list.append(filt.trigtypes)
        
    pulsetimes = np.concatenate(pulsetimes_list)
    pulseamps = np.concatenate(pulseamps_list)
    trigtimes = np.concatenate(trigtimes_list)
    trigamps = np.concatenate(trigamps_list)
    
    if len(pulseamps)==0:
        traces = np.array([])
        trigtypes = np.array([])
    else:
        traces = np.vstack([t for t in traces_list if len(t)>0])
        trigtypes = np.vstack([t for t in trigtypes_list if len(t)>0])
        
    if saveevents:
        if savepath is None or savename is None:
            print("savepath or savename has not been set, cannot save events.")
        else:
            saveevents_cdmsbats(pulsetimes, pulseamps, trigtimes, trigamps,
                                traces, trigtypes, savename, savepath, data["prop"], dumpnum=dumpnum, maxevts=maxevts)
    
    return pulsetimes, pulseamps, trigtimes, trigamps, traces, trigtypes


def saveevents_cdmsbats(pulsetimes, pulseamps, trigtimes, trigamps, traces, trigtypes, 
                        filename, filepath, pulses_prop, dumpnum=1, maxevts=1000):
    """
    Function for writing randoms events to MIDAS files for CDMS bats. Only works with the
    getChannels iotype for the rand_sections_wrapper.
    
    
    Parameters
    ----------
        pulsetimes : ndarray
            If we triggered on a pulse, the time of the pulse trigger in seconds. Otherwise this is zero.
        pulseamps : 
            If we triggered on a pulse, the optimum amplitude at the pulse trigger time. Otherwise this is zero.
        trigtimes : ndarray
            If we triggered due to ttl, the time of the ttl trigger in seconds. Otherwise this is zero.
        trigamps : 
            If we triggered due to ttl, the optimum amplitude at the ttl trigger time. Otherwise this is zero.
        traces : ndarray
            The corresponding trace for each detected event.
        trigtypes: ndarray
            Array of boolean vectors each of length 3. The first value indicates if the trace is a random or not.
            The second value indicates if we had a pulse trigger. The third value indicates if we had a ttl trigger.
        filename : str
            String to save the filename. Should follow CDMS format, which is 
            "[code][lasttwodigitsofyear][month][day]_[24hourclocktime]"
        filepath : str
            Full path to save the file to.
        pulses_prop : dict
            The dictionary from the getChannels function that stores all of the 
            experiment properties, i.e. the "prop" field of the output of getChannels
        dumpnum : int, optional
            The dump number that the file should start saving from and the event number should be determined by
        maxevts : int, optional
            The maximum number of events that should be stored in each dump
    
    """
    
    
    l = traces.shape[-1]
    
    srs = pulses_prop["SRS"][0][0][0][0].astype(float)
    rfb = pulses_prop["Rfb"][0][0][0][0].astype(float)
    turnratio = pulses_prop["turn_ratio"][0][0][0][0].astype(float)
    gain = srs*rfb*turnratio

    daqrange = np.diff(pulses_prop["daqrange"][0][0][0])[0]

    fs = pulses_prop["sample_rate"][0][0][0][0]
    
    mywriter = writer.DataWriter()
    
    settings = {'Z1': {'detectorType': 511}}
    settings['Z1']['phononTraceLength']=int(l)
    settings['Z1']['phononPreTriggerLength']=int(l//2)
    settings['Z1']['phononSampleRate']=int(fs)
    settings['Z1']['PA'] = {'driverGain': srs}
    settings['Z1']['PB'] = {'driverGain': srs}

    events = list()
    
    ii=0
    
    zipped = zip(pulsetimes, pulseamps, trigtimes, trigamps, traces, trigtypes)
    
    for pulsetime, pulseamp, trigtime, trigamp, trace, trigtype in zipped:

        # event admin information
        event_dict = {'SeriesNumber': int(filename.replace("_","")),
                      'EventNumber' : dumpnum*(10000)+ii,
                      'SimAvgX'     : 0,
                      'SimAvgY'     : round_sig(pulseamp, 6),
                      'SimAvgZ'     : round_sig(trigamp, 6)}



        # event trigger information
        trigger_dict = {'TriggerUnixTime1'  : 0,
                        'TriggerTime1'      : 0, 
                        'TriggerTimeFrac1'  : 0, 
                        'TriggerDetNum1'    : 0, 
                        'TriggerAmplitude1' : 0,
                        'TriggerStatus1'    : 3,
                        'TriggerUnixTime2'  : int(pulsetime),
                        'TriggerTime2'      : 0,
                        'TriggerTimeFrac2'  : int((pulsetime-int(pulsetime))/100e-9),
                        'TriggerStatus2'    : 1,
                        'TriggerUnixTime3'  : int(trigtime),
                        'TriggerTime3'      : 0,
                        'TriggerTimeFrac3'  : int((trigtime-int(trigtime))/100e-9),
                        'TriggerStatus3'    : 8}


        # Pulses
        channels_dict = {'PA' : (trace[0]*gain*1024*4/daqrange+2048).astype(np.int32),
                         'PB' : (trace[1]*gain*1024*4/daqrange+2048).astype(np.int32)}

        if trigtype[1] and trigtype[2]:
            event_dict['EventTime'] = int(trigtime)
            event_dict['TriggerType'] = 8
            trigger_dict['TriggerDetNum2'] = 1
            trigger_dict['TriggerAmplitude2'] = (pulseamp*gain*1024*4/daqrange+2048).astype(np.int32)
            trigger_dict['TriggerDetNum3'] = 1
            trigger_dict['TriggerAmplitude3'] = (trigamp*gain*1024*4/daqrange+2048).astype(np.int32)
        elif trigtype[2]:
            event_dict['EventTime'] = int(trigtime)
            event_dict['TriggerType'] = 8
            trigger_dict['TriggerDetNum2'] = 0
            trigger_dict['TriggerAmplitude2'] = 0
            trigger_dict['TriggerDetNum3'] = 1
            trigger_dict['TriggerAmplitude3'] = (trigamp*gain*1024*4/daqrange+2048).astype(np.int32)
        elif trigtype[1]:
            event_dict['EventTime'] = int(pulsetime)
            event_dict['TriggerType'] = 1
            trigger_dict['TriggerDetNum2'] = 1
            trigger_dict['TriggerAmplitude2'] = (pulseamp*gain*1024*4/daqrange+2048).astype(np.int32)
            trigger_dict['TriggerDetNum3'] = 0
            trigger_dict['TriggerAmplitude3'] = 0

        # Fill events
        events.insert(ii, {'event'  : event_dict,
                          'trigger' : trigger_dict, 
                          'Z1'      : channels_dict})

        ii+=1

        if ii==maxevts:
            # save events
            filename_out = f"{filename}_F{dumpnum:04}.mid"
            mywriter.open_file(filename_out, filepath)
            mywriter.write_settings_from_dict(settings)
            mywriter.write_events(events)
            mywriter.close_file()

            # start next dump
            dumpnum+=1
            ii=0

    if ii!=0:
        # save events
        filename_out = f"{filename}_F{dumpnum:04}.mid"
        mywriter.open_file(filename_out,filepath)
        mywriter.write_settings_from_dict(settings)
        mywriter.write_events(events)
        mywriter.close_file()



