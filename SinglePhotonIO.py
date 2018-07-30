from scipy.io import loadmat
import numpy as np

def temperature(V):
    p = np.array([-1.39947472e-28, 3.70470900e-23,-4.18400767e-18, 2.61290253e-13,-9.74453858e-09, 2.17094041e-04,-2.67885056e+00, 1.42124188e+04])
    q = np.array([-1.59753112, -0.28226513])
    return np.polyval(p, np.polyval(q, -1.0*np.abs(V))*1e3)

def getChannelsSingleFile(filename, verbose = False):
    if(verbose):
        print('Loading',filename)
    res = loadmat(filename, squeeze_me = False)
    prop = res['exp_prop']
    data = res['data_post']

    exp_prop = dict()
    for line in prop.dtype.names:
        try:
            val     = prop[line][0][0][0]
        except IndexError:
            val     = 'Nothing'
        if type(val) is str:
            exp_prop[line] = val
        elif val.size == 1:
            exp_prop[line] = val[0]
        else:
            exp_prop[line] = np.array(val, dtype = 'f')

    Gains = np.array(prop['SRS'][0][0][0], dtype = 'f')
    Rfbs = np.array(prop['Rfb'][0][0][0], dtype = 'f')
    Turns = np.array(prop['turn_ratio'][0][0][0], dtype = 'f')
    Fs = float(prop['sample_rate'][0][0][0])
    minnum = min(len(Gains), len(Rfbs), len(Turns))
    
    ch1 = data[:,:,0]
    ch2 = data[:,:,1]
    try:
        trig = data[:,:,2]
    except IndexError:
        trig = np.array([])
    ai0 = ch1[:]
    ai1 = ch2[:]
    ai2 = trig[:]
    try:
        ai3 = data[:, :, 3]
    except:
        pass
    
    try:
        ttable  = np.array([24*3600.0,3600.0,60.0,1.0])
        reltime = res['t_rel_trig'].squeeze()
        abstime = res['t_abs_trig'].squeeze()
        timestamp = abstime[:,2:].dot(ttable)+reltime
    except:
        timestamp = np.arange(0,len(ch1))
        if(verbose):
            print('No Timestamps Found')

    dVdI = Turns[:minnum]*Rfbs[:minnum]*Gains[:minnum]
    dIdV = 1.0/dVdI
    
    res = dict()
    res['A'] = ch1*dIdV[0]
    res['B'] = ch2*dIdV[1]
    res['Total'] = res['A']+res['B']
    if trig.size:
        res['T'] = trig*dIdV[2]
    else:
        res['T'] = trig
    res['dVdI'] = dVdI
    res['Fs'] = Fs
    res['prop'] = prop
    res['filenum'] = 1
    res['time'] = timestamp
    res['exp_prop'] = exp_prop
    res['ai0'] = ai0
    res['ai1'] = ai1
    res['ai2'] = ai2
    try:
        res['ai3'] = ai3
    except:
        pass
    return res

def getChannels(filelist,verbose=False):
    
    if(type(filelist) == str):
        return getChannelsSingleFile(filelist,verbose=verbose)
    else:
        res1=getChannelsSingleFile(filelist[0],verbose=verbose)
        combined=dict()
        combined['A']=[res1['A']]
        combined['B']=[res1['B']]
        combined['Total']=[res1['Total']]
        combined['T']=[res1['T']]
        combined['dVdI']=res1['dVdI']
        combined['Fs']=res1['Fs']
        combined['prop']=res1['prop']
        combined['time']=[res1['time']]

        for i in range(1,len(filelist)):
            try:
                res=getChannelsSingleFile(filelist[i],verbose=verbose)
                combined['A'].append(res['A'])
                combined['B'].append(res['B'])
                combined['Total'].append(res['Total'])
                combined['T'].append(res['T'])
                combined['time'].append(res['time'])
            except:
                print('Skipping '+filelist[i])

        combined['A']=np.concatenate(combined['A'])
        combined['B']=np.concatenate(combined['B'])
        combined['Total']=np.concatenate(combined['Total'])
        combined['T']=np.concatenate(combined['T'])
        combined['time']=np.concatenate(combined['time'])

        print(combined['T'].shape)
        
        combined['filenum']=len(filelist)
        
        return combined

def loadDataset(filelist):
    traces=getChannels(filelist,verbose=True)
    tlen=len(traces['A'][0])
    
    dataset=dict()
    dataset['Pulses']=dict()
    dataset['Noise']=dict()
    dataset['Info']=dict()
    
    chans=['A','B','T','Total']
    for chan in chans:
        dataset['Noise'][chan]=traces[chan][:,0:tlen/2]
        dataset['Pulses'][chan]=traces[chan][:,tlen/2:]
    
    for k in traces.keys():
        if k not in chans:
            dataset['Info'][k]=traces[k]
                        
    try:
        dataset['Info']['V_S1']=dataset['Info']['prop']['hvs1'][0][0][0][0]
        dataset['Info']['V_S2']=dataset['Info']['prop']['hvs2'][0][0][0][0]
        dataset['Info']['Voltage']=dataset['Info']['V_S2']-dataset['Info']['V_S1']
    except:
        print('No Voltage Fields')
        dataset['Info']['V_S1']='NA'
        dataset['Info']['V_S2']='NA'
        dataset['Info']['Voltage']='NA'

    try:
        dataset['Info']['LaserRate']=dataset['Info']['prop']['ltr'][0][0][0]
    except:
        print('No Laser Rate Field')
        dataset['Info']['LaserRate']='NA'

    try:
        dataset['Info']['LaserPulseWidth']=dataset['Info']['prop']['lwd'][0][0][0]
    except:
        print('No Laser PW Field')
        dataset['Info']['LaserPulseWidth']='NA'

    try:
        dataset['Info']['LaserPower']=dataset['Info']['prop']['lpk'][0][0][0]
    except:
        print('No Laser Power Field')
        dataset['Info']['LaserPower']='NA'

    try:
        dataset['Info']['LaserAttenuation']=dataset['Info']['prop']['laseratten'][0][0][0][0]
    except:
        print('No Attenuation Field')
        dataset['Info']['LaserAttenuation']='NA'
            
    try:
        dataset['Info']['TriggerThreshold']=dataset['Info']['prop']['threshtrig'][0][0][0][0]
    except:
        print('No Trigger Threshold Field')
        dataset['Info']['TriggerThreshold']='NA'

    try:
        dataset['Info']['Temp']=dataset['Info']['prop']['T_MC'][0][0][0][0]
    except:
        print('No Temperature Field')
        dataset['Info']['Temp']='NA'

    return dataset