import numpy as np
from scipy.signal import hilbert, find_peaks, resample, butter, filtfilt
import mne
import mne_bids
import pathlib
import os
from badsMarker import *
import json
import statistics
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from mne.preprocessing import ICA
from mne_icalabel import label_components
import asrpy
from scipy.stats import zscore
import ast

#Note that to maximise the effectiveness of normalisation, we do this before epoching.
#Then, to minimise loss of temporal resolution, we epoch before downsampling.


#Suppress the specific RuntimeWarning by message
warnings.filterwarnings(
    "ignore",
    message="No bad channels to interpolate. Doing nothing...",
    category=RuntimeWarning)
mne.set_log_level('WARNING')

def process_stimuli(stim_all, lpf_freq, ds_factor, fs):
    processed_stimuli = []
    
    for stim in stim_all:
        
        #Compute the envelope using the Hilbert transform
        a = np.abs(hilbert(stim)) #Mono -> only one stim channel -> no need to average
        
        #Find peaks in the envelope
        peaks, _ = find_peaks(a)
        
        #Prepare for interpolation
        b = np.concatenate(([0], a[peaks], [0]))
        iB = np.concatenate(([0], peaks, [len(stim) - 1]))
        
        #Interpolate the envelope
        c = np.interp(np.arange(len(stim)), iB, b)
        
        #Lowpass filter the interpolated signal
        nyquist = 0.5 * fs
        normal_cutoff = lpf_freq / nyquist
        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        d = filtfilt(b, a, c)
        
        #Downsample the filtered signal
        downsampled = resample(d, len(d) * ds_factor // fs)
        processed_stimuli.append(downsampled)
    
    return processed_stimuli

def preprocessAndEpoch(subject, para):   #Trim start and end
    modifiers = []
    if not para.interpBads: modifiers.append("retainedBads")
    if para.filterTimingOutliers: modifiers.append("removedTimeOutliers")
    if para.doICA and not para.stricterICA: modifiers.append("ICA")
    if para.stricterICA: modifiers.append("stricterICA")
    if para.doASR: modifiers.append("ASR")
    
    mod_tag = "_".join(modifiers) if modifiers else "raw"
    cache_file = os.path.join(para.cache_dir, f"{subject}_multStreamLast30s_scalp_{mod_tag}-epo.fif")
    
    if os.path.exists(cache_file):
        print(f"Loading cached epochs for subject {subject}")
        return mne.read_epochs(cache_file, preload=True)
    
    else:
        task = 'attnMultInstOBs'
        eeg_type = 'scalp'
        subjFolder = "sub-" + subject
        
        base_path = para.basePath
        bids_path = mne_bids.BIDSPath(root=para.dir_data, subject=subject, datatype='eeg', task=task, acquisition=eeg_type)
        eeg_raw = mne_bids.read_raw_bids(bids_path)
        eeg_raw.load_data()
        
        eeg_bpFiltered = eeg_raw.filter(l_freq=para.hpf, h_freq=para.lpf)
        
        #Handle bads by either marking and interpolating them, or keeping them:
        if para.interpBads == True:
            eeg_bpFiltered = badsMarker(eeg_bpFiltered, subjFolder, eeg_type, task)
            eeg_bpFiltered.set_eeg_reference(ref_channels="average")
            eeg_pastInterpStage = eeg_bpFiltered.copy().interpolate_bads()
            eeg_pastInterpStage.set_eeg_reference(ref_channels="average")       
        else:
            eeg_pastInterpStage = eeg_bpFiltered
            
        if para.doASR == True:
            asr = asrpy.ASR(sfreq=eeg_pastInterpStage.info["sfreq"], cutoff=20)
            asr.fit(eeg_pastInterpStage)
            eeg_pastASRstage = asr.transform(eeg_pastInterpStage)

        else:
            eeg_pastASRstage = eeg_pastInterpStage
            
        if para.doICA == True:
            ninterped = len(eeg_bpFiltered.info['bads'])
            nchans = eeg_pastASRstage.info['nchan']
            dims = nchans - ninterped - 1
            ica = ICA(n_components=dims, max_iter="auto", random_state=97, method='infomax', fit_params=dict(extended=True))
            ica.fit(eeg_pastASRstage)
        
            #ICLabel to estimate probabilities
            ic_labels = label_components(eeg_pastASRstage, ica, method="iclabel")
            
            #List to store indices of components to exclude
            exclusion_indices = []
            
            if para.stricterICA == False: #Only remove ones labelled as eye artifacts
            
            #Iterate through each prediction and label
                for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                    #Check if label is 'eye blink'
                    if label in ['eye blink']:
                        #Mark component for exclusion
                        exclusion_indices.append(i)
                
            elif para.stricterICA == True:
                #Threshold for exclusion
                threshold = 0.5
                
            #Iterate through each prediction and label
                for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                    #Check if label is not 'brain' or 'other' and probability is over threshold
                    if label not in ['brain'] or proba < threshold:
                        #Mark component for exclusion
                        exclusion_indices.append(i)
        
            print("Components to exclude:", exclusion_indices)
        
            eeg_preprocessed = eeg_pastASRstage.copy()
            ica.apply(eeg_preprocessed, exclude=exclusion_indices, n_pca_components=dims)
        
        if para.doICA != True:
            eeg_preprocessed = eeg_pastASRstage
            
        #Load event data
        pathToEventsFile = os.path.join(para.dir_data, subjFolder, 'eeg', f'{subjFolder}_task-{task}_acq-{eeg_type}_events.tsv')
        eventsData = pd.read_csv(pathToEventsFile, sep='\t')
    
        #Load event configuration
        pathToConfigFile = os.path.join(para.dir_data, 'task-attnMultInstOBs_events.json')
        with open(pathToConfigFile, 'r') as configFile:
            triggerConfig = json.load(configFile)
        
        eventsArray = np.column_stack([(eventsData['onset']*1000).astype(int), np.zeros(len(eventsData), dtype=int), eventsData['value']])
        unique_events_raw = np.unique(eventsArray[:, 2])
        possibleTrialStartIDs = np.arange(1, 145, 2)
        missing_events = set(possibleTrialStartIDs) - set(unique_events_raw)
        trialStartIDs = list(set(possibleTrialStartIDs) - missing_events)
        
        tmin = 35 + para.trial_trim_start
        tmax = tmin + para.eegPeriodAfterTrim
        epochs = mne.Epochs(eeg_preprocessed, eventsArray, event_id=trialStartIDs, 
                            tmin=tmin, tmax=tmax, baseline=None, detrend=1, preload=True)
    
        #Extract labels
        attd_inst = []
        setID = []
        for event_id in trialStartIDs:
            trigger_info = next((trigger for trigger in triggerConfig['triggers'] if trigger['value'] == event_id), None)
            if trigger_info:
                setID.append(trigger_info['additional_info'][0:5])
                if 'Vibr' in trigger_info['additional_info']:
                    attd_inst.append('Vibr')
                elif 'Harm' in trigger_info['additional_info']:
                    attd_inst.append('Harm')
                elif 'Keyb' in trigger_info['additional_info']:
                    attd_inst.append('Keyb')
        
        epochs.metadata = pd.DataFrame({'Set': setID, 'attd_inst': attd_inst})
        epochs.metadata.reset_index(drop=True, inplace=True)
        
        if para.filterTimingOutliers:
        #Filter out
            pathToErrorsDataFile = os.path.join(para.dir_data, "derivatives/significantEventTimingErrors_perSub", f'{subjFolder}_significantEventTimingErrors_triLen.tsv')
            trigErrorData = pd.read_csv(pathToErrorsDataFile, sep='\t')
            relevantRow = trigErrorData[(trigErrorData['Task']==task) & (trigErrorData['EEG Type']==eeg_type)]
            timingOutliers = relevantRow['Trials with |Δt - <Δt>|>10ms'].iloc[0]
            
            if pd.isna(timingOutliers): #Accounts for where there are none
                timingOutliers = []
            else:
                timingOutliers = ast.literal_eval(timingOutliers)  # safely convert to list
                timingOutliers = [int(i) for i in timingOutliers]  # ensure they're ints

            epochs.drop(timingOutliers)
            
        epochs_normalised = epochs.apply_function(lambda x: zscore(x, axis=-1))
        
        #Downsample EEG
        decimFactor = int(para.eegFs / para.ds)
        epochs_normalised.decimate(decim=decimFactor, offset=0, verbose=None)
    
        epochs_normalised.save(cache_file, overwrite=True)
        
        return epochs_normalised

def preprocessAndEpoch_MSfirst30s(subject, para):   #Trim start and end
    modifiers = []
    if not para.interpBads: modifiers.append("retainedBads")
    if para.filterTimingOutliers: modifiers.append("removedTimeOutliers")
    if para.doICA and not para.stricterICA: modifiers.append("ICA")
    if para.stricterICA: modifiers.append("stricterICA")
    if para.doASR: modifiers.append("ASR")
    
    mod_tag = "_".join(modifiers) if modifiers else "raw"
    cache_file = os.path.join(para.cache_dir, f"{subject}_multStream1st30s_scalp_{mod_tag}-epo.fif")
    
    if os.path.exists(cache_file):
        print(f"Loading cached epochs for subject {subject}")
        return mne.read_epochs(cache_file, preload=True)
    
    else:
        task = 'attnMultInstOBs'
        eeg_type = 'scalp'
        subjFolder = "sub-" + subject
        
        base_path = para.basePath
        bids_path = mne_bids.BIDSPath(root=para.dir_data, subject=subject, datatype='eeg', task=task, acquisition=eeg_type)
        eeg_raw = mne_bids.read_raw_bids(bids_path)
        eeg_raw.load_data()
        
        eeg_bpFiltered = eeg_raw.filter(l_freq=para.hpf, h_freq=para.lpf)
        
        #Handle bads by either marking and interpolating them, or keeping them:
        if para.interpBads == True:
            eeg_bpFiltered = badsMarker(eeg_bpFiltered, subjFolder, eeg_type, task)
            eeg_bpFiltered.set_eeg_reference(ref_channels="average")
            eeg_pastInterpStage = eeg_bpFiltered.copy().interpolate_bads()
            eeg_pastInterpStage.set_eeg_reference(ref_channels="average")       
        else:
            eeg_pastInterpStage = eeg_bpFiltered
            
        if para.doASR == True:
            asr = asrpy.ASR(sfreq=eeg_pastInterpStage.info["sfreq"], cutoff=20)
            asr.fit(eeg_pastInterpStage)
            eeg_pastASRstage = asr.transform(eeg_pastInterpStage)

        else:
            eeg_pastASRstage = eeg_pastInterpStage
            
        if para.doICA == True:
            ninterped = len(eeg_bpFiltered.info['bads'])
            nchans = eeg_pastASRstage.info['nchan']
            dims = nchans - ninterped - 1
            ica = ICA(n_components=dims, max_iter="auto", random_state=97, method='infomax', fit_params=dict(extended=True))
            ica.fit(eeg_pastASRstage)
        
            #ICLabel to estimate probabilities
            ic_labels = label_components(eeg_pastASRstage, ica, method="iclabel")
            
            #List to store indices of components to exclude
            exclusion_indices = []
            
            if para.stricterICA == False: #Only remove ones labelled as eye artifacts
            
            #Iterate through each prediction and label
                for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                    #Check if label is 'eye blink'
                    if label in ['eye blink']:
                        #Mark component for exclusion
                        exclusion_indices.append(i)
                
            elif para.stricterICA == True:
                #Threshold for exclusion
                threshold = 0.5
                
            #Iterate through each prediction and label
                for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                    #Check if label is not 'brain' or 'other' and probability is over threshold
                    if label not in ['brain'] or proba < threshold:
                        #Mark component for exclusion
                        exclusion_indices.append(i)
        
            print("Components to exclude:", exclusion_indices)
        
            eeg_preprocessed = eeg_pastASRstage.copy()
            ica.apply(eeg_preprocessed, exclude=exclusion_indices, n_pca_components=dims)
        
        if para.doICA != True:
            eeg_preprocessed = eeg_pastASRstage
            
        #Load event data
        pathToEventsFile = os.path.join(para.dir_data, subjFolder, 'eeg', f'{subjFolder}_task-{task}_acq-{eeg_type}_events.tsv')
        eventsData = pd.read_csv(pathToEventsFile, sep='\t')
    
        #Load event configuration
        pathToConfigFile = os.path.join(para.dir_data, 'task-attnMultInstOBs_events.json')
        with open(pathToConfigFile, 'r') as configFile:
            triggerConfig = json.load(configFile)
        
        eventsArray = np.column_stack([(eventsData['onset']*1000).astype(int), np.zeros(len(eventsData), dtype=int), eventsData['value']])
        unique_events_raw = np.unique(eventsArray[:, 2])
        possibleTrialStartIDs = np.arange(1, 145, 2)
        missing_events = set(possibleTrialStartIDs) - set(unique_events_raw)
        trialStartIDs = list(set(possibleTrialStartIDs) - missing_events)
        
        tmin = para.trial_trim_start
        tmax = tmin + para.eegPeriodAfterTrim
        epochs = mne.Epochs(eeg_preprocessed, eventsArray, event_id=trialStartIDs, 
                            tmin=tmin, tmax=tmax, baseline=None, detrend=1, preload=True)
    
        #Extract labels
        attd_inst = []
        setID = []
        for event_id in trialStartIDs:
            trigger_info = next((trigger for trigger in triggerConfig['triggers'] if trigger['value'] == event_id), None)
            if trigger_info:
                setID.append(trigger_info['additional_info'][0:5])
                if 'Vibr' in trigger_info['additional_info']:
                    attd_inst.append('Vibr')
                elif 'Harm' in trigger_info['additional_info']:
                    attd_inst.append('Harm')
                elif 'Keyb' in trigger_info['additional_info']:
                    attd_inst.append('Keyb')
        
        epochs.metadata = pd.DataFrame({'Set': setID, 'attd_inst': attd_inst})
        epochs.metadata.reset_index(drop=True, inplace=True)
        
        if para.filterTimingOutliers:
        #Filter out
            pathToErrorsDataFile = os.path.join(para.dir_data, "derivatives/significantEventTimingErrors_perSub", f'{subjFolder}_significantEventTimingErrors_triLen.tsv')
            trigErrorData = pd.read_csv(pathToErrorsDataFile, sep='\t')
            relevantRow = trigErrorData[(trigErrorData['Task']==task) & (trigErrorData['EEG Type']==eeg_type)]
            timingOutliers = relevantRow['Trials with |Δt - <Δt>|>10ms'].iloc[0]
            
            if pd.isna(timingOutliers): #Accounts for where there are none
                timingOutliers = []
            else:
                timingOutliers = ast.literal_eval(timingOutliers)  # safely convert to list
                timingOutliers = [int(i) for i in timingOutliers]  # ensure they're ints

            epochs.drop(timingOutliers)
    
        epochs_normalised = epochs.apply_function(lambda x: zscore(x, axis=-1))
        
        #Downsample EEG
        decimFactor = int(para.eegFs / para.ds)
        epochs_normalised.decimate(decim=decimFactor, offset=0, verbose=None)
        
        epochs_normalised.save(cache_file, overwrite=True)
        
        return epochs_normalised

def preprocessAndEpoch_singleStream(subject, para):
    modifiers = []
    if not para.interpBads: modifiers.append("retainedBads")
    if para.filterTimingOutliers: modifiers.append("removedTimeOutliers")
    if para.doICA and not para.stricterICA: modifiers.append("ICA")
    if para.stricterICA: modifiers.append("stricterICA")
    if para.doASR: modifiers.append("ASR")
    
    mod_tag = "_".join(modifiers) if modifiers else "raw"
    cache_file = os.path.join(para.cache_dir, f"{subject}_singStream_scalp_{mod_tag}-epo.fif")
    
    if os.path.exists(cache_file):
        print(f"Loading cached epochs for subject {subject}")
        return mne.read_epochs(cache_file, preload=True)
    
    else:
        task = 'attnOneInstNoOBs'
        eeg_type = 'scalp'
        subjFolder = "sub-" + subject
        
        base_path = pathlib.Path(__file__).resolve().parents[2]
        bids_path = mne_bids.BIDSPath(root=para.dir_data, subject=subject, datatype='eeg', task=task, acquisition=eeg_type)
        eeg_raw = mne_bids.read_raw_bids(bids_path)
        eeg_raw.load_data()
        
        eeg_bpFiltered = eeg_raw.filter(l_freq=para.hpf, h_freq=para.lpf)
        
        #Handle bads by either marking and interpolating them, or keeping them:
        if para.interpBads == True:
            eeg_bpFiltered = badsMarker(eeg_bpFiltered, subjFolder, eeg_type, task)
            eeg_bpFiltered.set_eeg_reference(ref_channels="average")
            eeg_pastInterpStage = eeg_bpFiltered.copy().interpolate_bads()
            eeg_pastInterpStage.set_eeg_reference(ref_channels="average")       
        else:
            eeg_pastInterpStage = eeg_bpFiltered
            
        if para.doASR == True:
            asr = asrpy.ASR(sfreq=eeg_pastInterpStage.info["sfreq"], cutoff=20)
            asr.fit(eeg_pastInterpStage)
            eeg_pastASRstage = asr.transform(eeg_pastInterpStage)

        else:
            eeg_pastASRstage = eeg_pastInterpStage
            
        if para.doICA == True:
            ninterped = len(eeg_bpFiltered.info['bads'])
            nchans = eeg_pastASRstage.info['nchan']
            dims = nchans - ninterped - 1
            ica = ICA(n_components=dims, max_iter="auto", random_state=97, method='infomax', fit_params=dict(extended=True))
            ica.fit(eeg_pastASRstage)
        
            #ICLabel to estimate probabilities
            ic_labels = label_components(eeg_pastASRstage, ica, method="iclabel")
            
            #List to store indices of components to exclude
            exclusion_indices = []
            
            if para.stricterICA == False: #Only remove ones labelled as eye artifacts
            
            #Iterate through each prediction and label
                for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                    #Check if label is 'eye blink'
                    if label in ['eye blink']:
                        #Mark component for exclusion
                        exclusion_indices.append(i)
                
            elif para.stricterICA == True:
                #Threshold for exclusion
                threshold = 0.5
                
            #Iterate through each prediction and label
                for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                    #Check if label is not 'brain' or 'other' and probability is over threshold
                    if label not in ['brain'] or proba < threshold:
                        #Mark component for exclusion
                        exclusion_indices.append(i)
        
            print("Components to exclude:", exclusion_indices)
        
            eeg_preprocessed = eeg_pastASRstage.copy()
            ica.apply(eeg_preprocessed, exclude=exclusion_indices, n_pca_components=dims)
        
        if para.doICA != True:
            eeg_preprocessed = eeg_pastASRstage
        
        #Load event data
        pathToEventsFile = os.path.join(para.dir_data, subjFolder, 'eeg', f'{subjFolder}_task-{task}_acq-{eeg_type}_events.tsv')
        eventsData = pd.read_csv(pathToEventsFile, sep='\t')
    
        #Load event configuration
        pathToConfigFile = os.path.join(para.dir_data, 'task-attnOneInstNoOBs_events.json')
        
        with open(pathToConfigFile, 'r') as configFile:
            triggerConfig = json.load(configFile)
        #MUST CONVERT eventsData['onset'] FROM SECONDS TO MS
        eventsArray = np.column_stack([(eventsData['onset']*para.eegFs).astype(int), np.zeros(len(eventsData), dtype=int), eventsData['value']])
        
        unique_events_raw = np.unique(eventsArray[:, 2])  #All event IDs in this particular task
        possibleTrialStartIDs = np.arange(1, 145, 2)  #ALL possible trial start IDs for all tasks
        missing_events = set(possibleTrialStartIDs) - set(unique_events_raw)  #Possible start IDs NOT in this particular task
        
        trialStartIDs = []
        for event in eventsArray:
            event_id = event[2]
            if event_id in possibleTrialStartIDs and event_id not in missing_events and event_id not in trialStartIDs:
                trialStartIDs.append(event_id)
        
        event_id = dict(zip(map(str, trialStartIDs), trialStartIDs))
        
        tmin = para.trial_trim_start
        tmax = tmin+para.eegPeriodAfterTrim
        #Create epochs
        epochs = mne.Epochs(eeg_preprocessed, eventsArray, event_id=event_id, 
                            tmin=tmin, tmax=tmax, baseline=None, detrend=1, preload=True)
        
        #Add labels:
                
        pathToAttendanceData = os.path.join(para.dir_data, subjFolder, 'beh',  f'{subjFolder}_task-{task}_beh.tsv')
    
        df = pd.read_csv(pathToAttendanceData,sep='\t')
        df = df[['stimuli','music_attended']]
        
        labels = []
        stimulus_all = []
        for i in range(0, len(df)-1): #Includes prac trial
            attdMus = df.iloc[i]['music_attended']
            if attdMus == "Yes":
                label = "attd"
            if attdMus == "No":
                label = "unattd"
            labels+= [label]
            
            stimulus = df.iloc[i]['stimuli'][-32:-22]
            stimulus_all += [stimulus]
            
        #Add metadata with labels to epochs
        epochs.metadata = pd.DataFrame({'stimulus': stimulus_all,'music_attd': labels})
        epochs.metadata.reset_index(drop=True, inplace=True)  #Reset index to ensure correct indexing
        
        if para.filterTimingOutliers:
        #Filter out
            pathToErrorsDataFile = os.path.join(para.dir_data, "derivatives/significantEventTimingErrors_perSub", f'{subjFolder}_significantEventTimingErrors_triLen.tsv')
            trigErrorData = pd.read_csv(pathToErrorsDataFile, sep='\t')
            relevantRow = trigErrorData[(trigErrorData['Task']==task) & (trigErrorData['EEG Type']==eeg_type)]
            timingOutliers = relevantRow['Trials with |Δt - <Δt>|>10ms'].iloc[0]
            
            if pd.isna(timingOutliers): #Accounts for where there are none
                timingOutliers = []
            else:
                timingOutliers = ast.literal_eval(timingOutliers)  # safely convert to list
                timingOutliers = [int(i) for i in timingOutliers]  # ensure they're ints

            epochs.drop(timingOutliers)
            
        epochs_normalised = epochs.apply_function(lambda x: zscore(x, axis=-1))
    
        #Downsample EEG
        decimFactor = int(para.eegFs / para.ds)
        epochs_normalised.decimate(decim=decimFactor, offset=0, verbose=None)
        
        epochs_normalised.save(cache_file, overwrite=True)

        return epochs_normalised
    
def preprocessAndEpoch_emoDec(subject, para):
    modifiers = []
    if not para.interpBads: modifiers.append("retainedBads")
    if para.filterTimingOutliers: modifiers.append("removedTimeOutliers")
    if para.doICA and not para.stricterICA: modifiers.append("ICA")
    if para.stricterICA: modifiers.append("stricterICA")
    if para.doASR: modifiers.append("ASR")
    
    mod_tag = "_".join(modifiers) if modifiers else "raw"
    cache_file = os.path.join(para.cache_dir, f"{subject}_emoDec_scalp_{mod_tag}-epo.fif")

    if os.path.exists(cache_file):
        print(f"Loading cached epochs for subject {subject}")
        return mne.read_epochs(cache_file, preload=True)
    
    else:
        task = 'emotion'
        eeg_type = 'scalp'
        subjFolder = "sub-" + subject
        
        base_path = pathlib.Path(__file__).resolve().parents[2]
        bids_path = mne_bids.BIDSPath(root=para.dir_data, subject=subject, datatype='eeg', task=task, acquisition=eeg_type)
        eeg_raw = mne_bids.read_raw_bids(bids_path)
        eeg_raw.load_data()
        
        eeg_bpFiltered = eeg_raw.filter(l_freq=para.hpf, h_freq=para.lpf)
        
        #Handle bads by either marking and interpolating them, or keeping them:
        if para.interpBads == True:
            eeg_bpFiltered = badsMarker(eeg_bpFiltered, subjFolder, eeg_type, task)
            eeg_bpFiltered.set_eeg_reference(ref_channels="average")
            eeg_pastInterpStage = eeg_bpFiltered.copy().interpolate_bads()
            eeg_pastInterpStage.set_eeg_reference(ref_channels="average")       
        else:
            eeg_pastInterpStage = eeg_bpFiltered
            
        if para.doASR == True:
            asr = asrpy.ASR(sfreq=eeg_pastInterpStage.info["sfreq"], cutoff=20)
            asr.fit(eeg_pastInterpStage)
            eeg_pastASRstage = asr.transform(eeg_pastInterpStage)

        else:
            eeg_pastASRstage = eeg_pastInterpStage
            
        if para.doICA == True:
            ninterped = len(eeg_bpFiltered.info['bads'])
            nchans = eeg_pastASRstage.info['nchan']
            dims = nchans - ninterped - 1
            ica = ICA(n_components=dims, max_iter="auto", random_state=97, method='infomax', fit_params=dict(extended=True))
            ica.fit(eeg_pastASRstage)
        
            #ICLabel to estimate probabilities
            ic_labels = label_components(eeg_pastASRstage, ica, method="iclabel")
            
            #List to store indices of components to exclude
            exclusion_indices = []
            
            if para.stricterICA == False: #Only remove ones labelled as eye artifacts
            
            #Iterate through each prediction and label
                for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                    #Check if label is 'eye blink'
                    if label in ['eye blink']:
                        #Mark component for exclusion
                        exclusion_indices.append(i)
                
            elif para.stricterICA == True:
                #Threshold for exclusion
                threshold = 0.5
                
            #Iterate through each prediction and label
                for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                    #Check if label is not 'brain' or 'other' and probability is over threshold
                    if label not in ['brain'] or proba < threshold:
                        #Mark component for exclusion
                        exclusion_indices.append(i)
        
            print("Components to exclude:", exclusion_indices)
        
            eeg_preprocessed = eeg_pastASRstage.copy()
            ica.apply(eeg_preprocessed, exclude=exclusion_indices, n_pca_components=dims)
        
        if para.doICA != True:
            eeg_preprocessed = eeg_pastASRstage
        
        #Load event data
        pathToEventsFile = os.path.join(para.dir_data, subjFolder, 'eeg', f'{subjFolder}_task-{task}_acq-{eeg_type}_events.tsv')
        eventsData = pd.read_csv(pathToEventsFile, sep='\t')
    
        #Load event configuration
        pathToConfigFile = os.path.join(para.dir_data, 'task-attnOneInstNoOBs_events.json')
        
        with open(pathToConfigFile, 'r') as configFile:
            triggerConfig = json.load(configFile)
        #MUST CONVERT eventsData['onset'] FROM SECONDS TO MS
        eventsArray = np.column_stack([(eventsData['onset']*para.eegFs).astype(int), np.zeros(len(eventsData), dtype=int), eventsData['value']])
        
        unique_events_raw = np.unique(eventsArray[:, 2])  #All event IDs in this particular task
        possibleTrialStartIDs = np.arange(1, 145, 2)  #ALL possible trial start IDs for all tasks
        missing_events = set(possibleTrialStartIDs) - set(unique_events_raw)  #Possible start IDs NOT in this particular task
        
        trialStartIDs = []
        for event in eventsArray:
            event_id = event[2]
            if event_id in possibleTrialStartIDs and event_id not in missing_events and event_id not in trialStartIDs:
                trialStartIDs.append(event_id)
        
        event_id = dict(zip(map(str, trialStartIDs), trialStartIDs))
        
        tmin = para.trial_trim_start
        tmax = tmin+para.eegPeriodAfterTrim
        
        #Create epochs
        epochs = mne.Epochs(eeg_preprocessed, eventsArray, event_id=event_id, 
                            tmin=tmin, tmax=tmax, baseline=None, detrend=1, preload=True)
        
        #Add labels:
                
        pathToAttendanceData = os.path.join(para.dir_data, subjFolder, 'beh',  f'{subjFolder}_task-{task}_beh.tsv')
    
        df = pd.read_csv(pathToAttendanceData,sep='\t')
        
        df = df[['stimuli']]
      
        stimulus_all = []
    
        for i in range(0, len(df)-1): #Includes prac trial
             
             stimulus = df.iloc[i]['stimuli'][-32:-22]
             stimulus_all += [stimulus]
             
         #Add metadata with labels to epochs
        epochs.metadata = pd.DataFrame({'stimulus': stimulus_all})
        epochs.metadata.reset_index(drop=True, inplace=True)  #Reset index to ensure correct indexing
    
        if para.filterTimingOutliers:
        #Filter out
            pathToErrorsDataFile = os.path.join(para.dir_data, "derivatives/significantEventTimingErrors_perSub", f'{subjFolder}_significantEventTimingErrors_triLen.tsv')
            trigErrorData = pd.read_csv(pathToErrorsDataFile, sep='\t')
            relevantRow = trigErrorData[(trigErrorData['Task']==task) & (trigErrorData['EEG Type']==eeg_type)]
            timingOutliers = relevantRow['Trials with |Δt - <Δt>|>10ms'].iloc[0]
            
            if pd.isna(timingOutliers): #Accounts for where there are none
                timingOutliers = []
            else:
                timingOutliers = ast.literal_eval(timingOutliers)  # safely convert to list
                timingOutliers = [int(i) for i in timingOutliers]  # ensure they're ints

            epochs.drop(timingOutliers)
            
        epochs_normalised = epochs.apply_function(lambda x: zscore(x, axis=-1))
    
        #Downsample EEG
        decimFactor = int(para.eegFs / para.ds)
        epochs_normalised.decimate(decim=decimFactor, offset=0, verbose=None)
        
        epochs_normalised.save(cache_file, overwrite=True)
        
        return epochs_normalised
    
#Plotting function for confusion matrix
def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def preprocessAndEpoch_ceegrid(subject, para):   #Trim start and end
    modifiers = []
    if not para.interpBads: modifiers.append("retainedBads")
    if para.filterTimingOutliers: modifiers.append("removedTimeOutliers")
    if para.doICA and not para.stricterICA: modifiers.append("ICA")
    if para.stricterICA: modifiers.append("stricterICA")
    if para.doASR: modifiers.append("ASR")
    if para.removeFpz != True:
        modifiers.append("FpzKept")
    
    mod_tag = "_".join(modifiers) if modifiers else "raw"
    cache_file = os.path.join(para.cache_dir, f"{subject}_multStreamLasthalf_ceegrid_{mod_tag}-epo.fif")
    
    if os.path.exists(cache_file):
        print(f"Loading cached epochs for subject {subject}")
        return mne.read_epochs(cache_file, preload=True)
    
    else:     
        task = 'attnMultInstOBs'
        eeg_type = 'ceegrid'
        subjFolder = "sub-" + subject
                
        base_path = para.basePath
        bids_path = mne_bids.BIDSPath(root=para.dir_data, subject=subject, datatype='eeg', task=task, acquisition=eeg_type)
        eeg_raw = mne_bids.read_raw_bids(bids_path)
        eeg_raw.load_data()
        
        if para.removeFpz == True:
            eeg_raw.drop_channels(['Fpz'])
            eeg_raw.set_eeg_reference(ref_channels="average") #Removed a channel, so given that we're using average ref it is best to reref
        
        eeg_bpFiltered = eeg_raw.filter(l_freq=para.hpf, h_freq=para.lpf)
        
        #Handle bads by either marking and interpolating them, or keeping them:
        if para.interpBads == True:
            eeg_bpFiltered = badsMarker(eeg_bpFiltered, subjFolder, eeg_type, task)
            eeg_bpFiltered.set_eeg_reference(ref_channels="average")
            eeg_pastInterpStage = eeg_bpFiltered.copy().interpolate_bads()
            eeg_pastInterpStage.set_eeg_reference(ref_channels="average")       
        else:
            eeg_pastInterpStage = eeg_bpFiltered
            
        if para.doASR == True:
            asr = asrpy.ASR(sfreq=eeg_pastInterpStage.info["sfreq"], cutoff=20)
            asr.fit(eeg_pastInterpStage)
            eeg_pastASRstage = asr.transform(eeg_pastInterpStage)

        else:
            eeg_pastASRstage = eeg_pastInterpStage
            
        if para.doICA == True:
            ninterped = len(eeg_bpFiltered.info['bads'])
            nchans = eeg_pastASRstage.info['nchan']
            dims = nchans - ninterped - 1
            ica = ICA(n_components=dims, max_iter="auto", random_state=97, method='infomax', fit_params=dict(extended=True))
            ica.fit(eeg_pastASRstage)
        
            #ICLabel to estimate probabilities
            ic_labels = label_components(eeg_pastASRstage, ica, method="iclabel")
            
            #List to store indices of components to exclude
            exclusion_indices = []
            
            if para.stricterICA == False: #Only remove ones labelled as eye artifacts
            
            #Iterate through each prediction and label
                for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                    #Check if label is 'eye blink'
                    if label in ['eye blink']:
                        #Mark component for exclusion
                        exclusion_indices.append(i)
                
            elif para.stricterICA == True:
                #Threshold for exclusion
                threshold = 0.5
                
            #Iterate through each prediction and label
                for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                    #Check if label is not 'brain' or 'other' and probability is over threshold
                    if label not in ['brain'] or proba < threshold:
                        #Mark component for exclusion
                        exclusion_indices.append(i)
        
            print("Components to exclude:", exclusion_indices)
        
            eeg_preprocessed = eeg_pastASRstage.copy()
            ica.apply(eeg_preprocessed, exclude=exclusion_indices, n_pca_components=dims)
        
        if para.doICA != True:
            eeg_preprocessed = eeg_pastASRstage
            
        #Load event data
        pathToEventsFile = os.path.join(para.dir_data, subjFolder, 'eeg', f'{subjFolder}_task-{task}_acq-{eeg_type}_events.tsv')
        eventsData = pd.read_csv(pathToEventsFile, sep='\t')
    
        #Load event configuration
        pathToConfigFile = os.path.join(para.dir_data, 'task-attnMultInstOBs_events.json')
        with open(pathToConfigFile, 'r') as configFile:
            triggerConfig = json.load(configFile)
        
        eventsArray = np.column_stack([(eventsData['onset']*1000).astype(int), np.zeros(len(eventsData), dtype=int), eventsData['value']])
        unique_events_raw = np.unique(eventsArray[:, 2])
        possibleTrialStartIDs = np.arange(1, 145, 2)
        missing_events = set(possibleTrialStartIDs) - set(unique_events_raw)
        trialStartIDs = list(set(possibleTrialStartIDs) - missing_events)
        
        tmin = 35 + para.trial_trim_start
        tmax = tmin+para.eegPeriodAfterTrim
        
        epochs = mne.Epochs(eeg_preprocessed, eventsArray, event_id=trialStartIDs, 
                            tmin=tmin, tmax=tmax, baseline=None, detrend=1, preload=True)
    
        #Extract labels
        attd_inst = []
        setID = []
        for event_id in trialStartIDs:
            trigger_info = next((trigger for trigger in triggerConfig['triggers'] if trigger['value'] == event_id), None)
            if trigger_info:
                setID.append(trigger_info['additional_info'][0:5])
                if 'Vibr' in trigger_info['additional_info']:
                    attd_inst.append('Vibr')
                elif 'Harm' in trigger_info['additional_info']:
                    attd_inst.append('Harm')
                elif 'Keyb' in trigger_info['additional_info']:
                    attd_inst.append('Keyb')
        
        epochs.metadata = pd.DataFrame({'Set': setID, 'attd_inst': attd_inst})
        epochs.metadata.reset_index(drop=True, inplace=True)
        
        if para.filterTimingOutliers:
        #Filter out
            pathToErrorsDataFile = os.path.join(para.dir_data, "derivatives/significantEventTimingErrors_perSub", f'{subjFolder}_significantEventTimingErrors_triLen.tsv')
            trigErrorData = pd.read_csv(pathToErrorsDataFile, sep='\t')
            relevantRow = trigErrorData[(trigErrorData['Task']==task) & (trigErrorData['EEG Type']==eeg_type)]
            timingOutliers = relevantRow['Trials with |Δt - <Δt>|>10ms'].iloc[0]
            
            if pd.isna(timingOutliers): #Accounts for where there are none
                timingOutliers = []
            else:
                timingOutliers = ast.literal_eval(timingOutliers)  # safely convert to list
                timingOutliers = [int(i) for i in timingOutliers]  # ensure they're ints

            epochs.drop(timingOutliers)
            
        epochs_normalised = epochs.apply_function(lambda x: zscore(x, axis=-1))
        
        #Downsample EEG
        decimFactor = int(para.eegFs / para.ds)
        epochs_normalised.decimate(decim=decimFactor, offset=0, verbose=None)
        
        epochs_normalised.save(cache_file, overwrite=True)
        
        return epochs_normalised

def preprocessAndEpoch_ceegrid_MSfirst30s(subject, para):   #Trim start and end
    modifiers = []
    if not para.interpBads: modifiers.append("retainedBads")
    if para.filterTimingOutliers: modifiers.append("removedTimeOutliers")
    if para.doICA and not para.stricterICA: modifiers.append("ICA")
    if para.stricterICA: modifiers.append("stricterICA")
    if para.doASR: modifiers.append("ASR")
    if para.removeFpz != True:
        modifiers.append("FpzKept")
    
    mod_tag = "_".join(modifiers) if modifiers else "raw"
    cache_file = os.path.join(para.cache_dir, f"{subject}_multStream1st30s_ceegrid_{mod_tag}-epo.fif")
    
    if os.path.exists(cache_file):
        print(f"Loading cached epochs for subject {subject}")
        return mne.read_epochs(cache_file, preload=True)
    
    else:     
        task = 'attnMultInstOBs'
        eeg_type = 'ceegrid'
        subjFolder = "sub-" + subject
        
        base_path = para.basePath
        bids_path = mne_bids.BIDSPath(root=para.dir_data, subject=subject, datatype='eeg', task=task, acquisition=eeg_type)
        eeg_raw = mne_bids.read_raw_bids(bids_path)
        eeg_raw.load_data()
        
        if para.removeFpz == True:
            eeg_raw.drop_channels(['Fpz'])
            eeg_raw.set_eeg_reference(ref_channels="average") #Removed a channel, so given that we're using average ref it is best to reref
            
        eeg_bpFiltered = eeg_raw.filter(l_freq=para.hpf, h_freq=para.lpf)
        
        #Handle bads by either marking and interpolating them, or keeping them:
        if para.interpBads == True:
            eeg_bpFiltered = badsMarker(eeg_bpFiltered, subjFolder, eeg_type, task)
            eeg_bpFiltered.set_eeg_reference(ref_channels="average")
            eeg_pastInterpStage = eeg_bpFiltered.copy().interpolate_bads()
            eeg_pastInterpStage.set_eeg_reference(ref_channels="average")       
        else:
            eeg_pastInterpStage = eeg_bpFiltered
            
        if para.doASR == True:
            asr = asrpy.ASR(sfreq=eeg_pastInterpStage.info["sfreq"], cutoff=20)
            asr.fit(eeg_pastInterpStage)
            eeg_pastASRstage = asr.transform(eeg_pastInterpStage)

        else:
            eeg_pastASRstage = eeg_pastInterpStage
            
        if para.doICA == True:
            ninterped = len(eeg_bpFiltered.info['bads'])
            nchans = eeg_pastASRstage.info['nchan']
            dims = nchans - ninterped - 1
            ica = ICA(n_components=dims, max_iter="auto", random_state=97, method='infomax', fit_params=dict(extended=True))
            ica.fit(eeg_pastASRstage)
        
            #ICLabel to estimate probabilities
            ic_labels = label_components(eeg_pastASRstage, ica, method="iclabel")
            
            #List to store indices of components to exclude
            exclusion_indices = []
            
            if para.stricterICA == False: #Only remove ones labelled as eye artifacts
            
            #Iterate through each prediction and label
                for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                    #Check if label is 'eye blink'
                    if label in ['eye blink']:
                        #Mark component for exclusion
                        exclusion_indices.append(i)
                
            elif para.stricterICA == True:
                #Threshold for exclusion
                threshold = 0.5
                
            #Iterate through each prediction and label
                for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                    #Check if label is not 'brain' or 'other' and probability is over threshold
                    if label not in ['brain'] or proba < threshold:
                        #Mark component for exclusion
                        exclusion_indices.append(i)
        
            print("Components to exclude:", exclusion_indices)
        
            eeg_preprocessed = eeg_pastASRstage.copy()
            ica.apply(eeg_preprocessed, exclude=exclusion_indices, n_pca_components=dims)
        
        if para.doICA != True:
            eeg_preprocessed = eeg_pastASRstage
            
        #Load event data
        pathToEventsFile = os.path.join(para.dir_data, subjFolder, 'eeg', f'{subjFolder}_task-{task}_acq-{eeg_type}_events.tsv')
        eventsData = pd.read_csv(pathToEventsFile, sep='\t')
    
        #Load event configuration
        pathToConfigFile = os.path.join(para.dir_data, 'task-attnMultInstOBs_events.json')
        with open(pathToConfigFile, 'r') as configFile:
            triggerConfig = json.load(configFile)
        
        eventsArray = np.column_stack([(eventsData['onset']*1000).astype(int), np.zeros(len(eventsData), dtype=int), eventsData['value']])
        unique_events_raw = np.unique(eventsArray[:, 2])
        possibleTrialStartIDs = np.arange(1, 145, 2)
        missing_events = set(possibleTrialStartIDs) - set(unique_events_raw)
        trialStartIDs = list(set(possibleTrialStartIDs) - missing_events)
        
        tmin = para.trial_trim_start
        tmax = tmin+para.eegPeriodAfterTrim
        
        epochs = mne.Epochs(eeg_preprocessed, eventsArray, event_id=trialStartIDs, 
                            tmin=tmin, tmax=tmax, baseline=None, detrend=1, preload=True)
    
        #Extract labels
        attd_inst = []
        setID = []
        for event_id in trialStartIDs:
            trigger_info = next((trigger for trigger in triggerConfig['triggers'] if trigger['value'] == event_id), None)
            if trigger_info:
                setID.append(trigger_info['additional_info'][0:5])
                if 'Vibr' in trigger_info['additional_info']:
                    attd_inst.append('Vibr')
                elif 'Harm' in trigger_info['additional_info']:
                    attd_inst.append('Harm')
                elif 'Keyb' in trigger_info['additional_info']:
                    attd_inst.append('Keyb')
        
        epochs.metadata = pd.DataFrame({'Set': setID, 'attd_inst': attd_inst})
        epochs.metadata.reset_index(drop=True, inplace=True)
        
        if para.filterTimingOutliers:
        #Filter out
            pathToErrorsDataFile = os.path.join(para.dir_data, "derivatives/significantEventTimingErrors_perSub", f'{subjFolder}_significantEventTimingErrors_triLen.tsv')
            trigErrorData = pd.read_csv(pathToErrorsDataFile, sep='\t')
            relevantRow = trigErrorData[(trigErrorData['Task']==task) & (trigErrorData['EEG Type']==eeg_type)]
            timingOutliers = relevantRow['Trials with |Δt - <Δt>|>10ms'].iloc[0]
            
            if pd.isna(timingOutliers): #Accounts for where there are none
                timingOutliers = []
            else:
                timingOutliers = ast.literal_eval(timingOutliers)  # safely convert to list
                timingOutliers = [int(i) for i in timingOutliers]  # ensure they're ints

            epochs.drop(timingOutliers)
    
        epochs_normalised = epochs.apply_function(lambda x: zscore(x, axis=-1))
        
        #Downsample EEG
        decimFactor = int(para.eegFs / para.ds)
        epochs_normalised.decimate(decim=decimFactor, offset=0, verbose=None)
        
        epochs_normalised.save(cache_file, overwrite=True)
        
        return epochs_normalised
    
def preprocessAndEpoch_singleStream_ceegrid(subject, para):
    modifiers = []
    if not para.interpBads: modifiers.append("retainedBads")
    if para.filterTimingOutliers: modifiers.append("removedTimeOutliers")
    if para.doICA and not para.stricterICA: modifiers.append("ICA")
    if para.stricterICA: modifiers.append("stricterICA")
    if para.doASR: modifiers.append("ASR")
    if para.removeFpz != True:
        modifiers.append("FpzKept")
    
    mod_tag = "_".join(modifiers) if modifiers else "raw"
    cache_file = os.path.join(para.cache_dir, f"{subject}_singStream_ceegrid_{mod_tag}-epo.fif")
    
    if os.path.exists(cache_file):
        print(f"Loading cached epochs for subject {subject}")
        return mne.read_epochs(cache_file, preload=True)
    
    else:
        task = 'attnOneInstNoOBs'
        eeg_type = 'ceegrid'
        subjFolder = "sub-" + subject
        
        base_path = pathlib.Path(__file__).resolve().parents[2]
        bids_path = mne_bids.BIDSPath(root=para.dir_data, subject=subject, datatype='eeg', task=task, acquisition=eeg_type)
        eeg_raw = mne_bids.read_raw_bids(bids_path)
        eeg_raw.load_data()
        
        if para.removeFpz == True:
            eeg_raw.drop_channels(['Fpz'])
            eeg_raw.set_eeg_reference(ref_channels="average") #Removed a channel, so given that we're using average ref it is best to reref
            
        eeg_bpFiltered = eeg_raw.filter(l_freq=para.hpf, h_freq=para.lpf)
        
        #Handle bads by either marking and interpolating them, or keeping them:
        if para.interpBads == True:
            eeg_bpFiltered = badsMarker(eeg_bpFiltered, subjFolder, eeg_type, task)
            eeg_bpFiltered.set_eeg_reference(ref_channels="average")
            eeg_pastInterpStage = eeg_bpFiltered.copy().interpolate_bads()
            eeg_pastInterpStage.set_eeg_reference(ref_channels="average")       
        else:
            eeg_pastInterpStage = eeg_bpFiltered
            
        if para.doASR == True:
            asr = asrpy.ASR(sfreq=eeg_pastInterpStage.info["sfreq"], cutoff=20)
            asr.fit(eeg_pastInterpStage)
            eeg_pastASRstage = asr.transform(eeg_pastInterpStage)

        else:
            eeg_pastASRstage = eeg_pastInterpStage
            
        if para.doICA == True:
            ninterped = len(eeg_bpFiltered.info['bads'])
            nchans = eeg_pastASRstage.info['nchan']
            dims = nchans - ninterped - 1
            ica = ICA(n_components=dims, max_iter="auto", random_state=97, method='infomax', fit_params=dict(extended=True))
            ica.fit(eeg_pastASRstage)
        
            #ICLabel to estimate probabilities
            ic_labels = label_components(eeg_pastASRstage, ica, method="iclabel")
            
            #List to store indices of components to exclude
            exclusion_indices = []
            
            if para.stricterICA == False: #Only remove ones labelled as eye artifacts
            
            #Iterate through each prediction and label
                for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                    #Check if label is 'eye blink'
                    if label in ['eye blink']:
                        #Mark component for exclusion
                        exclusion_indices.append(i)
                
            elif para.stricterICA == True:
                #Threshold for exclusion
                threshold = 0.5
                
            #Iterate through each prediction and label
                for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                    #Check if label is not 'brain' or 'other' and probability is over threshold
                    if label not in ['brain'] or proba < threshold:
                        #Mark component for exclusion
                        exclusion_indices.append(i)
        
            print("Components to exclude:", exclusion_indices)
        
            eeg_preprocessed = eeg_pastASRstage.copy()
            ica.apply(eeg_preprocessed, exclude=exclusion_indices, n_pca_components=dims)
        
        if para.doICA != True:
            eeg_preprocessed = eeg_pastASRstage
        
        #Load event data
        pathToEventsFile = os.path.join(para.dir_data, subjFolder, 'eeg', f'{subjFolder}_task-{task}_acq-{eeg_type}_events.tsv')
        eventsData = pd.read_csv(pathToEventsFile, sep='\t')
    
        #Load event configuration
        pathToConfigFile = os.path.join(para.dir_data, 'task-attnOneInstNoOBs_events.json')
        
        with open(pathToConfigFile, 'r') as configFile:
            triggerConfig = json.load(configFile)
        #MUST CONVERT eventsData['onset'] FROM SECONDS TO MS
        eventsArray = np.column_stack([(eventsData['onset']*para.eegFs).astype(int), np.zeros(len(eventsData), dtype=int), eventsData['value']])
        
        unique_events_raw = np.unique(eventsArray[:, 2])  #All event IDs in this particular task
        possibleTrialStartIDs = np.arange(1, 145, 2)  #ALL possible trial start IDs for all tasks
        missing_events = set(possibleTrialStartIDs) - set(unique_events_raw)  #Possible start IDs NOT in this particular task
        
        trialStartIDs = []
        for event in eventsArray:
            event_id = event[2]
            if event_id in possibleTrialStartIDs and event_id not in missing_events and event_id not in trialStartIDs:
                trialStartIDs.append(event_id)
        
        event_id = dict(zip(map(str, trialStartIDs), trialStartIDs))
        
        tmin = para.trial_trim_start
        tmax = tmin+para.eegPeriodAfterTrim
        
        #Create epochs
        epochs = mne.Epochs(eeg_preprocessed, eventsArray, event_id=event_id, 
                            tmin=tmin, tmax=tmax, baseline=None, detrend=1, preload=True)
    
        #Add labels:
                
        pathToAttendanceData = os.path.join(para.dir_data, subjFolder, 'beh',  f'{subjFolder}_task-{task}_beh.tsv')
    
        df = pd.read_csv(pathToAttendanceData,sep='\t')
        df = df[['stimuli','music_attended']]
        
        labels = []
        stimulus_all = []
        for i in range(0, len(df)-1): #Includes prac trial
            attdMus = df.iloc[i]['music_attended']
            if attdMus == "Yes":
                label = "attd"
            if attdMus == "No":
                label = "unattd"
            labels+= [label]
            
            stimulus = df.iloc[i]['stimuli'][-32:-22]
            stimulus_all += [stimulus]
            
        #Add metadata with labels to epochs
        epochs.metadata = pd.DataFrame({'stimulus': stimulus_all,'music_attd': labels})
        epochs.metadata.reset_index(drop=True, inplace=True)  #Reset index to ensure correct indexing
        
        if para.filterTimingOutliers:
        #Filter out
            pathToErrorsDataFile = os.path.join(para.dir_data, "derivatives/significantEventTimingErrors_perSub", f'{subjFolder}_significantEventTimingErrors_triLen.tsv')
            trigErrorData = pd.read_csv(pathToErrorsDataFile, sep='\t')
            relevantRow = trigErrorData[(trigErrorData['Task']==task) & (trigErrorData['EEG Type']==eeg_type)]
            timingOutliers = relevantRow['Trials with |Δt - <Δt>|>10ms'].iloc[0]
            
            if pd.isna(timingOutliers): #Accounts for where there are none
                timingOutliers = []
            else:
                timingOutliers = ast.literal_eval(timingOutliers)  # safely convert to list
                timingOutliers = [int(i) for i in timingOutliers]  # ensure they're ints

            epochs.drop(timingOutliers)
            
        epochs_normalised = epochs.apply_function(lambda x: zscore(x, axis=-1))
    
        #Downsample EEG
        decimFactor = int(para.eegFs / para.ds)
        epochs_normalised.decimate(decim=decimFactor, offset=0, verbose=None)
        
        epochs_normalised.save(cache_file, overwrite=True)
        
        return epochs_normalised

def preprocessAndEpoch_emoDec_ceegrid(subject, para):
    modifiers = []
    if not para.interpBads: modifiers.append("retainedBads")
    if para.filterTimingOutliers: modifiers.append("removedTimeOutliers")
    if para.doICA and not para.stricterICA: modifiers.append("ICA")
    if para.stricterICA: modifiers.append("stricterICA")
    if para.doASR: modifiers.append("ASR")
    if para.removeFpz != True:
        modifiers.append("FpzKept")
    
    mod_tag = "_".join(modifiers) if modifiers else "raw"
    cache_file = os.path.join(para.cache_dir, f"{subject}_emoDec_ceegrid_{mod_tag}-epo.fif")
    
    if os.path.exists(cache_file):
        print(f"Loading cached epochs for subject {subject}")
        return mne.read_epochs(cache_file, preload=True)
    
    else:
        task = 'emotion'
        eeg_type = 'ceegrid'
        subjFolder = "sub-" + subject
        
        base_path = pathlib.Path(__file__).resolve().parents[2]
        bids_path = mne_bids.BIDSPath(root=para.dir_data, subject=subject, datatype='eeg', task=task, acquisition=eeg_type)
        eeg_raw = mne_bids.read_raw_bids(bids_path)
        eeg_raw.load_data()
        
        if para.removeFpz == True:
            eeg_raw.drop_channels(['Fpz'])
            eeg_raw.set_eeg_reference(ref_channels="average") #Removed a channel, so given that we're using average ref it is best to reref
        
        eeg_bpFiltered = eeg_raw.filter(l_freq=para.hpf, h_freq=para.lpf)
        
        #Handle bads by either marking and interpolating them, or keeping them:
        if para.interpBads == True:
            eeg_bpFiltered = badsMarker(eeg_bpFiltered, subjFolder, eeg_type, task)
            eeg_bpFiltered.set_eeg_reference(ref_channels="average")
            eeg_pastInterpStage = eeg_bpFiltered.copy().interpolate_bads()
            eeg_pastInterpStage.set_eeg_reference(ref_channels="average")       
        else:
            eeg_pastInterpStage = eeg_bpFiltered
            
        if para.doASR == True:
            asr = asrpy.ASR(sfreq=eeg_pastInterpStage.info["sfreq"], cutoff=20)
            asr.fit(eeg_pastInterpStage)
            eeg_pastASRstage = asr.transform(eeg_pastInterpStage)

        else:
            eeg_pastASRstage = eeg_pastInterpStage
            
        if para.doICA == True:
            ninterped = len(eeg_bpFiltered.info['bads'])
            nchans = eeg_pastASRstage.info['nchan']
            dims = nchans - ninterped - 1
            ica = ICA(n_components=dims, max_iter="auto", random_state=97, method='infomax', fit_params=dict(extended=True))
            ica.fit(eeg_pastASRstage)
        
            #ICLabel to estimate probabilities
            ic_labels = label_components(eeg_pastASRstage, ica, method="iclabel")
            
            #List to store indices of components to exclude
            exclusion_indices = []
            
            if para.stricterICA == False: #Only remove ones labelled as eye artifacts
            
            #Iterate through each prediction and label
                for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                    #Check if label is 'eye blink'
                    if label in ['eye blink']:
                        #Mark component for exclusion
                        exclusion_indices.append(i)
                
            elif para.stricterICA == True:
                #Threshold for exclusion
                threshold = 0.5
                
            #Iterate through each prediction and label
                for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                    #Check if label is not 'brain' or 'other' and probability is over threshold
                    if label not in ['brain'] or proba < threshold:
                        #Mark component for exclusion
                        exclusion_indices.append(i)
        
            print("Components to exclude:", exclusion_indices)
        
            eeg_preprocessed = eeg_pastASRstage.copy()
            ica.apply(eeg_preprocessed, exclude=exclusion_indices, n_pca_components=dims)
        
        if para.doICA != True:
            eeg_preprocessed = eeg_pastASRstage
        
        #Load event data
        pathToEventsFile = os.path.join(para.dir_data, subjFolder, 'eeg', f'{subjFolder}_task-{task}_acq-{eeg_type}_events.tsv')
        eventsData = pd.read_csv(pathToEventsFile, sep='\t')
    
        #Load event configuration
        pathToConfigFile = os.path.join(para.dir_data, 'task-attnOneInstNoOBs_events.json')
        
        with open(pathToConfigFile, 'r') as configFile:
            triggerConfig = json.load(configFile)
        #MUST CONVERT eventsData['onset'] FROM SECONDS TO MS
        eventsArray = np.column_stack([(eventsData['onset']*para.eegFs).astype(int), np.zeros(len(eventsData), dtype=int), eventsData['value']])
        
        unique_events_raw = np.unique(eventsArray[:, 2])  #All event IDs in this particular task
        possibleTrialStartIDs = np.arange(1, 145, 2)  #ALL possible trial start IDs for all tasks
        missing_events = set(possibleTrialStartIDs) - set(unique_events_raw)  #Possible start IDs NOT in this particular task
        
        trialStartIDs = []
        for event in eventsArray:
            event_id = event[2]
            if event_id in possibleTrialStartIDs and event_id not in missing_events and event_id not in trialStartIDs:
                trialStartIDs.append(event_id)
        
        event_id = dict(zip(map(str, trialStartIDs), trialStartIDs))
        
        tmin = para.trial_trim_start
        tmax = tmin+ para.eegPeriodAfterTrim
        
        #Create epochs
        epochs = mne.Epochs(eeg_preprocessed, eventsArray, event_id=event_id, 
                            tmin=tmin, tmax=tmax, baseline=None, detrend=1, preload=True)
        
        #Add labels:
                
        pathToAttendanceData = os.path.join(para.dir_data, subjFolder, 'beh',  f'{subjFolder}_task-{task}_beh.tsv')
    
        df = pd.read_csv(pathToAttendanceData,sep='\t')
        
        df = df[['stimuli']]
      
        stimulus_all = []
    
        for i in range(0, len(df)-1): #Includes prac trial
             
             stimulus = df.iloc[i]['stimuli'][-32:-22]
             stimulus_all += [stimulus]
             
        #Add metadata with labels to epochs
        epochs.metadata = pd.DataFrame({'stimulus': stimulus_all})
        epochs.metadata.reset_index(drop=True, inplace=True)  #Reset index to ensure correct indexing
        
        if para.filterTimingOutliers:
        #Filter out
            pathToErrorsDataFile = os.path.join(para.dir_data, "derivatives/significantEventTimingErrors_perSub", f'{subjFolder}_significantEventTimingErrors_triLen.tsv')
            trigErrorData = pd.read_csv(pathToErrorsDataFile, sep='\t')
            relevantRow = trigErrorData[(trigErrorData['Task']==task) & (trigErrorData['EEG Type']==eeg_type)]
            timingOutliers = relevantRow['Trials with |Δt - <Δt>|>10ms'].iloc[0]
            
            if pd.isna(timingOutliers): #Accounts for where there are none
                timingOutliers = []
            else:
                timingOutliers = ast.literal_eval(timingOutliers)  # safely convert to list
                timingOutliers = [int(i) for i in timingOutliers]  # ensure they're ints

            epochs.drop(timingOutliers)
            
        epochs_normalised = epochs.apply_function(lambda x: zscore(x, axis=-1))
    
        #Downsample EEG
        decimFactor = int(para.eegFs / para.ds)
        epochs_normalised.decimate(decim=decimFactor, offset=0, verbose=None)
        
        epochs_normalised.save(cache_file, overwrite=True)
        
        return epochs_normalised