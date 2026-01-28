import os
import numpy as np
import soundfile as sf
import pathlib
import matplotlib.pyplot as plt
from processing_and_plotting_functions import *
from pymtrf import mtrf_train, mtrf_predict, mtrf_crossval
import mne
import warnings
from ptest_stuff import *
from statistics import mean
from sklearn import metrics

"Capitalising the word 'Set' for consistency with data from upstream"

"p-test stuff? Think just N is number of test trials, p_chance is 50%?"
"For at least one participant it says the accuracy threshold is >1 - THIS IS BC OF LOW N NUMBERS!"
"After adding LOMO stuff it seems to be getting vastly better results- overfit?"


#Maybe skim over/check preprocess_and_epoch/different versions of it


# ===============================================================================================================
# STEP 0: Define parameters
# ===============================================================================================================

class Parameters:
    def __init__(self):
        self.basePath = str(pathlib.Path(__file__).resolve().parents[2])
        self.dir_data = os.path.join(self.basePath, 'bids_dataset')
        self.dir_stim = os.path.join(str(pathlib.Path(__file__).parent.resolve().parent.resolve()), 'Stimuli')
        self.cache_dir = os.path.join(self.basePath, 'AAD cache')
        self.eeg_type = 'ceegrid'  # Define EEG type before filtering subjs
        if self.eeg_type == 'ceegrid':
            self.allSub = [f'{i:02d}' for i in range(1, 33) if i not in [4, 6, 16, 21]]
            self.removeFpz = True #Default for cEEGrid
        elif self.eeg_type == 'scalp':
            self.allSub = [f'{i:02d}' for i in range(1, 33) if i not in [4]]
            self.removeFpz = False #Only included here for completeness 
        self.ds = 25
        self.direction = -1
        self.tMin = 0
        self.tMax = 300
        self.lambda_reg = 10.0 ** np.arange(-7, 7, 2)
        self.hpf = 2
        self.lpf = 8
        self.eegFs = 1000
        self.trial_trim_start = 0.8
        self.trial_trim_end = 0.8
        self.eegPeriodAfterTrim = 30 - self.trial_trim_start - self.trial_trim_end
        self.filterTimingOutliers = True
        self.interpBads = True
        self.doICA = False
        self.stricterICA = False
        self.doASR = False
        self.show_plots = True

        if not (self.eegPeriodAfterTrim * self.ds).is_integer():
            print("Warning: chosen parameters cause slight frequency-resolution loss.")

para = Parameters()


# ===============================================================================================================
# STEP 1: Load & preprocess all stimuli once
# ===============================================================================================================

fs = 44100

#Load:
stim_all, processed_files = {}, set()
for root, _, files in os.walk(para.dir_stim):
    for fname in files:
        if not fname.endswith('.wav') or fname in processed_files:
            continue
        set_id = fname.split('-')[0]
        instr = fname.split('-')[-1].split('.')[0]
        
        stim_data, _ = sf.read(os.path.join(root, fname))
        stim_data = stim_data[int(fs*para.trial_trim_start):-int(fs*para.trial_trim_end)] #Trim fade-in & fade-out
        stim_all.setdefault(set_id, {'Harm': [], 'Keyb': [], 'Vibr': []})[instr].append(stim_data)
        
        processed_files.add(fname)
        
#Organise:
all_stimuli = {}
for set_id, instruments in stim_all.items():
    ordered = [instruments['Vibr'], instruments['Harm'], instruments['Keyb']]
    all_stimuli[set_id] = np.concatenate(ordered, axis=0)

#Preprocess to get the relevant feature envelopes:
processed_stim_all = {set_id: process_stimuli(data, para.lpf, para.ds, fs) for set_id, data in all_stimuli.items()}

# Pad or truncate to match EEG length
eeg_len = int(para.eegPeriodAfterTrim * para.ds) + 1
stim_orig = {instr: {} for instr in ('Vibr', 'Harm', 'Keyb')}
for set_id, proc in processed_stim_all.items():
    for idx, instr in enumerate(['Vibr', 'Harm', 'Keyb']):
        arr = proc[idx]
        stim_orig[instr][set_id] = np.pad(arr, (0, max(0, eeg_len - len(arr))))[:eeg_len]

print("Processed stimuli for all sets.")


# ===============================================================================================================
# STEP 2: Helper functions for leave-one-trial-out ridge parameter duning, training/averaging models, and 
# plotting stimulus feature predictions
# ===============================================================================================================

def find_best_lambda(EEG, Stim, para):
    EEG = np.transpose(EEG, (0, 2, 1))
    Stim = Stim[:, :, np.newaxis]
    r, *_ = mtrf_crossval(Stim, EEG, para.ds, para.direction, para.tMin, para.tMax, para.lambda_reg)
    mean_r = np.mean(r, axis=0)
    best_lambda = para.lambda_reg[np.argmax(mean_r)]
    return best_lambda, max(mean_r), mean_r

def train_and_average_models(eeg_data, stim_data, para, lambda_val):
    models = [mtrf_train(s[:, np.newaxis], e.T, para.ds, para.direction, para.tMin, para.tMax, lambda_val)[0] for s, e in zip(stim_data, eeg_data)]
    return np.mean(models, axis=0)

def plot_prediction(time_axis, pred, stim, label):
    plt.plot(time_axis, pred, label=f'Predicted {label}')
    plt.plot(time_axis, stim, linestyle='--', label=f'Stimulus ({label})')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()
    

# ===============================================================================================================
# STEP 3: Main loop for subj-dependent decoding, looping through all participants
# ===============================================================================================================

roc_auc_all_instrs_global = {'Harm': [], 'Vibr': [], 'Keyb': []}
N_test_trials_global = 0
R_vals_global = {'attd': {'Keyb': [], 'Harm':[], 'Vibr': []},
                 'unattd': {'Keyb': [], 'Harm':[], 'Vibr': []}}

for subj in para.allSub:
    print(f"Processing subject {subj}")
    
    # Load subj EEG
    if para.eeg_type == 'ceegrid':
        epochs_single = preprocessAndEpoch_singleStream_ceegrid(subj, para)
        epochs_multi = preprocessAndEpoch_ceegrid(subj, para)
        epochs_emo = preprocessAndEpoch_emoDec_ceegrid(subj, para)
    else:
        epochs_single = preprocessAndEpoch_singleStream(subj, para)
        epochs_multi = preprocessAndEpoch(subj, para)
        epochs_emo = preprocessAndEpoch_emoDec(subj, para)
    
    for instr in ['Vibr', 'Harm', 'Keyb']:
        train_data_pool = {'train_data': [], 'train_stim': [], 'Set':[]}
        test_data_pool = {'test_attd': [], 'test_unattd': [], 'test_attd_set': [], 'test_unattd_set': []}

        # Add multi-stream and emotion-decoding data to training set
        for ep in [epochs_multi[epochs_multi.metadata['attd_inst'] == instr], epochs_emo[epochs_emo.metadata['stimulus'].str.endswith(instr)]]:
            for i in range(len(ep)):
                set_id = ep.metadata.iloc[i]['stimulus'][:5] if 'stimulus' in ep.metadata.columns else ep.metadata.iloc[i]['Set']
                train_data_pool['train_data'].append(ep.get_data(copy=True)[0])
                train_data_pool['train_stim'].append(stim_orig[instr][set_id])
                train_data_pool['Set'].append(set_id)
                
        # Add SS-AAD attended trials to train data
        ss_attd = epochs_single[(epochs_single.metadata['stimulus'].str.endswith(instr)) & (epochs_single.metadata['music_attd'] == 'attd')]
        ss_attd_data = ss_attd.get_data(copy=True)
        ss_attd_meta = ss_attd.metadata               
                
        for i in range(len(ss_attd)):
            set_id = ss_attd_meta.iloc[i]['stimulus'][:5]
            train_data_pool['train_data'].append(ss_attd_data[i])
            train_data_pool['train_stim'].append(stim_orig[instr][set_id])
            train_data_pool['Set'].append(set_id)               
                
        #Prep SS-AAD unattended trials
        ss_unattd = epochs_single[(epochs_single.metadata['stimulus'].str.endswith(instr)) & (epochs_single.metadata['music_attd'] == 'unattd')]

        ss_attd_data = ss_attd.get_data(copy=True)
        ss_unattd_data = ss_unattd.get_data(copy=True)
        ss_attd_meta = ss_attd.metadata
        ss_unattd_meta = ss_unattd.metadata
        
        for i in range(len(ss_attd)):
            test_data_pool['test_attd'].append(ss_attd_data[i])
            test_data_pool['test_attd_set'].append(ss_attd_meta.iloc[i]['stimulus'][:5])
        
        for i in range(len(ss_unattd)):
            test_data_pool['test_unattd'].append(ss_unattd_data[i])
            test_data_pool['test_unattd_set'].append(ss_unattd_meta.iloc[i]['stimulus'][:5])                
                
        # Cross-validation for attended data:
        n_trials_attd = len(test_data_pool['test_attd'])
        
        for trial in range(n_trials_attd):
            test_eeg_attd = test_data_pool['test_attd'][trial].T     
            test_stim_attd_set = test_data_pool['test_attd_set'][trial]   
            test_stim_attd = stim_orig[instr][test_stim_attd_set].reshape(-1, 1)
                   
            # Prepare training and test data
            train_eeg = [d for i, d in enumerate(train_data_pool['train_data']) if train_data_pool['Set'][i] != test_stim_attd_set]
            train_stim = np.array([s for i, s in enumerate(train_data_pool['train_stim']) if train_data_pool['Set'][i] != test_stim_attd_set])
                        
            # Train model
            best_lambda, *_ = find_best_lambda(train_eeg, train_stim, para)
            model = train_and_average_models(train_eeg, train_stim, para, best_lambda)

            # Predict
            pred_attd, r_attd, *_ = mtrf_predict(test_stim_attd, test_eeg_attd, model, para.ds, para.direction, para.tMin, para.tMax, None)
            
            if para.show_plots:
                time_axis = np.arange(len(test_stim_attd))
                plot_prediction(time_axis, [x for x in pred_attd], test_stim_attd, instr)
                
            R_vals_global['attd'][instr].append(r_attd[0][0])
                        
        N_test_trials_global += n_trials_attd
    
    
        # Cross-validation for unattended data- using the same training data but still with LOMO:
        n_trials_unattd = len(test_data_pool['test_unattd'])
        
        for trial in range(n_trials_unattd):
            test_eeg_unattd = test_data_pool['test_unattd'][trial].T      
            test_stim_unattd_set = test_data_pool['test_unattd_set'][trial]  
            test_stim_unattd = stim_orig[instr][test_stim_unattd_set].reshape(-1, 1)
                         
            # Prepare training and test data
            train_eeg = [d for i, d in enumerate(train_data_pool['train_data']) if train_data_pool['Set'][i] != test_stim_unattd_set]
            train_stim = np.array([s for i, s in enumerate(train_data_pool['train_stim']) if train_data_pool['Set'][i] != test_stim_unattd_set])
                        
            # Train model
            best_lambda, *_ = find_best_lambda(train_eeg, train_stim, para)
            model = train_and_average_models(train_eeg, train_stim, para, best_lambda)         
            
            pred_unattd, r_unattd, *_ = mtrf_predict(test_stim_unattd, test_eeg_unattd, model, para.ds, para.direction, para.tMin, para.tMax, None)
            
            if para.show_plots:
                time_axis = np.arange(len(test_stim_unattd))
                plot_prediction(time_axis, [x for x in pred_unattd], test_stim_unattd, instr)
                
            R_vals_global['unattd'][instr].append(r_unattd[0][0])
            
        N_test_trials_global += n_trials_unattd
               
        
        #NOTE- AS R VALS ARE GLOBAL, THESE WILL BE FOR ALL PARTICIPANTS SO FAR..
        y = np.concatenate([np.zeros(len(R_vals_global['unattd'][instr]),dtype=int), np.ones(len(R_vals_global['attd'][instr]),dtype=int)]) 
        scores = np.concatenate([R_vals_global['unattd'][instr],R_vals_global['attd'][instr]])

        fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
        roc_auc = metrics.roc_auc_score(y,scores)
        roc_auc_all_instrs_global[instr].append(roc_auc)
        
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        display.plot()
        plt.show()
        
        
# ===============================================================================================================
# STEP 4: Calculate global R-values, ROC AUC score, chance ROC AUC and p < 0.05 ROC AUC
# ===============================================================================================================

avg_attd = mean([r for vals in R_vals_global['attd'].values() for r in vals])  
avg_unattd = mean([r for vals in R_vals_global['unattd'].values() for r in vals])

mean_roc_auc_vibr = np.mean(roc_auc_all_instrs_global['Vibr'])
mean_roc_auc__harm = np.mean(roc_auc_all_instrs_global['Harm'])
mean_roc_auc_keyb = np.mean(roc_auc_all_instrs_global['Keyb'])
all_roc_auc_scores = roc_auc_all_instrs_global['Vibr'] + roc_auc_all_instrs_global['Harm'] + roc_auc_all_instrs_global['Keyb']
mean_roc_auc_all = np.mean(all_roc_auc_scores)

crit_acc_global = dataset_p_values_calculator({"N": N_test_trials_global, "p_chance": 0.5})


# ===============================================================================================================
# STEP 5: Print summary
# ===============================================================================================================

print("\n=== GLOBAL SUMMARY ===")
print(f"Average attended R: {avg_attd}")
print(f"Average unattended R: {avg_unattd}")
print(f"Global mean ROC AUC (Vibr): {mean_roc_auc_vibr:.4f}")
print(f"Global mean ROC AUC (Harm): {mean_roc_auc__harm:.4f}")
print(f"Global mean ROC AUC (Keyb): {mean_roc_auc_keyb:.4f}")
print(f"Overall global ROC AUC (all instruments): {mean_roc_auc_all:.4f}")
print(f"Global p < 0.05 accuracy threshold: {crit_acc_global:.4f}")