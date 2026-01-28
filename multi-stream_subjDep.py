import os
import numpy as np
import soundfile as sf
import pathlib
import mne
import warnings
import matplotlib.pyplot as plt
from pymtrf import mtrf_train, mtrf_predict, mtrf_crossval
from processing_and_plotting_functions import *
from ptest_stuff import *

"Capitalising the word 'Set' for consistency with data from upstream"


# ===============================================================================================================
# STEP 0: Define parameters
# ===============================================================================================================

class Parameters:
    def __init__(self):
        self.basePath = str(pathlib.Path(__file__).resolve().parents[2])
        self.dir_data = os.path.join(self.basePath, 'bids_dataset')
        self.dir_stim = os.path.join(str(pathlib.Path(__file__).parent.resolve().parent.resolve()), 'Stimuli')
        self.cache_dir = os.path.join(self.basePath, 'AAD cache')
        self.eeg_type = 'scalp'  # Define EEG type before filtering subjs
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
        self.filterTimingOutliers = False
        self.interpBads = True
        self.doICA = False
        self.stricterICA = False
        self.doASR = True
        self.show_plots = False

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
        
        if instr != 'Harm': #Harmonica pieces excluded from this analysis for simplicity   
            stim_data, _ = sf.read(os.path.join(root, fname))
            stim_data = stim_data[int(fs * para.trial_trim_start) : -int(fs * para.trial_trim_end)] #Trim fade-in & fade-out
            stim_all.setdefault(set_id, {'Keyb': [], 'Vibr': []})[instr].append(stim_data)
        
        processed_files.add(fname)

#Organise:
all_stimuli = {}
for set_id, instruments in stim_all.items():
    ordered = [instruments['Vibr'], instruments['Keyb']]
    all_stimuli[set_id] = np.concatenate(ordered, axis=0)

#Preprocess to get the relevant feature envelopes:
processed_stim_all = {set_id: process_stimuli(data, para.lpf, para.ds, fs) for set_id, data in all_stimuli.items()}

# Pad or truncate to match EEG length
eeg_len = int(para.eegPeriodAfterTrim * para.ds) + 1
stim_orig = {instr: {} for instr in ('Vibr', 'Keyb')}
for set_id, proc in processed_stim_all.items():
    for idx, instr in enumerate(['Vibr', 'Keyb']):
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
    r, _, _, _, _ = mtrf_crossval(Stim, EEG, para.ds, para.direction, para.tMin, para.tMax, para.lambda_reg)
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

R_vals = {
    'attd': {'Keyb': [], 'Vibr': []},
    'unattd': {'Keyb': [], 'Vibr': []}
}
all_accuracies = []
all_p_chances = []
N_total = 0

for subj in para.allSub:
    print(f"Processing subject {subj}")
    
    # Load subj's EEG data
    if para.eeg_type == 'ceegrid':
        epochs = preprocessAndEpoch_ceegrid(subj, para)
        epochs_single = preprocessAndEpoch_singleStream_ceegrid(subj, para)
        epochs_emo = preprocessAndEpoch_emoDec_ceegrid(subj, para)
    elif para.eeg_type == 'scalp':
        epochs = preprocessAndEpoch(subj, para)
        epochs_single = preprocessAndEpoch_singleStream(subj, para)
        epochs_emo = preprocessAndEpoch_emoDec(subj, para)

    combined = {'Keyb': {'data': [], 'stim': [], 'set_id': []},
                'Vibr': {'data': [], 'stim': [], 'set_id': []}}

    # Multi-stream (AAD)
    for instr in ('Keyb', 'Vibr'):
        subset = epochs[epochs.metadata['attd_inst'] == instr]
        data = subset.get_data(copy=True)
        for i in range(len(subset)):
            combined[instr]['data'].append(data[i])
            set_id = subset.metadata.iloc[i]['Set']
            combined[instr]['stim'].append(stim_orig[instr][set_id])
            combined[instr]['set_id'].append(set_id)

    # Single-stream (attended only)
    for instr in ('Keyb', 'Vibr'):
        cond = (epochs_single.metadata['stimulus'].str.endswith(instr)) & \
               (epochs_single.metadata['music_attd'] == 'attd')
        subset = epochs_single[cond]
        data = subset.get_data(copy=True)
        for i in range(len(subset)):
            combined[instr]['data'].append(data[i])
            set_id = subset.metadata.iloc[i]['stimulus'][:5]
            combined[instr]['stim'].append(stim_orig[instr][set_id])
            combined[instr]['set_id'].append(set_id)

    # Emotion-decoding
    for instr in ('Keyb', 'Vibr'):
        cond = epochs_emo.metadata['stimulus'].str.endswith(instr)
        subset = epochs_emo[cond]
        data = subset.get_data(copy=True)
        for i in range(len(subset)):
            combined[instr]['data'].append(data[i])
            set_id = subset.metadata.iloc[i]['stimulus'][:5]
            combined[instr]['stim'].append(stim_orig[instr][set_id])
            combined[instr]['set_id'].append(set_id)

    acc_correct = 0
    acc_total = 0

    for instr in ('Keyb', 'Vibr'):
        subset = epochs[epochs.metadata['attd_inst'] == instr]
        data = subset.get_data(copy=True)

        for i in range(len(subset)):
            set_id = subset.metadata.iloc[i]['Set']
            test_eeg = data[i].T
            test_stim = stim_orig[instr][set_id].reshape(-1, 1)

            # Attended inst model
            train_data = [d for j, d in enumerate(combined[instr]['data']) if combined[instr]['set_id'][j] != set_id]
            train_stim = np.array([s for j, s in enumerate(combined[instr]['stim']) if combined[instr]['set_id'][j] != set_id])
            lambda_attd, _, _ = find_best_lambda(train_data, train_stim, para)
            model_attd = train_and_average_models(train_data, train_stim, para, lambda_attd)
            pred_attd, r_attd, _, _ = mtrf_predict(test_stim, test_eeg, model_attd, para.ds, para.direction, para.tMin, para.tMax, None)

            # Unattended inst model (opposite instrument)
            unattd_instr = 'Vibr' if instr == 'Keyb' else 'Keyb'
            unattd_stim_test = stim_orig[unattd_instr][set_id].reshape(-1, 1)
            unattd_train_data = [d for j, d in enumerate(combined[unattd_instr]['data']) if combined[unattd_instr]['set_id'][j] != set_id]
            unattd_train_stim = np.array([s for j, s in enumerate(combined[unattd_instr]['stim']) if combined[unattd_instr]['set_id'][j] != set_id])
            lambda_unattd, _, _ = find_best_lambda(unattd_train_data, unattd_train_stim, para)
            model_unattd = train_and_average_models(unattd_train_data, unattd_train_stim, para, lambda_unattd)
            pred_unattd, r_unattd, _, _ = mtrf_predict(unattd_stim_test, test_eeg, model_unattd, para.ds, para.direction, para.tMin, para.tMax, None)

            # Save R-values
            R_vals['attd'][instr].append(r_attd[0][0])
            R_vals['unattd'][instr].append(r_unattd[0][0])

            if para.show_plots:
                time_axis = np.arange(len(test_stim))
                print(f"\nTrial {i+1} ({instr}) — Set {set_id}")
                plot_prediction(time_axis, pred_attd, test_stim, instr)
                plot_prediction(time_axis, pred_unattd, unattd_stim_test, unattd_instr)

            acc_correct += int(r_attd[0][0] > r_unattd[0][0])
            
            acc_total += 1

    acc = acc_correct / acc_total
    all_accuracies.append(acc)

    # Estimate chance accuracy
    n_keyb = len(combined['Keyb']['data']) - 1
    n_vibr = len(combined['Vibr']['data']) - 1
    N_prior = n_keyb + n_vibr
    p_keyb_prior = n_keyb / N_prior
    p_vibr_prior = n_vibr / N_prior

    n_keyb_test = len(epochs[epochs.metadata['attd_inst'] == 'Keyb'])
    n_vibr_test = len(epochs[epochs.metadata['attd_inst'] == 'Vibr'])
    N_post = n_keyb_test + n_vibr_test
    p_keyb_post = n_keyb_test / N_post
    p_vibr_post = n_vibr_test / N_post

    p_chance = p_keyb_prior * p_keyb_post + p_vibr_prior * p_vibr_post
    all_p_chances.append(p_chance)
    N_total += N_post

    crit_acc = dataset_p_values_calculator({"N": N_post, "p_chance": p_chance})
    print(f"Subj {subj} | Accuracy: {acc:.4f} | Critical (p<0.05): {crit_acc:.4f}")


# ===============================================================================================================
# STEP 4: Calculate summary accuracy, p-chance, p < 0.05 value
# ===============================================================================================================

global_acc = np.mean(all_accuracies)
mean_p_chance = np.mean(all_p_chances)
crit_acc_global = dataset_p_values_calculator({"N": N_total, "p_chance": mean_p_chance})


# ===============================================================================================================
# STEP 5: Print summary
# ===============================================================================================================

print("\n=== GLOBAL SUMMARY ===")
print(f"Mean R (Keyb attended): {np.mean(R_vals['attd']['Keyb']):.4f}")
print(f"Mean R (Keyb unattended): {np.mean(R_vals['unattd']['Keyb']):.4f}")
print(f"Mean R (Vibr attended): {np.mean(R_vals['attd']['Vibr']):.4f}")
print(f"Mean R (Vibr unattended): {np.mean(R_vals['unattd']['Vibr']):.4f}")
print(f"Global decoding accuracy: {global_acc:.4f}")
print(f"Global p < 0.05 accuracy threshold: {crit_acc_global:.4f}")