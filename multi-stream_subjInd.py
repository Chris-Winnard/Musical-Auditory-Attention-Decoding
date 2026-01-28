import os
import numpy as np
import soundfile as sf
import pathlib
import mne
import warnings
import matplotlib.pyplot as plt
from processing_and_plotting_functions import *
from pymtrf import mtrf_train, mtrf_predict, mtrf_crossval
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
        self.eeg_type = 'scalp'  # Define EEG type before filtering subjects
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
        self.doICA = True
        self.stricterICA = True
        self.doASR = False
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
        
#Organise stimuli:
all_stimuli = {}
for set_id, instruments in stim_all.items():
    ordered = [instruments['Vibr'], instruments['Keyb']]
    all_stimuli[set_id] = np.concatenate(ordered, axis=0)

#Preprocess to get the relevant feature envelopes:
processed_stim_all = {set_id: process_stimuli(data, para.lpf, para.ds, fs) for set_id, data in all_stimuli.items()}

# Now split processed_stim_all back into per-instrument arrays, padded/truncated to match EEG length
eeg_len = int(para.eegPeriodAfterTrim * para.ds) + 1
stim_orig = {instr: {} for instr in ('Vibr', 'Keyb')}

for set_id, proc in processed_stim_all.items():
    # proc is a list/tuple [Vibr_proc, Keyb_proc]
    for idx, instr in enumerate(['Vibr', 'Keyb']):
        arr = proc[idx]
        if arr.shape[0] < eeg_len:
            padded = np.zeros(eeg_len)
            padded[: arr.shape[0]] = arr
            stim_orig[instr][set_id] = padded
        else:
            stim_orig[instr][set_id] = arr[:eeg_len]

print("Processed stimuli for all sets.")


# ===============================================================================================================
# STEP 2: Helper functions for excluding data (necessary for leave-one-person-and-movie-out cross-val), leave-one
# -trial-out ridge parameter duning, training/averaging models, and plotting stimulus feature predictions
# ===============================================================================================================

def exclude_LOPMO(data_list, stim_list, meta_list, test_meta):
    """
    Given lists of data, stimulus, and meta (each aligned), remove all entries
    that match either Set or Subject in test_meta.
    Return (filtered_data_list, filtered_stim_array).
    """
    filtered_data = []
    filtered_stim = []
    for d, s, m in zip(data_list, stim_list, meta_list):
        if (m['Set'] != test_meta['Set']) and (m['Subject'] != test_meta['Subject']):
            filtered_data.append(d)
            filtered_stim.append(s)
    return filtered_data, np.array(filtered_stim)


def find_best_lambda(EEG_list, Stim_array, para):
    """
    Run mtrf_crossval on all trials (EEG_list, Stim_array) to pick best λ.
    EEG_list: shape [n_trials, n_channels, n_samples] → need to transpose to [trials, samples, channels].
    Stim_array: shape [n_trials, n_samples], will be reshaped to [trials, samples, 1].
    Returns (best_lambda, max_mean_r, all_mean_r_values).
    """
    # Reformat EEG to [trials, samples, channels]
    EEG = np.transpose(np.stack(EEG_list), (0, 2, 1))
    # Reformat Stim to [trials, samples, 1]
    Stim = Stim_array[:, :, np.newaxis]

    # Sanity checks
    assert Stim.ndim == 3 and Stim.shape[2] == 1
    assert EEG.ndim == 3
    assert Stim.shape[0] == EEG.shape[0] and Stim.shape[1] == EEG.shape[1]

    fs, lambdas, tMin, tMax, direction = para.ds, para.lambda_reg, para.tMin, para.tMax, para.direction
    r_vals, _, _, _, _ = mtrf_crossval(Stim, EEG, fs, direction, tMin, tMax, lambdas)

    # mean r over trials for each λ
    mean_r = np.mean(r_vals, axis=0)
    best_idx = np.argmax(mean_r)
    return lambdas[best_idx], mean_r[best_idx], mean_r

def train_and_average_models(eeg_list, stim_array, para, lambda_val):
    """
    Given lists of single-trial EEG ([channels, samples]) and stim ([samples, 1]), train one
    model per trial and average the TRF weights.
    """
    models = []
    for i in range(len(eeg_list)):
        # Note: eeg_list[i] has shape [channels, samples], so we transpose for mtrf_train
        stim = stim_array[i][:, np.newaxis]  # [samples, 1]
        eeg = eeg_list[i].T                  # [samples, channels]

        # sanity checks
        assert stim.ndim == 2 and stim.shape[1] == 1
        assert eeg.ndim == 2 and eeg.shape[0] == stim.shape[0]
        model, _, _ = mtrf_train(stim, eeg, para.ds, para.direction, para.tMin, para.tMax, lambda_val)
        models.append(model)

    # stack and average across trials → shape [channels, time_lags]
    return np.mean(np.stack(models, axis=0), axis=0)

def plot_prediction(time_axis, pred, stim, label):
    plt.plot(time_axis, pred, label=f'Predicted {label}')
    plt.plot(time_axis, stim, linestyle='--', label=f'Stimulus ({label})')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
# ===============================================================================================================
# STEP 3: Build combined data/meta/stim for both instruments
# ===============================================================================================================

# Accumulate across all subjects:
combined = {'Keyb': {'multistream_epochs': [],
                     'all_data': [],    # list of arrays [channels, samples]
                     'all_meta': [],    # list of dicts {'Set':…, 'Subject':…}
                     'all_stim': []},     # list of [samples] arrays
            'Vibr': {'multistream_epochs': [],
                     'all_data': [],
                     'all_meta': [],
                     'all_stim': []}}

for subject in para.allSub:
    print(f"Processing subject {subject}")

    # Load subject's EEG data
    if para.eeg_type == 'ceegrid':
        epochs = preprocessAndEpoch_ceegrid(subject, para)
        epochs_single = preprocessAndEpoch_singleStream_ceegrid(subject, para)
        epochs_emo = preprocessAndEpoch_emoDec_ceegrid(subject, para)
    elif para.eeg_type == 'scalp':
        epochs = preprocessAndEpoch(subject, para)
        epochs_single = preprocessAndEpoch_singleStream(subject, para)
        epochs_emo = preprocessAndEpoch_emoDec(subject, para)
        
    # ------------------------------
    # (A) Multi-stream: separate Keyb vs. Vibr
    # ------------------------------
    for instr in ('Keyb', 'Vibr'):
        # Create a boolean mask where 'attd_inst' equals this instrument
        inst_mask = (epochs.metadata['attd_inst'] == instr)
        inst_epochs = epochs[inst_mask]  # this is an Epochs object filtered for this instr

        combined[instr]['multistream_epochs'].append(inst_epochs)

        data_array = inst_epochs.get_data(copy=True)  # shape [n_trials_this_inst, n_channels, n_samples]
        # We’ll transpose to [n_trials, samples, channels] when needed; for now store as is
        for trial_idx in range(len(inst_epochs)):
            row = inst_epochs.metadata.iloc[trial_idx]
            set_id = row['Set']
            combined[instr]['all_meta'].append({'Set': set_id, 'Subject': subject})
            # Extract the trial’s EEG as [channels, samples]
            single_trial_eeg = data_array[trial_idx, :, :]  # [n_channels, n_samples]
            combined[instr]['all_data'].append(single_trial_eeg)
            combined[instr]['all_stim'].append(stim_orig[instr][set_id])
            
    # ------------------------------
    # (B) Single-stream (attd-only) training data
    # ------------------------------
    for instr in ('Keyb', 'Vibr'):
        cond = (
            epochs_single.metadata['stimulus'].str.endswith(instr)
            & (epochs_single.metadata['music_attd'] == 'attd')
        )
        single_epochs = epochs_single[cond]
        data_array = single_epochs.get_data(copy=True)

        for trial_idx in range(len(single_epochs)):
            row = single_epochs.metadata.iloc[trial_idx]
            set_id = row['stimulus'][:5]  # first 5 chars = set ID
            combined[instr]['all_meta'].append({'Set': set_id, 'Subject': subject})
            single_trial_eeg = data_array[trial_idx, :, :]
            combined[instr]['all_data'].append(single_trial_eeg)
            combined[instr]['all_stim'].append(stim_orig[instr][set_id])

    # ------------------------------
    # (C) Emotion-dec training data (both instruments)
    # ------------------------------
    for instr in ('Keyb', 'Vibr'):
        cond_emo = epochs_emo.metadata['stimulus'].str.endswith(instr)
        emo_epochs = epochs_emo[cond_emo]
        data_array = emo_epochs.get_data(copy=True)

        for trial_idx in range(len(emo_epochs)):
            row = emo_epochs.metadata.iloc[trial_idx]
            set_id = row['stimulus'][:5]
            combined[instr]['all_meta'].append({'Set': set_id, 'Subject': subject})
            single_trial_eeg = data_array[trial_idx, :, :]
            combined[instr]['all_data'].append(single_trial_eeg)
            combined[instr]['all_stim'].append(stim_orig[instr][set_id])

# Concatenate multi-stream epochs into a single Epochs object per instrument
for instr in ('Keyb', 'Vibr'):
    combined[instr]['multistream_epochs'] = mne.concatenate_epochs(combined[instr]['multistream_epochs'])
    
    # Convert the accumulated lists into numpy arrays:
    #   all_data: list of [channels, samples] → stack to [n_trials, channels, samples]
    #   all_stim: list of [samples]       → stack to [n_trials, samples]
    combined[instr]['all_data'] = np.stack(combined[instr]['all_data'], axis=0)
    combined[instr]['all_stim'] = np.stack(combined[instr]['all_stim'], axis=0)


# ===============================================================================================================
# STEP 4: Generic leave-one-trial-out testing per instrument
# ===============================================================================================================

accuracy_counts = {'correct': 0, 'total': 0}
R_vals = {'attd': {'Keyb': [], 'Vibr': []},
          'unattd': {'Keyb': [], 'Vibr': []}}

def run_LOPMO_trial(instr, trial_idx):
    """
    Runs one leave-one-person-and-movie-out iteration for instrument = instr ('Keyb' or 'Vibr')
    at position trial_idx (0-based within combined[instr]['multistream_epochs']).
    Returns: (is_correct, r_attd, r_unattd).
    """
    inst_epochs = combined[instr]['multistream_epochs']
    all_data = combined[instr]['all_data']      # shape [N_all, channels, samples]
    all_stim = combined[instr]['all_stim']      # shape [N_all, samples]
    all_meta = combined[instr]['all_meta']      # list of dicts

    # Get the metadata row for this held-out trial
    row = inst_epochs.metadata.iloc[trial_idx]
    set_id = row['Set']
    # Use the same index into all_meta to get the correct Subject (since all align by trial_idx)
    subject_id = combined[instr]['all_meta'][trial_idx]['Subject']
    test_meta = {'Set': set_id, 'Subject': subject_id}

    # Extract test EEG and stimulus for this trial
    data_array = inst_epochs.get_data(copy=True)  # shape [n_trials_this_inst, channels, samples]
    test_eeg = data_array[trial_idx, :, :].T      # [samples, channels]
    test_stim = stim_orig[instr][test_meta['Set']].reshape(-1, 1)  # [samples,1]

    # Train model for attended instrument, excluding any same Set/Subject
    train_data, train_stim = exclude_LOPMO(all_data, all_stim, all_meta, test_meta)
    lambda_attd, _, _ = find_best_lambda(train_data, train_stim, para)
    model_attd = train_and_average_models(train_data, train_stim, para, lambda_attd)

    # Predict on the held-out trial with attended inst model
    pred_attd, r_attd, _, _ = mtrf_predict(test_stim, test_eeg, model_attd, para.ds,
                                   para.direction, para.tMin, para.tMax, None)

    # Unattended inst model (opposite instrument)
    unattd_instr = 'Vibr' if instr == 'Keyb' else 'Keyb'
    unattd_data = combined[unattd_instr]['all_data']
    unattd_stim = combined[unattd_instr]['all_stim']
    unattd_meta = combined[unattd_instr]['all_meta']
    
    unattd_train_data, unattd_train_stim = exclude_LOPMO(unattd_data, unattd_stim, unattd_meta, test_meta)
    lambda_unattd, _, _ = find_best_lambda(unattd_train_data, unattd_train_stim, para)
    model_unattd = train_and_average_models(unattd_train_data, unattd_train_stim, para, lambda_unattd)

    unattd_stim_test = stim_orig[unattd_instr][test_meta['Set']].reshape(-1, 1)
    pred_unattd, r_unattd, _, _ = mtrf_predict(unattd_stim_test, test_eeg, model_unattd, para.ds,
                                     para.direction, para.tMin, para.tMax, None)

    if para.show_plots:
        time_axis = np.arange(len(test_stim))
        print(f"\nTrial {i+1} ({instr}) — Set {set_id}")
        plot_prediction(time_axis, pred_attd, test_stim, instr)
        plot_prediction(time_axis, pred_unattd, unattd_stim_test, unattd_instr)
        
    # Classification: if r_attd > r_unattd, we say “correct”
    is_correct = (r_attd[0][0] > r_unattd[0][0])

    return is_correct, r_attd[0][0], r_unattd[0][0]


# Run LOPMO for both instruments
for instr in ('Keyb', 'Vibr'):
    inst_epochs = combined[instr]['multistream_epochs']
    n_trials = len(inst_epochs)
    print(f"Running LOPMO for {instr}, total trials: {n_trials}")

    for trial_idx in range(n_trials):
        correct, r_attd, r_unattd = run_LOPMO_trial(instr, trial_idx)
        
        accuracy_counts['correct'] += int(bool(correct))
        accuracy_counts['total'] += 1
        R_vals['attd'][instr].append(r_attd)
        R_vals['unattd'][instr].append(r_unattd)

# Final accuracy
accuracy = accuracy_counts['correct'] / accuracy_counts['total']


# ===============================================================================================================
# STEP 5: Calculate summary R-values, p-chance, p < 0.05 value
# ===============================================================================================================

# Average R-values
mean_r = {'Keyb': {'attd': np.mean(R_vals['attd']['Keyb']),
                   'unattd': np.mean(R_vals['unattd']['Keyb'])},
          'Vibr': {'attd': np.mean(R_vals['attd']['Vibr']),
                   'unattd': np.mean(R_vals['unattd']['Vibr'])}}

# Prior: counts of (all_data) minus one because of LOPMO removal
vibr_len_prior = combined['Vibr']['all_data'].shape[0] - 1
keyb_len_prior = combined['Keyb']['all_data'].shape[0] - 1
N_prior = vibr_len_prior + keyb_len_prior
p_vibr_prior = vibr_len_prior / N_prior
p_keyb_prior = keyb_len_prior / N_prior

# Posterior: counts of multi-stream trials
len_vibr_post = len(combined['Vibr']['multistream_epochs'])
len_keyb_post = len(combined['Keyb']['multistream_epochs'])
N_post = len_vibr_post + len_keyb_post
p_vibr_post = len_vibr_post / N_post
p_keyb_post = len_keyb_post / N_post

#p_chance and p < 0.05 value
p_chance = p_vibr_prior * p_vibr_post + p_keyb_prior * p_keyb_post
info_dict = {"N": N_post, "p_chance": p_chance}
accuracy_critical = dataset_p_values_calculator(info_dict)


# ===============================================================================================================
# STEP 6: Print summary
# ===============================================================================================================

print(f"Mean R (Keyb attended): {mean_r['Keyb']['attd']:.4f}")
print(f"Mean R (Keyb unattended): {mean_r['Keyb']['unattd']:.4f}")
print(f"Mean R (Vibr attended): {mean_r['Vibr']['attd']:.4f}")
print(f"Mean R (Vibr unattended): {mean_r['Vibr']['unattd']:.4f}")
print(f"p < 0.05 threshold accuracy: {accuracy_critical}")
print(f"Overall decoding accuracy: {accuracy:.4f}")