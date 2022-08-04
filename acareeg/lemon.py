# Authors: Christian O'Reilly <christian.oreilly@gmail.com>
# License: MIT

import mne
from pymatreader import read_mat
import numpy as np
from pathlib import Path

from .utils import get_epochs_to_trials


def get_montage_lemon(subject, root_path, montage_rel_path="EEG_MPILMBB_LEMON/EEG_Localizer_BIDS_ID/",
                      parse_pattern_montage="sub-{subject}/sub-{subject}.mat"):
    path_montage = Path(root_path) / montage_rel_path
    file_name = path_montage / parse_pattern_montage.format(subject=subject)
    if file_name.exists():
        montage_mat_file = read_mat(str(file_name))

        head_points = {}
        for ch_name in np.unique(montage_mat_file["HeadPoints"]["Label"]):
            inds = np.where(np.array(montage_mat_file["HeadPoints"]["Label"]) == ch_name)[0]
            head_points[ch_name] = montage_mat_file["HeadPoints"]["Loc"][:, inds].mean(1)

        ch_names = [ch_name.split("_")[-1] for ch_name in montage_mat_file["Channel"]["Name"]]
        ch_names = [ch_name if ch_name[:2] != "FP" else "Fp" + ch_name[2:] for ch_name in ch_names]

        ch_pos = dict(zip(ch_names, montage_mat_file["Channel"]["Loc"]))

        return mne.channels.make_dig_montage(ch_pos=ch_pos,
                                             nasion=head_points["NA"],
                                             lpa=head_points["LPA"],
                                             rpa=head_points["RPA"])

    return mne.channels.make_standard_montage('standard_1020')


def get_events_lemon(raw, event_ids=None):
    if event_ids is None:
        event_ids = {"EO": 1, "EC": 2}

    annot_sample = []
    annot_id = []
    freq = raw.info["sfreq"]

    annot_map = {'2': "EO", '3': 'EO', '4': 'EC'}

    for a in raw.annotations:
        if a["description"] in annot_map:
            annot_sample.append(int(a["onset"] * freq))
            annot_id.append(event_ids[annot_map[a["description"]]])

    return np.array([annot_sample, [0] * len(annot_sample), annot_id], dtype=int).T


def get_epochs_lemon(subject, root_path, event_id, tmin=0, tmax=2 - 1/250, baseline=None,
                     eeg_rel_path="EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed"):
    montage = get_montage_lemon(subject, root_path)
    raw = mne.io.read_raw_eeglab(str(Path(root_path) / eeg_rel_path / f"sub-{subject}_{event_id}.set"), verbose=False)
    raw.set_montage(montage, on_missing="warn", verbose=False)

    events = get_events_lemon(raw)
    return mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=baseline, event_repeated="drop", verbose=False)


def get_trials_lemon(subject, root_path, event_ids=("EO", "EC"), use_csd=False):
    epochs_lst = []
    for event_id in event_ids:
        epochs = get_epochs_lemon(subject, root_path, event_id)
        epochs.load_data()
        epochs = epochs.pick("eeg")
        assert(epochs.info["sfreq"] == 250)
        assert(len(epochs.times) == 500)
        epochs_lst.append(epochs)
    return get_epochs_to_trials(epochs_lst, subject, use_csd=use_csd)
