import os
import pandas as pd
import numpy as np
import mne
from collections import OrderedDict
from mne.io.eeglab.eeglab import _check_load_mat, _get_info
from mne.preprocessing import read_ica_eeglab
from pathlib import Path

from .infantmodels import compute_sources


models_path = Path("/home/christian/ni_connectivity_models")


chan_mapping = {"E17": "NAS", "E22": "Fp1", "E9": "Fp2", "E11": "Fz", "E124": "F4", "E122": "F8", "E24": "F3",
                "E33": "F7", "E36": "C3", "E45": "T7", "E104": "C4", "E108": "T8", "E52": "P3", "E57": "LM",
                "E58": "P7", "E92": "P4", "E100": "RM", "E96": "P8", "E62": "Pz", "E70": "O1", "E75": "Oz",
                "E83": "O2"}

event_id = {
    ("london", 6): {"rst": 1},
    ("london", 12): {"base": 0, "eeg1": 1, "eeg2": 2, "eeg3": 3},
    "washington": {
        "cov": {"cov": 0},
        "videos": {"cov": 0, "Toys": 1, "Socl": 2},
        "videos_only": {"Toys": 1, "Socl": 2},
        "all": {"cov": 0, "Toys": 1, "Socl": 2}
    }
}

# As per https://github.com/methlabUZH/automagic/wiki/How-to-start
# These number are coherent with the map shown in geodesic-sensor-net.pdf
eog_channels = ["E1", "E8", "E14", "NAS", "E21", "E25", "E32", "E125", "E126", "E127", "E128"]

montage_name = "GSN-HydroCel-129-montage.fif"

datasets = ["london", "washington"]
ages = {"london": ["06", "12"], "washington": ["06", "12", "18"]}
line_freqs = {"london": 50.0, "washington": 60.0}

patterns = {"london": "derivatives/lossless/sub-s*/ses-m{age}/eeg/sub-s*_ses-m{age}_eeg_qcr.set",
            "washington": "derivatives/lossless/sub-s*/ses-m{age}/eeg/sub-s*_ses-m{age}_task-*_eeg_qcr.set"}


def preprocess(raw, notch_width=None, line_freq=50.0):
    if notch_width is None:
        notch_width = np.array([1.0, 0.1, 0.01, 0.1])

    notch_freqs = np.arange(line_freq, raw.info["sfreq"]/2.0, line_freq)
    raw.notch_filter(notch_freqs, picks=["eeg", "eog"], fir_design='firwin',
                     notch_widths=notch_width[:len(notch_freqs)], verbose=None)


def mark_bad_channels(raw, file_name, mark_to_remove=("manual", "rank")):
    raw_eeg = _check_load_mat(file_name, None)
    info, _, _ = _get_info(raw_eeg)
    print("############ file_name", file_name, type(file_name))
    print("############ raw_eeg", raw_eeg.keys(), type(raw_eeg))
    chan_info = raw_eeg.marks["chan_info"]

    mat_chans = np.array(info["ch_names"])
    assert (len(chan_info["flags"][0]) == len(mat_chans))

    if len(np.array(chan_info["flags"]).shape) > 1:
        ind_chan_to_drop = np.unique(np.concatenate([np.where(flags)[0] for flags, label in zip(chan_info["flags"],
                                                                                                chan_info["label"])
                                                     if label in mark_to_remove]))
    else:
        ind_chan_to_drop = np.where(chan_info["flags"])[0]

    bad_chan = [chan for chan in mat_chans[ind_chan_to_drop]]

    raw.info['bads'].extend(bad_chan)


def add_bad_segment_annot(raw, file_name, mark_to_remove=("manual",)):
    raw_eeg = _check_load_mat(file_name, None)
    time_info = raw_eeg.marks["time_info"]

    if len(np.array(time_info["flags"]).shape) > 1:
        ind_time_to_drop = np.unique(np.concatenate([np.where(flags)[0] for flags, label in zip(time_info["flags"],
                                                                                                time_info["label"])
                                                     if label in mark_to_remove]))
    else:
        ind_time_to_drop = np.where(time_info["flags"])[0]

    ind_starts = np.concatenate(
        [[ind_time_to_drop[0]], ind_time_to_drop[np.where(np.diff(ind_time_to_drop) > 1)[0] + 1]])
    ind_ends = np.concatenate([ind_time_to_drop[np.where(np.diff(ind_time_to_drop) > 1)[0]], [ind_time_to_drop[-1]]])
    durations = (ind_ends + 1 - ind_starts) / raw.info["sfreq"]
    onsets = ind_starts / raw.info["sfreq"]

    for onset, duration in zip(onsets, durations):
        raw.annotations.append(onset, duration, description="bad_lossless_qc")


def remove_rejected_ica_components(raw, file_name, inplace=True):
    raw_eeg = _check_load_mat(file_name, None)
    mark_to_remove = ["manual"]
    comp_info = raw_eeg.marks["comp_info"]

    if len(np.array(comp_info["flags"]).shape) > 1:
        ind_comp_to_drop = np.unique(np.concatenate([np.where(flags)[0] for flags, label in zip(comp_info["flags"],
                                                                                                comp_info["label"])
                                                     if label in mark_to_remove]))
    else:
        ind_comp_to_drop = np.where(comp_info["flags"])[0]

    if inplace:
        read_ica_eeglab(file_name).apply(raw, exclude=ind_comp_to_drop)
    else:
        read_ica_eeglab(file_name).apply(raw.copy(), exclude=ind_comp_to_drop)


def preprocessed_raw(path, line_freq, montage=None, verbose=False, rename_channel=False, apply_ica=True):
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=verbose)
    raw.set_montage(montage, verbose=verbose)

    preprocess(raw, line_freq=line_freq, notch_width=np.array([0.0, 0.1, 0.01, 0.1]))

    raw = raw.filter(1, None, fir_design='firwin', verbose=verbose)

    mark_bad_channels(raw, path)
    add_bad_segment_annot(raw, path)
    if apply_ica:
        remove_rejected_ica_components(raw, path, inplace=True)

    raw = raw.interpolate_bads(reset_bads=True, verbose=verbose)

    if rename_channel:
        raw.rename_channels({ch: ch2 for ch, ch2 in chan_mapping.items() if ch in raw.ch_names})

    raw.set_channel_types({ch: "eog" for ch in eog_channels if ch in raw.ch_names})

    return raw


def process_events_london_resting_state(raw, age):
    annot_sample = []
    annot_id = []
    freq = raw.info["sfreq"]

    if age == 6:
        rst0 = []
        rst1 = []
        for a in raw.annotations:
            if a["description"] == 'Rst0':
                rst0.append(a["onset"])
            if a["description"] == 'Rst1':
                rst1.append(a["onset"])

        if len(rst0) == 0 and len(rst1) == 0:
            return None

        annot_sample = np.concatenate([np.arange(start, stop - 0.999, 1.0) for start, stop in zip(rst0, rst1)])
        annot_sample = (annot_sample * freq).astype(int)
        annot_id = [1] * len(annot_sample)

    elif age == 12:
        #annots = [OrderedDict((("onset", 0), ("duration", 0), ("description", "base"), ('orig_time', None)))]
        #annots.extend([a for a in raw.annotations
        #               if a["description"] in ["eeg1", "eeg2", "eeg3"]])
        annots = [a for a in raw.annotations if a["description"] in ["eeg1", "eeg2", "eeg3"]]
        if len(annots) == 0:
            return None

        if raw.annotations[-1]["onset"] > annots[-1]["onset"]:
            end = np.min([raw.annotations[-1]["onset"], annots[-1]["onset"] + 50.])
            annots.append(OrderedDict((("onset", end), ("duration", 0),
                                       ("description", "end"), ('orig_time', None))))

        for annot, next_annot in zip(annots[:-1], annots[1:]):
            annot_sample.append(np.arange(int(annot["onset"] * freq),
                                          int(next_annot["onset"] * freq),
                                          int(tmax * freq)))
            annot_id.extend(event_id[("london", age)][annot["description"]] * np.ones(len(annot_sample[-1])))

        annot_sample = np.concatenate(annot_sample)

    else:
        raise ValueError("Invalid value for session.")

    return np.array([annot_sample, [0] * len(annot_sample), annot_id], dtype=int).T


def process_events_washington_resting_state(raw):
    annot_sample = []
    annot_id = []
    freq = raw.info["sfreq"]

    # COV AND VIDEOS
    #annots = [OrderedDict((("onset", 0), ("duration", 0), ("description", "cov"), ('orig_time', None)))]
    #annots.extend([a for a in raw.annotations if a["description"] in ["Toys", "EndM", "Socl"]])
    annots = [a for a in raw.annotations if a["description"] in ["Toys", "EndM", "Socl"]]
    if len(annots) == 0:
        return None

    annots.append(OrderedDict((("onset", annots[-1]["onset"] + 50.), ("duration", 0),
                               ("description", "end"), ('orig_time', None))))

    for annot, next_annot in zip(annots[:-1], annots[1:]):
        if annot["description"] == "EndM":
            continue

        annot_sample.append(np.arange(int(annot["onset"] * freq),
                                      int(next_annot["onset"] * freq),
                                      int(tmax * freq)))
        id_ = event_id["washington"]["videos"][annot["description"]]
        annot_id.extend(id_ * np.ones(len(annot_sample[-1])))

    annot_sample = np.concatenate(annot_sample)

    return np.array([annot_sample, [0] * len(annot_sample), annot_id], dtype=int).T


def process_events_resting_state(raw, dataset, age):
    if dataset == "london":
        return process_events_london_resting_state(raw, age)
    if dataset == "washington":
        return process_events_washington_resting_state(raw)


def process_epochs(raw, dataset, age, events, tmin=0, tmax=1):

    freq = raw.info["sfreq"]
    if dataset == "london":
        filtered_event_id = {key: val for key, val in event_id[("london", age)].items() if val in events[:, 2]}
        if len(filtered_event_id):
            # "tmax = tmax[dataset] - 1.0 / freq" because MNE is inclusive on the last point and we don't want that
            # "baseline=None" because the baseline is corrected by a 1Hz high-pass on the raw data
            return mne.Epochs(raw, events, filtered_event_id, tmin=tmin,
                              tmax=tmax - 1.0 / freq, baseline=None,
                              preload=True, reject_by_annotation=True)
        return None

    elif dataset == "washington":

        filtered_event_id = {key: val for key, val in event_id[dataset]["videos"].items() if val in events[:, 2]}
        if len(filtered_event_id):
            return mne.Epochs(raw, events, filtered_event_id, tmin=tmin,
                                       tmax=tmax - 1.0 / freq, baseline=None,
                                       preload=True, reject_by_annotation=True)
        return None


def get_resting_stage_epochs(subject, dataset, age, bids_root="/project/def-emayada/eegip/",
                             subjects_dir=None, montage_name="HGSN129-montage.fif"):

    eeg_path = Path(bids_root) / dataset / "derivatives" / "lossless" / f"sub-s{subject}" / f"ses-m{age:02}" / "eeg"
    eeg_path = list(eeg_path.glob("*_qcr.set"))[0]

    montage = None
    if subjects_dir is None:
        subjects_dir = Path(os.environ["SUBJECTS_DIR"])
        template = f"ANTS{age}-0Months3T"
        montage_path = Path(subjects_dir) / template / "montages" / montage_name
        if montage_path.exists():
            montage = mne.channels.read_dig_fif(str(montage_path))
            montage.ch_names = ["E" + str(int(ch_name[3:])) for ch_name in montage.ch_names]
            montage.ch_names[128] = "Cz"

    if montage is None:
        montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

    raw = preprocessed_raw(eeg_path, line_freqs[dataset], montage)
    events = process_events_resting_state(raw, dataset, age)
    if events is None:
        return
    return process_epochs(raw, dataset, age, events)


def get_connectivity(epochs, age, fmin=(4, 8, 12, 30, 4), fmax=(8, 12, 30, 100, 100),
                        bands=("theta", "alpha", "beta", "gamma", "broadband"), con_name="ciplv",
                        mode='multitaper', faverage=True, return_type="df"):

    label_ts, anat_label = compute_sources(epochs, age, return_labels=True, return_xr=False)
    label_names = [label.name for label in anat_label]

    sfreq = epochs.info['sfreq']  # the sampling frequency

    dfs = []
    con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(label_ts, method=con_name,
                                                                                    mode=mode, sfreq=sfreq, fmin=fmin,
                                                                                    fmax=fmax, faverage=faverage)

    if return_type == "df":
        for mat, band in zip(con.transpose(2, 0, 1), bands):
            mat = pd.DataFrame(mat) + np.triu(mat * np.nan)
            mat.index = label_names
            mat.columns = label_names
            df = mat.reset_index().melt(id_vars="index").dropna()
            df.columns = ["region1", "region2", "con"]
            df["con_name"] = con_name
            df["band"] = band
            df["age"] = age
            dfs.append(df)

        return pd.concat(dfs)

    if return_type == "xr":
        return xr.DataArray(con,
                            dims=("region1", "region2", "band"),
                            coords={"region1": label_names,
                                    "region2": label_names,
                                    "band": bands})
