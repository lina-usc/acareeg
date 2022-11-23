# Authors: Christian O'Reilly <christian.oreilly@gmail.com>
# License: MIT

import os
import pandas as pd
import numpy as np
import mne
from collections import OrderedDict
from mne.io.eeglab.eeglab import _check_load_mat, _get_info
from mne.preprocessing import read_ica_eeglab
from pathlib import Path
import xarray as xr
import paramiko
import getpass
from platform import node

from .infantmodels import compute_sources


chan_mapping = {"E17": "NAS", "E22": "Fp1", "E9": "Fp2", "E11": "Fz", "E124": "F4", "E122": "F8", "E24": "F3",
                "E33": "F7", "E36": "C3", "E45": "T7", "E104": "C4", "E108": "T8", "E52": "P3", "E57": "LM",
                "E58": "P7", "E92": "P4", "E100": "RM", "E96": "P8", "E62": "Pz", "E70": "O1", "E75": "Oz",
                "E83": "O2"}

event_id = {
    ("london", 6): {"rst": 1},
    ("london", 12): {"base": 0, "eeg1": 1, "eeg2": 2, "eeg3": 3, "eeg4": 4},
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


def get_mastersheet(mastersheet_filename="mastersheet_latest.xlsx", **kwargs):
    mastersheet_path = '/lustre03/project/def-emayada/notebooks/eegip_mastersheet/current_mastersheet/'
    return get_file(mastersheet_path, mastersheet_filename, **kwargs)


def get_measure(measure='mullen', **kwargs):
    measure_path = '/lustre03/project/def-emayada/notebooks/eegip_mastersheet'\
                   '/output_mastersheet_excel_file/frontend_sheets/'
    measure_file_name = f'eegip_{measure}.xlsx'
    return get_file(measure_path, measure_file_name, **kwargs)


def get_file(file_path, file_name, host="narval.calculquebec.ca", port=22, force_download=False):

    file_path = Path(file_path)

    # if the user is on the narval cluster
    if 'narval' in node():
        # grab the file locally
        return pd.read_excel(file_path / file_name, index_col=0, header=0)

    if not Path(file_name).exists() or force_download:
        username = input(f'Enter your username for {host}:')
        password = getpass.getpass(f'Enter password for {host}:')

        transport = paramiko.Transport((host, port))
        transport.connect(None, username, password)

        with paramiko.SFTPClient.from_transport(transport) as sftp_client:
            sftp_client.get(str(file_path / file_name), file_name)

    return pd.read_excel(file_name, index_col=0, header=0)


def preprocess(raw, notch_width=None, line_freq=50.0):
    if notch_width is None:
        notch_width = np.array([1.0, 0.1, 0.01, 0.1])

    notch_freqs = np.arange(line_freq, raw.info["sfreq"]/2.0, line_freq)
    raw.notch_filter(notch_freqs, picks=["eeg", "eog"], fir_design='firwin',
                     notch_widths=notch_width[:len(notch_freqs)], verbose=None)


def mark_bad_channels(raw, file_name, mark_to_remove=("manual", "rank")):
    raw_eeg = _check_load_mat(file_name, None)
    info, _, _ = _get_info(raw_eeg)
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


def preprocessed_raw(path, line_freq, montage=None, verbose=False, rename_channel=False, apply_ica=True,
                     interp_bad_ch=True, reset_bads=True):
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=verbose)
    raw.set_montage(montage, verbose=verbose)

    preprocess(raw, line_freq=line_freq, notch_width=np.array([0.0, 0.1, 0.01, 0.1]))

    raw = raw.filter(1, None, fir_design='firwin', verbose=verbose)

    mark_bad_channels(raw, path)
    add_bad_segment_annot(raw, path)
    if apply_ica:
        remove_rejected_ica_components(raw, path, inplace=True)

    if interp_bad_ch:
        raw = raw.interpolate_bads(reset_bads=reset_bads, verbose=verbose)

    if rename_channel:
        raw.rename_channels({ch: ch2 for ch, ch2 in chan_mapping.items() if ch in raw.ch_names})

    raw.set_channel_types({ch: "eog" for ch in eog_channels if ch in raw.ch_names})

    return raw


def process_events_london_resting_state(raw, age, tmax=1):
    annot_sample = []
    annot_id = []
    freq = raw.info["sfreq"]

    if age == 6:
        """
         For London 6m, resting-state data are marked by pairs of Rst0-Rst1 events marking
         the begining and the end of every resting-state trial. We reject any trial that has
         a duration smaller than 10s.
        """

        rst0 = []
        rst1 = []
        for a in raw.annotations:
            if a["description"] == 'Rst0':
                rst0.append(a["onset"])
            if a["description"] == 'Rst1':
                rst1.append(a["onset"])

        if len(rst0) == 0 and len(rst1) == 0:
            return None
        assert(len(rst0) == len(rst1))

        # Keep only trials that are at least 10 s long.
        # All trials are defined by Rst0-Rst1 pair of events.
        annot_sample = np.concatenate([np.arange(start*freq, (stop-tmax)*freq, tmax*freq)
                                       for start, stop in zip(rst0, rst1)
                                       if stop - start > 10.0]).astype(int)
        annot_id = [1] * len(annot_sample)

    elif age == 12:
        """
         For London 12m, resting-state data are marked only by Rst0 markers at the begining of
         every trials. Trials and consecutives, so we can use the following Rst0 maker as being the 
         end of the current trials. The last trials is ended with the last occurence of a SWIR or SWEN
         event. 
         
         Trials shorter than 10s are rejected. Trials longer than 50s are truncated at 50s.
         
         Events eeg1, eeg2 and eeg3, when present, are at the same time as the Rst0 events. They 
         are used to set the type of event (type of video that was presented). If no eegx event
         is available for a Rst0, as type eeg4 is set, which indicate a resting-state event using
         an unknown video.
        """

        onsets = [annot["onset"] for annot in raw.annotations if annot["description"] == "Rst0"]
        if len(onsets) == 0:
            return None

        final_swir_swen_onset = [annot["onset"]
                                 for annot in raw.annotations
                                 if annot["description"] in ["SWEN", "SWIR"]][-1]
        durations = list(np.diff(onsets)) + [final_swir_swen_onset - onsets[-1]]

        # Truncate longer trials at 50s
        durations = [min(d, 50) for d in durations]

        # Reject any trial that last less than 10s
        durations, onsets = zip(*[(d, onset) for d, onset in zip(durations, onsets) if d >= 10])

        # Find eeg1, eeg2, eeg3 events clost (less than 5s appart) for Rst0 and attribute this event
        # type. Else, attribute eeg4 (unpecified video type).
        eegx_annots = [(a["description"], a["onset"])
                       for a in raw.annotations
                       if a["description"] in ["eeg1", "eeg2", "eeg3"]]
        if len(eegx_annots):
            eegx_desc, eegx_onsets = zip(*eegx_annots)
        else:
            eegx_desc = []
            eegx_onsets = []

        trial_types = []
        for onset in onsets:
            inds = np.where(np.abs(eegx_onsets-np.array(onset))<0.01)[0]
            if len(inds) == 0:
                trial_types.append("eeg4")
            elif len(inds) == 1:
                trial_types.append(eegx_desc[inds[0]])
            else:
                print(eegx_onsets)
                print(onsets, final_swir_swen_onset)
                raise RuntimeError

        annot_id = []
        for onset, duration, trial_type in zip(onsets, durations, trial_types):
            annot_sample.append(np.arange(int(onset * freq),
                                          int((onset+duration-tmax) * freq),
                                          int(tmax * freq)))
            id_ = event_id[("london", age)][trial_type]
            annot_id.extend(id_ * np.ones(len(annot_sample[-1])))
        annot_sample = np.concatenate(annot_sample)
    else:
        raise ValueError("Invalid value for session.")

    return np.array([annot_sample, [0] * len(annot_sample), annot_id], dtype=int).T

def process_events_seattle_resting_state(*args, **kwargs):
    process_events_washington_resting_state(*args, **kwargs)

def process_events_washington_resting_state(raw, tmax=1):
    """
     The Seattle dataset comes with sequence of events like:
      ['Socl', 'Socl', 'EndM', 'Toys', 'Toys', 'EndM', 'Socl', 'Socl', 'EndM', 'Toys', 'Toys', 'EndM']
     The EndM markers can be ingnored and the pairs of either Socl or Toys indicate start-stop timing.
    """

    annot_sample = []
    annot_id = []
    freq = raw.info["sfreq"]

    annots = [a for a in raw.annotations if a["description"] in ["Toys", "Socl"]]
    if len(annots) == 0:
        return None

    """
     There are three problematic series (subject age series):
        102 12 ['Socl']
        915 6 ['Toys', 'Toys', 'EndM', 'Socl']
        915 12 ['Socl']
     For all these cases the number of Socl/Toys is odd and the problematic event is the last one.
     If we compare the time of the end of the file (subject age onset end_recording):
        102 12 558.386 589.998
        915 6 260.166 464.998
        915 12 108.34 142.998
     In two cases, the end of the file "close" those events but in one case, it does not (not until 200s, which
     is too long. So we will close these open events at the end of the file or 60s after the onset, whichever
     comes first.
     """
    if len(annots) % 2 == 1:
        annots.append(OrderedDict((("onset", min(annots[-1]["onset"] + 60., raw.times[-1])),
                                   ("duration", 0),
                                   ("description", annots[-1]["description"]),
                                   ('orig_time', None))))

    for annot, next_annot in zip(annots[:-1:2], annots[1::2]):
        annot_sample.append(np.arange(int(annot["onset"] * freq),
                                      int((next_annot["onset"] - tmax) * freq),
                                      int(tmax * freq)))
        id_ = event_id["washington"]["videos"][annot["description"]]
        annot_id.extend(id_ * np.ones(len(annot_sample[-1])))

    annot_sample = np.concatenate(annot_sample)

    return np.array([annot_sample, [0] * len(annot_sample), annot_id], dtype=int).T


def process_events_resting_state(raw, dataset, age, tmax=1):
    if dataset == "london":
        return process_events_london_resting_state(raw, age, tmax=tmax)
    if dataset == "washington":
        return process_events_washington_resting_state(raw, tmax=tmax)


def process_epochs(raw, dataset, age, events, tmin=0, tmax=1, verbose=None):

    freq = raw.info["sfreq"]
    if dataset == "london":
        filtered_event_id = {key: val for key, val in event_id[("london", age)].items() if val in events[:, 2]}
        if len(filtered_event_id):
            # "baseline=None" because the baseline is corrected by a 1Hz high-pass on the raw data
            return mne.Epochs(raw, events, filtered_event_id, tmin=tmin,
                              tmax=tmax - 1.0 / freq, baseline=None,
                              preload=True, reject_by_annotation=True,
                              verbose=verbose)
        return None

    elif dataset == "washington":

        filtered_event_id = {key: val for key, val in event_id[dataset]["videos"].items() if val in events[:, 2]}
        if len(filtered_event_id):
            return mne.Epochs(raw, events, filtered_event_id, tmin=tmin,
                              tmax=tmax - 1.0 / freq, baseline=None,
                              preload=True, reject_by_annotation=True,
                              verbose=verbose)
        return None


def get_resting_state_epochs(subject, dataset, age, bids_root="/project/def-emayada/eegip/",
                             subjects_dir=None, montage_name="HGSN129-montage.fif", tmax=1,
                             rename_channel=False, apply_ica=True, interp_bad_ch=True, reset_bads=True):



    subject = int(subject)
    age = int(age)
    eeg_path = Path(bids_root) / dataset / "derivatives" / "lossless" / f"sub-s{subject:03}" / f"ses-m{age:02}" / "eeg"
    eeg_path = list(eeg_path.glob("*_qcr.set"))
    if len(eeg_path) == 0:
        return None
    eeg_path = eeg_path[0]

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

    raw = preprocessed_raw(eeg_path, line_freqs[dataset], montage, rename_channel=rename_channel, 
                           apply_ica=apply_ica, interp_bad_ch=interp_bad_ch, reset_bads=reset_bads)
    events = process_events_resting_state(raw, dataset, age, tmax=tmax)
    if events is None:
        return
    return process_epochs(raw, dataset, age, events, tmax=tmax)


def get_connectivity(epochs, age, fmin=(4, 8, 12, 30, 4), fmax=(8, 12, 30, 100, 100),
                     bands=("theta", "alpha", "beta", "gamma", "broadband"), con_name="ciplv",
                     mode='multitaper', faverage=True, return_type="df", minimal_snr=None,
                     verbose=True, template=None, src_kwargs=None):

    if src_kwargs is None:
        src_kwargs = {}

    label_ts, anat_label = compute_sources(epochs, age, template=template, return_labels=True, return_xr=False,
                                           minimal_snr=minimal_snr, verbose=verbose, **src_kwargs)
    label_names = [label.name for label in anat_label]

    sfreq = epochs.info['sfreq']
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


def get_event_counts(dataset, bids_root="/project/def-emayada/eegip/"):
    root = Path(bids_root) / dataset / "derivatives" / "lossless"
    results = {"subjects": [], "ages": []}
    for age in ["06", "12", "18"]:
        for path in root.glob(f"sub-s*/ses-m{age}/eeg/sub-s*_ses-m{age}_eeg_qcr.set"):
            results["subjects"].append(str(path)[-23:-20])
            results["ages"].append(age)
            raw = mne.io.read_raw_eeglab(path)
            event_names, counts = np.unique([annot["description"] for annot in raw.annotations], return_counts=True)
            event_counts = dict(list(zip(event_names, counts)))
            for event_name in event_names:
                if event_name not in results:
                    if len(results["ages"]) == 1:
                        results[event_name] = []
                    else:
                        results[event_name] = [0]*(len(results["ages"])-1)
            for event_name in results:
                if event_name in ["ages", "subjects"]:
                    continue
                if event_name in event_counts:
                    results[event_name].append(event_counts[event_name])
                else:
                    results[event_name].append(0)

    return pd.DataFrame(results).sort_values(["ages", "subjects"])
