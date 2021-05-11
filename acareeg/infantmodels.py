from pathlib import Path
import os
import mne
import xarray as xr
from mne.io.eeglab.eeglab import _check_load_mat, _get_info
from mne.preprocessing import read_ica_eeglab

import numpy as np
from collections import OrderedDict
import pandas as pd
import neurokit as nk
import nolds

from acareeg.eegip import mark_bad_channels, add_bad_segment_annot, remove_rejected_ica_components


def get_bem_artifacts(template, montage_name="HGSN129-montage.fif", subjects_dir=None):

    if subjects_dir is None:
        subjects_dir = Path(os.environ["SUBJECTS_DIR"])

    montage = mne.channels.read_dig_fif(str(Path(subjects_dir) / template / "montages" / montage_name))
    trans = mne.channels.compute_native_head_t(montage)
    bem_model = mne.read_bem_surfaces(str(Path(subjects_dir) / template / "bem" / f"{template}-5120-5120-5120-bem.fif"))
    bem_solution = mne.read_bem_solution(
        str(Path(subjects_dir) / template / "bem" / f"{template}-5120-5120-5120-bem-sol.fif"))
    surface_src = mne.read_source_spaces(str(Path(subjects_dir) / template / "bem" / f"{template}-oct-6-src.fif"))

    return montage, trans, bem_model, bem_solution, surface_src


def get_head_models(ages=("6mo", "12mo", "18mo"), subjects_dir=None):
    for age in ages:
        mne.datasets.fetch_infant_template(age, subjects_dir=subjects_dir)


def validate_model(template):
    get_head_models()

    montage, trans, bem_model, bem_solution, surface_src = get_bem_artifacts(template)
    montage.ch_names = ["E" + str(int(ch_name[3:])) for ch_name in montage.ch_names]
    montage.ch_names[128] = "Cz"

    info = mne.create_info(montage.ch_names, sfreq=256, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((len(montage.ch_names), 1)), info, copy=None, verbose=False).set_montage(montage)

    fig = mne.viz.plot_alignment(raw.info, trans=trans, subject=template,
                                 subjects_dir=subjects_dir, surfaces='head',
                                 show_axes=True, dig="fiducials", eeg="projected",
                                 coord_frame='mri', mri_fiducials=True,
                                 src=surface_src, bem=bem_solution)
    time.sleep(1.0)
    fig.plotter.off_screen = True
    mne.viz.set_3d_view(figure=fig, azimuth=135, elevation=80, distance=0.6)
    time.sleep(1.0)
    fig.plotter.screenshot(f"coregistration_{template}_1.png")

    mne.viz.set_3d_view(figure=fig, azimuth=45, elevation=80, distance=0.6)
    time.sleep(1.0)
    fig.plotter.screenshot(f"coregistration_{template}_2.png")

    mne.viz.set_3d_view(figure=fig, azimuth=270, elevation=80, distance=0.6)
    time.sleep(1.0)
    fig.plotter.screenshot(f"coregistration_{template}_3.png")


def process_sources(epochs, trans, surface_src, bem_solution, fwd_mindist=0, diag_cov=False,
                    cov_method="auto", loose=0.0, inv_method="eLORETA", lambda2=0.1, pick_ori=None,
                    return_generator=True):

    fwd = mne.make_forward_solution(epochs.info, trans, surface_src, bem_solution, mindist=fwd_mindist)
    noise_cov = mne.compute_covariance(epochs, method=cov_method)
    if diag_cov:
        noise_cov = noise_cov.as_diag()

    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov, loose=loose)
    return mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, method=inv_method, lambda2=lambda2,
                                                 pick_ori=pick_ori, return_generator=return_generator)


def sources_to_labels(stcs, age=None, template=None, parc='aparc',
                      mode='mean_flip', allow_empty=True, return_generator=False,
                      subjects_dir=None):

        if template is None:
            if age is not None:
                template = f"ANTS{age}-0Months3T"
            else:
                raise ValueError("The age or the template must be specified.")

        montage, trans, bem_model, bem_solution, surface_src = get_bem_artifacts(template, subjects_dir=subjects_dir)

        anat_label = mne.read_labels_from_annot(template, subjects_dir=subjects_dir, parc=parc)
        label_ts = mne.extract_label_time_course(stcs, anat_label, surface_src, mode=mode,
                                                 allow_empty=allow_empty, return_generator=return_generator)
        return label_ts, anat_label


def compute_sources(epochs, age, subjects_dir=None, template=None, return_labels=False):
    get_head_models()

    if template is None:
        template = f"ANTS{age}-0Months3T"
    if subjects_dir is None:
        subjects_dir = Path(os.environ["SUBJECTS_DIR"])

    montage, trans, bem_model, bem_solution, surface_src = get_bem_artifacts(template, subjects_dir=subjects_dir)
    montage.ch_names = ["E" + str(int(ch_name[3:])) for ch_name in montage.ch_names]
    montage.ch_names[128] = "Cz"
    epochs.set_montage(montage)

    if return_labels:
        stcs = process_sources(epochs, trans, surface_src, bem_solution, return_generator=True)
        label_ts, anat_label = sources_to_labels(stcs, age=age)
        sources_xr = xr.DataArray(np.array(label_ts),
                                  dims=("epoch", "region", "time"),
                                  coords={"epoch": np.arange(len(label_ts)),
                                          "region": [label.name for label in anat_label],
                                          "time": epochs.times})
        return sources_xr

    stcs = process_sources(epochs, trans, surface_src, bem_solution, return_generator=False)
    stcs = np.stack([stc.data for stc in stcs])
    sources_xr = xr.DataArray(stcs,
                              dims=("epoch", "source", "time"),
                              coords={"epoch": np.arange(len(stcs)),
                                      "source": np.arange(stcs.shape[1]),
                                      "time": epochs.times})
    return sources_xr
