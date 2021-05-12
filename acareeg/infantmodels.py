# Authors: Christian O'Reilly <christian.oreilly@gmail.com>
# License: MIT

from pathlib import Path
import os
import mne
import time
import xarray as xr
import numpy as np
import os.path as op

from .simulation import get_epochs_sim


def get_bem_artifacts(template, montage_name="HGSN129-montage.fif", subjects_dir=None):

    if subjects_dir is None:
        subjects_dir = Path(os.environ["SUBJECTS_DIR"])
    if template == "fsaverage":
        fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
        trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
        surface_src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
        bem_solution = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
        bem_model = None
        montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
        return montage, trans, bem_model, bem_solution, surface_src


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


def validate_model(age=None, template=None, subjects_dir=None):
    get_head_models()

    if template is None:
        if age is not None:
            template = f"ANTS{age}-0Months3T"
        else:
            raise ValueError("The age or the template must be specified.")

    montage, trans, bem_model, bem_solution, surface_src = get_bem_artifacts(template, subjects_dir=subjects_dir)
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


def process_sources(epochs, trans, surface_src, bem_solution, fwd_mindist=5.0, diag_cov=True,
                    cov_method="auto", loose=0.0, inv_method="eLORETA", lambda2=1e-4, pick_ori=None,
                    return_generator=True, return_fwd=False):

    fwd = mne.make_forward_solution(epochs.info, trans, surface_src, bem_solution, mindist=fwd_mindist, eeg=True)
    noise_cov = mne.compute_covariance(epochs, method=cov_method)
    if diag_cov:
        noise_cov = noise_cov.as_diag()

    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov, loose=loose)
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, method=inv_method, lambda2=lambda2,
                                                 pick_ori=pick_ori, return_generator=return_generator)
    if return_fwd:
        return stcs, fwd
    return stcs


def sources_to_labels(stcs, age=None, template=None, parc='aparc', mode='mean_flip',
                      allow_empty=True, return_generator=False, subjects_dir=None):
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


def compute_sources(epochs, age, subjects_dir=None, template=None, return_labels=False, return_xr=True,
                    loose=0.0, inv_method="eLORETA", lambda2=1e-4, minimal_snr=None, verbose=True):
    get_head_models()

    if template is None:
        template = f"ANTS{age}-0Months3T"
    if subjects_dir is None:
        subjects_dir = Path(os.environ["SUBJECTS_DIR"])

    montage, trans, bem_model, bem_solution, surface_src = get_bem_artifacts(template, subjects_dir=subjects_dir)
    if template != "fsaverage":
        montage.ch_names = ["E" + str(int(ch_name[3:])) for ch_name in montage.ch_names]
        montage.ch_names[128] = "Cz"
    epochs.set_montage(montage)

    if minimal_snr is not None:
        nb_epochs = min(5, len(epochs))
        stcs, fwd = process_sources(epochs[:nb_epochs], trans, surface_src, bem_solution, return_generator=False,
                                    loose=loose, inv_method=inv_method, lambda2=lambda2, return_fwd=True)
        snr = get_epochs_sim(epochs[:nb_epochs], stcs, fwd, return_snr=True)[1]
        if verbose:
            print(f"## Average SNR for the EGG reconstruction from estimated sources: {np.mean(snr)} dB")
        if np.mean(snr) < minimal_snr:
            print(snr)
            raise RuntimeError("Insufficient SNR detected. You should normally expect SNR > 15 dB. If the "
                               "values obtained are smaller than that, you should check the coregistration "
                               "of your electrode montage to your head model by "
                               f"calling acareeg.infantmodels.validate_model(age={age}).")

    stcs = process_sources(epochs, trans, surface_src, bem_solution, return_generator=True,
                                loose=loose, inv_method=inv_method, lambda2=lambda2)

    if return_labels:
        label_ts, anat_label = sources_to_labels(stcs, age=age, subjects_dir=subjects_dir)
        if return_xr:
            return xr.DataArray(np.array(label_ts),
                                dims=("epoch", "region", "time"),
                                coords={"epoch": np.arange(len(label_ts)),
                                        "region": [label.name for label in anat_label],
                                        "time": epochs.times})
        return label_ts, anat_label

    if return_xr:
        stcs = np.stack([stc.data for stc in stcs])
        return xr.DataArray(stcs,
                            dims=("epoch", "source", "time"),
                            coords={"epoch": np.arange(len(stcs)),
                                    "source": np.arange(stcs.shape[1]),
                                    "time": epochs.times})
    return stcs
