# Authors: Christian O'Reilly <christian.oreilly@gmail.com>
# License: MIT

from pathlib import Path
import os
import mne
import time
import xarray as xr
import numpy as np
import os.path as op
import pandas as pd

from .simulation import get_epochs_sim


ages_dic = {0.5:'2wk', 1:'1mo', 2:'2mo', 3:'3mo', 4.5:'4.5mo', 6:'6mo', 7.5:'7.5mo', 9:'9mo', 10.5:'10.5mo', 12:'12mo', 15:'15mo', 18:'18mo', 24:'2yr'}

def get_bem_artifacts(template, montage_name="HGSN129-montage.fif", subjects_dir=None, include_vol_src=True,
                      labels_vol=('Left-Amygdala', 'Left-Caudate', 'Left-Hippocampus', 'Left-Pallidum',
                                  'Left-Putamen', 'Left-Thalamus', 'Right-Amygdala', 'Right-Caudate',
                                  'Right-Hippocampus', 'Right-Pallidum', 'Right-Putamen', 'Right-Thalamus'),
                      force_vol_computation=False):

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

    if not include_vol_src:
        surface_src = mne.read_source_spaces(str(Path(subjects_dir) / template / "bem" / f"{template}-oct-6-src.fif"))
        return montage, trans, bem_model, bem_solution, surface_src

    mixed_src_file_path = Path(subjects_dir) / template / "bem" / f"{template}-mixed-src.fif"
    if not mixed_src_file_path.exists() or force_vol_computation:
        surface_src = mne.read_source_spaces(str(Path(subjects_dir) / template / "bem" / f"{template}-oct-6-src.fif"))
        volume_src = mne.setup_volume_source_space(template, pos=5.0, bem=bem_solution,
                                                add_interpolator=True, volume_label=labels_vol)
        mixed_src = surface_src + volume_src
        mixed_src.save(str(mixed_src_file_path), overwrite=True)

    mixed_src = mne.read_source_spaces(str(mixed_src_file_path))

    return montage, trans, bem_model, bem_solution, mixed_src


def get_head_models(ages=("6mo", "12mo", "18mo"), subjects_dir=None):
    for age in ages:
        mne.datasets.fetch_infant_template(age, subjects_dir=subjects_dir)


def validate_model(age=None, template=None, subjects_dir=None):
    get_head_models(age, subjects_dir)
    template = __validate_template__(age, template, subjects_dir)

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
                    cov_method="auto", loose="auto", fixed=True, inv_method="eLORETA", lambda2=1e-4, pick_ori=None,
                    return_generator=True, return_fwd=False, include_vol_src=True):

    fwd = mne.make_forward_solution(epochs.info, trans, surface_src, bem_solution, mindist=fwd_mindist, eeg=True)
    noise_cov = mne.compute_covariance(epochs, method=cov_method)
    if diag_cov:
        noise_cov = noise_cov.as_diag()

    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov, loose=loose, fixed=fixed)
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, method=inv_method, lambda2=lambda2,
                                                 pick_ori=pick_ori, return_generator=return_generator)
    if return_fwd:
        return stcs, fwd
    return stcs

def __validate_template__(age=None, template=None, subjects_dir=None):
    if template is None:
        if age is not None:
            template = f"ANTS{age}-0Months3T"
        else:
            raise ValueError("The age or the template must be specified.")

    if subjects_dir is None:
        subjects_dir = Path(os.environ["SUBJECTS_DIR"])

    if not (Path(subjects_dir) / template).exists():
        raise ValueError(f"The template {template} is not available in {subjects_dir}. If this template is one "
                         "of the standard infant templates, you can downloaded it using "
                         "mne.datasets.fetch_infant_template")
    return template

def region_centers_of_masse(age=None, template=None, parc="aparc", surf_name="pial",
                            subjects_dir=None, include_vol_src=True):
    template = __validate_template__(age, template, subjects_dir)
    montage, trans, bem_model, bem_solution, src = get_bem_artifacts(template, subjects_dir=subjects_dir,
                                                                     include_vol_src=include_vol_src)
    center_of_masses_dict = {}
    if include_vol_src:
        for src_obj in src[2:]:
            roi_str = src_obj["seg_name"]
            if 'left' in roi_str.lower():
                roi_str = roi_str.replace('Left-', '') + '-lh'
            elif 'right' in roi_str.lower():
                roi_str = roi_str.replace('Right-', '') + '-rh'

            center_of_masses_dict[roi_str] = np.average(src_obj['rr'][src_obj["vertno"]], axis=0)

    for label in mne.read_labels_from_annot(template, subjects_dir=subjects_dir, parc=parc):
        ind_com = np.where(label.vertices == label.center_of_mass(subject=template, subjects_dir=subjects_dir))[0]
        if len(label.pos[ind_com, :]):
            center_of_masses_dict[label.name] = label.pos[ind_com, :][0]

    center_of_masses_df = pd.DataFrame(center_of_masses_dict).T.reset_index()
    center_of_masses_df.columns = ["region", "x", "y", "z"]
    center_of_masses_df["template"] = template
    return center_of_masses_df


def sources_to_labels(stcs, age=None, template=None, parc='aparc', mode='mean_flip',
                      allow_empty=True, return_generator=False, subjects_dir=None,
                      include_vol_src=True):
    template = __validate_template__(age, template, subjects_dir)
    montage, trans, bem_model, bem_solution, src = get_bem_artifacts(template, subjects_dir=subjects_dir,
                                                                     include_vol_src=include_vol_src)

    labels_parc = mne.read_labels_from_annot(template, subjects_dir=subjects_dir, parc=parc)
    labels_ts = mne.extract_label_time_course(stcs, labels_parc, src, mode=mode, allow_empty=allow_empty,
                                             return_generator=return_generator)

    if include_vol_src:
        labels_aseg = mne.get_volume_labels_from_src(src, template, subjects_dir)
        labels = labels_parc + labels_aseg
    else:
        labels = labels_parc

    return labels_ts, labels


def compute_sources(epochs, age, subjects_dir=None, template=None, return_labels=False,
                    return_xr=True, loose="auto", fixed=True, inv_method="eLORETA", pick_ori=None,
                    lambda2=1e-4, minimal_snr=None, verbose=True, include_vol_src=True):
    get_head_models([ages_dic[age]], subjects_dir)
    template = __validate_template__(age, template, subjects_dir)
    if subjects_dir is None:
        subjects_dir = Path(os.environ["SUBJECTS_DIR"])

    montage, trans, bem_model, bem_solution, src = get_bem_artifacts(template, subjects_dir=subjects_dir,
                                                                     include_vol_src=include_vol_src)
    if template != "fsaverage":
        montage.ch_names = ["E" + str(int(ch_name[3:])) for ch_name in montage.ch_names]
        montage.ch_names[128] = "Cz"
    epochs.set_montage(montage)

    process_src_kwargs = dict(loose=loose, fixed=fixed, inv_method=inv_method, pick_ori=pick_ori,
                              lambda2=lambda2)
    if minimal_snr is not None:
        nb_epochs = min(5, len(epochs))
        stcs, fwd = process_sources(epochs[:nb_epochs], trans, src, bem_solution, return_generator=False,
                                    return_fwd=True, **process_src_kwargs)
        snr = get_epochs_sim(epochs[:nb_epochs], stcs, fwd, return_snr=True)[1]
        if verbose:
            print(f"## Average SNR for the EGG reconstruction from estimated sources: {np.mean(snr)} dB")
        if np.mean(snr) < minimal_snr:
            print(snr)
            raise RuntimeError("Insufficient SNR detected. You should normally expect SNR > 15 dB. If the "
                               "values obtained are smaller than that, you should check the coregistration "
                               "of your electrode montage to your head model by "
                               f"calling acareeg.infantmodels.validate_model(age={age}).")

    stcs = process_sources(epochs, trans, src, bem_solution, return_generator=True, **process_src_kwargs)

    if return_labels:
        label_ts, anat_label = sources_to_labels(stcs, age=age, subjects_dir=subjects_dir,
                                                 include_vol_src=include_vol_src)
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
