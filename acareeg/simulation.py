# Authors: Christian O'Reilly <christian.oreilly@gmail.com>
# License: MIT

import mne
import numpy as np
import scipy
import tqdm


def get_epochs_sim(epochs, stcs, fwd, noise_cov=None, return_snr=False):
    raw_sim = mne.simulation.simulate_raw(info=epochs.info, stc=stcs, forward=fwd)
    if noise_cov is not None:
        raw_sim = mne.simulation.add_noise(raw_sim, noise_cov)
    nb_epochs = len(stcs)
    nb_samples = len(epochs.times)
    events = np.array([np.arange(0, nb_epochs * nb_samples, nb_samples), [0] * nb_epochs, [1] * nb_epochs]).T
    epochs_sim = mne.Epochs(raw_sim, events, tmin=epochs.times[0], tmax=epochs.times[-1],
                            baseline=None, preload=True, reject_by_annotation=False)
    if return_snr:
        snr = []
        for epoch, epoch_sim in zip(epochs, epochs_sim):
            snr.append(10*np.log10((np.abs((epoch + epoch_sim)).mean()/2)/np.abs(epoch-epoch_sim).mean()))

        return epochs_sim, snr
    return epochs_sim


def rand_phase(y):
    cf1 = np.fft.rfft(y)
    amp1 = np.abs(cf1)
    theta2 = np.random.rand(*cf1.shape) * 2 * np.pi
    cf2 = amp1 * np.cos(theta2) + 1j * amp1 * np.sin(theta2)
    return np.fft.irfft(cf2)


def randomize_sources(epochs, stcs, fwd, random_type="PRLC", seed=4234, prlc_params=None):

    if prlc_params is None:
        raise ValueError("If using PRLC, a random_type dict must be passed.")
    if "kdtree" not in prlc_params:
        raise ValueError("If using PRLC, a prlc_params dict must be passed containing a kdtree item dict.")
    if seed is not None:
        np.random.seed(seed)

    rand_stcs = stcs.copy()
    for stc in rand_stcs:
        if random_type == "PRLC":  # Phase-randomized locally-correlated surrogate
            stc.data = rand_phase(stc.data)
            get_corrected_sources_one_epoch(stc, seed=None, **prlc_params)

        elif random_type == "phase":
            stc.data = rand_phase(stc.data)

        elif random_type == "spatial":
            # Shuffling the sources
            nb_sources = fwd["src"][0]["nuse"] + fwd["src"][1]["nuse"]
            inds = np.arange(nb_sources)
            np.random.shuffle(inds)
            stc.data = stc.data[inds]
        else:
            raise ValueError("random_type can only be 'PRLC', 'spatial', or 'phase'")

        correct_source_amplitude(epochs, rand_stcs, fwd)


def correct_source_amplitude(epochs, stcs, fwd):
    # Simulate EEG from sources
    epochs_sim = get_epochs_sim(epochs, stcs, fwd)

    # Compute a correction factor so that the average amplitude of the simulated EEG
    # match the average amplitude of the recorded EEG
    amp_factor = np.abs(epochs.get_data()).mean() / np.abs(epochs_sim.get_data()).mean()

    # Apply the correction factor to the sources
    for stc in stcs:
        stc.data = stc.data * amp_factor


def get_corrected_sources_one_epoch(stc, kdtree=None, a=-0.1217, b=0.13275, c=0.35, seed=None):
    if seed is not None:
        np.random.seed(seed)
    old_norms = np.abs(stc.data).sum(1)
    new_sources = []
    for ind, (p, inds_lst, dists_lst) in enumerate(zip(kdtree.data, kdtree.inds, kdtree.dists)):
        selected = np.random.choice(range(len(inds_lst)), int(c * len(inds_lst)))
        weights = np.exp(a * dists_lst[selected]) * b
        new_sources.append((stc.data[np.array(inds_lst)[selected], :].T * weights).sum(1) + stc.data[ind].T)

    stc.data = (np.stack(np.array(new_sources)).T / np.abs(new_sources).sum(1) * old_norms).T


def get_vertice_pos(fwd):
    return np.vstack((fwd["src"][0]['rr'][fwd["src"][0]["vertno"]],
                      fwd["src"][1]['rr'][fwd["src"][1]["vertno"]]))


def get_model_kdtree(fwd, dist_weight_max=20.0, progress="tqdm_notebook"):
    pos = get_vertice_pos(fwd)
    kdtree = scipy.spatial.KDTree(pos)
    kdtree.inds = kdtree.query_ball_point(kdtree.data, dist_weight_max/1000)

    if progress == "tqdm_notebook":
        iter_ = tqdm.notebook.tqdm(list(enumerate(kdtree.inds)))
    elif progress == "tqdm":
        iter_ = tqdm.tqdm(list(enumerate(kdtree.inds)))
    else:
        iter_ = enumerate(kdtree.inds)

    for ind, inds_lst in iter_:
        inds_lst.remove(ind)
    kdtree.dists = np.array([np.sqrt(np.sum((kdtree.data[inds_lst] - p)**2, axis=1))*1000
                             for p, inds_lst in list(zip(kdtree.data, kdtree.inds))])
    return kdtree
