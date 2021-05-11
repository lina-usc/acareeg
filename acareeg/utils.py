# Authors: Christian O'Reilly <christian.oreilly@gmail.com>
# License: MIT

import mne
import numpy as np
import xarray as xr


def get_epochs_to_trials(epochs_lst, subject, use_csd=False):
    data = []
    offset = 0
    if isinstance(epochs_lst, mne.epochs.Epochs):
        epochs_lst = [epochs_lst]

    for epochs in epochs_lst:
        if use_csd:
            ch_pos = np.array(list(epochs.get_montage().get_positions()["ch_pos"].values()))
            channels_to_drop = np.array(epochs.ch_names)[np.any(np.isnan(ch_pos), axis=1)]
            epochs.drop_channels(channels_to_drop)
            epochs = mne.preprocessing.compute_current_source_density(epochs)

        data.append(xr.DataArray(epochs.get_data()[:, :, :, None],
                                 dims=("epoch", "channel", "time", "subject"),
                                 coords={"epoch": np.arange(len(epochs)) + offset,
                                         "channel": epochs.ch_names,
                                         "time": np.arange(0, 2, 1/250),
                                         "subject": [subject]}))
        offset += len(epochs)
    return xr.concat(data, dim="epoch")
