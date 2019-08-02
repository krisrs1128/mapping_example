#!/usr/bin/env python
import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import os.path
import pandas as pd


class LCMPatches(Dataset):
    """
    Landcover Mapping Data
    Returns the aerial image, the NAIP low-res map, and the ground truth. The
    n_in_mem parameter controls the increments over which the next set of
    patches are loaded into memory.
    Example
    -------
    >>> path = "/data/" # suppose bound lcm/ here in singularity instance
    >>> lcm = LCMPatches(path)
    >>> x, _, y = lcm[0]
    """
    def __init__(self, path, mode="train", n_in_mem=200):
        super(LCMPatches).__init__()
        self.cur_ids = []
        self.mode = mode
        self.n_in_mem = n_in_mem
        self.patches = {}
        self.path = path

        # read in the metadata
        metadata_paths = glob.glob("{}*patches.csv".format(path))
        metadata = [pd.read_csv(p) for p in metadata_paths]
        metadata = pd.concat(metadata, sort=False)
        metadata["train"] = metadata["patch_fn"].str.contains("train")
        metadata["region"] = metadata["patch_fn"].str[:2]

        # subset to train / validation sets
        train_ix = metadata["train"].values
        if self.mode == "train":
            metadata = metadata.loc[train_ix, :]
        else:
            metadata = metadata.loc[np.bitwise_not(train_ix), :]

        # order in which to return patches
        metadata.insert(0, "sampler_id", range(len(metadata)))
        self.metadata = metadata


    def __len__(self):
        return len(self.metadata)


    def update_queue(self, ix):
        start = ix - ix % self.n_in_mem
        self.cur_ids = list(range(start, start + self.n_in_mem))
        cur_meta = self.metadata.loc[self.metadata["sampler_id"].isin(self.cur_ids), :]
        self.patches = {
            row["sampler_id"]:
            np.load(os.path.join(self.path, row["patch_fn"]))
            for _, row in cur_meta.iterrows()
        }


    def __getitem__(self, ix):
        """
        For each patch, the returned elements are,
          - image: The 3-channel aerial image.
          - low_res: The low-res NAIP landcover labeling
          - high_res: A ground truth landcover labeling
        """
        # update currently loaded dictionary of patches
        if ix not in self.cur_ids:
            self.update_queue(ix)

        # split patch into x, y_naip, y_high_res
        patch = torch.Tensor(self.patches[ix].squeeze())
        return patch[:3], patch[4], (patch[5] == self.k).int()


class LCMBinary(LCMPatches):
    """
    Version of LCMPatches with binary response: is label == k?
    """
    def __init__(self, path, mode="train", n_in_mem=25, k=1):
        self.k = k
        super(LCMBinary, self).__init__(path, mode, n_in_mem)


    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, ix):
        # update currently loaded dictionary of patches
        if ix not in self.cur_ids:
            self.update_queue(ix)

        # split patch into x, y_naip, y_high_res
        patch = torch.Tensor(self.patches[ix].squeeze())
        return patch[:3], patch[4], (patch[5] == self.k).float()
