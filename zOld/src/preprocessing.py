"""
Preprocessing Script for Lung Nodule Detection
-------------------------------------------------

Description:
This script is part of a pipeline for lung nodule detection using CT scans.
It performs a series of operations to prepare the raw CT scan data for further
analysis and model training. Specifically, the script handles:
- Reading and parsing raw metadata and annotations
- Extracting cubic chunks (candidates) from CT scans based on annotations
- Converting patient coordinates to voxel coordinates and vice versa
- Caching results for performance optimization
- Initializing a dataset suitable for PyTorch's DataLoader

Modules Used:
- SimpleITK for reading medical images (.mhd files)
- numpy for numerical operations
- csv for reading comma-separated values files
- functools for function-level optimizations
- torch for PyTorch specific tasks

Functions:
- read_csv: Reads a CSV file and returns its content
- parse_diameter_dict: Parses nodule information to create a diameter dictionary
- parse_candidate_info: Parses candidate information and matches to nodules
- read_mhd: Reads .mhd files and returns a CtData namedtuple
- get_raw_candidate: Extracts cubic chunk from CT scan
- get_ct_data: Retrieves CtData namedtuple for a specific series_uid (LRU cached)
- get_candidate_info: Retrieves a list of CandidateInfoTuple (LRU cached)
- get_ct_candidate: Retrieves a specific CT scan candidate (Memoized)

Classes:
- LunaDataset: A PyTorch Dataset class for the Luna16 dataset

Logging:
- All major steps are logged, and a log file is generated as 'preprocessing.log'
"""


import functools
import csv
from collections import namedtuple
from dataclasses import dataclass
from typing import List
import glob
import SimpleITK as sitk
import numpy as np
import logging
import math
import pickle
import copy
import torch
import torch.nn.functional as F
import random
from torch.utils.data import Dataset

from utils.util import XyzTuple, xyz_to_irc, setup_logger
from utils.disk import getCache

logger = setup_logger(__name__, "logs/preprocessing.log", level=logging.ERROR)

raw_cache = getCache("ct_raw")

ANNOTATIONS_CSV = "Data/annotations.csv"
CANDIDATES_CSV = "Data/candidates.csv"

CandidateInfoTuple = namedtuple(
    "CandidateInfoTuple", "is_nodule_bool, diameter_mm, series_uid, center_xyz"
)

# CT Data structure
CtData = namedtuple("CtData", ["hu_a", "origin_xyz", "vxSize_xyz", "direction_a"])
# Note: _a indicates arrays, _xyz indicates patient corrdinates, _irc indicates voxel coordinates
# _t indicates torch tensors


def read_csv(filepath: str):
    """
    Reads a CSV file and returns its content as a list of lists. Each inner list represents a row
    in the CSV, and the items within the inner list represent the fields in that row.

    Parameters:
        filepath: A string representing the path to the CSV file.

    Returns:
        A list of lists containing the rows and fields of the CSV file.
    """
    try:
        with open(filepath, "r") as f:
            return list(csv.reader(f))
    except FileNotFoundError:
        print(f"File {filepath} not found")


def parse_diameter_dict(rows: List[List[str]]):
    """
        Function takes the rows of a CSV file (presumably read by read_csv) and parses it to create a
        dictionary (diameter_dict). Each key in this dictionary is a series_uid and its corresponding
    value is a list of tuples, where each tuple contains the center coordinates and diameter of a
    nodule.

    Parameters:
        rows: A list of lists where each inner list represents a row in the CSV file. It assumes that the first row is a header and starts processing from the second row.

    Returns:
        A dictionary containing parsed nodule information, indexed by series_uid."""

    diameter_dict = {}
    for series_uid, *coords, diameter in rows[1:]:
        diameter_dict.setdefault(series_uid, []).append(
            (tuple(map(float, coords)), float(diameter))
        )
    return diameter_dict


def parse_candidate_info(diameter_dict, rows):
    """
    This function takes a dictionary of diameter information and a list of rows to generate a list of candidates, each represented as a CandidateInfoTuple. It matches candidates to nodules based on their center coordinates and sets the candidate diameter to the nodule diameter if they are considered a match.

    Parameters:
        diameter_dict: A dictionary where each key is a series_uid and the corresponding value is a list of tuples containing nodule center coordinates and diameter.
        rows: A list of lists, where each inner list represents a row from a CSV file containing candidate information.
    Returns:
        A sorted list of CandidateInfoTuple instances. Each tuple contains:
        is_nodule_bool: A boolean indicating whether the candidate is a nodule.
        cand_diameter: The diameter of the candidate. Set to nodule diameter if it matches any nodule, otherwise 0.0.
        series_uid: The unique identifier for the series to which this candidate belongs.
        center_xyz: The center coordinates of the candidate.
    """
    candidate_info_list = []
    for row in rows[1:]:
        series_uid = row[0]
        is_nodule_bool = bool(int(row[4]))
        cand_center_xyz = tuple([float(x) for x in row[1:4]])

        cand_diameter = 0.0
        # iterate over all annotations that have the same `series_id` as
        # the current candidate
        for anno_tuple in diameter_dict.get(series_uid, []):
            anno_center_xyz, anno_diameter = anno_tuple
            # if the absolute difference in any dimension is greater
            # than a quarter of the annotation diameter then the current
            # candidate is not considered to be a match for this annotation.
            if all(
                abs(cand_center_xyz[i] - anno_center_xyz[i]) <= anno_diameter / 4
                for i in range(3)
            ):
                cand_diameter = anno_diameter
                break

        candidate_info_list.append(
            CandidateInfoTuple(
                is_nodule_bool, cand_diameter, series_uid, cand_center_xyz
            )
        )
    return sorted(candidate_info_list, reverse=True)


def read_mhd(mhd_path):
    # sitk implicitly consumes the .raw file in addition to the passed-in .mhd file
    logger.info("Reading .mhd file.")
    ct_mhd = sitk.ReadImage(mhd_path)
    logger.debug(f"ct_mhd details: {ct_mhd}")

    logger.info("Converting to numpy array and clipping values")
    ## load mdh to numpy array
    ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

    ## clip values to be between range. Shape ct_a is (135, 512, 512)
    ct_a.clip(-1000, 1000, ct_a)
    logger.debug(f"ct_a shape: {ct_a.shape}")

    logger.info("Extracting metadata.")
    origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
    vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())

    # transformation matrix is originally a 9 element array. Here we transform it to 3x3 matrix
    direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    return CtData(
        hu_a=ct_a,
        origin_xyz=origin_xyz,
        vxSize_xyz=vxSize_xyz,
        direction_a=direction_a,
    )


def get_raw_candidate(ct_data, center_xyz, width_irc):
    """
    Extracts a cubic chunk of CT scan data centered around a specified point.

    Parameters:
        ct_data (CtData): Namedtuple containing the CT scan data and metadata.
        center_xyz (tuple): Coordinates (x, y, z) representing the center in the patient coordinate system.
        width_irc (tuple): Width in voxels along each axis (i, r, c).

    Returns:
        ct_chunk (ndarray): The cubic chunk of CT scan data.
        center_irc (tuple): The center of the candidate in voxel coordinates.
    """
    logger.info("Converting XYZ to IRC coordinates")
    center_irc = xyz_to_irc(
        center_xyz, ct_data.origin_xyz, ct_data.vxSize_xyz, ct_data.direction_a
    )

    slice_list = []
    for axis, center_val in enumerate(center_irc):
        start_ndx = int(round(center_val - width_irc[axis] / 2))
        end_ndx = int(start_ndx + width_irc[axis])

        assert center_val >= 0 and center_val < ct_data.hu_a.shape[axis], repr(
            [
                center_xyz,
                ct_data.origin_xyz,
                ct_data.vxSize_xyz,
                center_irc,
                axis,
            ]
        )

        logger.debug(f"Calculating slice indices for axis {axis}")
        if start_ndx < 0:
            logger.warning("Start index is negative. Clipping to zero.")
            start_ndx = 0
            end_ndx = int(width_irc[axis])

        if end_ndx > ct_data.hu_a.shape[axis]:
            logger.warning("End index exceeds CT shape. Adjusting.")
            end_ndx = ct_data.hu_a.shape[axis]
            start_ndx = int(ct_data.hu_a.shape[axis] - width_irc[axis])

        slice_list.append(slice(start_ndx, end_ndx))

    logger.info("Extracting CT chunk.")
    ct_chunk = ct_data.hu_a[tuple(slice_list)]

    return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def get_ct_data(series_uid):
    mhd_path = glob.glob(f"Data/LUNA16/{series_uid}.mhd")[0]
    return read_mhd(mhd_path)


@functools.lru_cache(1)
def get_candidate_info():
    annotations_rows = read_csv(ANNOTATIONS_CSV)
    candidate_rows = read_csv(CANDIDATES_CSV)
    diameter_dict = parse_diameter_dict(annotations_rows)

    return parse_candidate_info(diameter_dict, candidate_rows)


@raw_cache.memoize(typed=True)
def get_ct_candidate(series_uid, center_xyz, width_irc):
    """
    Retrieves a CT scan candidate based on the given series UID and other parameters.

    Parameters:
    series_uid (str): The unique identifier for the CT scan series.
    center_xyz (tuple): The x, y, z coordinates of the center in patient coordinates.
    width_irc (tuple): The width, height, and depth of the region of interest.

    Returns:
    tuple: ct_chunk (array-like), the extracted region of interest; center_irc (tuple), coordinates voxel coordinates.

    Memoization:
    The function uses @raw_cache.memoize to cache the results based on the input parameters.
    This avoids recomputing the same data, leading to performance gains especially for
    computationally expensive operations.
    """
    ct_data = get_ct_data(series_uid)
    ct_chunk, center_irc = get_raw_candidate(ct_data, center_xyz, width_irc)
    return ct_chunk, center_irc


def augment_ct_candidate(
    augmentation_dict, series_uid, center_xyz, width_irc, use_cache=True
):
    # decide to fetch cached CT or load anew
    if use_cache:
        ct_chunk, center_irc = get_ct_candidate(series_uid, center_xyz, width)
    else:
        ct = get_ct_data(seried_uid)
        ct_chunk, center_irc = get_raw_candidate(ct, center_xyz, width)

    # convert the CT to a tensor and add two new axes to fir the model input requirements.
    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    transform_t = torch.eye(4)  # create identity matrix

    for i in range(3):
        # The following code block applies random transformations based on the specified augmentations.
        # Each augmentation (flip, offset, scale, rotate) is applied with a certain probability or range.
        if "flip" in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i, i] *= -1
        if "offset" in augmentation_dict:
            offset = augmentation_dict["offset"]
            random = random.random() * 2 - 1
            transform_t[i, 3] += random * offset
        if "scale" in augmentation_dict:
            scale = augmentation_dict["scale"]
            random = random.random() * 2 - 1
            transform_t[i, i] *= 1.0 + random * scale

    if "rotate" in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation = torch.tensor(
            [
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        transform_t @= rotation

    # loop over spatial dimensions (ignoring batch and channel) and apply and apply augmentations
    affine_t = F.affine_grid(
        transform_t[:3].unsqueeze(0).to(torch.float32),
        ct_t.size(),
        align_corners=False,
    )

    augmented_chunk = F.grid_sample(
        ct_t,
        affine_t,
        padding_mode="border",
        align_corners=False,
    ).to("cpu")

    if "noise" in augmentation_dict:
        noise = torch.randn_like(augmented_chunk)
        noise *= augmentation_dict["noise"]
        augmented_chunk += noise

    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    def __init__(self, val_stride=0, is_val_set_bool=None, series_uid=None, ratio=0):
        self.ratio = ratio
        logger.info("Initializing LunaDataset")

        # copy return value so that cached copy won't be impacted by altering self.candidate_info_list
        self.candidate_info_list = copy.copy(get_candidate_info())
        logger.info(
            f"Initial Candidate Info List length: {len(self.candidate_info_list)}"
        )

        # if provided with a single series_uid, return nodules only from that id.
        if series_uid:
            self.candidate_info_list = [
                x for x in self.candidate_info_list if x.series_uid == series_uid
            ]
        logger.info(
            f"Filtered by seried_uid: {series_uid}. \
            New candidate info list length: {len(self.candidate_info_list)}"
        )

        # check if this instance of LunaDataset is intended to be a validation set.
        if is_val_set_bool:
            assert val_stride > 0, val_stride
            # filter the candidate_info_list to include only every val_stride-th element.
            # this effectively creates a validation set from the original dataset.
            self.candidate_info_list = self.candidate_info_list[::val_stride]
            assert self.candidate_info_list
            logger.info(
                f"Created validation set with stride {val_stride}.\
                New candidate info list length: {len(self.candidate_info_list)}"
            )
        # ff it's not a validation set but val_stride is still greater than zero
        elif val_stride > 0:
            # delete every val_stride-th element from the candidate_info_list.
            # this creates a training set that doesn't overlap with the validation set.
            del self.candidate_info_list[::val_stride]
            assert self.candidate_info_list
            logger.info(
                f"Created testset by removing every {val_stride}-th.\
                New candidate info list length: {len(self.candidate_info_list)}"
            )

        # create dedicated lists of positive and negative training samples
        self.pos_list = [x for x in self.candidate_info_list if x.is_nodule_bool]

        self.neg_list = [x for x in self.candidate_info_list if not x.is_nodule_bool]

    def shuffle_samples(self):
        # randomize the order of the samples presented
        if self.ratio:
            random.shuffle(self.neg_list)
            random.shuffle(self.pos_list)

    def __len__(self):
        # hardcoding the dataset length to 200,000 to streamline the training process:
        # this ensures consistent epoch durations for better comparison across runs,
        # increases the frequency of positive samples to address class imbalance,
        # and optimizes computational resource usage for more efficient training iterations.
        if self.ratio:
            return 200000
            # return 2000 # this is for the small sample for testing
        else:
            return len(self.candidate_info_list)

    def __getitem__(self, ndx):
        # is self.ratio is true then (self.ratio + 1) = 2 and we balance the data
        # half positive and half negative samples
        if self.ratio:
            pos_id = ndx // (self.ratio + 1)

            # nonzero remainder means index should be negative sample
            if ndx % (self.ratio + 1):
                neg_id = ndx - 1 - pos_id
                # overflow wraparound
                # `%=` operator ensures the index stays within the list bounds by
                # wrapping around if it exceeds the list size.
                neg_id %= len(self.neg_list)
                candidate_info_tuple = self.neg_list[neg_id]
            else:
                pos_id %= len(self.pos_list)
                candidate_info_tuple = self.pos_list[pos_id]
        else:
            # get the information of the candidate at index ndx
            candidate_info_tuple = self.candidate_info_list[ndx]

        # define the dimensions of the CT chunk to be extracted
        width_irc = (32, 48, 48)

        candidate_a, center_irc = get_ct_candidate(
            candidate_info_tuple.series_uid, candidate_info_tuple.center_xyz, width_irc
        )

        # convert PyTorch tensor
        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        # add the channel dimension using unsqueeze
        candidate_t = candidate_t.unsqueeze(0)

        # create a one-hot encoded tensor to represent whether the candidate is a nodule or not.
        # one-hot encoding is used here because nn.CrossEntropyLoss expects one output value per class.
        # this tensor will be used as the target label for training.
        positive_t = torch.tensor(
            [
                not candidate_info_tuple.is_nodule_bool,
                candidate_info_tuple.is_nodule_bool,
            ],
            dtype=torch.long,
        )

        return (
            candidate_t,
            positive_t,
            candidate_info_tuple.series_uid,
            torch.tensor(center_irc),
        )


# if __name__ == "__main__":
#    # Save the datasets
#     def save_dataset_with_pickle(dataset, save_path):
#         with open(save_path, 'wb') as f:
#             pickle.dump(dataset, f)
#         print(f"Dataset saved to {save_path}")

#     train_dataset = LunaDataset(val_stride=10, is_val_set_bool=False)
#     val_dataset = LunaDataset(val_stride=10, is_val_set_bool=True)

#     save_dataset_with_pickle(train_dataset, "Data/saved_data/train_dataset.pkl")
#     save_dataset_with_pickle(val_dataset, "Data/saved_data/val_dataset.pkl")
