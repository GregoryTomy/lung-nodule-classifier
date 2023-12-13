import copy
import csv
import functools
import glob
import os
import random
from collections import namedtuple
import SimpleITK as sitk
import numpy as np
import torch
import torch.cuda
from torch.utils.data import Dataset
from util.disk import get_cache
from util.util import XyzTuple, xyz2irc
from util.config_log import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

#change the string here to create a new cache directory
raw_cache = get_cache("full")


ANNOTATIONS_CSV = "Data/annotations.csv"
CANDIDATES_CSV = "Data/candidates_V2.csv"



CandidateInfoTuple = namedtuple(
    "CandidateInfoTuple", "is_nodule_bool, diameter_mm, series_uid, center_xyz"
)

def read_csv(filepath):
    """
    Reads a CSV file and returns its content as a list of lists. Each inner list represents a row
    in the CSV, and the items within the inner list represent the fields in that row.
    """
    try:
        with open(filepath, "r") as f:
            return list(csv.reader(f))
    except FileNotFoundError:
        print(f"File {filepath} not found")


def parse_diameter_dict(rows):
    """
    Function takes the rows of a CSV file (presumably read by read_csv) and parses it to create a
    dictionary (diameter_dict). 
    """

    diameter_dict = {}
    for series_uid, *coords, diameter in rows[1:]:
        diameter_dict.setdefault(series_uid, []).append(
            (tuple(map(float, coords)), float(diameter))
        )
    return diameter_dict

def parse_candidate_info(diameter_dict, rows):
    """
    This function takes a dictionary of diameter information and a list of rows to generate a list 
    of candidates, each represented as a CandidateInfoTuple. It matches candidates to nodules based 
    on their center coordinates and sets the candidate diameter to the nodule diameter if they are 
    considered a match.
    """
    candidate_info_list = []
    for row in rows[1:]:
        series_uid = row[0]
        is_nodule_bool = bool(int(row[4]))
        cand_center_xyz = tuple(float(x) for x in row[1:4])

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

@functools.lru_cache(1)
def get_candidate_info_list():
    annotations_rows = read_csv(ANNOTATIONS_CSV)
    candidate_rows = read_csv(CANDIDATES_CSV)

    diameter_dict = parse_diameter_dict(annotations_rows)
    candidate_info_list =parse_candidate_info(diameter_dict, candidate_rows)
   
    return candidate_info_list


class Ct:
    """
    The Ct class is designed for handling and preprocessing CT data.
    """
    def __init__(self, series_uid):
        mhd_path = glob.glob("Data/LUNA16/{}.mhd".format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        # load mdh to numpy array
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # clip values to be between range. Shape ct_a is (135, 512, 512)
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vx_size_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def get_raw_candidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vx_size_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_IDX = int(round(center_val - width_irc[axis] / 2))
            end_IDX = int(start_IDX + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr(
                [
                    self.series_uid,
                    center_xyz,
                    self.origin_xyz,
                    self.vx_size_xyz,
                    center_irc,
                    axis,
                ]
            )

            if start_IDX < 0:
                start_IDX = 0
                end_IDX = int(width_irc[axis])

            if end_IDX > self.hu_a.shape[axis]:
                end_IDX = self.hu_a.shape[axis]
                start_IDX = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_IDX, end_IDX))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def get_ct_data(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def get_ct_candidate(series_uid, center_xyz, width_irc):
    ct = get_ct_data(series_uid)
    ct_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    return ct_chunk, center_irc


class LunaDataset(Dataset):
    def __init__(
        self,
        val_stride=0,
        is_val_set_bool=None,
        series_uid=None,
        sortby_str="random",
        ratio_int=0,
        candidate_info_list=None,
    ):
        self.ratio_int = ratio_int

        if candidate_info_list:
            self.candidate_info_list = copy.copy(candidate_info_list)
            self.use_cache = False
        else:
            self.candidate_info_list = copy.copy(get_candidate_info_list())
            self.use_cache = True

        if series_uid:
            self.candidate_info_list = [
                x for x in self.candidate_info_list if x.series_uid == series_uid
            ]

        if is_val_set_bool:
            assert val_stride > 0, val_stride
            self.candidate_info_list = self.candidate_info_list[::val_stride]
            assert self.candidate_info_list
        elif val_stride > 0:
            del self.candidate_info_list[::val_stride]
            assert self.candidate_info_list

        if sortby_str == "random":
            random.shuffle(self.candidate_info_list)
        elif sortby_str == "series_uid":
            self.candidate_info_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == "label_and_size":
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.negative_list = [
            nt for nt in self.candidate_info_list if not nt.is_nodule_bool
        ]
        self.pos_list = [nt for nt in self.candidate_info_list if nt.is_nodule_bool]

        log.info(
            f"{self!r}: {len(self.candidate_info_list)} {'validation' if is_val_set_bool else 'training'} samples, "
            f"{len(self.negative_list)} neg, {len(self.pos_list)} pos, "
            f"{'{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'} ratio"
        )
         
    def shuffle_samples(self):
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.pos_list)

    def __len__(self):
        if self.ratio_int:
            return 20000
        return len(self.candidate_info_list)

    def __getitem__(self, IDX):
        # is self.ratio is true then (self.ratio + 1) = 2 and we balance the data
        # half positive and half negative samples
        if self.ratio_int:
            pos_IDX = IDX // (self.ratio_int + 1)

            # nonzero remainder means index should be negative sample
            if IDX % (self.ratio_int + 1):
                # overflow wraparound
                # `%=` operator ensures the index stays within the list bounds by
                # wrapping around if it exceeds the list size.
                neg_IDX = IDX - 1 - pos_IDX
                neg_IDX %= len(self.negative_list)
                candidate_info_tuple = self.negative_list[neg_IDX]
            else:
                pos_IDX %= len(self.pos_list)
                candidate_info_tuple = self.pos_list[pos_IDX]
        else:
            # get the information of the candidate at index IDX
            candidate_info_tuple = self.candidate_info_list[IDX]

        # define the dimensions of the CT chunk to be extracted
        width_irc = (32, 48, 48)

        # if caching is enabled (use_cache is True), retrieve the candidate data from the cache.
        if self.use_cache:
            candidate_a, center_irc = get_ct_candidate(
                candidate_info_tuple.series_uid, candidate_info_tuple.center_xyz, width_irc,
            )

            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            # add the channel dimension using unsqueeze
            candidate_t = candidate_t.unsqueeze(0)
        # If caching is not used, compute the candidate data directly without using the cache.
        else:
            ct = get_ct_data(candidate_info_tuple.series_uid)
            candidate_a, center_irc = ct.get_raw_candidate(candidate_info_tuple.center_xyz, width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        # create a one-hot encoded tensor to represent whether the candidate is a nodule or not.
        # one-hot encoding is used here because nn.CrossEntropyLoss expects one output value per class.
        # this tensor will be used as the target label for training.
        pos_t = torch.tensor(
            [not candidate_info_tuple.is_nodule_bool, candidate_info_tuple.is_nodule_bool],
            dtype=torch.long,
        )

        return (
            candidate_t,
            pos_t,
            candidate_info_tuple.series_uid,
            torch.tensor(center_irc),
        )
