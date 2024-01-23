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


ANNOTATIONS_CSV = "Data/annotations_with_malignancy.csv"
CANDIDATES_CSV = "Data/candidates.csv"



CandidateInfoTuple = namedtuple(
    "CandidateInfoTuple", "is_nodule_bool, has_annotation_bool, is_mal_bool , diameter_mm, series_uid, center_xyz"
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


@functools.lru_cache(1)
def get_candidate_info_list():
    annotation_rows = read_csv(ANNOTATIONS_CSV)
    candidate_rows = read_csv(CANDIDATES_CSV)

    candidate_info_list = []
    # loop over the annotations to get the actual nodules
    for row in annotation_rows[1:]:
        series_uid = row[0]
        annotations_center_xyz = tuple([float(x) for x in row[1:4]])
        annotations_diameter_mm = float(row[4])
        is_malignant_bool = {"False": False, "True": True}[row[5]]

        candidate_info_list.append(CandidateInfoTuple(
            True, True, is_malignant_bool, annotations_diameter_mm, series_uid, annotations_center_xyz
        ))

    # loop over the candidates to get but only for non-nodules since we have nodules from the annotations
    for row in candidate_rows[1:]:
        series_uid = row[0]
        is_nodule_bool = bool(int(row[4]))
        candidate_center_xyz = tuple([float(x) for x in row[1:4]])
        
        # as these are not nodules, the nodules specific info is fulled as False and 0
        if not is_nodule_bool:
            candidate_info_list.append(CandidateInfoTuple(
                False, False, False, 0.0, series_uid, candidate_center_xyz
            )) 

    candidate_info_list.sort(reverse=True)

    return candidate_info_list
        # cand_diameter = 0.0
        # # iterate over all annotations that have the same `series_uid` as
        # # the current candidate
        # for anno_tuple in diameter_dict.get(series_uid, []):
        #     anno_center_xyz, anno_diameter = anno_tuple
        #     # if the absolute difference in any dimension is greater
        #     # than a quarter of the annotation diameter then the current
        #     # candidate is not considered to be a match for this annotation.
        #     if all(
        #         abs(cand_center_xyz[i] - anno_center_xyz[i]) <= anno_diameter / 4
        #         for i in range(3)
        #     ):
        #         cand_diameter = anno_diameter
        #         break

        # candidate_info_list.append(
        #     CandidateInfoTuple(
        #         is_nodule_bool, cand_diameter, series_uid, cand_center_xyz
        #     )
        # )
    # return sorted(candidate_info_list, reverse=True)


@functools.lru_cache(1)
def get_candidate_info_dict():
    """
    Same information as get_candidate_info_list but grouped by series uid.
    """
    candidate_info_list = get_candidate_info_list()
    candidate_info_dict = {}

    for candidate_info_tuple in candidate_info_list:
        candidate_info_dict.setdefault(candidate_info_tuple.series_uid, []).append(candidate_info_tuple)

    return candidate_info_dict


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
        # ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vx_size_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        candidate_info_list = get_candidate_info_dict()[self.series_uid]
        # print(len(candidate_info_list))
        # print("\n")
        # print(candidate_info_list)

        if not candidate_info_list:
            raise ValueError("No candidates values found")

        # get the nodules for the particular series uid
        self.positive_info_list = [
            candidate_tuple for candidate_tuple in candidate_info_list if candidate_tuple.is_nodule_bool
        ]

        # print("\n ##################")
        # print(self.positive_info_list)
        # print("\n ##################")
        # # if not self.positive_info_list:
        # #     raise ValueError("No positive values found")
            
        self.positive_mask = self.build_annotation_mask(self.positive_info_list) 
        # print(self.positive_mask)
        # print("\n ##################")
        self.positive_indexes = (self.positive_mask.sum(axis=(1,2)).nonzero()[0].tolist()) # take indices of mask slices with nonzero count and make into list

    def build_annotation_mask(self, positive_info_list, threshold_hu = -700):
        bounding_box_a = np.zeros_like(self.hu_a, dtype=bool) 

        for candidate_info_tuple in positive_info_list:
            center_irc = xyz2irc(
                candidate_info_tuple.center_xyz,
                self.origin_xyz,
                self.vx_size_xyz,
                self.direction_a
            )
            center_index = int(center_irc.index)
            center_row = int(center_irc.row)
            center_col = int(center_irc.col)

            # perform the search
            index_radius = 2
            try:
                while self.hu_a[center_index + index_radius, center_row, center_col] > threshold_hu \
                    and self.hu_a[center_index - index_radius, center_row, center_col] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2 
            try:
                while self.hu_a[center_index , center_row + row_radius , center_col] > threshold_hu \
                    and self.hu_a[center_index , center_row - row_radius, center_col] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2 
            try:
                while self.hu_a[center_index , center_row , center_col + col_radius] > threshold_hu \
                    and self.hu_a[center_index , center_row , center_col - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            bounding_box_a[
                center_index - index_radius: center_index + index_radius + 1,
                center_row - row_radius: center_row + row_radius + 1,
                center_col - col_radius: center_col + col_radius + 1
                ] = True

        mask_a = bounding_box_a & (self.hu_a > threshold_hu)    # cleanup by taking the intersection of the bounding box and tissue denser than the threshold

        return mask_a

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
        positive_chunk = self.positive_mask[tuple(slice_list)]   # added here to cache the postive mask chunks

        return ct_chunk, positive_chunk, center_irc


@functools.lru_cache(1, typed=True)
def get_ct_data(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def get_ct_candidate(series_uid, center_xyz, width_irc):
    ct = get_ct_data(series_uid)
    ct_chunk, positive_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    return ct_chunk, positive_chunk, center_irc

@raw_cache.memoize(typed=True)
def get_ct_sample_size(series_uid):
    ct = Ct(series_uid)

    return int(ct.hu_a.shape[0]), ct.positive_indexes

class Luna2dSegmentationDataset(Dataset):
    def __init__(
        self,
        val_stride=0,
        is_val_set_bool=None,
        series_uid=None,
        context_slices_count = 3,
        full_CT_bool = False,
    ):

        self.context_slices_count = context_slices_count
        self.full_CT_bool = full_CT_bool

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(get_candidate_info_dict().keys())

        if is_val_set_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]   # starting with a series list with all series, we keep only the val_strideth element
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]  # if training, we delete every val_strideth element.
            assert self.series_list
        
        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indexes = get_ct_sample_size(series_uid)

            if self.full_CT_bool:   # if true, use every slice in the CT for the dataset (for end-to-end performance evaluation)
                self.sample_list += [(series_uid, slice_ndx) for slice_ndx in range(index_count)]
            else:   # used for validation during training, limiting to CT slices with a positive mask present.j
                self.sample_list += [(series_uid, slice_ndx) for slice_ndx in positive_indexes]

        self.candidate_info_list = get_candidate_info_list()    # cached

        series_set = set(self.series_list)  # made a set for faster lookup
        self.candidate_info_list = [candidate_info_tuple for candidate_info_tuple in self.candidate_info_list \
            if candidate_info_tuple.series_uid in series_set]   # get nodule candidates with series_uid included in the series list
        
        self.positive_list = [candidate_info_tuple for candidate_info_tuple in self.candidate_info_list \
            if candidate_info_tuple.is_nodule_bool] # get only actual nodules. used in balancing the data.
        

        log.info(
            f"{self!r}: {len(self.series_list)}"
            f"{{None: 'general', True: 'validation', False: 'training}}[is_val_set_bool]"
            f"series, {len(self.sample_list)} slices, {len(self.positive_list)} nodules"
        )
         
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, NDX):
        # NDX % len(sample_list) used to ensure index stays within the bounds of a list
        # if index is equal or greater than the length, it wraps around to the beginnign of the list
        series_uid, slice_ndx = self.sample_list[NDX % len(self.sample_list)]
        
        return self.get_item_full_slice(series_uid, slice_ndx)

    def get_item_full_slice(self, series_uid, slice_ndx):
        ct = get_ct_data(series_uid)
        ct_t = torch.zeros((self.context_slices_count * 2 + 1, 512, 512)) # preallocate the output

        start_ndx = slice_ndx - self.context_slices_count
        end_ndx = slice_ndx + self.context_slices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            # when we reach beyond the boudns of the ct_a, we duplicate the first or last slice.
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))
            
        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

        return ct_t, pos_t, ct.series_uid, slice_ndx    # ct.series_uid and slice_ndx are for debugging and display


class TrainingLunda2dSegmentationDataset(Luna2dSegmentationDataset):
    """
    The trianing dataset will be 64x64 crops around the positive candidates. 
    These 64x64 pathches will be taken randomly from a 96x96 crop centered around the nodule.
    Three slices of context in both direction (Z) will be used as additional channels.
    """

    def _init__(self, *args, **kwargs):
        super()._init__(**args, **kwargs)

        self.ratio_int = 2
    
    def __len__(self):
        return 300000

    def shuffle_samples(self):
        random.shuffle(self.candidate_info_list)
        random.shuffle(self.positive_list)
    
    def get_training_crop(self, candidate_info_tuple):
        ct_a, postive_a, center_irc = get_ct_candidate(
            candidate_info_tuple.series_uid, candidate_info_tuple.center_xyz, (7, 96, 96)
        )

        positive_a = postive_a[3:4]

        row_offset = random.randrange(0,32)
        column_offset = random.randrange(0, 32)

        ct_t = torch.from_numpy(ct_a[:, row_offset:row_offset+64, column_offset:column_offset+64]).to(torch.float32)
        positive_t = torch.from_numpy(positive_a[:, row_offset:row_offset+64, column_offset:column_offset+64]).to(torch.long)

        slice_ndx = center_irc.index
        
        return ct_t, positive_t, candidate_info_tuple.series_uid, slice_ndx

    def __getitem__(self, NDX):
        candidate_info_tuple = self.positive_list[NDX % len(self.positive_list)]
        return self.get_training_crop(candidate_info_tuple)