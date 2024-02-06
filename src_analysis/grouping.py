import sys
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import scipy.ndimage.morphology as morphology
import scipy.ndimage.measurements as measurements

from src_segmentation.model_seg import UNetWrapper
from src_segmentation.preprocess_seg import Luna2dSegmentationDataset
from src_classification.model_cls import LunaModel
from src_classification.preprocess_cls import CandidateInfoTuple, LunaDataset, get_candidate_info_dict, get_candidate_info_list, get_ct_data
from util.util import xyz2irc, irc2xyz, enum_estimate
from util.config_log import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def match_and_score(detections, truth, threshold=0.5, threshold_malignant=0.5):
    """
    Returns a 3x4 confusion matrix. If one true nodule matches multiple detections, the highest
    detection is considered. If one detection matches several true nodule annotations, it counts
    for all of them.
    """
    true_nodules = [c for c in truth if c.is_nodule_bool]
    truth_diameters = np.array([c.diameter_mm for c in true_nodules])
    truth_xyz = np.array([c.center_xyz for c in true_nodules])

    detected_xyz = np.array([n[2] for n in detections])
    # detection classes will contain
    # 1) detected by seg but filtered by cls
    # 2) detected as benign nodule (or nodule if no malignancy model is used)
    # 3) detected as malignant nodule (if applicable)
    detected_classes = np.array([
        1 if d[0] < threshold else (2 if d[1] < threshold else 3) for d in detections
    ])

    confusion_matrix = np.zeros((3,4), dtype=np.int_)
    if len(detected_xyz) == 0:
        for tn in true_nodules:
            confusion_matrix[2 if tn.is_mal_bool else 1, 0] += 1
    elif len(truth_xyz) == 0:
        for dc in detected_classes:
            confusion_matrix[0, dc] += 1
    else:
        normalized_distances = np.linalg.norm(truth_xyz[:, None] - detected_xyz[None], ord=2, axis=-1) / truth_diameters[:, None]
        matches = (normalized_distances < 0.7)
        unmatched_detections = np.ones(len(detections), dtype=np.bool_)
        matched_true_nodules = np.zeros(len(true_nodules), dtype=np.int_)
        for i_tn, i_detection in zip(*matches.nonzero()):
            matched_true_nodules[i_tn] = max(matched_true_nodules[i_tn], detected_classes[i_detection])
            unmatched_detections[i_detection] = False
        
        for ud, dc in zip(unmatched_detections, detected_classes):
            if ud:
                confusion_matrix[0, dc] += 1
        for tn, dc in zip(true_nodules, matched_true_nodules):
            confusion_matrix[2 if tn.is_mal_bool else 1, dc] += 1
    
    return confusion_matrix


def print_confusion(label, confusions, do_mal):
    row_labels = ['Non-Nodules', 'Benign', 'Malignant']

    if do_mal:
        col_labels = ['', 'Complete Miss', 'Filtered Out', 'Pred. Benign', 'Pred. Malignant']
    else:
        col_labels = ['', 'Complete Miss', 'Filtered Out', 'Pred. Nodule']
        confusions[:, -2] += confusions[:, -1]
        confusions = confusions[:, :-1]
    cell_width = 16
    f = '{:>' + str(cell_width) + '}'
    print(label)
    print(' | '.join([f.format(s) for s in col_labels]))
    for i, (l, r) in enumerate(zip(row_labels, confusions)):
        r = [l] + list(r)
        if i == 0:
            r[1] = ''
        print(' | '.join([f.format(i) for i in r]))

class NoduleAnalysisApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument(
            '-b', '--batch-size', help='Batch size to use for training', default=32, type=int,
        )

        parser.add_argument(
            '-nw', '--num-workers', help='Number of worker processes for data loading', default=8, type=int,
        )

        parser.add_argument(
            '--tb-dir', default='', help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument(
            "--run-validation", help="Run over validation set rather than a single CT", action="store_true", default=False
        )

        parser.add_argument(
            "--include-train", help="Include data from the training set. (validation data is default)", action="store_true", default=False
        )

        parser.add_argument(
            "-sm",
            "--segmentation-model-path",
            help="Path to saved segmentation model",
            nargs='?',
            defualt="final_models/seg_2024-01-22_11.44_final_seg_300000.best.state",
        )

        parser.add_argument(
            "-cm",
            "--classification-model-path",
            help="Path to saved classification model",
            nargs='?',
            defualt="final_models/cls_2024-01-26_10.26._final-nodule-nonnodule.best.state",
        )
        
        parser.add_argument(
            "--malignancy-model-path",
            help="Path to the saved malignancy classification model",
            nargs='?',
            default="final_models/cls_2024-01-26_10.26._final-nodule-nonnodule.best.state",
        )

        parser.add_argument(
            "series_uid",
            nargs='?',
            default=None,
            help="Series UID to use."
        )

        self.cli_args = parser.parse_args(sys_argv)

        if not (bool(self.cli_args.series_uid) ^ (self.cli_args.run_validation)):
            raise Exception("Only one of series_uid or --run_validation should be specified")

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
    
        self.segmentation_model, self.classification_model, self.malignancy_model = self.init_models()

    
    def init_models(self):
        log.debug(self.cli_args.segmentation_model_path)
        segmentation_model_dict = torch.load(self.cli_args.segmentation_model_path)     # load the pretrained segmentation model dictionary

        segmentation_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode="upconv",
        )

        segmentation_model.load_state_dict(segmentation_model_dict["model_state"])
        segmentation_model.eval()

        log.debug(self.cli_args.classification_model_path)
        classification_model_dict = torch.load(self.cli_args.classification_model_path)

        classification_model = LunaModel()
        classification_model.load_state_dict(classification_model_dict["model_state"])
        classification_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                classification_model = nn.DataParallel(classification_model)
            
            segmentation_model.to(self.device)
            classification_model.to(self.device)

        if self.cli_args.malignancy_path:
            malignancy_model = LunaModel()
            malignancy_dict = torch.load(self.cli_args.malignancy_path)
            malignancy_model.load_state_dict(malignancy_dict["model_state"])
            malignancy_model.eval()
            
            if self.use_cuda:
                malignancy_model.to(self.device)
        else:
            malignancy_model = None

        return segmentation_model, classification_model, malignancy_model

    def init_segmentation_dataloader(self, series_uid):
        segmentation_dataset = Luna2dSegmentationDataset(
            context_slices_count=3,
            series_uid=series_uid,
            full_CT_bool=True,
        )

        segmentation_dataloader = DataLoader(
            segmentation_dataset,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count()if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return segmentation_dataloader

    def segment_CT(self, ct, series_uid):
        """Loads a CT from a single series_uid and returns each slice, one per call"""
        with torch.no_grad():
            output_a = np.zeros_like(ct.hu_a, dtype=np.float32)    # array to hold probability annotations
            segmentation_dataloader = self.init_segmentation_dataloader(series_uid)
            
            # loop over CTs in batches
            for input_t, _, _, slice_list in segmentation_dataloader:
                input_g = input_t.to(self.device)
                predictions_g = self.segmentation_model(input_g)

                for i, slice_ndx in enumerate(slice_list):  # copy each element to the output array
                    output_a[slice_ndx] = predictions_g[i].cpu().numpy() 
        
            mask_a = output_a > 0.5
            # Applies binary erosion to refine segmentation, removing boundary voxels and small components (< 3x3x3 voxels).
            # Ensures segmentation precision by retaining voxels only if all neighbors are flagged.
            mask_a = morphology.binary_erosion(mask_a, iterations=1)    ##

            return mask_a

    def group_segmentation_output(self, series_uid, ct, clean_a):
        """
        The function will take all non-zero pixels that share an edge with another non-zero pixel
        and group them together.
        """
        # assign each voxel the label of the group it belongs to
        candidate_label_a, candidate_count = measurements.label(clean_a)
        center_irc_list = measurements.center_of_mass(
            ct.hu_a.clip(-1000, 1000) + 1001,   # offset so that mass is non-negative as expected by the function
            labels=candidate_label_a,
            index=np.arange(1, candidate_count + 1),
        )

        candidate_info_list = []
        for i, center_irc in enumerate(center_irc_list):
            center_xyz = irc2xyz(
                center_irc,
                ct.origin_xyz,
                ct.vx_size_xyz,
                ct.direction_a,
            )

            assert np.all(np.isfinite(center_irc)), repr(['irc', center_irc, i, candidate_count])
            assert np.all(np.isfinite(center_xyz)), repr(['xyz', center_xyz])

            candidate_info_tuple = CandidateInfoTuple(
                False, False, False, 0.0, series_uid, center_xyz
            )
            candidate_info_list.append(candidate_info_tuple)

        return candidate_info_list

    def init_classification_dataloader(self, candidate_info_list):
        classification_dataset = LunaDataset(
            sortby_str="series_uid",
            candidate_info_list=candidate_info_list,
        )

        classification_dataloader = DataLoader(
            classification_dataset,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        
        return classification_dataloader

    def classify_candidates(self, ct, candidate_info_list):
        """
        This function crops the candidate regions identified by the segmentation model to be fed
        into the classification model
        """
        classification_dataloader = self.init_classification_dataloader(candidate_info_list)
        classifications_list = []

        for batch_IDX, batch_tuple in enumerate(classification_dataloader):
            # print(len(batch_tuple))
            input_t, _, series_list, center_list = batch_tuple
            input_g = input_t.to(self.device)

            with torch.no_grad():
                _, probability_nodule_g = self.classification_model(input_g)    # run input through nodule non-nodule classification model
                if self.malignancy_model is not None:
                    _, probability_mal_g = self.malignancy_model(input_g)
                else:
                    probability_mal_g = torch.zeros_like(probability_nodule_g)

            zip_iter = zip(
                center_list,
                probability_nodule_g[:, 1].tolist(),
                probability_mal_g[:, 1].tolist(),
            )

            for center_irc, probability_nodule, probability_malignant in zip_iter:
                center_xyz = irc2xyz(
                    center_irc,
                    direction_a=ct.direction_a,
                    origin_xyz=ct.origin_xyz,
                    vxsize_xyz=ct.vx_size_xyz,
                )

                classification_tuple = (probability_nodule, probability_malignant, center_xyz, center_irc)
                classifications_list.append(classification_tuple)
        
        return classifications_list


    def log_results(self, mode_str, filtered_list, series_to_diagnosis_dict, positive_set):
        count_dict = {
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
        }

        for series_uid in filtered_list:
            probability_float, center_irc = series_to_diagnosis_dict.get(series_uid, (0.0, None))
            if center_irc is not None:
                center_irc = tuple(int(x.item()) for x in center_irc)
            positive_bool = series_uid in positive_set    
            prediction_bool = probability_float > 0.5
            correct_bool = positive_bool == prediction_bool

            if positive_bool and prediction_bool:
                count_dict["true_positive"] += 1
            if not positive_bool and not prediction_bool:
                count_dict["true_negative"] += 1
            if not positive_bool and prediction_bool:
                count_dict["false_positive"] += 1
            if positive_bool and not prediction_bool:
                count_dict["false_negative"] += 1
        
        total_count = sum(count_dict.values())
        percent_dict = {k: v / (total_count or 1) * 100 for k, v in count_dict.items()}

        precision = percent_dict["precision"] = count_dict["true_positive"] / ((count_dict["true_positive"] + count_dict["false_positive"]) or 1)
        recall = percent_dict["recall"] = count_dict["true_positive"] / ((count_dict["true_positive"] + count_dict["false_negative"]) or 1)
        percent_dict["f1_score"] = 2 * (precision * recall) / ((precision + recall) or 1)

        log.info(
            f"{mode_str} true positive:{percent_dict['true_positive']:.1f}%, "
            f"true negative:{percent_dict['true_negative']:.1f}%, "
            f"false positive:{percent_dict['false_positive']:.1f}%, "
            f"false negative:{percent_dict['false_negative']:.1f}%"
        )

        log.info(
            f"{mode_str} precision:{percent_dict['precision']:.3f}, "
            f"recall:{percent_dict['recall']:.3f}, F1: {percent_dict['f1_score']:.3f}"
        )

    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")

        validation_dataset = LunaDataset(
            val_stride=10,
            is_val_set_bool=True,
        )

        validation_set = set(
            candidate_info_tuple.series_uid for candidate_info_tuple in validation_dataset.candidate_info_list
        )

        positive_set = set(
            candidate_info_tuple.series_uid 
            for candidate_info_tuple in get_candidate_info_list() 
            if candidate_info_tuple.is_nodule_bool
        )

        if self.cli_args.series_uid:
            series_set = set(self.cli_args.series_uid.split(','))
        else:
            series_set = set(
                candidate_info_tuple.series_uid
                for candidate_info_tuple in get_candidate_info_list()
            )

        if self.cli_args.include_train:
            train_list = sorted(series_set - validation_set)
        else:
            train_list = []
        validation_list = sorted(series_set & validation_set)

        candidate_info_dict = get_candidate_info_dict()
        series_iter = enum_estimate(
            validation_list + train_list,
            "Series"
        )
        all_confusion = np.zeros((3, 4), dtype=np.int_)
        for _, series_uid in series_iter:
            ct = get_ct_data(series_uid)
            mask_a = self.segment_CT(ct, series_uid)

            candidate_info_list = self.group_segmentation_output(
                series_uid, ct, mask_a,
            )
            classifications_list = self.classify_candidates(
                ct, candidate_info_list
            )

            if not self.cli_args.run_validation:
                print(f"Found nodule candidates in {series_uid}:")
                for probability, probability_malignant, center_xyz, center_irc in classifications_list:
                    if probability > 0.5:
                        string = f"nodule probability {probability:.3f}, "
                        if self.malignancy_model:
                            string += f"malignancy probability {probability_malignant:.3f}, "
                        string += f"center xyz {center_xyz}"
                        print(string)

            if series_uid in candidate_info_dict:
                one_confusion = match_and_score(
                    classifications_list, candidate_info_dict[series_uid]
                )
                all_confusion += one_confusion
                print_confusion(
                    series_uid, one_confusion, self.malignancy_model is not None
                )

        print_confusion(
            "Total", all_confusion, self.malignancy_model is not None
        )




if __name__ == "__main__":
    NoduleAnalysisApp().main()