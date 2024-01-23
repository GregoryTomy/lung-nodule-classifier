"""
Main model training and evaluation module.
"""
import argparse
import datetime
import os
import sys
import shutil
import hashlib
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from util.util import enum_estimate
from util.focal_loss import FocalLoss
from .preprocess import Luna2dSegmentationDataset, TrainingLunda2dSegmentationDataset, get_ct_data
from util.config_log import logging
from .model import UNetWrapper, SegmentationAugmentation

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)  

METRICS_LABEL_IDX=0
METRICS_PRED_IDX=1
METRICS_LOSS_IDX=2
METRICS_SIZE = 3
METRICS_TP_IDX = 4
METRICS_FN_IDX = 5
METRICS_FP_IDX = 6

METRICS_SIZE = 10

class SegmentationTrainingApp:
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
            '-e','--epochs', help='Number of epochs to train for', default=1,type=int,
        )

        parser.add_argument(
            '--balanced', help="Balance the training data 1:1", action='store_true', default=False,
        )

        parser.add_argument(
            '--augment-all', help="Augment the training data.", action='store_true', default=False,
        )

        parser.add_argument('--augment-flip',
            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--augment-offset',
            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--augment-scale',
            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--augment-rotate',
            help="Augment the training data by randomly rotating the data around the head-foot axis.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--augment-noise',
            help="Augment the training data by randomly adding noise to the data.",
            action="store_true",
            default=False
        )

        parser.add_argument(
            '-wp', '--init-weights-path', help="Path to inital wrights file to load before training.", default=None, type=str,
        )

        parser.add_argument('--tb-dir', default='', help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument(
            'comment', help="Comment suffix for Tensorboard run.", nargs='?', default='',
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M')

        self.trn_writer = None
        self.val_writer = None
        self.total_train_samples_count = 0

        self.augmentation_dict = {}
        if self.cli_args.augment_all or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augment_all or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.03
        if self.cli_args.augment_all or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augment_all or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augment_all or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.optimizer = self.initialize_optimizer()
        self.segmentation_model, self.augmentation_model = self.initialize_model()

        # if self.initial_weights_path and os.path.isfile(self.initial_weights_path):
        #     log.info(f"loading initial weights from {self.initial_weights_path}")
        #     checkpoint = torch.load(self.initial_weights_path)
        #     if 'model_state_dict' in checkpoint:
        #         self.model.load_state_dict(checkpoint["model_state_dict"])
        #     else:
        #         self.model.load_state_dict(checkpoint)
        #         # self.model.load_state_dict(torch.load(self.initial_weights_path))

        # # self.checkpoint_dir = "model_checkpoints"
        # os.makedirs(self.checkpoint_dir, exist_ok=True)
        # self.best_model_state = None


    def initialize_model(self):
        segmentation_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4, # first layer will have 2 ** 4 = 16 filters which doubles with each downsampling.
            padding=True,   # padding to ensure outpute image is the same size as the input
            batch_norm=True,    # batch normalization after every activation
            up_mode='upconv'    # upsampling is an upconvulutional layer
        )

        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            segmentation_model= segmentation_model.to(self.device)
            augmentation_model= augmentation_model.to(self.device)

        return segmentation_model, augmentation_model

    def initialize_optimizer(self):
        return Adam(self.segmentation_model.parameters())

    def init_train_dl(self):
        train_ds = TrainingLunda2dSegmentationDataset( 
            val_stride=10,
            is_val_set_bool=False,
            context_slices_count=3,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def init_val_dl(self):
        val_ds = Luna2dSegmentationDataset(
            val_stride=10,
            is_val_set_bool=True,
            context_slices_count=3,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def init_tensorboard(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_dir, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '_train_' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '_val_' + self.cli_args.comment)

    def dice_loss(self, predictions_g, label_g, epsilon=1):
        # Sum over everything except the batch dimension to get the positively labeled, (softly) positively detected,
        # and (softly) correct positives per batch item
        dice_label_g = label_g.sum(dim=[1, 2, 3])
        dice_prediction_g = predictions_g.sum(dim=[1, 2, 3])
        dice_correct_g = (predictions_g * label_g).sum(dim=[1, 2, 3])

        # add the epsilon to handle cases with no predictions or labels
        dice_ratio_g = (2 * dice_correct_g + epsilon) / (dice_prediction_g + dice_label_g + epsilon)

        return 1 - dice_ratio_g

    def compute_batch_loss(self, batch_IDX, batch_tup, batch_size, metrics_g, classfication_threshold=0.5):
        input_t, label_t, _, _ = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)


        if self.segmentation_model.training and self.augmentation_dict:
            input_g = self.augmentation_model(input_g, label_g)

        predictions_g = self.segmentation_model(input_g)

        dice_loss_g = self.dice_loss(predictions_g, label_g)

        # focuses on tumor pixels. the multiplaction of predictions and labels leaves only
        # tumor pizels (label = 1) to contribute to the loss. This isolates the false negatives
        # (tumor but predicted as non-tumor) in the loss calculation
        false_negative_loss_g = self.dice_loss(predictions_g * label_g, label_g)    # predictions * labels returns true positives

        start_IDX = batch_IDX * batch_size
        end_IDX = start_IDX + input_t.size(0)

        with torch.no_grad():
            predictions_bool = (predictions_g[:, 0:1] > classfication_threshold).to(torch.float32)

            true_positives = (predictions_bool * label_g).sum(dim=[1, 2, 3])
            false_negatives = ((1 - predictions_bool) * label_g).sum(dim=[1, 2, 3])
            false_positives = (predictions_bool * (~label_g)).sum(dim=[1, 2, 3])

            metrics_g[METRICS_LOSS_IDX, start_IDX, end_IDX] = dice_loss_g
            metrics_g[METRICS_TP_IDX, start_IDX, end_IDX] = true_positives
            metrics_g[METRICS_FN_IDX, start_IDX, end_IDX] = false_negatives
            metrics_g[METRICS_FP_IDX, start_IDX, end_IDX] = false_positives
        
        # we weight the loss such that it is 8 times more important to correctly classify
        # the entire population of the positive class than the negative class. We are willing
        # to trade away many correctly classified negative pixels for one correctly classified 
        # postiive pixel. Note, this is a tradeoff and we will expect a high false positive rate.
        # Adam optimizer works in our favor as using SGD would push the model to overpredict and
        # return every pixel as positive.
        return dice_loss_g.mean() + false_negative_loss_g.mean() * 8

    def log_images(self, epoch_IDX, mode_str, dataloader):
        self.segmentation_model.eval()

        images = sorted(dataloader.dataset.series_list)[:12] # get 12 CTs
        for series_IDX, series_uid in enumerate(images):
            ct = get_ct_data(series_uid)

            for slice_IDX in range(6):
                # select six equidistant slices throughout the CT
                ct_IDX = slice_IDX * (ct.hu_a.shape[0] - 1) // 5
                sample_tuple = dataloader.dataset.get_item_full_slice(series_uid, ct_IDX)

                ct_t, label_t, series_uid, ct_IDX = sample_tuple

                input_g = ct_t.to(self.device).unsqueeze(0)
                label_g = label_t.to(self.device).unsqueeze(0)

                prediction_g = self.segmentation_model(input_g)[0]
                prediction_a = prediction_g.to("cpu").detach().numpy()[0] > 0.5
                label_a = label_g.cpu().numpy()[0][0] > 0.5

                ct_t[:-1, :, :] /= 2000
                ct_t[:-1, :, :] += 0.5

                ct_slice_a = ct_t[dataloader.dataset.context_slices_count].numpy()

                image_a = np.zeros((512, 512, 3), dtype=np.float32)
                image_a[:, :, :] = ct_slice_a.reshape((512, 512, 1))        # copy graysvale CT slice into each RGB channel
                image_a[:, :, 0] += prediction_a & (1 - label_a)            # false positives are flagged as red
                image_a[:, :, 0] += (1 - prediction_a) & label_a   
                image_a[:, :, 1] += ((1 - prediction_a) & label_a) * 0.5        # false negatives are flagged as orange


                image_a[: , :, 1] += prediction_a & label_a             # true positives are flagged as green
                image_a *= 0.5                                          # reduce intensity
                image_a.clip(0, 1, image_a)                             # ensure values are within range to be rendered properly (augmentation might create out of bound values)

                writer = getattr(self, mode_str + "_writer")
                writer.add_image(
                    f"{mode_str}/{series_IDX}_prediction_{slice_IDX}",
                    image_a,
                    self.total_train_samples_count,
                    dataformats="HWC",                                  # lets tensorboard know that the RGB channels is the third axis
                )

                # save the ground truth of the image used for training.
                if epoch_IDX == 1:
                    image_a = np.zeroes((512, 512, 3), dtype=np.float32)
                    image_a[:, :, :] = ct_slice_a.reshape((512, 512, 1))
                    image_a[:, :, 1] += label_a

                    image_a *= 0.5
                    image_a[image_a < 0] = 0
                    image_a[image_a > 1] = 1
                    
                    writer.add_image(
                        f"{mode_str}/{series_IDX}_prediction_{slice_IDX}",
                        image_a,
                        self.total_train_samples_count,
                        dataformats="HWC",
                    )
                
                writer.flush()

    def train_one_epoch(self, epoch_IDX, train_dl):
        self.segmentation_model.train()
        train_dl.dataset.shuffle_samples()
        train_metrics = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enum_estimate(
            train_dl,
            f"E{epoch_IDX} Training.",
            start_IDX=train_dl.num_workers,
        )

        for batch_IDX, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.compute_batch_loss(
                batch_IDX,
                batch_tup,
                train_dl.batch_size,
                train_metrics,
            )

            loss_var.backward()
            self.optimizer.step()

        self.total_train_samples_count += len(train_dl.dataset)

        return train_metrics.to('cpu')


    def val_one_epoch(self, epoch_IDX, val_dl):
        with torch.no_grad():
            self.segmentation_model.eval()
            val_metrics = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enum_estimate(
                val_dl,
                f"E{epoch_IDX} Validation."
                start_IDX=val_dl.num_workers,
            )
            for batch_IDX, batch_tup in batch_iter:
                self.compute_batch_loss(
                    batch_IDX,
                    batch_tup,
                    val_dl.batch_size,
                    val_metrics,
                )

        return val_metrics.to('cpu')


    def log_metrics(
            self,
            epoch_IDX,
            mode_str,
            metrics_t,
    ):
        self.init_tensorboard()
        log.info("E{} {}".format(epoch_IDX,type(self).__name__,))

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        all_label_count = sum_a[METRICS_TP_IDX] + sum_a[METRICS_FN_IDX]

        # false_pos_count = neg_count - neg_correct
        # false_neg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_IDX].mean()

        metrics_dict["percent_all/true_positive"] = sum_a[METRICS_TP_IDX] / (all_label_count or 1) * 100   # handle division by zero.
        metrics_dict["percent_all/false_negative"] = sum_a[METRICS_FN_IDX] / (all_label_count or 1) * 100 
        metrics_dict["percent_all/false_positives"] = sum_a[METRICS_FP_IDX] / (all_label_count or 1) * 100 
    
        # metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_IDX, neg_label_mask].mean()
        # metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_IDX, pos_label_mask].mean()

        # metrics_dict['correct/all'] = (pos_correct + neg_correct) / metrics_t.shape[1] * 100
        # metrics_dict['correct/neg'] = (neg_correct) / neg_count * 100
        # metrics_dict['correct/pos'] = (pos_correct) / pos_count * 100

        precision = metrics_dict['pr/precision'] = \
            sum_a[METRICS_TP_IDX] / ((sum_a[METRICS_TP_IDX] + sum_a[METRICS_FP_IDX] or 1))
        recall    = metrics_dict['pr/recall'] = \
            sum_a[METRICS_TP_IDX] / ((sum_a[METRICS_TP_IDX] + sum_a[METRICS_FN_IDX] or 1))

        metrics_dict['pr/f1_score'] = \
            2 * (precision * recall) / ((precision + recall) or 1)
        
        log.info(
            f"E{epoch_IDX} {mode_str:8} {metrics_dict['loss/all']:.4f} loss, "
            f"{metrics_dict['pr/precision']:.4f} precision, "
            f"{metrics_dict['pr/recall']:.4f} recall, "
            f"{metrics_dict['f/f1_score']:.4f} f1 score, "
        )

        log.info(
            f"E{epoch_IDX} {mode_str + '_all':8} "
            f"{metrics_dict['loss/all']:.4f} loss, "
            f"{metrics_dict['percent_all/true_positive']:-5.1f}% true positive, {metrics_dict['percent_all/false_negative']:-5.1f}% false_negative, {metrics_dict['percent_all/false_positive']:-9.1f}% false positive"
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar("seg_" + key, value, self.total_train_samples_count)

        writer.flush() # make sure that all pending or buffered data points are written to disk by SummaryWriter

        return metrics_dict["pr/recall"]


    def save_model(self, type_str, epoch_IDX, is_best=False):
        """
        Model parameters are saved (as opposed to the model instance) allowing
        for loading into models that expect parameters of the same shape
        """
        file_path = os.path.join(
            'segmentation_models',
            self.cli_args.tb_prefix,
            f"{type_str}_{self.time_str}_{self.cli_args.comment}_{self.total_train_samples_count}.state"
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)  # mode assigns permissions (owner can read, write, and execute)

        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module    # remove the DataParallel wrapper

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_IDX,
            'total_training_samples_count': self.total_train_samples_count, # training samples exposed to the model so far
        }

        torch.save(state, file_path)

        log.info(f"Saving model params to {file_path}")

        if is_best:
            best_path = os.path.join(
                'segmentation_models',
                self.cli_args.tb_prefix,
                f"{type_str}_{self.time_str}_{self.cli_args.comment}_{self.total_train_samples_count}.best.state"
            )
            shutil.copyfile(file_path, best_path)
            log.info(f"Saving model params to {best_path}")


    def train(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.init_train_dl()
        val_dl = self.init_val_dl()
        
        best_score = 0.0
        self.validation_epochs = 5  # validate every n epochs
        for epoch_IDX in range(1, self.cli_args.epochs + 1):

            log.info(
                f"Epoch {epoch_IDX} of {self.cli_args.epochs}, "
                f"{len(train_dl)}/{len(val_dl)} batches of size "
                f"{self.cli_args.batch_size}*{(torch.cuda.device_count() if self.use_cuda else 1)}"
            )

            train_metrics = self.train_one_epoch(epoch_IDX, train_dl)
            self.log_metrics(epoch_IDX, 'trn', train_metrics)
            
            if epoch_IDX == 1 or epoch_IDX % self.validation_epochs == 0:
                val_metrics = self.val_one_epoch(epoch_IDX, val_dl)
                score = self.log_metrics(epoch_IDX, 'val', val_metrics)
                best_score = max(score, best_score)

                self.save_model('seg', epoch_IDX, score == best_score)
                self. log_images(epoch_IDX, 'trn', train_dl)
                self.log_images(epoch_IDX, 'val', val_dl)

        self.trn_writer.close()
        self.val_writer.close()


    def evaluate_model(self):
        log.info("Evaluating final model")

        val_dl = self.init_val_dl()
        val_metrics = self.val_one_epoch(0, val_dl)
        val_metrics_np = val_metrics.cpu().numpy()

        df = pd.DataFrame({
        'Actual': val_metrics_np[METRICS_LABEL_IDX, :],
        'Predicted': val_metrics_np[METRICS_PRED_IDX, :]
        })

        df.to_csv("val_out_df.csv", index=False)
        
        froc = self.calculate_froc(val_metrics, 0)
        self.log_metrics(0, 'val', val_metrics)

        if hasattr(self, 'val_writer'):
            self.val_writer.close()
       

if __name__ == '__main__':
    SegmentationTrainingApp.train()

