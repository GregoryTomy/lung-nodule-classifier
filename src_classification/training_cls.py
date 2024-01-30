"""
Main model training and evaluation module.
"""
import argparse
import datetime
import os
import sys
import shutil
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from util.util import enum_estimate
from .preprocess_cls import LunaDataset
from util.config_log import logging
from .model_cls import LunaModel, CTAugmentation

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

METRICS_LABEL_IDX=0
METRICS_PRED_IDX=1
METRICS_LOSS_IDX=2
METRICS_SIZE = 3

class LunaTrainingApp:
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

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model, self.aug_model = self.initialize_model()
        self.initial_weights_path = self.cli_args.init_weights_path
        self.optimizer = self.initialize_optimizer()

        if self.initial_weights_path and os.path.isfile(self.initial_weights_path):
            log.info(f"Loading initial weights from {self.initial_weights_path}")
            checkpoint = torch.load(self.initial_weights_path)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
                # self.model.load_state_dict(torch.load(self.initial_weights_path))

        # self.checkpoint_dir = "model_checkpoints"
        # os.makedirs(self.checkpoint_dir, exist_ok=True)
        # self.best_model_state = None

        # early stopping initialization
        # self.best_froc = float(0)
        # self.best_f2 = float(0)
        # self.epochs_no_improve = 0
        # self.early_stopping_patience = 3


    def initialize_model(self):
        model = LunaModel()
        aug_model = CTAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                aug_model = nn.DataParallel(aug_model)
            model = model.to(self.device)
            aug_model = aug_model.to(self.device)

        return model, aug_model

    def initialize_optimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def init_train_dl(self):
        train_ds = LunaDataset(
            val_stride=10,
            is_val_set_bool=False,
            ratio_int=int(self.cli_args.balanced),
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
        val_ds = LunaDataset(
            val_stride=10,
            is_val_set_bool=True,
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
                log_dir=log_dir + '-train-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val-' + self.cli_args.comment)

    def compute_batch_loss(self, batch_IDX, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup
        
        # label_t = label_t[:, 1].unsqueeze(1).float()

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)


        if self.model.training and self.augmentation_dict:
            input_g = self.aug_model(input_g)

        # logits_g = self.model(input_g)
        logits_g, probability_g = self.model(input_g)

        # probability_g = torch.sigmoid(logits_g[:, 1])
        # print(logits_g.shape)
        # print(label_g.shape)
        # exit()
        # loss_func = FocalLoss(reduction="none")
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func( 
            logits_g,
            # logits_g[:, 1],
            label_g[:,1],
            # label_g.float() # floats are necessary fot binary_cross_entropy in loss calculation
        )

        start_IDX = batch_IDX * batch_size
        end_IDX = start_IDX + label_t.size(0)

        metrics_g[METRICS_LABEL_IDX, start_IDX:end_IDX] = label_g[:, 1]
        # metrics_g[METRICS_LABEL_IDX, start_IDX:end_IDX] = label_g
        metrics_g[METRICS_PRED_IDX, start_IDX:end_IDX] = probability_g[:, 1]
        metrics_g[METRICS_LOSS_IDX, start_IDX:end_IDX] = loss_g

        return loss_g.mean()


    def train_one_epoch(self, epoch_IDX, train_dl):
        self.model.train()
        train_dl.dataset.shuffle_samples()
        train_metrics = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enum_estimate(
            train_dl,
            "E{} Training".format(epoch_IDX),
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
            self.model.eval()
            val_metrics = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enum_estimate(
                val_dl,
                "E{} Validation ".format(epoch_IDX),
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

    def calculate_froc(self, val_metrics, epoch_IDX):
        thresholds = [1/8, 1/4, 1/2, 1, 2, 4, 8]
        sensitivities = []
        fp_per_scan = []
        
        total_positives = int(val_metrics[METRICS_LABEL_IDX].sum())
        total_scans = len(val_metrics[METRICS_LABEL_IDX])

        # torch.sort returns a nameduple(values/indices)
        sorted_predictions, sorted_indices = torch.sort(val_metrics[METRICS_PRED_IDX], descending=True)
        sorted_labels = val_metrics[METRICS_LABEL_IDX][sorted_indices] 

        # calculate the cumulative false positives. 1 - sorted flips the 0, 1 label
        fps = torch.cumsum(1 - sorted_labels, dim=0)
        

        for fp_rate in thresholds:
            num_fp_allowed = int(fp_rate * total_scans)
            threshold_indices = (fps <= num_fp_allowed).nonzero(as_tuple=True)
            
            if threshold_indices[0].numel() == 0:
                # if no indices are found, no false positives are allowed
                threshold_index = 0
            else:
                # otherwise take the last index where the cumulative false positives are less than or equal to allowed number
                threshold_index = threshold_indices[0][-1].item()
            # get the corresponding threshold value
            threshold = sorted_predictions[threshold_index].item()

            tp = ((val_metrics[METRICS_PRED_IDX] >= threshold) & (val_metrics[METRICS_LABEL_IDX] == 1))
            fp = ((val_metrics[METRICS_PRED_IDX] >= threshold) & (val_metrics[METRICS_LABEL_IDX] == 0))

            tp = tp.sum()
            fp = fp.sum()

            # print(f"true positives: {tp}")
            # print(f"false positives: {fp}")

            sensitivity = tp / total_positives
            average_fp_per_scan = fp / total_scans
            
            sensitivities.append(sensitivity)
            fp_per_scan.append(average_fp_per_scan)

        froc_score = np.mean(sensitivities)
        log.info("Epoch {} FROC score: {}".format(epoch_IDX, froc_score))

        return froc_score , sensitivities, fp_per_scan



    def log_metrics(
            self,
            epoch_IDX,
            mode_str,
            metrics_t,
            classfication_threshold=0.5,
    ):
        self.init_tensorboard()
        log.info("E{} {}".format(
            epoch_IDX,
            type(self).__name__,
        ))

        neg_label_mask = metrics_t[METRICS_LABEL_IDX] <= classfication_threshold
        neg_pred_mask = metrics_t[METRICS_PRED_IDX] <= classfication_threshold

        pos_label_mask = ~neg_label_mask
        pos_pred_mask = ~neg_pred_mask

        neg_count = int(neg_label_mask.sum())
        pos_count = int(pos_label_mask.sum())

        true_neg_count = neg_correct = int((neg_label_mask & neg_pred_mask).sum())
        true_pos_count = pos_correct = int((pos_label_mask & pos_pred_mask).sum())

        false_pos_count = neg_count - neg_correct
        false_neg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_IDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_IDX, neg_label_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_IDX, pos_label_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / metrics_t.shape[1] * 100
        metrics_dict['correct/neg'] = (neg_correct) / neg_count * 100
        metrics_dict['correct/pos'] = (pos_correct) / pos_count * 100

        precision = metrics_dict['pr/precision'] = \
            true_pos_count / np.float32(true_pos_count + false_pos_count)
        recall    = metrics_dict['pr/recall'] = \
            true_pos_count / np.float32(true_pos_count + false_neg_count)

        metrics_dict['f/f1_score'] = \
            2 * (precision * recall) / (precision + recall)
        
        f2 = metrics_dict['f/f2_score'] = \
            5 * (precision * recall) / (4 * precision + recall)
        
        metrics_dict['f/f05_score'] = \
            (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall)

        metrics_dict['pr/fpr'] = false_pos_count / np.float32(false_pos_count + true_neg_count)

        metrics_dict['pr/FROC'], _, _ = self.calculate_froc(metrics_t, epoch_IDX)
        
        log.info(
            f"E{epoch_IDX} {mode_str:8} {metrics_dict['loss/all']:.4f} loss, "
            f"{metrics_dict['correct/all']:-5.1f}% correct, "
            f"{metrics_dict['pr/precision']:.4f} precision, "
            f"{metrics_dict['pr/recall']:.4f} recall, "
            f"{metrics_dict['f/f1_score']:.4f} f1 score, "
            f"{metrics_dict['f/f2_score']:.4f} f2 score, "
            f"{metrics_dict['f/f05_score']:.4f} f0.5 score, "
            f"{metrics_dict['pr/fpr']:.4f} fpr, "
            f"{metrics_dict['pr/FROC']:.4f} FROC"
        )

        log.info(
            f"E{epoch_IDX} {mode_str + '_neg':8} {metrics_dict['loss/neg']:.4f} loss, "
            f"{metrics_dict['correct/neg']:-5.1f}% correct ({neg_correct} of {neg_count})"
        )

        log.info(
            f"E{epoch_IDX} {mode_str + '_pos':8} {metrics_dict['loss/pos']:.4f} loss, "
            f"{metrics_dict['correct/pos']:-5.1f}% correct ({pos_correct} of {pos_count})"
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.total_train_samples_count)

        score = metrics_dict["f/f1_score"]

        return score  

    def save_model(self, type_str, epoch_IDX, is_best=False):
        """
        Model parameters are saved (as opposed to the model instance) allowing
        for loading into models that expect parameters of the same shape
        """
        file_path = os.path.join(
            'classification_models',
            self.cli_args.tb_dir,
            f"{type_str}_{self.time_str}_{self.cli_args.comment}_{self.total_train_samples_count}.state"
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)  # mode assigns permissions (owner can read, write, and execute)

        model = self.model
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
                'classification_models',
                self.cli_args.tb_dir,
                f"{type_str}_{self.time_str}_{self.cli_args.comment}_{self.total_train_samples_count}.best.state"
            )
            shutil.copyfile(file_path, best_path)
            log.info(f"Saving model params to {best_path}")

    def train(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.init_train_dl()
        val_dl = self.init_val_dl()

        best_score = 0.0
        validation_epochs = 5
        for epoch_IDX in range(1, self.cli_args.epochs + 1):

            log.info(
                f"Epoch {epoch_IDX} of {self.cli_args.epochs}, "
                f"{len(train_dl)}/{len(val_dl)} batches of size "
                f"{self.cli_args.batch_size}*{(torch.cuda.device_count() if self.use_cuda else 1)}"
            )

            train_metrics = self.train_one_epoch(epoch_IDX, train_dl)
            self.log_metrics(epoch_IDX, 'trn', train_metrics)
            
            if epoch_IDX == 1 or epoch_IDX % validation_epochs == 0:
                val_metrics = self.val_one_epoch(epoch_IDX, val_dl)
                # froc, _, _ = self.calculate_froc(val_metrics, epoch_IDX)
                score = self.log_metrics(epoch_IDX, 'val', val_metrics)
                best_score = max(score, best_score)
                self.save_model('cls', epoch_IDX, score==best_score)

        #         # Early stopping check
        #         if self.f2 > self.best_f2:
        #             self.best_f2 = self.f2
        #             self.epochs_no_improve = 0
        #             self.best_model_state ={
        #                 "model_state_dict": self.model.state_dict(),
        #                 "optimizer_state_dict": self.optimizer.state_dict(),
        #                 "best_f2": self.best_f2,
        #                 "epoch": epoch_IDX,
        #             }
        #         else:
        #             self.epochs_no_improve += 1
        #             log.info(f"No improvement in validation loss for {self.epochs_no_improve} epochs")
        #             if self.epochs_no_improve >= self.early_stopping_patience:
        #                 log.info("Stopping early due to no improvement in F2")
        #                 if self.best_model_state is not None:
        #                     best_model_path = os.path.join(self.checkpoint_dir, f"model_froc_{self.best_froc:.4f}.pth")
        #                     log.info(f"Saving best model to {best_model_path}")
        #                     torch.save(self.best_model_state, best_model_path) 
        #                 break

        # # if early stopping as not been triggered, save the last best model
        # if self.epochs_no_improve < self.early_stopping_patience:
        #     log.info("Training completed without early stopping")
        #     if self.best_model_state is not None:
        #         best_model_path = os.path.join(self.checkpoint_dir, f"model_froc_{self.best_froc:.4f}.pth")
        #         log.info(f"Saving best model to {best_model_path}")
        #         torch.save(self.best_model_state, best_model_path) 
                
        if hasattr(self, 'trn_writer'):
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
    LunaTrainingApp().train()
