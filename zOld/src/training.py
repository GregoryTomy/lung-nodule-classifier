import argparse
import sys
import os
import datetime as dt
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import logging
import time
import pickle
import numpy as np

from torch.utils.data import DataLoader
from src.preprocessing import LunaDataset
from utils.util import setup_logger, log_progress

logger = setup_logger(__name__, "logs/training.log", level=logging.INFO)


METRICS_LABEL_IDX = 0
METRICS_PRED_IDX = 1
METRICS_LOSS_IDX = 2
METRICS_SIZE = 3


def load_dataset_with_pickle(load_path):
    with open(load_path, "rb") as f:
        dataset = pickle.load(f)
    return dataset


class LunaTraining:
    def __init__(self, model, optimizer, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-nw",
            "--num-workers",
            help="Number of worker processes to use for data loading",
            default=8,
            type=int,
        )
        parser.add_argument(
            "-e", "--epochs", help="Number of epochs to train for.", default=1, type=int
        )

        parser.add_argument(
            "-b", "--batch-size", help="batch size for training.", default=32, type=int
        )

        parser.add_argument(
            "-bl",
            "--balanced",
            help="balance the training data. 50% positive, 50%negative.",
            action="store_true",
            default=False,
        )

        parser.add_argument(
            "comment", help="Comment for Tensorboard run.", nargs="?", default=""
        )

        parser.add_argument(
            "-a",
            "--augmented",
            help="Augment the training data.",
            action="store_true",
            default=False,
        )

        parser.add_argument(
            "--augment_flip",
            help="Augment the training data by randomly flipping the data.",
            action="store_true",
            default=False,
        )

        parser.add_argument(
            "--augment_offset",
            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
            action="store_true",
            default=False,
        )

        parser.add_argument(
            "--augment_scale",
            help="Augment the training data by increasing/decreasing the size of the image.",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--augment_rotate",
            help="Augment the training data by randomly rotating the data.",
            action="store_true",
            default=False,
        )

        parser.add_argument(
            "--augment_noise",
            help="Augment the training data by randomly adding noise to the data.",
            action="store_true",
            default=False,
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = dt.datetime.now().strftime("'%m-%d_%H.%M'")
        self.train_writer = None
        self.val_writer = None

        self.augmentation_dict = {}
        # the values here work emperically and are not optimized
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict["flip"] = True
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict["scale"] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict["offset"] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict["rotate"] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict["noise"] = 25.0

        # check if CUDA is available, if so set device to CUDA else CPU
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self._load_model(model)
        self.optimizer = optimizer

        self.total_training_samples_count = 0

    ### NOTE: model passed into the class will be instantiated outside
    def _load_model(self, model):
        """
        Load the model to the appropriate device(s).

        Parameters:
        - model_class: PyTorch model class to be instantiated.

        Returns:
        - model: The initialized PyTorch model.
        """
        if self.use_cuda:
            logger.info(f"Using CUDA with {torch.cuda.device_count()} devices")

            # if multiple GPUs available, distribute betweeen them
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)  # or use DistributedDataParallel?
            model = model.to(self.device)
        return model

    def _init_traindl(self):
        """
        Initializes the training DataLoader using LunaDataset and batch size information.

        Returns:
        - train_dl: DataLoader object for training data.
        """
        train_data = LunaDataset(
            val_stride=10, is_val_set_bool=False, ratio=int(self.cli_args.balanced)
        )
        # train_data = load_dataset_with_pickle("Data/saved_data/train_dataset.pkl")

        batch_size = self.cli_args.batch_size
        # Adjust the batch size for multi-GPU training. In Data Parallelism, each GPU will
        # receive a mini-batch of (batch_size / num_gpus). To keep the effective batch size the same,
        # we scale up the original batch size by the number of available GPUs. This ensures that each
        # GPU receives a mini-batch of the original size.
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_data,
            batch_size=batch_size,
            # num_workers=0,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        return train_dl

    def _init_valdl(self):
        """
        Initializes the validation DataLoader using LunaDataset and batch size information.

        Returns:
        - val_dl: DataLoader object for validation data.
        """
        val_data = LunaDataset(val_stride=10, is_val_set_bool=True)
        # val_data = load_dataset_with_pickle("Data/saved_data/val_dataset.pkl")

        batch_size = self.cli_args.batch_size
        # see _initTrainDl for explanation below
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_data,
            batch_size=batch_size,
            # num_workers=0,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def _init_tensorboard(self):
        if self.train_writer is None:
            log_dir = os.path.join("runs", self.time_str)

            self.train_writer = SummaryWriter(log_dir=log_dir + "-train" + self.cli_args.comment)

            self.val_writer = SummaryWriter(log_dir=log_dir + "-val" + self.cli_args.comment)

    def do_training(self, epoch_id, train_dl):
        self.model.train()

        train_metrics = torch.zeros(
            METRICS_SIZE, len(train_dl.dataset), device=self.device
        )
        start_time = time.time()
        num_batches = len(train_dl)

        for batch_idx, batch_tuple in enumerate(train_dl):
            self.optimizer.zero_grad()

            loss = self.compute_batch_loss(
                batch_idx, batch_tuple, train_dl.batch_size, train_metrics
            )

            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                log_progress(epoch_id, batch_idx, num_batches, start_time)

        self.total_training_samples_count += len(train_dl.dataset)

        return train_metrics.to("cpu")

    def do_validation(self, epoch_id, val_dl):
        # no grad to turn off updating network weights
        with torch.no_grad():
            self.model.eval()
            val_metrics = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            start_time = time.time()
            num_batches = len(val_dl)

            for batch_idx, batch_tuple in enumerate(val_dl):
                self.compute_batch_loss(
                    batch_idx, batch_tuple, val_dl.batch_size, val_metrics
                )

                if batch_idx % 10 == 0:
                    log_progress(epoch_id, batch_idx, num_batches, start_time)

        return val_metrics.to("cpu")

    def compute_batch_loss(self, batch_idx, batch_tuple, batch_size, metrics):
        """
        Compute the loss for a batch of data and update the metrics tensor.

        Parameters:
        - self: The context object which likely has access to the model, the device, etc.
        - batch_idx (int): The index of the current batch within the epoch.
        - batch_tuple (tuple): A tuple containing the batch data. Expected to have input data, labels, and possibly other information.
        - batch_size (int): The number of samples in the batch.
        - metrics (torch.Tensor): A tensor to record metrics like actual labels, predicted probabilities, and losses for analysis.

        Returns:
        - torch.Tensor: The mean loss for the batch, which will be used for the backward pass and gradient descent.
        """

        input_t, label_t, _, _ = batch_tuple

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, prob_g = self.model(input_g)

        # reduction none gives loss per sample
        loss_func = nn.CrossEntropyLoss(reduction="none")
        loss_g = loss_func(logits_g, label_g[:, 1])

        start_idx = batch_idx * batch_size
        actual_batch_size = label_t.size(0)
        end_idx = start_idx + actual_batch_size

        # detach since metrics don't need to hold gradients
        metrics[METRICS_LABEL_IDX, start_idx:end_idx] = label_g[:, 1].detach()
        metrics[METRICS_PRED_IDX, start_idx:end_idx] = prob_g[:, 1].detach()
        metrics[METRICS_LOSS_IDX, start_idx:end_idx] = loss_g.detach()

        return loss_g.mean()

    def log_metrics(self, epoch_id, mode_str, metrics, classification_threshold=0.5):
        self._init_tensorboard()
        # mode_str tells us if the metrics are for training or validation

        negative_label_mask = metrics[METRICS_LABEL_IDX] <= classification_threshold
        negative_pred_mask = metrics[METRICS_PRED_IDX] <= classification_threshold
        positive_label_mask = ~negative_label_mask
        positive_pred_mask = ~negative_pred_mask

        negative_count = int(negative_label_mask.sum())
        positive_count = int(positive_label_mask.sum())

        true_neg_count = int((negative_label_mask & negative_pred_mask).sum())
        true_pos_count = int((positive_label_mask & positive_pred_mask).sum())

        false_pos_count = negative_count - true_neg_count
        false_neg_count = positive_count - true_pos_count

        metrics_dict = {}
        metrics_dict["loss/all"] = metrics[
            METRICS_LOSS_IDX
        ].mean()  # average loss over entire epoch
        metrics_dict["loss/neg"] = metrics[
            METRICS_LOSS_IDX, negative_label_mask
        ].mean()  #
        metrics_dict["loss/pos"] = metrics[METRICS_LOSS_IDX, positive_label_mask].mean()
        metrics_dict["correct/all"] = (
            (true_pos_count + true_neg_count) / metrics.shape[1] * 100
        )
        metrics_dict["correct/neg"] = true_neg_count / np.float32(negative_count) * 100
        metrics_dict["correct/pos"] = true_pos_count / np.float32(positive_count) * 100

        precision = metrics_dict["m/precision"] = true_pos_count / np.float32(
            true_pos_count + false_pos_count
        )

        recall = metrics_dict["m/recall"] = true_pos_count / np.float32(
            true_pos_count + false_neg_count
        )

        f1score = metrics_dict["m/f1score"] = (2 * recall * precision) / (
            precision + recall
        )

        # logger.info(
        #     f"E{epoch_id} {mode_str:8} {metrics_dict['loss/all']:.4f} loss, "
        #     f"{metrics_dict['correct/all']:.1f}% correct, "
        #     f"{mode_str + '_neg':8} {metrics_dict['loss/neg']:.4f} loss, "
        #     f"{metrics_dict['correct/neg']:.1f}% correct ({true_neg_count} of {negative_count}), "
        #     f"{mode_str + '_pos':8} {metrics_dict['loss/pos']:.4f} loss, "
        #     f"{metrics_dict['correct/pos']:.1f}% correct ({true_pos_count} of {positive_count})"
        # )
        # logging info overall
        logger.info(
            f"E{epoch_id} {mode_str:8} {metrics_dict['loss/all']:.4f} loss, "
            f"correct {metrics_dict['correct/all']:-5.1f}% , "
            f"precision {metrics_dict['m/precision']:.4f}, "
            f"recall {metrics_dict['m/recall']:.4f} , "
            f"f1 score{metrics_dict['m/f1score']:.4f} "
        )

        # logging the correctly identified positive and negative labels
        logger.info(
            f"E{epoch_id} {mode_str}_neg {metrics_dict['loss/neg']:.4f} loss, "
            f"correct: {metrics_dict['correct/neg']:-5.1f}% | ({true_neg_count} of {negative_count})"
        )

        logger.info(
            f"E{epoch_id} {mode_str}_pos {metrics_dict['loss/pos']:.4f} loss, "
            f"correct: {metrics_dict['correct/pos']:-5.1f}% | ({true_pos_count} of {positive_count})"
        )
        writer = getattr(self, mode_str + "_writer")

        for k, v in metrics_dict.items():
            # argument 1 is the tag and tells which graph the values are being added to
            # key names are correct/pos form and the grouping will be by the substring before /
            # argument 2 are the y values of our data points
            # argument 3 are the x axis values which we use with total training samples count
            writer.add_scalar(k, v, self.total_training_samples_count)

    def train(self):
        logger.info(f"Starting training with {type(self).__name__ , self.cli_args}")

        train_dl = self._init_traindl()
        val_dl = self._init_valdl()

        for epoch_id in range(1, self.cli_args.epochs + 1):
            logger.info(
                f"Epoch {epoch_id} of {self.cli_args.epochs}, "
                f"{len(train_dl)}/{len(val_dl)} batches of size "
                f"{self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1)}"
            )

            train_metrics = self.do_training(epoch_id, train_dl)
            self.log_metrics(epoch_id, "train", train_metrics)

            val_metrics = self.do_validation(epoch_id, val_dl)
            self.log_metrics(epoch_id, "val", val_metrics)
