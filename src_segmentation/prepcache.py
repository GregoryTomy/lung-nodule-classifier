"""
This module is used to initialize and manage the preparation of data caching for training. 
"""

import argparse
import sys
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from util.util import enum_estimate
from .preprocess import LunaDataset
from util.config_log import logging
from .model import LunaModel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class PrepCache:
    @classmethod
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--batch-size",
            help="Batch size to use for training",
            default=600,
            type=int,
        )
        parser.add_argument(
            "--num-workers",
            help="Number of worker processes for data loading",
            default=8,
            type=int,
        )

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        self.prep_dl = DataLoader(
            LunaDataset(
                sortby_str="series_uid",
            ),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        batch_iter = enum_estimate(
            self.prep_dl,
            "Stuffing cache",
            start_IDX=self.prep_dl.num_workers,
        )
        for _ in batch_iter:
            pass


if __name__ == "__main__":
    PrepCache().main()
