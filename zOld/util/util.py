from collections import namedtuple
import numpy as np
import time
import datetime as dt
import logging

XyzTuple = namedtuple("XyzTuple", ["x", "y", "z"])
IrcTuple = namedtuple("IrcTuple", ["index", "row", "column"])


def xyz_to_irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    """
    Convert coordinates from XYZ to IRC.

    Parameters:
    - coord_xyz: tuple or list or np.array
        Coordinates in XYZ format.
    - origin_xyz: tuple or list or np.array
        The origin offset in XYZ format.
    - vxSize_xyz: tuple or list or np.array
        The voxel sizes in each XYZ dimension.
    - direction_a: np.array
        The direction (transformation) matrix that maps voxel indices to physical coordinates.

    Returns:
    - IrcTuple: tuple
        Coordinates in IRC format.
    """
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    # Step 1: subtract the origin offset
    # Step 2: multiply with the inverse of the direction matrix
    # Step 3: scale back usign the inverse voxel sizes
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)

    # Step 4: flip coordinates from CRI to IRC when returning
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))


def log_progress(epoch_idx, batch_idx, num_batches, start_time):
    elapsed_time = time.time() - start_time
    batches_left = num_batches - batch_idx
    estimated_total_time = elapsed_time / (batch_idx + 1) * num_batches
    estimated_end_time = start_time + estimated_total_time
    estimated_time_left = estimated_end_time - time.time()

    end_time_str = dt.datetime.fromtimestamp(estimated_end_time).strftime(
        "'%Y-%m-%d %H:%M:%S'"
    )
    estimated_time_left_str = str(dt.timedelta(seconds=estimated_time_left)).split(".")[
        0
    ]

    logging.info(
        f"Epoch {epoch_idx}, Batch {batch_idx}/{num_batches}: "
        f"{batches_left} batches left, "
        f"Estimated completion at {end_time_str}, "
        f"Time left: {estimated_time_left_str}"
    )


def setup_logger(name, log_file, level=logging.INFO):
    """Set up a logger for a particular module.

    Args:
    name (str): Name of the logger.
    log_file (str): File path for the logger to output logs.
    level (logging.level): Logging level, e.g., logging.INFO, logging.WARNING.
    """
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    
    # create a logger with the specified name
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
