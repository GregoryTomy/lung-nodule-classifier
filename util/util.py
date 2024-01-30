"""
Helper functions for the model.
"""

from collections import namedtuple
import datetime
import time
import numpy as np
from util.config_log import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

XyzTuple = namedtuple('XyzTuple', ['x', 'y', 'z'])
IrcTuple = namedtuple('IrcTuple', ['index', 'row', 'col'])

def xyz2irc(coord_xyz, origin_xyz, vxsize_xyz, direction_a):
    """
    Convert coordinates from XYZ to IRC.

    Parameters:
    - coord_xyz: tuple or list or np.array
        Coordinates in XYZ format.
    - origin_xyz: tuple or list or np.array
        The origin offset in XYZ format.
    - vxsize_xyz: tuple or list or np.array
        The voxel sizes in each XYZ dimension.
    - direction_a: np.array
        The direction (transformation) matrix that maps voxel indices to physical coordinates.

    Returns:
    - IrcTuple: tuple
        Coordinates in IRC format.
    """
    origin_a = np.array(origin_xyz)
    vxsize_a = np.array(vxsize_xyz)
    coord_a = np.array(coord_xyz)
    # Step 1: subtract the origin offset
    # Step 2: multiply with the inverse of the direction matrix
    # Step 3: scale back usign the inverse voxel sizes
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxsize_a
    cri_a = np.round(cri_a)

    # Step 4: flip coordinates from CRI to IRC when returning
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))

def irc2xyz(coord_irc, origin_xyz, vxsize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxsize_a = np.array(vxsize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxsize_a)) + origin_a

    return IrcTuple(*coords_xyz)

def enum_estimate(
        iter,
        desc_str,
        start_IDX=0,
        print_IDX=4,
        backoff=None,
        iter_len=None,
):
    """
    Enumerates over an iterable with logging of progress and estimated completion time.
    This code is adapted from Eli Stevens work.
    Parameters:
        iter (iterable): The iterable to be enumerated.
        desc_str (str): Description string for logging.
        start_IDX (int, optional): Starting index for enumeration. Defaults to 0.
        print_IDX (int, optional): Index at which to start printing progress logs. Defaults to 4.
        backoff (int, optional): Factor for increasing print_IDX to reduce frequency of logs as progress is made. Defaults to 2.
        iter_len (int, optional): Length of the iterable. If not provided, it's calculated.

    Yields:
        tuple: (current index, item) from the iterable.
    """

    if iter_len is None:
        iter_len = len(iter)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_IDX < start_IDX * backoff:
        print_IDX *= backoff

    log.warning(f"{desc_str} ----/{iter_len}, starting")


    start_ts = time.time()
    for (current_IDX, item) in enumerate(iter):
        yield (current_IDX, item)
        if current_IDX == print_IDX: 
            duration_sec = ((time.time() - start_ts)/ (current_IDX - start_IDX + 1)* (iter_len-start_IDX))

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            log.info(
                f"{desc_str} {current_IDX:-4}/{iter_len}, done at {str(done_dt).rsplit('.', 1)[0]}, {str(done_td).rsplit('.', 1)[0]}"
             )

            print_IDX *= backoff

        if current_IDX + 1 == start_IDX:
            start_ts = time.time()

    log.warning(f"{desc_str} ----/{iter_len}, done at {str(datetime.datetime.now()).rsplit('.', 1)[0]}")