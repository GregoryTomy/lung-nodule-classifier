"""
This script module enhances data caching by integrating Gzip compression, specifically designed 
for efficient storage and retrieval of large binary data types. It features a custom GzipDisk 
class that extends the diskcache.Disk class, enabling Gzip compression of cached data. 
This code is heavily adapted from the work of Eli Stevens.
"""

import gzip
from cassandra.cqltypes import BytesType
from diskcache import FanoutCache, Disk
from diskcache.core import io
from io import BytesIO
from diskcache.core import MODE_BINARY


from util.config_log import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class GzipDisk(Disk):
    def store(self, value, read, key=None):
        """
        Overriding the store method to add Gzip compression.
        """
        if type(value) is BytesType:
            if read:
                value = value.read()
                read = False

            # compress the value using Gzip
            str_io = BytesIO()
            gz_file = gzip.GzipFile(mode='wb', compresslevel=1, fileobj=str_io)

            # handle large values by breaking them into chunks
            for offset in range(0, len(value), 2**30):
                gz_file.write(value[offset:offset+2**30])
            gz_file.close()

            value = str_io.getvalue()

        return super(GzipDisk, self).store(value, read)


def get_cache(scope_str):
    """
    Function to create a FanoutCache instance.
    """
    return FanoutCache(
        'Data/cache/' + scope_str, disk=GzipDisk, shards=64,timeout=1, size_limit=3e11,
    )
