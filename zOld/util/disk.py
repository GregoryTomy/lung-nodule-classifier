import gzip
from cassandra.cqltypes import BytesType
from diskcache import FanoutCache, Disk
from diskcache.core import io
from io import BytesIO
from diskcache.core import MODE_BINARY

CHUNK_SIZE = 2 ** 30  # 1 GB
CACHE_SIZE_LIMIT = 3e10  # 30 GB
GZIP_COMPRESS_LEVEL = 1

class GzipDisk(Disk):
    def store(self, value, read, key=None):
        """Stores a value into the disk with optional Gzip compression."""
        if self.is_bytes_type(value):
            if read:
                value = value.read()
                read = False
            value = self.compress(value)
        return super(GzipDisk, self).store(value, read)

    @staticmethod
    def is_bytes_type(value):
        return type(value) is BytesType

    @staticmethod
    def compress(value):
        str_io = BytesIO()
        with gzip.GzipFile(mode='wb', compresslevel=GZIP_COMPRESS_LEVEL, fileobj=str_io) as gz_file:
            for offset in range(0, len(value), CHUNK_SIZE):
                gz_file.write(value[offset:offset+CHUNK_SIZE])
        return str_io.getvalue()

    def fetch(self, mode, filename, value, read):
        """Fetches a value from the disk with optional Gzip decompression."""
        value = super(GzipDisk, self).fetch(mode, filename, value, read)
        if mode == MODE_BINARY:
            value = self.decompress(value)
        return value

    @staticmethod
    def decompress(value):
        str_io = BytesIO(value)
        with gzip.GzipFile(mode='rb', fileobj=str_io) as gz_file:
            read_csio = BytesIO()
            while True:
                uncompressed_data = gz_file.read(CHUNK_SIZE)
                if uncompressed_data:
                    read_csio.write(uncompressed_data)
                else:
                    break
        return read_csio.getvalue()

def getCache(scope_str):
    """Initializes and returns a FanoutCache with GzipDisk as the storage backend."""
    return FanoutCache(f'/scratch/alpine/nito4059/data-unversioned/cache/{scope_str}',
                       disk=GzipDisk,
                       shards=64,
                       timeout=1,
                       size_limit=CACHE_SIZE_LIMIT)
