import gin
import time
import hashlib


@gin.configurable
def random_output_dir():
    output_dir = "data/output/%s" % hashlib.md5(str(time.time()).encode()).hexdigest()
    return output_dir
