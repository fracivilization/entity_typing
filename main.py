import gin
import sys
from src import *
from hashlib import md5
import time
from loguru import logger

if __name__ == "__main__":
    gin.parse_config_file(sys.argv[1])
    # translate_ner_dataset()
    pass
    span_inside_classifier = SpanInsideClassifier()
