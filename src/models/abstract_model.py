from collections import defaultdict
from datasets import DatasetDict, Dataset, Sequence, Value
from typing import Dict, List, Tuple
from datasets.info import DatasetInfo
from dataclasses import dataclass, field
import numpy as np
from transformers.modeling_outputs import SequenceClassifierOutput
import torch


@dataclass
class SequenceClassifierOutputPlus(SequenceClassifierOutput):
    feature_vecs: torch.Tensor = None


@dataclass
class SpanClassifierOutput:
    label: str
    logits: np.array = None


@dataclass
class SpanClassifierDataTrainingArguments:
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )


class SpanClassifier:
    def __init__(
        self,
        span_classification_datasets: DatasetDict,
        data_args: SpanClassifierDataTrainingArguments,
    ) -> None:
        self.span_classification_datasets = span_classification_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=False,  # because it might have a problem
        )
        self.argss = []

    def predict(self, tokens: List[str], start: int, end: int) -> SpanClassifierOutput:
        raise NotImplementedError

    def batch_predict(
        self, tokens: List[List[str]], start: List[int], end: List[int]
    ) -> List[SpanClassifierOutput]:
        assert len(tokens) == len(start)
        assert len(start) == len(end)
        raise NotImplementedError

    def preprocess_function(self, example: Dict) -> Dict:
        """preprocess_function for encoding
        Args:
            example (Dict): {"tokens": List[List[str]], "start": List[int], "end": List[int], "label": List[int]}
        Returns:
            ret_dict (Dict): {"label": List[int], "...": ...}
        """
        raise NotImplementedError


def translate_into_orig_train_args(training_args):
    from transformers import TrainingArguments as OrigTrainingArguments

    train_dict = training_args.to_dict()
    del train_dict["_n_gpu"]
    if train_dict["log_level"] == -1:
        train_dict["log_level"] = "passive"
    if train_dict["log_level_replica"] == -1:
        train_dict["log_level_replica"] = "passive"
    return OrigTrainingArguments(**train_dict)
