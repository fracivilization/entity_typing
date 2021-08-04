from datasets import Dataset, DatasetDict
from .models.abstract_model import SpanClassifier
import gin
from collections import defaultdict
from tqdm import tqdm
from loguru import logger


@gin.configurable
class SpanClassificationTestor:
    def __init__(
        self,
        span_classification_datasets: DatasetDict,
        span_classifier_model: SpanClassifier,
    ) -> None:
        self.test_span_classification_dataset = span_classification_datasets["test"]
        self.span_classifier_model = span_classifier_model
        self.label_list = self.test_span_classification_dataset.features["label"].names
        self.predictions = self.load_predictions()
        pass

    def load_predictions(self) -> Dataset:
        predictions = defaultdict(list)
        outputs = self.span_classifier_model.batch_predict(
            self.test_span_classification_dataset["tokens"],
            self.test_span_classification_dataset["start"],
            self.test_span_classification_dataset["end"],
        )
        predictions = {
            "tokens": self.test_span_classification_dataset["tokens"],
            "start": self.test_span_classification_dataset["start"],
            "end": self.test_span_classification_dataset["end"],
            "pred_label": [o.label for o in outputs],
            "gold_label": [
                self.label_list[label]
                for label in self.test_span_classification_dataset["label"]
            ],
        }
        predictions = Dataset.from_dict(predictions)
        return predictions

    def evaluate(self):
        from sklearn.metrics import classification_report

        tp = sum(
            pl == gl
            for gl, pl in zip(
                self.predictions["gold_label"], self.predictions["pred_label"]
            )
            if gl != "O"
        )
        if tp == 0:
            precision, recall, f1 = 0, 0, 0
        else:
            precision = tp / sum(pl != "O" for pl in self.predictions["pred_label"])
            recall = tp / sum(gl != "O" for gl in self.predictions["gold_label"])
            f1 = 2 / (1 / precision + 1 / recall)
        logger.info("P/R/F=%.2f/%.2f/%.2f" % (100 * precision, 100 * recall, 100 * f1))
        logger.info(
            classification_report(
                self.predictions["gold_label"], self.predictions["pred_label"]
            )
        )
