from datasets import DatasetDict, Dataset
import gin
from datasets import load_dataset
import datasets
from seqeval.metrics.sequence_labeling import get_entities
from collections import defaultdict

load_dataset = gin.configurable(load_dataset)
from hashlib import md5
import os


@gin.configurable
def translate_ner_dataset(ner_dataset: DatasetDict):
    hash_name = md5(
        ("translate_ner_dataset" + str(ner_dataset.cache_files)).encode()
    ).hexdigest()
    output_dir = "data/buffer/%s" % hash_name
    if not os.path.exists(output_dir):
        new_tag_names = ner_dataset["train"].features["ner_tags"].feature.names
        label_names = list(set([l for l, s, e in get_entities(new_tag_names)]))
        features = datasets.Features(
            {
                "tokens": datasets.Sequence(datasets.Value("string")),
                "start": datasets.Value("int32"),
                "end": datasets.Value("int32"),
                "label": datasets.ClassLabel(names=label_names),
            }
        )
        new_dataset_dict = dict()
        for key, split in ner_dataset.items():
            new_split = defaultdict(list)
            for snt in split:
                ner_tags = [new_tag_names[l] for l in snt["ner_tags"]]
                for l, s, e in get_entities(ner_tags):
                    new_split["tokens"].append(snt["tokens"])
                    new_split["start"].append(s)
                    new_split["end"].append(e + 1)
                    new_split["label"].append(l)
            new_dataset_dict[key] = Dataset.from_dict(new_split, features=features)
        translated_dataset = DatasetDict(new_dataset_dict)
        translated_dataset.save_to_disk(output_dir)
    translated_dataset = DatasetDict.load_from_disk(output_dir)
    return translated_dataset
