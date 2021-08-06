from copy import deepcopy
from transformers.models.bert.modeling_bert import BertForTokenClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
from typing import Dict, List
import gin
from .abstract_model import (
    SpanClassifier,
    SpanClassifierOutput,
    SpanClassifierDataTrainingArguments,
    SequenceClassifierOutputPlus,
    translate_into_orig_train_args,
)


class BertForSpanInsConClassification(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = torch.nn.Linear(2 * config.hidden_size, config.num_labels)

    def forward(
        self,
        span_inscon_input_ids=None,
        span_inscon_attention_mask=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        # return_dict = self.config.use_return_dict

        minibatch_size, max_seq_lens = span_inscon_input_ids.shape
        start_mask = (span_inscon_input_ids == 2).reshape(
            minibatch_size, max_seq_lens, -1
        )
        end_mask = (span_inscon_input_ids == 3).reshape(
            minibatch_size, max_seq_lens, -1
        )
        outputs = self.bert(
            span_inscon_input_ids,
            attention_mask=span_inscon_attention_mask,
        )
        sequence_output = outputs.last_hidden_state
        start_vectors = (sequence_output * start_mask).sum(dim=1)
        end_vectors = (sequence_output * end_mask).sum(dim=1)
        feature_vecs = torch.cat([start_vectors, end_vectors], dim=1)
        sequence_output = self.dropout(feature_vecs)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     # Only keep active parts of the loss
        #     if attention_mask is not None:
        #         active_loss = attention_mask.view(-1) == 1
        #         active_logits = logits.view(-1, self.num_labels)
        #         active_labels = torch.where(
        #             active_loss,
        #             labels.view(-1),
        #             torch.tensor(loss_fct.ignore_index).type_as(labels),
        #         )
        #         loss = loss_fct(active_logits, active_labels)
        #     else:
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        # return TokenClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        return SequenceClassifierOutputPlus(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            feature_vecs=feature_vecs,
        )


from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
import os
from datasets import ClassLabel, load_dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    set_seed,
)
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import torch

span_start_token = "[unused1]"
span_end_token = "[unused2]"


@gin.configurable
@dataclass
class SpanInsConClassificationModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    saved_param_path: Optional[str] = field(
        default=None,
        metadata={"help": "Fine-Tuned parameters. If there is, load this parameter."},
    )


@gin.configurable
@dataclass
class SpanInsConClassificationDataTrainingArguments(
    SpanClassifierDataTrainingArguments
):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(
        default="ner", metadata={"help": "The name of the task (ner, pos...)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    # overwrite_cache: bool = field(
    #     default=False,
    #     metadata={"help": "Overwrite the cached training and evaluation sets"},
    # )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Max sequence length in training."},
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


from datasets import Dataset, DatasetDict, DatasetInfo, Sequence, Value

mask_token = "[MASK]"
from tqdm import tqdm


@gin.configurable
class SpanInsConClassifier(SpanClassifier):
    def __init__(
        self,
        span_classification_datasets: DatasetDict,
        model_args: SpanInsConClassificationModelArguments,
        data_args: SpanInsConClassificationDataTrainingArguments,
        training_args: TrainingArguments,
    ) -> None:
        """[summary]
        Args:
            span_classification_datasets (DatasetDict): with context tokens
        """
        self.model_args = model_args
        self.data_args = data_args
        training_args = translate_into_orig_train_args(training_args)
        self.training_args = training_args
        logger.info("Start Loading BERT")
        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome."
            )
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info("Training/evaluation parameters %s", training_args)
        # Set seed before initializing model.
        set_seed(training_args.seed)

        datasets = span_classification_datasets
        datasets = DatasetDict(
            {"train": datasets["train"], "validation": datasets["validation"]}
        )
        if training_args.do_train:
            column_names = datasets["train"].column_names
            features = datasets["train"].features
        else:
            column_names = datasets["validation"].column_names
            features = datasets["validation"].features
        # text_column_name = "tokens" if "tokens" in column_names else column_names[0]
        # label_column_name = (
        #     f"{data_args.task_name}_tags"
        #     if f"{data_args.task_name}_tags" in column_names
        #     else column_names[1]
        # )

        label_list = features["label"].names
        self.label_list = label_list
        num_labels = len(label_list)

        # Load pretrained model and tokenizer
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            additional_special_tokens=[span_start_token, span_end_token],
        )
        model = BertForSpanInsConClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        if model_args.saved_param_path:
            model.load_state_dict(torch.load(model_args.saved_param_path))
        # Preprocessing the dataset
        # Padding strategy
        self.tokenizer = tokenizer
        self.model = model
        self.mask_id = self.tokenizer.vocab[mask_token]

        span_classification_datasets = DatasetDict(
            {
                "train": span_classification_datasets["train"],
                "validation": span_classification_datasets["validation"],
            }
        )
        super().__init__(span_classification_datasets, data_args)
        self.argss += [model_args, data_args]
        tokenized_datasets = self.span_classification_datasets

        # Data collator
        # data_collator = DataCollatorForTokenClassification(tokenizer)

        # Metrics
        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            return {
                "accuracy_score": accuracy_score(true_labels, true_predictions),
                "precision": precision_score(true_labels, true_predictions),
                "recall": recall_score(true_labels, true_predictions),
                "f1": f1_score(true_labels, true_predictions),
            }

        # Initialize our Trainer
        from transformers import default_data_collator

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"]
            if training_args.do_train
            else None,
            eval_dataset=tokenized_datasets["validation"]
            if training_args.do_eval
            else None,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
        )
        self.trainer = trainer

        # Training
        if training_args.do_train:
            trainer.train(
                model_path=model_args.model_name_or_path
                if os.path.isdir(model_args.model_name_or_path)
                else None
            )
            trainer.save_model()  # Saves the tokenizer too for easy upload

    def predict(self, tokens: List[str], start: int, end: int) -> SpanClassifierOutput:
        context_tokens = self.get_spanned_token(tokens, start, end)
        tokenized_context = self.tokenizer(
            context_tokens,
            # padding="max_length",
            # truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            # max_length=self.data_args.max_length,
            is_split_into_words=True,
            return_offsets_mapping=True,
        )
        tokenized_context
        kwargs = {
            "span_context_input_ids": torch.LongTensor(
                [tokenized_context["input_ids"]]
            ).to(self.model.device),
            "span_context_attention_mask": torch.LongTensor(
                [tokenized_context["attention_mask"]]
            ).to(self.model.device),
        }
        outputs = self.model(**kwargs)
        return SpanClassifierOutput(
            label=self.label_list[outputs.logits[0].argmax()],
            logits=outputs.logits[0].cpu().detach().numpy(),
        )

    def get_spanned_token(self, tokens: List[str], start: int, end: int):
        return (
            tokens[:start]
            + [span_start_token]
            + tokens[start:end]
            + [span_end_token]
            + tokens[end:]
        )

    def get_spanned_tokens(
        self, tokens: List[List[str]], start: List[int], end: List[int]
    ):
        context_tokens = [
            self.get_spanned_token(tok, s, e) for tok, s, e in zip(tokens, start, end)
        ]
        return context_tokens

    def batch_predict(
        self, tokens: List[List[str]], start: List[int], end: List[int]
    ) -> List[SpanClassifierOutput]:
        assert len(tokens) == len(start)
        assert len(start) == len(end)
        context_tokens = self.get_spanned_tokens(tokens, start, end)
        # max_context_len = max(
        #     len(
        #         self.tokenizer(
        #             cont_tok,
        #             is_split_into_words=True,
        #         )["input_ids"]
        #     )
        #     for cont_tok in context_tokens
        # )
        tokenized_contexts = self.tokenizer(
            context_tokens,
            padding="max_length",
            max_length=216,
            truncation=True,
            is_split_into_words=True,
        )
        dataset = Dataset.from_dict(
            {
                "span_inscon_input_ids": tokenized_contexts["input_ids"],
                "span_inscon_attention_mask": tokenized_contexts["attention_mask"],
            }
        )
        outputs = self.trainer.predict(dataset)
        logits = outputs.predictions
        if isinstance(logits, tuple):
            assert logits[0].shape[1] == len(self.label_list)
            logits = logits[0]
        ret_list = []
        for logit in logits:
            ret_list.append(
                SpanClassifierOutput(
                    label=self.label_list[logit.argmax()], logits=logit
                )
            )
        return ret_list

    def preprocess_function(self, example: Dict) -> Dict:
        context_tokens = self.get_spanned_tokens(
            example["tokens"], example["start"], example["end"]
        )
        tokenized_inputs = self.tokenizer(
            context_tokens,
            padding="max_length",
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            # return_offsets_mapping=True,
            max_length=self.data_args.max_length,
        )
        return {
            "span_inscon_input_ids": tokenized_inputs["input_ids"],
            "span_inscon_attention_mask": tokenized_inputs["attention_mask"],
            "label": example["label"],
        }
