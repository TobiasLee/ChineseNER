# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """
import logging
import os
import sys
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from losses import DiceLoss, FocalLoss, LabelSmoothingCrossEntropy
from torch.nn import CrossEntropyLoss
from ray import tune
import ray
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils_ner import Split, TokenClassificationDataset, TokenClassificationTask
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import CLIReporter
from transformers import AutoConfig, TrainingArguments, glue_tasks_num_labels
from ray.tune.integration.wandb import wandb_mixin
import wandb

logger = logging.getLogger(__name__)
os.environ["WANDB_API_KEY"] = "a05e10410445270f685a5a963afab96fdbfc2acc"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    task_type: Optional[str] = field(
        default="NER", metadata={"help": "Task type to fine tune in training (e.g. NER, POS, etc)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    loss_type: Optional[str] = field(
        default='CrossEntropyLoss', metadata={"help": "The loss used during training."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


class TuneTransformerTrainer(Trainer):
    def __init__(self, model, args, compute_metrics, train_dataset=None, eval_dataset=None, loss_type='CrossEntropyLoss'):
        Trainer.__init__(self, model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                         compute_metrics=compute_metrics)
        if loss_type == 'DiceLoss':
            self.loss_fct = DiceLoss()
        elif loss_type == 'FocalLoss':
            self.loss_fct = FocalLoss()
        elif loss_type == 'LabelSmoothingCrossEntropy':
            self.loss_fct = LabelSmoothingCrossEntropy()
        elif loss_type == 'CrossEntropyLoss':
            self.loss_fct = CrossEntropyLoss()
        else:
            raise ValueError("Doesn't support such loss type")

    def get_optimizers(self, num_training_steps):
        self.current_optimizer, self.current_scheduler = super().get_optimizers(num_training_steps)
        return self.current_optimizer, self.current_scheduler

    def evaluate(self, eval_dataset=None, ):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.prediction_loop(eval_dataloader, description="Evaluation")
        self.log(output.metrics)

        if self.optimizer and self.lr_scheduler:
            self.save_state()

        tune.report(**output.metrics)

        return output.metrics

    def save_state(self):
        with tune.checkpoint_dir(step=self.state.global_step) as checkpoint_dir:
            self.args.output_dir = checkpoint_dir
            # This is the directory name that Huggingface requires.
            output_dir = os.path.join(
                self.args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
            self.save_model(output_dir)
            if self.is_world_master():
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def compute_loss(self, model, inputs):
        outputs = model(**inputs)
        logits = outputs[1]  # [bsz, max_token_len, class_num]
        labels = inputs['labels']  # [bsz, max_token_len]
        attention_mask = inputs['attention_mask']  # [bsz, max_token_len]
        loss = None
        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, model.module.num_labels)  # [bsz * max_token_len, class_num]
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.loss_fct.ignore_index).type_as(labels)
                )  # [bsz * max_token_len]
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(logits.view(-1, model.module.num_labels), labels.view(-1))
        return loss


def get_datasets(config, task):
    labels = task.get_labels(config['labels'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'])
    train_dataset = TokenClassificationDataset(
        token_classification_task=task,
        data_dir=config['data_dir'],
        tokenizer=tokenizer,
        labels=labels,
        model_type=config['model_name_or_path'],
        max_seq_length=config['max_seq_length'],
        overwrite_cache=config['overwrite_cache'],
        mode=Split.train,
    )
    eval_dataset = TokenClassificationDataset(
        token_classification_task=task,
        data_dir=config['data_dir'],
        tokenizer=tokenizer,
        labels=labels,
        model_type=config['model_name_or_path'],
        max_seq_length=config['max_seq_length'],
        overwrite_cache=config['overwrite_cache'],
        mode=Split.dev,
    )
    return train_dataset, eval_dataset


@wandb_mixin
def train_transformer(config, checkpoint_dir=None):
    module = import_module("tasks")
    try:
        token_classification_task_clazz = getattr(module, config['task_type'])
        token_classification_task: TokenClassificationTask = token_classification_task_clazz()
    except AttributeError:
        raise ValueError(
            f"Task {config['task_type']} needs to be defined as a TokenClassificationTask subclass in {module}. "
            f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
        )
    train_dataset, eval_dataset = get_datasets(config, token_classification_task)
    labels = token_classification_task.get_labels(config['labels'])
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)
    cache_dir = config['cache_dir']
    loss_type = config['loss_type']

    training_args = TrainingArguments(
        output_dir=tune.get_trial_dir(),
        learning_rate=config["learning_rate"],
        do_train=True,
        do_eval=True,
        evaluate_during_training=True,
        # Run eval after every epoch.
        eval_steps=(len(train_dataset) // (config["per_gpu_train_batch_size"])) + 1,
        # We explicitly set save to 0, and do checkpointing in evaluate instead
        save_steps=0,
        num_train_epochs=config["num_epochs"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_gpu_train_batch_size"],
        per_device_eval_batch_size=config["per_gpu_val_batch_size"],
        warmup_steps=0,
        weight_decay=config["weight_decay"],
        logging_dir="./logs",  # TODO 没加load_best_model_at_end等
    )

    model_name_or_path = recover_checkpoint(checkpoint_dir, config["model_name_or_path"])

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=cache_dir,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=cache_dir,
    )

    def _align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

    def _compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = _align_predictions(p.predictions, p.label_ids)
        return {
            "accuracy_score": accuracy_score(out_label_list, preds_list) * 100,
            "precision": precision_score(out_label_list, preds_list) * 100,
            "recall": recall_score(out_label_list, preds_list) * 100,
            "f1": f1_score(out_label_list, preds_list) * 100,
        }

    # Use our modified TuneTransformerTrainer
    tune_trainer = TuneTransformerTrainer(
        model=model,
        args=training_args,
        compute_metrics=_compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss_type=loss_type
    )
    tune_trainer.train(model_name_or_path)


def eval_transformer(config, analysis):
    loss_type = config['loss_type']
    module = import_module("tasks")
    try:
        token_classification_task_clazz = getattr(module, config['task_type'])
        token_classification_task: TokenClassificationTask = token_classification_task_clazz()
    except AttributeError:
        raise ValueError(
            f"Task {config['task_type']} needs to be defined as a TokenClassificationTask subclass in {module}. "
            f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
        )
    _, test_dataset = get_datasets(config, token_classification_task)
    labels = token_classification_task.get_labels(config['labels'])
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}

    def _align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

    def _compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = _align_predictions(p.predictions, p.label_ids)
        return {
            "accuracy_score": accuracy_score(out_label_list, preds_list) * 100,
            "precision": precision_score(out_label_list, preds_list) * 100,
            "recall": recall_score(out_label_list, preds_list) * 100,
            "f1": f1_score(out_label_list, preds_list) * 100,
        }
    best_config = analysis.get_best_config(metric="eval_f1", mode="max")
    logger.info(best_config)
    best_checkpoint = recover_checkpoint(
        analysis.get_best_trial(metric="eval_f1", mode="max").checkpoint.value)
    logger.info(best_checkpoint)
    best_model = AutoModelForTokenClassification.from_pretrained(best_checkpoint).to("cuda")

    test_args = TrainingArguments(output_dir="./best_model_results", )

    test_trainer = TuneTransformerTrainer(
        model=best_model,
        args=test_args,
        compute_metrics=_compute_metrics,
        loss_type=loss_type
    )

    metrics = test_trainer.evaluate(test_dataset)
    logger.info(metrics)


def recover_checkpoint(tune_checkpoint_dir, model_name=None):
    if tune_checkpoint_dir is None or len(tune_checkpoint_dir) == 0:
        return model_name
    # Get subdirectory used for Huggingface.
    subdirs = [
        os.path.join(tune_checkpoint_dir, name)
        for name in os.listdir(tune_checkpoint_dir)
        if os.path.isdir(os.path.join(tune_checkpoint_dir, name))
    ]
    # There should only be 1 subdir.
    assert len(subdirs) == 1, subdirs
    return subdirs[0]


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    ray.init(log_to_driver=True, ignore_reinit_error=True, num_gpus=8, num_cpus=40, lru_evict=True)
    # ray.init(log_to_driver=True, ignore_reinit_error=True, num_cpus=40, local_mode=True)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    config = {
        # These 3 configs below were defined earlier
        "task_type": model_args.task_type,
        "model_name_or_path": model_args.model_name_or_path,
        'cache_dir': model_args.cache_dir,
        'loss_type': model_args.loss_type,
        'labels': data_args.labels,
        'data_dir': data_args.data_dir,
        'max_seq_length': data_args.max_seq_length,
        'overwrite_cache': data_args.overwrite_cache,
        "per_gpu_val_batch_size": 32,
        "per_gpu_train_batch_size": tune.choice([8, 16, 32]),
        "learning_rate": tune.uniform(1e-5, 5e-5),
        "weight_decay": tune.uniform(0.0, 0.3),
        "num_epochs": tune.choice([5, 10, 15]),
        "max_steps": -1,  # We use num_epochs instead.
        "wandb": {
            "project": "pbt_bert_ner",
            "reinit": True,
            "allow_val_change": True
        }
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_f1",
        mode="max",
        perturbation_interval=2,
        hyperparam_mutations={
            "weight_decay": lambda: tune.uniform(0.0, 0.3).func(None),
            "learning_rate": lambda: tune.uniform(1e-5, 5e-5).func(None),
            "per_gpu_train_batch_size": [8, 16, 32],
        })

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_gpu_train_batch_size": "train_bs/gpu",
            "num_epochs": "num_epochs"
        },
        metric_columns=[
            "accuracy_score", "precision", "recall", "f1"
        ])

    logger.info("begin tuning")

    analysis = tune.run(
        train_transformer,
        resources_per_trial={
            "cpu": 4,
            "gpu": 4
        },
        config=config,
        num_samples=30,  # TODO
        scheduler=scheduler,
        keep_checkpoints_num=30,  # TODO
        checkpoint_score_attr="training_iteration",
        progress_reporter=reporter,
        local_dir="./ray_results/",
        name="pbt_bert_ner",
        queue_trials=True)

    logger.info("evaluate best model")

    eval_transformer(config, analysis)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
