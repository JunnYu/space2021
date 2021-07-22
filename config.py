import argparse
import os
from pathlib import Path

import torch
from transformers import SchedulerType


def get_version(base_path):
    max_version = 0
    for p in Path(base_path).glob("version_*"):
        version = int(str(p).split("_")[-1])
        max_version = max(version, max_version)
    max_version = max_version + 1
    return "version_" + str(max_version)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a multilabel text classification task."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="task1",
        help="task.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/train.csv",
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default="data/dev.csv",
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Whether or not overwrite cache examples and features",
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Evaluate during training.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug.",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="hfl/chinese-roberta-wwm-ext",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        help="model_type.",
    )
    parser.add_argument(
        "--pooler_type",
        type=str,
        default="cls",
        help="pooler_type.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether or not use fp16.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-6,
        help="adam_epsilon.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.02, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=20,
        help="logging_steps.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_radio",
        type=float,
        default=0.05,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--log_dir", type=str, default="outputs/logs", help="Tensorboard log dir."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/weights",
        help="Where to store the final model.",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument("--alpha", type=float, default=4.0, help="alpha.")
    parser.add_argument(
        "--rdrop",
        action="store_true",
        help="rdrop",
    )
    args = parser.parse_args()

    #################################################################################
    args.task = "task1"
    args.model_name_or_path = "junnyu/roformer_chinese_base"
    args.model_type = "roformer"
    args.pooler_type = "cls"
    args.overwrite_cache = True
    args.fp16 = True
    args.evaluate_during_training = True
    args.per_device_train_batch_size = 16
    args.per_device_eval_batch_size = 64
    args.learning_rate = 2e-5
    args.num_train_epochs = 20
    args.max_length = 512
    args.num_warmup_radio = 0.05
    args.seed = 42
    args.rdrop = False
    args.alpha = 0.5

    #################################################################################
    base_path = (
        args.task + "/" + args.model_name_or_path.replace("/", "_").replace("-", "_")
    )
    args.version = get_version(base_path)
    # mkdir
    args.output_dir = f"{base_path}/{args.version}/weights"
    os.makedirs(args.output_dir, exist_ok=True)
    args.log_dir = f"{base_path}/{args.version}/logs"
    os.makedirs(args.log_dir, exist_ok=True)
    args.train_file = f"data/{args.task}/train.csv"
    args.validation_file = f"data/{args.task}/dev.csv"
    args.cache_dir = "/hy-nas/models"

    torch.save(args, os.path.join(args.log_dir, "args.pt"))

    return args
