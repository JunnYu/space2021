import logging
import math
import os

import torch
import transformers
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    BertTokenizerFast,
    DataCollatorWithPadding,
    RoFormerTokenizerFast,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from chinesebert import ChineseBertTokenizerFast, DataCollatorForChineseBERT
from config import parse_args
from model import RDropModelForSequenceClassification
from utils import compute_metric, load_dataset

logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    tb_writer = SummaryWriter(log_dir=args.log_dir)
    accelerator = Accelerator(fp16=args.fp16)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(os.path.dirname(args.output_dir), "run.log"),
                mode="w",
                encoding="utf-8",
            )
        ],
    )
    logger.info(accelerator.state)

    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    model = RDropModelForSequenceClassification(
        args.model_name_or_path,
        model_type=args.model_type,
        pooler_type=args.pooler_type,
        alpha=args.alpha,
        # model kwargs
        problem_type="single_label_classification",
        num_labels=2,
        cache_dir=args.cache_dir,
    )

    if args.model_type == "chinesebert":
        tokenizer_cls = ChineseBertTokenizerFast
    elif args.model_type in ["bert", "roberta"]:
        tokenizer_cls = BertTokenizerFast
    elif args.model_type in ["roformer", "wobert"]:
        tokenizer_cls = RoFormerTokenizerFast
    else:
        raise ValueError(
            "model_type must be in chinesebert/bert/roberta/wobert/roformer"
        )
    tokenizer = tokenizer_cls.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir
    )

    train_dataset = load_dataset(args, tokenizer, is_training=True)
    eval_dataset = load_dataset(args, tokenizer, is_training=False)

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        dcls = (
            DataCollatorForChineseBERT
            if args.model_type == "chinesebert"
            else DataCollatorWithPadding
        )
        data_collator = dcls(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.num_warmup_radio * args.max_train_steps),
        num_training_steps=args.max_train_steps,
    )

    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous train batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Instantaneous eval batch size per device = {args.per_device_eval_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    tr_loss, logging_loss, eval_best_metric = 0.0, 0.0, 0.0

    for epoch in range(args.num_train_epochs):
        # train
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch, rdrop=args.rdrop)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            tr_loss += loss.item()
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if args.logging_steps > 0 and completed_steps % args.logging_steps == 0:
                    tb_writer.add_scalar(
                        "lr", lr_scheduler.get_last_lr()[0], completed_steps
                    )
                    tb_writer.add_scalar(
                        "train/loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        completed_steps,
                    )
                    logging_loss = tr_loss

            if completed_steps >= args.max_train_steps:
                break

        # eval
        if args.evaluate_during_training:
            predictions_list = []
            labels_list = []
            model.eval()
            with torch.no_grad():
                for step, batch in enumerate(eval_dataloader):
                    outputs = model(**batch)
                    predictions = outputs.logits.argmax(dim=-1)

                    predictions_list.extend(
                        accelerator.gather(predictions).cpu().numpy()
                    )
                    labels_list.extend(
                        accelerator.gather(batch["labels"]).cpu().numpy()
                    )

            eval_results = compute_metric(predictions_list, labels_list)
            tb_writer.add_scalar("eval/f1", eval_results["f1"], completed_steps)
            tb_writer.add_scalar("eval/acc", eval_results["acc"], completed_steps)
            eval_best_metric = max(eval_best_metric, eval_results["f1"])

            logger.info(
                f"epoch {epoch}: {eval_results}, eval_best_metric {eval_best_metric}"
            )

            if args.output_dir is not None:
                save_path = os.path.join(args.output_dir, f"ckpt-epoch-{epoch}.bin")
                logger.info(f"Save model weights as {save_path}.")
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), save_path)


if __name__ == "__main__":
    main()
