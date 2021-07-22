import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertTokenizerFast,
    DataCollatorWithPadding,
    RoFormerTokenizerFast,
)

from chinesebert import ChineseBertTokenizerFast, DataCollatorForChineseBERT
from model import RDropModelForSequenceClassification
from utils import load_dataset


def predict(model_dir, args_dir, test_dir, result_dir, task=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = torch.load(f"{str(args_dir)}/args.pt")
    args.test_file = test_dir
    args.max_length = 512
    args.overwrite_cache = True
    args.task = f"task{str(task)}"

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

    tokenizer = tokenizer_cls.from_pretrained(args.model_name_or_path)
    test_dataset = load_dataset(args, tokenizer=tokenizer, is_test=True)
    dcls = (
        DataCollatorForChineseBERT
        if args.model_type == "chinesebert"
        else DataCollatorWithPadding
    )
    data_collator = dcls(tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=data_collator,
    )

    model = RDropModelForSequenceClassification(
        args.model_name_or_path,
        model_type=args.model_type,
        pooler_type=args.pooler_type,
        cache_dir=args.cache_dir,
        num_labels=2,
    )
    if os.path.exists(model_dir):
        model.load_state_dict(torch.load(model_dir))
    else:
        print("model_dir", model_dir, "不存在")
    model.eval()
    model.to(device)

    pred_tags = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            batch_output = outputs.logits.argmax(dim=-1).cpu().numpy()
            pred_tags.extend(batch_output)

    with open(test_dir, "r", encoding="utf8") as fr:
        items = json.load(fr)

    for idx, item in enumerate(items):
        item[f"judge{str(task)}"] = int(pred_tags[idx])

    with open(result_dir, "w", encoding="utf8") as fw:
        json.dump(items, fw, indent=2, ensure_ascii=False)


def task1(model_dir, result_dir):
    args_dir = model_dir.parent.parent / "logs"
    predict(
        model_dir=model_dir,
        args_dir=args_dir,
        test_dir="raw_data/pack1/task1-dev.json",
        result_dir=os.path.join(result_dir, "task1-dev-with-answer.json"),
        task=1,
    )
    predict(
        model_dir=model_dir,
        args_dir=args_dir,
        test_dir="raw_data/pack1/task3-dev.json",
        result_dir=os.path.join(result_dir, "task3-dev-task1-pred.json"),
        task=1,
    )


def task2(model_dir, result_dir):
    args_dir = model_dir.parent.parent / "logs"
    predict(
        model_dir=model_dir,
        args_dir=args_dir,
        test_dir="raw_data/pack1/task2-dev.json",
        result_dir=os.path.join(result_dir, "task2-dev-with-answer.json"),
        task=2,
    )
    predict(
        model_dir=model_dir,
        args_dir=args_dir,
        test_dir="raw_data/pack1/task3-dev.json",
        result_dir=os.path.join(result_dir, "task3-dev-task2-pred.json"),
        task=2,
    )


if __name__ == "__main__":
    print("task1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for p in Path(".").glob("task1/**/*.bin"):
        newpath = p.parent.parent / "preds"
        newpath.mkdir(exist_ok=True, parents=True)
        print(f"loading {str(p)}")
        task1(p, result_dir=newpath)

    print("task2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for p in Path(".").glob("task2/**/*.bin"):
        newpath = p.parent.parent / "preds"
        newpath.mkdir(exist_ok=True, parents=True)
        print(f"loading {str(p)}")
        task2(p, result_dir=newpath)
