import json
import logging
import os

import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset


def compute_metric(predictions, labels):
    f1 = f1_score(labels, predictions, average="micro")
    acc = accuracy_score(labels, predictions)
    return {"f1": f1, "acc": acc}


class DummyDataset(Dataset):
    def __init__(self, features, is_test=False):
        super().__init__()
        self.features = features
        self.is_test = is_test

    def __getitem__(self, index):
        f = self.features[index]
        inputs = {}
        for k, v in f.items():
            if self.is_test and k == "labels":
                continue
            else:
                inputs[k] = torch.tensor(
                    v, dtype=torch.float if k == "attention_mask" else torch.long
                )
        return inputs

    def __len__(self):
        return len(self.features)


class InputExamples:
    def __init__(self, id, text, labels):
        self.id = id
        self.text = text
        self.labels = labels


def read_examples(input_file, debug=False, is_test=False, task="task1"):
    if is_test:
        with open(input_file, "r", encoding="utf8") as fr:
            items = json.load(fr)
        examples = []
        for index, item in enumerate(items):
            if task == "task1":
                text = item["context"]
            else:
                text = item["reason"] + item["context"]
            example = InputExamples(id=index, text=text, labels=0)

            examples.append(example)
    else:
        df = pd.read_csv(input_file, converters={"labels": eval})
        examples = []
        for index, row in df.iterrows():
            if task == "task1":
                text = row["context"]
            else:
                text = row["reason"] + row["context"]
            example = InputExamples(id=index, text=text, labels=int(row["judge"]))

            if debug and index >= 256:
                print(index, len(examples))
                break
            examples.append(example)

    return examples


def convert_examples_to_features(
    examples, tokenizer, max_length, padding, is_test=False
):
    features = []
    for example in tqdm(examples):
        if example.id % 500 == 0:
            logger.info(
                "Converting %s/%s",
                example.id,
                len(examples),
            )

        inputs = tokenizer(
            example.text, max_length=max_length, padding=padding, truncation=True
        )

        if not is_test:
            inputs["labels"] = example.labels

        if example.id < 10:
            logger.info("*** Example ***")
            logger.info("example.id: %s" % (example.id))
            logger.info("input_ids: %s" % " ".join([str(x) for x in inputs.input_ids]))
            if not is_test:
                logger.info("labels: %s" % inputs.labels)

        features.append(inputs)

    return features


def load_dataset(
    args,
    tokenizer,
    is_training=True,
    is_test=False,
):

    if is_training:
        input_file = args.train_file
        dataset_type = "train"
    else:
        input_file = args.validation_file
        dataset_type = "dev"

    if is_test:
        input_file = args.test_file
        dataset_type = "test"

    cached_features_file = os.path.join(
        os.path.dirname(input_file),
        "cached_features_{}".format(dataset_type),
    )
    cached_examples_file = os.path.join(
        os.path.dirname(input_file),
        "cached_examples_{}".format(dataset_type),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("从 %s 加载特征文件", cached_features_file)
        examples = torch.load(cached_examples_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("根据输入文件 %s 生成特征", input_file)
        examples = read_examples(
            input_file, debug=args.debug, is_test=is_test, task=args.task
        )
        features = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_length=args.max_length,
            padding="max_length" if args.pad_to_max_length else False,
            is_test=is_test,
        )

        logger.info("Saving examples into cached file %s", cached_examples_file)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(examples, cached_examples_file)
        torch.save(features, cached_features_file)

    dataset = DummyDataset(features, is_test=is_test)

    return dataset
