# space2021
space2021



# 数据
```bash
data
    -task1
        -train.csv
        -dev.csv
    -task2
        -train.csv
        -dev.csv
    -task3
        -dev.csv
raw_data
    -pack1
        -task1-dev.json
        -task1-train-with-answer.json
        -task2-dev.json
        -task2-train-with-answer.json
        -task3-dev.json
    -pack1.5
        -task1-dev-with-answer.json
        -task2-dev-with-answer.json
        -task3-dev-with-answer.json

# train.csv格式如下
# task1没有reason，因此处理的时候直接设置为“空”，task2格式一致，有reason。
data/task1/train.csv
context,reason,judge
后来，在全行上下寻找“金穗”“的过程中，为了能吸引“金穗”“的注意，钱鹤鸣把小红莲亲属的来信和小红莲的照片贴到了办公室底的宣传栏上，看谁来取。他这时模模糊糊猜想可能是张培英。,空,0
在××戏院里，我看见住着几十位伤兵，中间有五六个重伤的兵士，或在腰上，或在腿上，中着炮弹；还有正在生病的。我们找他们的管事人，想商量一个办法，据说他安住在城下旅馆里。在戏院里的一角上，用两张椅子并起来，铺着一点稻草，一个面黄肌瘦的兵，裹着一条灰色的毯子，勉强撑起半截身子招呼我们，说他腿上受着重伤，而且又病了，睡在这儿冷得发抖，“能求你替我想想法子吗？”在他那双大而黑的眼睛里，带着失望与希求的神色，闪着晶莹的泪光。我们随即跑到医院里，请他们立刻教人去那儿检查，把重病的抬到医院里去。,空,0

```
# logs
## task1
version1 表示 cls

version2 表示 mean

version3 表示 cls+rdrop 1.0

version4 表示 mean+rdrop 1.0

## task2
version1 表示 cls

version2 表示 mean

version3 表示 cls+rdrop 0.5

version4 表示 mean+rdrop 0.5

# train
```bash
# (1) 准备数据
# (2) 修改config.py里面配置
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
# (3) 然后train
python train.py
```

# predict

```bash
# (4) 预测
python predict.py
```

# results

|                                   | Task1          |          |            | Task2          |          |            | Task3     |         |         |
|-----------------------------------|----------------|----------|------------|----------------|----------|------------|-----------|---------|---------|
| 模型                                | 方法             | Accuracy | best epoch | 方法             | Accuracy | best epoch | Precision | Recall  | F1      |
| hfl/chinese-roberta-wwm-ext       | cls            | 0.6514   | epoch 15   | cls            | 0.7476   | epoch 11   | 0.5847    | 0.3942  | 0.4709  |
|                                   | mean           | 0.6737   | epoch 10   | mean           | 0.7299   | epoch 12   | 0.5798    | 0.4401  | 0.5004  |
|                                   | cls+rdrop=1.0  | 0.6303   | epoch 10   | cls+rdrop=0.5  | 0.7524   | epoch 12   | 0.5717    | 0.3607  | 0.4424  |
|                                   | mean+rdrop=1.0 | 0.6687   | epoch 17   | mean+rdrop=0.5 | 0.7246   | epoch 9    | 0.5707    | 0.4443  | 0.4996  |
| junnyu/roformer_chinese_base      | cls            | 0.6600   | epoch 14   | cls            | 0.7768   | epoch 17   | 0.5857    | 0.4234  | 0.4915  |
|                                   | mean           | 0.6663   | epoch 10   | mean           | 0.7720   | epoch 16   | 0.6061    | 0.4694  | 0.5290  |
|                                   | cls+rdrop=1.0  | 0.6873   | epoch 7    | cls+rdrop=0.5  | 0.7514   | epoch 5    | 0.5830    | 0.4694  | 0.5201  |
|                                   | mean+rdrop=1.0 | 0.6849   | epoch 15   | mean+rdrop=0.5 | 0.7371   | epoch 8    | 0.6310    | 0.4930  | 0.5536  |
| junnyu/ChineseBERT-base           | cls            | 0.6911   | epoch 12   | cls            | 0.7486   | epoch 13   | 0.5828    | 0.4708  | 0.5208  |
|                                   | mean           | 0.6849   | epoch 12   | mean           | 0.7648   | epoch 4    | 0.6387    | 0.4777  | 0.5466  |
|                                   | cls+rdrop=1.0  | 0.6749   | epoch 16   | cls+rdrop=0.5  | 0.7658   | epoch 18   | 0.5891    | 0.4652  | 0.5198  |
|                                   | mean+rdrop=1.0 | 0.6824   | epoch 13   | mean+rdrop=0.5 | 0.7557   | epoch 16   | 0.6438    | 0.4833  | 0.5521  |
| junnyu/roformer_chinese_char_base | cls            | 0.6576   | epoch 8    | cls            | 0.7749   | epoch 10   | 0.6411    | 0.4081  | 0.4987  |
|                                   | mean           | 0.6749   | epoch 10   | mean           | 0.7778   | epoch 8    | 0.6039    | 0.4735  | 0.5308  |
|                                   | cls+rdrop=1.0  | 0.6985   | epoch 17   | cls+rdrop=0.5  | 0.7720   | epoch 16   | 0.5938    | 0.5070  | 0.5470  |
|                                   | mean+rdrop=1.0 | 0.6563   | epoch 11   | mean+rdrop=0.5 | 0.7754   | epoch 16   | 0.6098    | 0.4485  | 0.5169  |
| junnyu/wobert_chinese_plus_base   | cls            | 0.6725   | epoch 17   | cls            | 0.7409   | epoch 10   | 0.5930    | 0.4972  | 0.5409  |
|                                   | mean           | 0.6551   | epoch 16   | mean           | 0.7682   | epoch 7    | 0.5506    | 0.5000  | 0.5241  |
|                                   | cls+rdrop=1.0  | 0.6687   | epoch 6    | cls+rdrop=0.5  | 0.7500   | epoch 18   | 0.5707    | 0.4721  | 0.5168  |
|                                   | mean+rdrop=1.0 | 0.6873   | epoch 12   | mean+rdrop=0.5 | 0.7500   | epoch 19   | 0.5846    | 0.4666  | 0.5190  |
| hfl/chinese-roberta-wwm-ext-large | cls            | 0.6873   | epoch 5    | cls            | 0.7783   | epoch 19   | 0.6343    | 0.5460  | 0.5868  |
|                                   | mean           | 0.6898   | epoch 9    | mean           | 0.7725   | epoch 16   | 0.6305    | 0.5181  | 0.5688  |
|                                   | cls+rdrop=1.0  | 0.6700   | epoch 19   | cls+rdrop=0.5  | 0.7763   | epoch 9    | 0.5861    | 0.5167  | 0.5492  |
|                                   | mean+rdrop=1.0 | 0.6861   | epoch 12   | mean+rdrop=0.5 | 0.7749   | epoch 19   | 0.5867    | 0.5279  | 0.5557  |
| junyu/uer_large                   | cls            | 0.6923   | epoch 8    | cls            | 0.7969   | epoch 13   | 0.6104    | 0.4889  | 0.5429  |
|                                   | mean           | 0.6675   | epoch 12   | mean           | 0.7941   | epoch 14   | 0.5815    | 0.4471  | 0.5055  |
|                                   | cls+rdrop=1.0  | 0.6836   | epoch 13   | cls+rdrop=0.5  | 0.7854   | epoch 15   | 0.6057    | 0.5028  | 0.5495  |
|                                   | mean+rdrop=1.0 | 0.6799   | epoch 19   | mean+rdrop=0.5 | 0.7773   | epoch 7    | 0.5690    | 0.5627  | 0.5658  |
