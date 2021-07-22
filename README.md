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
data/task1/train.csv
context,reason,judge
后来，在全行上下寻找“金穗”“的过程中，为了能吸引“金穗”“的注意，钱鹤鸣把小红莲亲属的来信和小红莲的照片贴到了办公室底的宣传栏上，看谁来取。他这时模模糊糊猜想可能是张培英。,空,0
在××戏院里，我看见住着几十位伤兵，中间有五六个重伤的兵士，或在腰上，或在腿上，中着炮弹；还有正在生病的。我们找他们的管事人，想商量一个办法，据说他安住在城下旅馆里。在戏院里的一角上，用两张椅子并起来，铺着一点稻草，一个面黄肌瘦的兵，裹着一条灰色的毯子，勉强撑起半截身子招呼我们，说他腿上受着重伤，而且又病了，睡在这儿冷得发抖，“能求你替我想想法子吗？”在他那双大而黑的眼睛里，带着失望与希求的神色，闪着晶莹的泪光。我们随即跑到医院里，请他们立刻教人去那儿检查，把重病的抬到医院里去。,空,0
```

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

