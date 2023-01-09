import os
from types import SimpleNamespace


cfg = SimpleNamespace(**{})

# data path
cfg.data_dir = "/raid/rsna/"
cfg.fold = 0

cfg.test_df = cfg.data_dir + "sample_submission.csv"
cfg.output_dir = "./output/weights/"

# dataset
cfg.batch_size = 4
cfg.val_batch_size = 64
cfg.img_size = (256, 256)
cfg.train_cache_rate = 0.0
cfg.val_cache_rate = 0.0

# training
cfg.lr = 3e-4
cfg.min_lr = 1e-5
cfg.weight_decay = 0
cfg.epochs = 15
cfg.seed = -1
cfg.eval_epochs = 1
cfg.start_cal_metric_epoch = 1
cfg.warmup = 1

# ressources
cfg.mixed_precision = True
cfg.gpu = 0
cfg.device = "cuda:%d" % cfg.gpu
cfg.num_workers = 8
cfg.weights = None

basic_cfg = cfg
