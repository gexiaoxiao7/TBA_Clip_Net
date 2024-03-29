import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()
# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.ROOT = ''
_C.DATA.TRAIN_FILE = ''
_C.DATA.VAL_FILE = ''
_C.DATA.DATASET = 'hmdb51'
_C.DATA.NUM_FRAMES = 30
_C.DATA.INPUT_SIZE = 224
_C.DATA.NUM_CLASSES = 51
_C.DATA.LABEL_LIST = 'labels/hmdb51_org_base_labels.csv'
_C.DATA.IF_TEACHER = 1
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.ARCH = 'ViT-B/32'
_C.MODEL.YOLO = 'Yolo-model/yolov8n.pt'
_C.MODEL.OUTPUT = 'models'
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
TRAINER = True
_C.TRAIN.EPOCHS = 30
_C.TRAIN.IF_PRETRAINED = 1
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.001
_C.TRAIN.LR = 8.e-6
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.ACCUMULATION_STEPS = 1
_C.TRAIN.LR_SCHEDULER = 'cosine'
_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.OPT_LEVEL = 'O1'
_C.TRAIN.AUTO_RESUME = 0
_C.TRAIN.USE_CHECKPOINT = 0

# -----------------------------------------------------------------------------
# Trainer settings
# -----------------------------------------------------------------------------
_C.TRAINER = CN()
_C.TRAINER.TRANS_FRAMES = 1
_C.TRAINER.SAVE_FREQ = 10
_C.TRAINER.PRINT_FREQ = 5
# -----------------------------------------------------------------------------
# Output settings
# -----------------------------------------------------------------------------
_C.OUTPUT = ''

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    _update_config_from_file(config, args.config)

    config.defrost()
    if args.batch_size is not None:
        config.TRAIN.BATCH_SIZE = args.batch_size
    if args.if_teacher is not None:
        config.DATA.IF_TEACHER = args.if_teacher
    if args.num_frames is not None:
        config.DATA.NUM_FRAMES = args.num_frames
    if args.arch is not None:
        config.MODEL.ARCH = args.arch
    if args.trans_frames is not None:
        config.TRAINER.TRANS_FRAMES = args.trans_frames
    if args.output is not None:
        config.OUTPUT= args.output
    # set local rank for distributed training
    # config.LOCAL_RANK = args.local_rank
    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)
    return config