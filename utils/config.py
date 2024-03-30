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
_C.DATA.TEST_FILE = ''
_C.DATA.DATASET = 'hmdb51'
_C.DATA.NUM_FRAMES = 30
_C.DATA.INPUT_SIZE = 224
_C.DATA.NUM_CLASSES = 51
_C.DATA.LABEL_LIST = 'labels/hmdb51_org_base_labels.csv'
_C.DATA.IF_TEACHER = 1
_C.DATA.SHOTS = 2
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.ARCH = 'ViT-B/32'
_C.MODEL.YOLO = 'Yolo-model/yolov8n.pt'
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
TRAINER = True
_C.TRAIN.EPOCHS = 30
_C.TRAIN.IF_TEST = 1
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
# Tip-adapter settings
# -----------------------------------------------------------------------------
_C.TIP_ADAPTER = CN()
_C.TIP_ADAPTER.CACHE_DIR = ''
_C.TIP_ADAPTER.LOAD_CACHE = 0
_C.TIP_ADAPTER.AUGMENT_EPOCH = 10
_C.TIP_ADAPTER.INIT_BETA = 1
_C.TIP_ADAPTER.INIT_ALPHA = 3
_C.TIP_ADAPTER.LOAD_PRE_FEAT = 0
# -----------------------------------------------------------------------------
# Output settings
# -----------------------------------------------------------------------------
_C.OUTPUT = ''
_C.SEARCH_HP = 1
_C.SEARCH_SCALE = [7,3]
_C.SEARCH_STEP = [200,20]
_C.PREFIX = ''
_C.SUFFIX = ''
_C.TEMPORAL_POOLING = ''
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