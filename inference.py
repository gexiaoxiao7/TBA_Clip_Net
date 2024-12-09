from utils.logger import create_logger
import argparse
import torch
import yaml


def load_model(cfg):
    with open(cfg) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    prompt_learner = torch.load(config['prompt_learner'])
    attention_model = torch.load(config['attention_model'])
    cache_keys = torch.load(config['cache_keys'])
    cache_values = torch.load(config['cache_values'])
    tip_adapter = torch.load(config['tip_adapter'])

    return prompt_learner, attention_model, cache_keys, cache_values, tip_adapter



def main(opt):
    prompt_learner, attention_model, cache_keys, cache_values, tip_adapter = load_model(opt.config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/inference.yaml')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--prompt_learner', '-pl', type=str)
    parser.add_argument('--attention_model', '-at', type=str)
    parser.add_argument('--cache_keys', '-ck', type=str)
    parser.add_argument('--cache_values', '-cv', type=str)
    parser.add_argument('--tip_adapter', '-ta', type=str)
    opt = parser.parse_args()

    logger = create_logger(output_dir='train_output', dist_rank=0, name=f"inference")
    logger.info(opt)

    main(opt)