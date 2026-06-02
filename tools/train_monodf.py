
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime
import torch
import copy

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger, set_random_seed
from lib.helpers.save_helper import load_checkpoint


def da_init(model, model_cfg, seed, logger):
    set_random_seed(seed)
    ref_cfg = copy.deepcopy(model_cfg)
    ref_cfg['use_ogm'] = False
    ref_cfg['use_gqr'] = False
    ref_cfg['enable_geometry_output'] = False

    ref_model, _ = build_model(ref_cfg)
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    ref_state = ref_model.state_dict()
    target_state = target.state_dict()

    copied = []
    skipped = []
    for key, value in ref_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key].copy_(value)
            copied.append(key)
        else:
            skipped.append(key)
    target.load_state_dict(target_state, strict=False)


def parse_args():
    parser = argparse.ArgumentParser(description='MonoDF training entry')
    parser.add_argument('--config', dest='config', default='configs/monodf.yaml',
                        help='YAML config path')
    parser.add_argument('-e', '--evaluate_only', action='store_true', default=False,
                        help='evaluation only')
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint path to resume from (optimizer + epoch + best result restored)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='if set, all artifacts go to outputs/<model_name>/experiments/<exp_name>/')
    return parser.parse_args()


def setup_output_dir(cfg, model_name, exp_name):
    base_dir = os.path.join('./' + cfg['trainer']['save_path'], model_name)
    if exp_name:
        output_dir = os.path.join(base_dir, 'experiments', exp_name)
    else:
        output_dir = base_dir
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    args = parse_args()
    assert os.path.exists(args.config), f'config not found: {args.config}'
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))
    model_name = cfg['model_name']
    output_dir = setup_output_dir(cfg, model_name, args.exp_name)
    config_snapshot = os.path.join(output_dir, 'config.yaml')
    if os.path.abspath(args.config) != os.path.abspath(config_snapshot):
        shutil.copyfile(args.config, config_snapshot)

    log_file = os.path.join(output_dir, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)
    logger.info('Output dir: %s' % output_dir)

    train_loader, test_loader = build_dataloader(cfg['dataset'])

    model, criterion = build_model(cfg['model'])
    if bool(cfg['model'].get('da_init', False)):
        da_init(model, cfg['model'], cfg.get('random_seed', 444), logger)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_ids = list(map(int, str(cfg['trainer']['gpu_ids']).split(',')))
    if len(gpu_ids) == 1:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)

    if args.evaluate_only:
        tester = Tester(cfg=cfg['tester'], model=model, dataloader=test_loader,
                        logger=logger, train_cfg=cfg['trainer'], model_name=model_name)
        tester.output_dir = output_dir
        tester.test()
        return

    optimizer = build_optimizer(cfg['optimizer'], model)
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg['trainer'], model=model, optimizer=optimizer,
                      train_loader=train_loader, test_loader=test_loader,
                      lr_scheduler=lr_scheduler, warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger, loss=criterion, model_name=model_name)
    trainer.output_dir = output_dir

    if args.resume:
        assert os.path.exists(args.resume), f'resume ckpt not found: {args.resume}'
        epoch, best_result, best_epoch = load_checkpoint(
            model=model.to(device),
            optimizer=optimizer,
            filename=args.resume,
            map_location=device,
            logger=logger)
        trainer.epoch = epoch
        trainer.best_result = best_result
        trainer.best_epoch = best_epoch
        lr_scheduler.last_epoch = epoch - 1
        logger.info('Resumed from %s (epoch=%d, best=%.4f@%d)' % (args.resume, epoch, best_result, best_epoch))

    tester = Tester(cfg=cfg['tester'], model=trainer.model, dataloader=test_loader,
                    logger=logger, train_cfg=cfg['trainer'], model_name=model_name)
    tester.output_dir = output_dir
    if cfg['dataset']['test_split'] != 'test':
        trainer.tester = tester

    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))
    trainer.train()

    if cfg['dataset']['test_split'] == 'test':
        return

    logger.info('###################  Testing  ##################')
    tester.test()


if __name__ == '__main__':
    main()
