import os
import os.path as osp
import yaml

def _check_dir(dir, make_dir=True):
    if not osp.exists(dir):
        if make_dir:
            print('Create directory {}'.format(dir))
            os.mkdir(dir)
        else:
            raise Exception('Directory not exist {}'.format(dir))

def get_train_config(config_file='config/train_config.yaml'):
    with open(config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)

    _check_dir(cfg['dataset']['data_root'], make_dir=False)
    _check_dir(cfg['ckpt_root'])

    return cfg


if __name__ == '__main__':
    cfg = get_train_config('train_config.yaml')
    print(cfg['dataset'])
    print(cfg['augment_data'])
