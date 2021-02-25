import os
import sys

sys.path.append("/src")

import toml
import tensorflow as tf
from tensorflow import keras

from utils import tf_utils
from model import MonoNet
from utils import helpers
from utils.dataset import load_dataset

tf.random.set_seed(1)


def train():
    model = MonoNet(cfg['model'])

    if cfg['std']['pretrain']:
        model.load_weights(cfg['std']['pretrain'])
        print('[info] pretrained weighted loaded from : {}'.format(cfg['std']['pretrain']))

    train_ds = load_dataset(cfg['std']['train_file'], cfg)
    val_ds = load_dataset(cfg['std']['val_file'], cfg, repeat=False)

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir='/src/logs/{}'.format(cfg['std']['log_code']), update_freq='epoch'),
        keras.callbacks.ModelCheckpoint(
            '/src/logs/{}/model/weights.ckpt'.format(cfg['std']['log_code']), save_weights_only=True, save_best_only=True)
    ]

    helpers.dump_config(cfg)
    tf_utils.print_summary(model, cfg)

    model.compile(
        keras.optimizers.Adam(learning_rate=cfg['model']['lr']),
        keras.optimizers.Adam(learning_rate=cfg['model']['lr']),
        eager=True
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=callbacks,
        epochs=cfg['model']['epochs'],
        steps_per_epoch=2975
    )

if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    try:
        cfg = toml.load(sys.argv[1])
    except:
        raise ValueError("No config file passed.")

    train()
