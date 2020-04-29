# https://machinetalk.org/2019/03/18/introduction-to-tensorflow-datasets/
# https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html



# downloaded datasets are loacated here: ~/tensorflow_datasets
# downloader wont redownload if there is a dataset already here....


import os
import time
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from pprint import pprint


import tensorflow as tf
import tensorflow_datasets as tfds
print(tf.__version__)


output_dir = "nmt"
en_vocab_file = os.path.join(output_dir, "en_vocab")
zh_vocab_file = os.path.join(output_dir, "zh_vocab")
checkpoint_path = os.path.join(output_dir, "checkpoints")
log_dir = os.path.join(output_dir, 'logs')
download_dir = "tensorflow-datasets/downloads"

if not os.path.exists(output_dir):
  os.makedirs(output_dir)


tmp_builder = tfds.builder("wmt19_translate/zh-en")
pprint(tmp_builder.subsets)



config = tfds.translate.wmt.WmtConfig(
  version=tfds.core.Version('0.0.3', experiments={tfds.core.Experiment.S3: False}),
  language_pair=("zh", "en"),
  subsets={
    tfds.Split.TRAIN: ["newscommentary_v14"]
  }
)
builder = tfds.builder("wmt_translate", config=config)
builder.download_and_prepare(download_dir=download_dir)


