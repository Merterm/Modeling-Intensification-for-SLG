# coding: utf-8
"""
Collection of helper functions
"""
import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
from logging import Logger
from typing import Callable, Optional, List
import numpy as np

import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
import gzip
import pickle
from torchtext.data import Dataset
import yaml
from vocabulary import Vocabulary
from sklearn.preprocessing import MinMaxScaler
from dtw import dtw
import sys
# sys.path.append("/home/carla/MyProjects/SLP-FE-Linguistics/SLT")

# from SLT.signjoey.vocabulary import build_vocab
# from SLT.signjoey.model import build_model
# from SLT.signjoey.prediction import test

class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """

def make_model_dir(model_dir: str, overwrite=False, model_continue=False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :param model_continue: whether to continue from a checkpoint
    :return: path to model directory
    """
    # If model already exists
    if os.path.isdir(model_dir):

        # If model continuing from checkpoint
        if model_continue:
            # Return the model_dir
            return model_dir

        # If set to not overwrite, this will error
        if not overwrite:
            raise FileExistsError(
                "Model directory exists and overwriting is disabled.")

        # If overwrite, recursively delete previous directory to start with empty dir again
        for file in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        shutil.rmtree(model_dir, ignore_errors=True)

    # If model directly doesn't exist, make it and return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    fh = logging.FileHandler(
        "{}/{}".format(model_dir, log_file))
    fh.setLevel(level=logging.DEBUG)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logging.getLogger("").addHandler(sh)
    logger.info("Progressive Transformers for End-to-End SLP")
    return logger


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg") -> None:
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')

    return torch.from_numpy(mask) == 0 # Turns it into True and False's

# Subsequent mask of two sizes
def uneven_subsequent_mask(x_size: int, y_size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, x_size, y_size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0  # Turns it into True and False's

def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")


def get_latest_checkpoint(ckpt_dir, post_fix="_every" ) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory, of either every validation step or best
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir: directory of checkpoint
    :param post_fixe: type of checkpoint, either "_every" or "_best"

    :return: latest checkpoint file
    """
    # Find all the every validation checkpoints
    list_of_files = glob.glob("{}/*{}.ckpt".format(ckpt_dir,post_fix))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location='cuda' if use_cuda else 'cpu')
    return checkpoint

def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

# Find the best timing match between a reference and a hypothesis, using DTW
def calculate_dtw(references, hypotheses):
    """
    Calculate the DTW costs between a list of references and hypotheses

    :param references: list of reference sequences to compare against
    :param hypotheses: list of hypothesis sequences to fit onto the reference

    :return: dtw_scores: list of DTW costs
    """
    # Euclidean norm is the cost function, difference of coordinates
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    dtw_scores = []

    # Remove the BOS frame from the hypothesis
    hypotheses = hypotheses[:, 1:]

    # For each reference in the references list
    for i, ref in enumerate(references):
        # Cut the reference down to the max count value
        _ , ref_max_idx = torch.max(ref[:, -1], 0)
        if ref_max_idx == 0: ref_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        ref_count = ref[:ref_max_idx,:-1].cpu().numpy()

        # Cut the hypothesis down to the max count value
        hyp = hypotheses[i]
        _, hyp_max_idx = torch.max(hyp[:, -1], 0)
        if hyp_max_idx == 0: hyp_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        hyp_count = hyp[:hyp_max_idx,:-1].cpu().numpy()

        # Calculate DTW of the reference and hypothesis, using euclidean norm
        d, cost_matrix, acc_cost_matrix, path = dtw(ref_count, hyp_count, dist=euclidean_norm)

        # Normalise the dtw cost by sequence length
        d = d/acc_cost_matrix.shape[0]

        dtw_scores.append(d)

    # Return dtw scores and the hypothesis with altered timing
    return dtw_scores

def calculate_backtranslation_BLUE(hypotheses, config, logger, type):

    slt_cfg_file = config["slt"]["config"]
    slt_ckpt = config["slt"]["ckpt"]
    slt_cfg = load_config(slt_cfg_file)
    slt_model = slt_cfg["training"]["model_dir"].split('/')[-1]
    output_path = os.path.join(config["training"]["model_dir"], 'SLT_output')

    # Save data in skels format
    save_predictions(config, hypotheses, type)

    # Save Test data in slt format
    data_path = config["data"][type]
    extension = ".pred." + config["data"]["trg"] + "_TG2S"
    feature_size = slt_cfg["data"]["feature_size"]
    output_file = os.path.join(os.path.abspath(output_path), slt_model) + '.' + type

    # save best validation results in SLT format
    try:
        output_file_dev = os.path.join(os.path.abspath(output_path), slt_model) + '.' + 'dev'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        save_pred_slt_format(config["data"]["dev"], output_file_dev, extension, feature_size)
    except:
        print('no saved skeletons from dev set')


    save_pred_slt_format(data_path, output_file, extension, feature_size)
    
    # Compute Blue Scores
    test(slt_cfg_file, slt_ckpt, data_path, output_file,output_path, logger)


def save_predictions(config, hypotheses, type):
        if type == "val":
            type = "dev"

        with open(config["data"][type]+ ".pred." + config["data"]["trg"] + "_TG2S", "w") as out_skels:
            for sample in hypotheses:

                for frame in sample:
                    line = " ".join(str('{:.5f}'.format(item)) for item in frame.tolist())
                    out_skels.write(line)
                    out_skels.write(' ')
                out_skels.write('\n')

def create_dataset_file(annotations, filename):
    with gzip.open(filename, "wb") as f:
        pickle.dump(annotations, f)

def save_pred_slt_format(data_path, output_file, extension, feature_size):

    text_file = data_path + '.text'
    gloss_file = data_path +  '.gloss'
    skels_file = data_path + extension
    files_file = data_path + '.files'

    annotations = []

    with open(text_file, 'r') as t_f, \
        open(gloss_file, 'r') as g_f, \
        open(skels_file, 'r') as s_f, \
        open(files_file, 'r') as f_f:
        text = t_f.readlines()
        gloss = g_f.readlines()
        skels = s_f.readlines()
        files = f_f.readlines()

        reps = []
        for i in range(len(text)):
            temp = {"name": files[i].strip('\n'), "signer": "Signer00", "gloss": gloss[i].strip('\n'),
                    "text": text[i].strip('\n')}
            video_skels = skels[i].strip()
            temp_skels = [float(num) for num in video_skels.split(' ')]
            temp_skels_array = np.array(temp_skels).reshape(-1, int(feature_size)+1)
            temp["sign"] = torch.Tensor(temp_skels_array[:,0:-1])
            if temp["name"] not in reps:
                annotations.append(temp)
                reps.append(temp["name"])



    create_dataset_file(annotations, output_file)

# def load_slt_model(slt_cfg_file, slt_ckpt):

#     cfg = load_config(slt_cfg_file)

#     gls_vocab_file = cfg.get("gls_vocab", None)
#     txt_vocab_file = cfg.get("txt_vocab", None)

#     gls_vocab = build_vocab(
#         field="gls",
#         min_freq=0,
#         max_size=3000,
#         dataset=None,
#         vocab_file=gls_vocab_file,
#     )
#     txt_vocab = build_vocab(
#         field="txt",
#         min_freq=0,
#         max_size=3000,
#         dataset=None,
#         vocab_file=txt_vocab_file,
#     )
#     do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
#     do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0

#     model = build_model(
#         cfg=cfg["model"],
#         gls_vocab=gls_vocab,
#         txt_vocab=txt_vocab,
#         sgn_dim=sum(cfg["data"]["feature_size"])
#         if isinstance(cfg["data"]["feature_size"], list)
#         else cfg["data"]["feature_size"],
#         do_recognition=do_recognition,
#         do_translation=do_translation,
#     )

#     # load model state from disk
#     use_cuda = cfg["training"].get("use_cuda", False)
#     model_checkpoint = load_checkpoint(slt_ckpt, use_cuda=use_cuda)
#     model.load_state_dict(model_checkpoint["model_state"])

#     return model

def load_scaler(data_path):
    train_data = open(data_path, "r")
    num_list = [float(num) for num in train_data.read().split()]
    # create scaler
    scaler = MinMaxScaler(feature_range=(-1,1))
    # fit scaler on data
    scaler.fit(np.array(num_list).reshape(-1,1))
    return scaler