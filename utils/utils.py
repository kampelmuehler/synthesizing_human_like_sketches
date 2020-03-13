import sys
import datetime
import logging
import shutil
from os.path import join, isdir, isfile
from os import makedirs
import torch
import torch.nn as nn


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


# from https://github.com/naoto0804/pytorch-AdaIN
def calc_mean_std(feat, eps=1e-5, device='cuda:0'):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


# from https://github.com/naoto0804/pytorch-AdaIN
def adaptive_instance_normalization_mean_std(content_feat, style_mean, style_std):
    content_mean, content_std = calc_mean_std(content_feat)
    # double unsqueeze to match dimensions
    style_mean = style_mean[(..., None, None)]
    style_std = style_std[(..., None, None)]
    normalized_feat = (content_feat - content_mean.expand_as(content_feat)) / content_std.expand_as(content_feat)
    return normalized_feat * style_std.expand_as(content_feat) + style_mean.expand_as(content_feat)


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Source: http://code.activestate.com/recipes/577058/
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError(f"invalid default answer: {default}")

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def get_logger(outpath, test=False, prefix=''):
    outpath = f'{outpath}'
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                join(outpath, 'logs' if test is False else '',
                     f'{"testresults_" if test is True else ""}'
                     f'{f"{prefix}_" if prefix != "" else ""}'
                     f'{datetime.datetime.now().replace(microsecond=0).isoformat()}.log')),
            logging.StreamHandler(sys.stdout)
        ])
    return logging.getLogger('sketch_generator' if test is False else 'sketch_generator_test')


def setup_output(outpath, overwrite_protection=True):
    outpath = f'{outpath}'
    # make sure not to overwrite any past experiments
    if isdir(outpath):
        if overwrite_protection is True:
            overwrite = query_yes_no(f'Output path {outpath} exists. Overwrite?', default='no')
        else:
            overwrite = True
        if overwrite is True:
            shutil.rmtree(outpath)
        else:
            print(f'Output path {outpath} exists. No overwrite requested. Aborting')
            sys.exit(-1)

    # create directories for output
    makedirs(join(outpath, 'logs'))
    makedirs(join(outpath, 'tensorboard'))
    makedirs(join(outpath, 'models'))


def listGPUs():
    # (IDs are not consistent with nvidia-smi which lists ind PCI_BUS_ORDER
    # but default is FASTEST_FIRST which makes more sense)
    print(f'{torch.cuda.device_count()} GPU(s) available:')
    for gpu in range(torch.cuda.device_count()):
        print(f'\tGPU {gpu}: {torch.cuda.get_device_name(gpu)}')


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1)).view(in_feat.size()[0], 1, in_feat.size()[2],
                                                                  in_feat.size()[3])
    return in_feat / (norm_factor.expand_as(in_feat) + eps)


def cos_sim(in0, in1):
    in0_norm = normalize_tensor(in0)
    in1_norm = normalize_tensor(in1)
    N = in0.size()[0]
    X = in0.size()[2]
    Y = in0.size()[3]

    return torch.mean(torch.mean(torch.sum(in0_norm * in1_norm, dim=1).view(N, 1, X, Y), dim=2).view(N, 1, 1, Y),
                      dim=3).view(N)


def correct_topk(output_classifier, target_classifier, maxk):
    _, pred_topk = output_classifier.topk(maxk, 1, True, True)
    pred_topk = pred_topk.t()
    correct_topk = pred_topk.eq(target_classifier.view(1, -1).expand_as(pred_topk))
    return correct_topk.sum().item()
