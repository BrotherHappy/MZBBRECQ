import torch
import torch.nn as nn
import argparse
import os
import random
import numpy as np
import time
import hubconf
from quant import *
from data.imagenet import build_imagenet_data
from quant.block_recon import compute_hamming_loss
from quant.adaptive_rounding import AdaRoundQuantizer
from loguru import logger

model = torch.load("MZBBRECQ/resnet50-W8-A8-brecqTrue-202404090131/qnn.pt")

def exam_quantized(model:torch.nn.Module,img=torch.randn(1,3,224,224)):
    model = model.cuda()
    img = img.cuda()
    ret_dict = dict()
    def get_hook(name):
        def hook(m:nn.Conv2d,i,o):
            x = i[0]
            d = dict()
            d['w_scale'] = m.weight_quantizer.delta
            d['weight'] = m.weight
            d['qweight'] = m.weight_fake_quant.int_repr(m.weight).round()
            ret_dict[name] = d
        return hook
    hooks = list()
    for name,m in model.named_modules():
        if isinstance(m,QuantModule):
            if isinstance(m.weight_quantizer,AdaRoundQuantizer):
                hooks.append(m.register_forward_hook(get_hook(name)))

    model(img)
    for hook in hooks:
        hook.remove()
    return ret_dict
    


if __name__ == "__main__":
    exam_quantized(model)

