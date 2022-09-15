#!/usr/bin/env python
# coding=utf-8
# File Name: evaluate.py
# Author: Lixin Zou
# Mail: zoulixin15@gmail.com
# Created Time: Tue Sep 13 23:21:03 2022

from baseline_model.utils.sys_tools import find_class
import torch
import numpy as np
import warnings
import sys
from metrics import *
from Transformer4Ranking.model import *
from dataloader import *
from args import config

warnings.filterwarnings('ignore')
print(config)
exp_settings = config.exp_settings

token_encoder = TransformerModel(
    ntoken=config.ntokens, 
    hidden=config.emb_dim, 
    nhead=config.nhead, 
    nlayers=config.nlayers, 
    dropout=config.dropout,
    mode='finetune'
)

method_str = exp_settings['method_name']
if method_str not in ['IPWrank', 'DLA', 'RegressionEM', 'PairDebias', 'NavieAlgorithm']:
    print("please choose a method in 'IPWrank', 'DLA', 'RegressionEM', 'PairDebias', 'NavieAlgorithm'")
    sys.exit()
model = find_class('baseline_model.learning_algorithm.'+method_str)\
                  (exp_settings=exp_settings, encoder_model=token_encoder)

if config.evaluate_model_path != "":
    # load model
    model.model.load_state_dict(torch.load(config.evaluate_model_path))

# load dataset
test_annotate_dataset = TestDataset(config.test_annotate_path, max_seq_len=config.max_seq_len, data_type='annotate')
test_annotate_loader = DataLoader(test_annotate_dataset, batch_size=config.eval_batch_size)
# evaluate
total_scores = []

for test_data_batch in test_annotate_loader:
    feed_input = build_feed_dict(test_data_batch)
    score = model.get_scores(feed_input)
    score = score.cpu().detach().numpy().tolist()
    total_scores += score

with open(config.result_path, "w") as f:
    f.writelines("\n".join(map(str, total_scores)))

