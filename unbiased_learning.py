# -*- encoding: utf-8 -*-
'''
@Time    :   2022/06/12 14:49:28
@Author  :   Chu Xiaokai 
@Contact :   xiaokaichu@gmail.com
'''
from baseline_model.utils.sys_tools import find_class
import torch
import numpy as np
import warnings
import sys
from metrics import *
from Transformer4Ranking.model import *
from dataloader import *
from args import config

random.seed(config.seed+1)
np.random.seed(config.seed+1)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
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

# load pretrained model
if config.init_parameters != "":
    print('load warm up model ', config.init_parameters)
    token_encoder.load_state_dict(torch.load(config.init_parameters))

method_str = exp_settings['method_name']
if method_str not in ['IPWrank', 'DLA', 'RegressionEM', 'PairDebias', 'NavieAlgorithm']:
    print("please choose a method in 'IPWrank', 'DLA', 'RegressionEM', 'PairDebias', 'NavieAlgorithm'")
    sys.exit()

model = find_class('baseline_model.learning_algorithm.'+method_str)\
                  (exp_settings=exp_settings, encoder_model=token_encoder)

train_dataset = TrainDataset(config.train_datadir, max_seq_len=config.max_seq_len, buffer_size=config.buffer_size)
train_data_loader = DataLoader(train_dataset, batch_size=config.train_batch_size)
vaild_annotate_dataset = TestDataset(config.valid_annotate_path, max_seq_len=config.max_seq_len, data_type='annotate')
vaild_annotate_loader = DataLoader(vaild_annotate_dataset, batch_size=config.eval_batch_size) 
vaild_click_dataset = TestDataset(config.valid_click_path, max_seq_len=config.max_seq_len, data_type='click', buffer_size=100000)
vaild_click_loader = DataLoader(vaild_click_dataset, batch_size=config.eval_batch_size) 

idx = 0
for train_batch in train_data_loader:
    loss = model.train(build_feed_dict(train_batch))  

    if idx % config.log_interval == 0:
        print(f'{idx:5d}th step | loss {loss:5.6f}')

    if idx % config.eval_step == 0:

        # ------------   evaluate on annotated data -------------- # 
        total_scores = []
        for test_data_batch in vaild_annotate_loader:
            feed_input = build_feed_dict(test_data_batch)
            score = model.get_scores(feed_input)
            score = score.cpu().detach().numpy().tolist()
            total_scores += score

        result_dict_ann = evaluate_all_metric(
            qid_list=vaild_annotate_dataset.total_qids, 
            label_list=vaild_annotate_dataset.total_labels, 
            score_list=total_scores, 
            freq_list=vaild_annotate_dataset.total_freqs
        )
        print(
            f'{idx}th step valid annotate | '
            f'dcg@10: all {result_dict_ann["all_dcg@10"]:.6f} | '
            f'high {result_dict_ann["high_dcg@10"]:.6f} | '
            f'mid {result_dict_ann["mid_dcg@10"]:.6f} | '
            f'low {result_dict_ann["low_dcg@10"]:.6f} | '
            f'pnr {result_dict_ann["pnr"]:.6f}'
        )

        # ------------   evaluate on click data -------------- # 
        total_scores = []
        for test_data_batch in vaild_click_loader:
            feed_input = build_feed_dict(test_data_batch)
            score = model.get_scores(feed_input)
            score = score.cpu().detach().numpy().tolist()
            total_scores += score

        result_dict_click = evaluate_all_metric(
            qid_list=vaild_click_dataset.total_qids, 
            label_list=vaild_click_dataset.total_labels, 
            score_list=total_scores, 
            freq_list=None
        )
        print(
            f'{idx}th step valid click | '
            f'dcg@3 {result_dict_click["all_dcg@3"]:.6f} | '
            f'dcg@5 {result_dict_click["all_dcg@5"]:.6f} | '
            f'dcg@10 {result_dict_click["all_dcg@10"]:.6f} | '
            f'pnr {result_dict_click["pnr"]:.6f}'
        )
        if idx % config.save_step == 0 and idx > 0:
            torch.save(model.state_dict(),
                      'save_model/save_steps{}_{:.5f}_{:5f}.model'.format(idx, result_dict_ann['pnr'], result_dict_click['pnr'])
            )
        idx += 1
