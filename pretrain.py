# -*- encoding: utf-8 -*-
'''
@Time    :   2022/06/10 15:51:44
@Author  :   Chu Xiaokai
@Contact :   xiaokaichu@gmail.com
'''
import time
import sys
import os
sys.path.append(os.getcwd())
from dataloader import *
from Transformer4Ranking.model import *
import torch
from torch import nn
from torch.utils.data import  DataLoader 
from metrics import evaluate_all_metric
from args import config

# control seed
# 生成随机数，以便固定后续随机数，方便复现代码
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

# load dataset 
train_dataset = TrainDataset(config.train_datadir, max_seq_len=config.max_seq_len, buffer_size=config.buffer_size)
train_data_loader = DataLoader(train_dataset, batch_size=config.train_batch_size)
vaild_annotate_dataset = TestDataset(config.valid_annotate_path, max_seq_len=config.max_seq_len, data_type='annotate')
vaild_annotate_loader = DataLoader(vaild_annotate_dataset, batch_size=config.eval_batch_size) 
vaild_click_dataset = TestDataset(config.valid_click_path, max_seq_len=config.max_seq_len, data_type='click', buffer_size=100000)
vaild_click_loader = DataLoader(vaild_click_dataset, batch_size=config.eval_batch_size) 

model = TransformerModel(
    ntoken=config.ntokens, 
    hidden=config.emb_dim, 
    nhead=config.nhead, 
    nlayers=config.nlayers, 
    dropout=config.dropout,
    mode='pretrain'
)

if torch.cuda.device_count() >= config.n_gpus > 1:
    print("Let's use", config.n_gpus, "GPUs!")
    model = nn.DataParallel(model, device_ids = list(range(config.n_gpus)))
model.cuda()

# load pretrained model
if config.init_parameters != "":
    print('load warm up model ', config.init_parameters)
    if config.n_gpus > 1:
        model.load_state_dict(torch.load(config.init_parameters))
    else:
        print(torch.load(config.init_parameters).keys())
        model.load_state_dict(torch.load(config.init_parameters), strict=False)

# init optimization
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scheduler = get_linear_schedule_with_warmup(
                optimizer, \
                num_warmup_steps=config.warmup_steps,\
                num_training_steps=config.max_steps,
            )

# train model
model.train()  # turn on train mode
log_interval = config.log_interval
total_loss = 0
total_ctr_loss = 0.0
total_mlm_loss = 0.0 
start_time = time.time()
criterion = nn.BCEWithLogitsLoss()

idx = 0
for src_input, src_segment, src_padding_mask, click_label in train_data_loader:
    model.train()
    optimizer.zero_grad()
    masked_src_input, mask_label = mask_data(src_input)

    score, mlm_loss = model(
        src=masked_src_input,   # mask data
        src_segment=src_segment, 
        src_padding_mask=src_padding_mask, 
        mlm_label=mask_label, 
    )
            
    mlm_loss = torch.mean(mlm_loss)
    click_label = click_label.cuda()
    ctr_loss = criterion(score, click_label)

    loss = mlm_loss + ctr_loss

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) 
    loss.backward()
    optimizer.step()
    scheduler.step()
    total_ctr_loss += ctr_loss.item()
    total_mlm_loss += mlm_loss.item()
    total_loss += loss.item()

    # log time
    if idx % log_interval == 0:
        lr = scheduler.get_last_lr()[0]
        ms_per_batch = (time.time() - start_time) * 1000 / log_interval
        cur_loss = total_loss / log_interval
        cur_ctr_loss = total_ctr_loss / log_interval
        cur_mlmloss = total_mlm_loss / log_interval
        print(
            f'{idx:5d}th step | '
            f'lr {lr:.3e} | ms/batch {ms_per_batch:5.2f} | '
            f'ctr {cur_ctr_loss:5.5f} | mlm {cur_mlmloss:5.5f}')
        total_mlm_loss = 0
        total_ctr_loss = 0
        total_loss = 0
        start_time = time.time()

    # evaluate
    if idx % config.eval_step == 0:
        all_ndcg_list = []
        model.eval()

        # ------------   evaluate on annotated data -------------- # 
        total_scores = []
        for src_input, src_segment, src_padding_mask in vaild_annotate_loader:
            score = model(src=src_input, src_segment=src_segment, src_padding_mask=src_padding_mask).cpu().detach().numpy().tolist()
            total_scores += score

        result_dict_ann = evaluate_all_metric(
            qid_list=vaild_annotate_dataset.total_qids, 
            label_list=vaild_annotate_dataset.total_labels, 
            score_list=total_scores, 
            freq_list=vaild_annotate_dataset.total_freqs
        )
        print(
            f'{idx}th step valid annotate | '
            f'dcg@10: all {result_dict_ann["all_dcg@10"]:.5f} | '
            f'high {result_dict_ann["high_dcg@10"]:.5f} | '
            f'mid {result_dict_ann["mid_dcg@10"]:.5f} | '
            f'low {result_dict_ann["low_dcg@10"]:.5f} | '
            f'pnr {result_dict_ann["pnr"]:.5f}'
        )

        # ------------   evaluate on click data -------------- # 
        total_scores = []
        for src_input, src_segment, src_padding_mask in vaild_click_loader:
            score = model(src=src_input, src_segment=src_segment, src_padding_mask=src_padding_mask).cpu().detach().numpy().tolist()
            total_scores += score
        result_dict_click = evaluate_all_metric(
            qid_list=vaild_click_dataset.total_qids, 
            label_list=vaild_click_dataset.total_labels, 
            score_list=total_scores, 
            freq_list=None
        )
        print(
            f'{idx}th step valid click |'
            f'dcg@3 {result_dict_click["all_dcg@3"]:.5f} | '
            f'dcg@5 {result_dict_click["all_dcg@5"]:.5f} | '
            f'dcg@10 {result_dict_click["all_dcg@10"]:.5f} | '
            f'pnr {result_dict_click["pnr"]:.5f}'
        )

        if idx % config.save_step == 0 and idx > 0:
            torch.save(model.state_dict(), 
                      'save_model/save_steps{}_{:.5f}_{:5f}.model'.format(idx, result_dict_ann['pnr'], result_dict_click['pnr'])
            )
    idx += 1