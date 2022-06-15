# -*- encoding: utf-8 -*-
'''
@Time    :   2022/06/10 15:51:44
@Author  :   Chu Xiaokai
@Contact :   xiaokaichu@gmail.com
'''
import math
import torch
import torch.nn.functional as F
import os
import random
from torch.utils.data import Dataset, DataLoader, IterableDataset
import gzip
from functools import reduce
from args import config

# --------------- data process for  masked language modeling (MLM) ----------------  #
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

def mask_data(seq, mask_ignore_token_ids=[config._CLS_, config._SEP_, config._PAD_],
                   mask_token_id=config._MASK_,
                   mask_prob=0.1,
                   pad_token_id=config._PAD_,
                   replace_prob=1.0
    ):
    no_mask = mask_with_tokens(seq, mask_ignore_token_ids)
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)
    masked_seq = seq.clone()
    labels = seq.masked_fill(~mask, pad_token_id)  # use pad to fill labels
    replace_prob = prob_mask_like(seq, replace_prob)
    mask = mask * replace_prob
    masked_seq = masked_seq.masked_fill(mask, mask_token_id)
    return masked_seq, labels


# ----------------------  DataLoader ----------------------- #
def process_data(query, title, content, max_seq_len):
    """ process [query, title, content] into a tensor 
        [CLS] + query + [SEP] + title + [SEP] + content + [SEP] + [PAD]
    """
    data = [config._CLS_]
    segment = [0]

    data = data + [int(item) + 10 for item in query.split(b'\x01')] # query
    data = data + [config._SEP_]
    segment = segment + [0] * (len(query.split(b'\x01')) + 1)

    data = data + [int(item) + 10 for item in title.split(b'\x01')] # content
    data = data + [config._SEP_] # sep defined as 1
    segment = segment + [1] * (len(title.split(b'\x01')) + 1)

    data = data + [int(item) + 10 for item in content.split(b'\x01')] # content
    data = data + [config._SEP_]
    segment = segment + [1] * (len(content.split(b'\x01')) + 1)

    # padding 
    padding_mask = [False] * len(data)
    if len(data) < max_seq_len: 
        padding_mask += [True] * (max_seq_len - len(data))
        data += [config._PAD_] * (max_seq_len - len(data))
    else:
        padding_mask = padding_mask[:max_seq_len]
        data = data[:max_seq_len]

    # segment id
    if len(segment) < max_seq_len:
        segment += [1] * (max_seq_len-len(segment))
    else:
        segment = segment[:max_seq_len]

    padding_mask = torch.BoolTensor(padding_mask)
    data = torch.LongTensor(data)
    segment = torch.LongTensor(segment)
    return data, segment, padding_mask

    
class TrainDataset(IterableDataset):
    def __init__(self, directory_path, buffer_size=100000, max_seq_len=128):
        self.directory_path = directory_path
        self.buffer_size = buffer_size 
        self.files = os.listdir(self.directory_path)
        random.shuffle(self.files)
        self.cur_query = "#"
        self.max_seq_len = max_seq_len

    def __iter__(self):
        buffer = []
        for file in self.files:
            print('load file', file)
            if file[-3:] != '.gz' or file == 'part-00000.gz':  # part-00000.gz is for evaluation
                continue
            with gzip.open(os.path.join(self.directory_path, file), 'rb') as f:
                for line in f.readlines():
                    line_list = line.strip(b'\n').split(b'\t')
                    if len(line_list) == 3:  # new query 
                        self.cur_query = line_list[1]
                    elif len(line_list) > 6:  # urls 
                        position, title, content, click_label = line_list[0], line_list[2], line_list[3], line_list[5]
                        try:
                            src_input, segment, src_padding_mask = process_data(self.cur_query, title, content, self.max_seq_len)
                            buffer.append([src_input, segment, src_padding_mask, float(click_label)])
                        except:
                            pass
                    
                    if len(buffer) >= self.buffer_size:
                        random.shuffle(buffer)

                        for record in buffer:
                            yield record


class TestDataset(Dataset):

    def __init__(self, fpath, max_seq_len, data_type, buffer_size=300000):
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        
        if data_type == 'annotate':
            self.buffer, self.total_qids, self.total_labels, self.total_freqs = self.load_annotate_data(fpath)
        elif data_type == 'click':
            self.buffer, self.total_qids, self.total_labels = self.load_click_data(fpath)


    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        return self.buffer[index]

    def load_annotate_data(self, fpath):
        print('load annotated data from ', fpath)
        total_qids = []
        buffer = []
        total_labels = []
        total_freqs = []
        for line in open(fpath, 'rb'):
            line_list = line.strip(b'\n').split(b'\t')
            qid, query, title, content, label, freq = line_list
            if 0 <= int(freq) <= 2:  # high freq 
                freq = 0
            elif 3 <= int(freq) <= 6:  # mid freq 
                freq = 1
            elif 7 <= int(freq):  # tail
                freq = 2 
            total_qids.append(int(qid))
            total_labels.append(int(label))
            total_freqs.append(freq)
            src_input, src_segment, src_padding_mask = process_data(query, title, content, self.max_seq_len)
            buffer.append([src_input, src_segment, src_padding_mask])

        return buffer, total_qids, total_labels, total_freqs
    
    def load_click_data(self, fpath):
        print('load logged click data from ', fpath)
        with gzip.open(fpath, 'rb') as f:
            buffer = []
            total_qids = []
            total_labels = []
            cur_qids = 0
            for line in f.readlines():
                line_list = line.strip(b'\n').split(b'\t')
                if len(line_list) == 3:  # new query 
                    self.cur_query = line_list[1]
                    cur_qids += 1
                elif len(line_list) > 6:  # urls 
                    position, title, content, click_label = line_list[0], line_list[2], line_list[3], line_list[5]
                    try:
                        src_input, src_segment, src_padding_mask = process_data(self.cur_query, title, content, self.max_seq_len)
                        buffer.append([src_input, src_segment, src_padding_mask])
                        total_qids.append(cur_qids)
                        total_labels.append(int(click_label))
                    except:
                        pass
            
                if len(buffer) >= self.buffer_size:  # we use 300,000 click records for test
                    break
        
        return buffer, total_qids, total_labels


def build_feed_dict(data_batch):
    if len(data_batch) == 4:  # for training
        src, src_segment, src_padding_mask, click_label = data_batch
    elif len(data_batch) == 3:  # for validation
        src, src_segment, src_padding_mask = data_batch

    feed_dict = {
        'src': src,
        'src_segment': src_segment,
        'src_padding_mask': src_padding_mask,
    }

    if len(data_batch) == 4:
        click_label = click_label.numpy().reshape(-1, 10).T
        for i in range(10):
            feed_dict['label'+str(i)] = click_label[i]
    return feed_dict