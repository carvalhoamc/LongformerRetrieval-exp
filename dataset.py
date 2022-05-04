import sys
sys.path += ["./"]
import os
import math
import json
import torch
import pickle
import random
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from typing import List
import itertools
logger = logging.getLogger(__name__)
from star_tokenizer import RobertaTokenizer
import time

class TextTokenIdsCache:
    def __init__(self, data_dir, prefix):
        meta = json.load(open(f"{data_dir}/{prefix}_meta"))
        self.total_number = meta['total_number']
        self.max_seq_len = meta['embedding_size']
        try:
            self.ids_arr = np.memmap(f"{data_dir}/{prefix}.memmap", 
                shape=(self.total_number, self.max_seq_len), 
                dtype=np.dtype(meta['type']), mode="r")
            self.lengths_arr = np.load(f"{data_dir}/{prefix}_length.npy")
        except FileNotFoundError:
            self.ids_arr = np.memmap(f"{data_dir}/memmap/{prefix}.memmap", 
                shape=(self.total_number, self.max_seq_len), 
                dtype=np.dtype(meta['type']), mode="r")
            self.lengths_arr = np.load(f"{data_dir}/memmap/{prefix}_length.npy")
        assert len(self.lengths_arr) == self.total_number
        
    def __len__(self):
        return self.total_number
    
    def __getitem__(self, item):
        return self.ids_arr[item, :self.lengths_arr[item]]

class SequenceDataset(Dataset):
    def __init__(self, ids_cache, max_seq_length):
        self.ids_cache = ids_cache
        self.max_seq_length = max_seq_length
        
    def __len__(self):  
        return len(self.ids_cache)

    def __getitem__(self, item):
        input_ids = self.ids_cache[item].tolist()
        seq_length = min(self.max_seq_length-1, len(input_ids)-1)
        input_ids = [input_ids[0]] + input_ids[1:seq_length] + [input_ids[-1]]
        attention_mask = [1]*len(input_ids)

        ret_val = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": item,
        }
        return ret_val

class SubsetSeqDataset:
    def __init__(self, subset: List[int], ids_cache, max_seq_length):
        self.subset = sorted(list(subset))
        self.alldataset = SequenceDataset(ids_cache, max_seq_length)
        
    def __len__(self):  
        return len(self.subset)

    def __getitem__(self, item):
        return self.alldataset[self.subset[item]]

def load_rel(rel_path):
    reldict = defaultdict(list)
    for line in tqdm(open(rel_path), desc=os.path.split(rel_path)[1]):
        qid, _, pid, _ = line.split()
        qid, pid = int(qid), int(pid)
        reldict[qid].append((pid))
    return dict(reldict)

def load_rank(rank_path):
    rankdict = defaultdict(list)
    for line in tqdm(open(rank_path), desc=os.path.split(rank_path)[1]):
        qid, pid, _ = line.split()
        qid, pid = int(qid), int(pid)
        rankdict[qid].append(pid)
    return dict(rankdict)

def pack_tensor_2D(lstlst, default, dtype, length=None):
    batch_size = len(lstlst)
    length = length if length is not None else max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor

def get_collate_function(max_seq_length):
    cnt = 0
    def collate_function(batch):
        nonlocal cnt
        length = None
        if cnt < 10:
            length = max_seq_length
            cnt += 1

        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids, default=1, 
                dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0, 
                dtype=torch.int64, length=length),
        }
        ids = [x['id'] for x in batch]
        return data, ids
    return collate_function

class TrainInbatchDataset(Dataset):
    def __init__(self, rel_file, queryids_cache, docids_cache, 
            max_query_length, max_doc_length):
        self.query_dataset = SequenceDataset(queryids_cache, max_query_length)
        self.doc_dataset = SequenceDataset(docids_cache, max_doc_length)
        self.reldict = load_rel(rel_file)
        self.qids = sorted(list(self.reldict.keys()))

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid = self.qids[item]
        pid = random.choice(self.reldict[qid])
        query_data = self.query_dataset[qid]
        passage_data = self.doc_dataset[pid]
        return query_data, passage_data

class TrainInbatchWithHardDataset(TrainInbatchDataset):
    def __init__(self, rel_file, rank_file, map_file, queryids_cache,
            docids_cache, hard_num,
            max_query_length, max_doc_length):
        TrainInbatchDataset.__init__(self, 
            rel_file, queryids_cache, docids_cache, 
            max_query_length, max_doc_length)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
        self.map_dict = json.load(open(map_file))
        self.hierarchical_score = json.load(score_file)
        self.rankdict = json.load(open(rank_file))
        assert hard_num > 0
        self.hard_num = hard_num

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid = self.qids[item]
        pid = random.choice(self.reldict[qid])
        query_data = self.query_dataset[qid]
        passage_data = self.doc_dataset[pid]
        hardpids = random.sample(self.rankdict[str(qid)], self.hard_num)
        hard_passage_data = [self.doc_dataset[hardpid] for hardpid in hardpids]
        sentence_span_passage = self.map_dict[pid]['sentence']
        para_span_passage = self.map_dict[pid]['para']
        sentence_span_hard = [self.map_dict[hardpid]['sentence'] for hardpid in hardpids]
        para_span_hard = [self.map_dict[hardpid]['para'] for hardpid in hardpids]
        sentence_score = self.hierarchical_score[pid]['sentence']
        para_score = self.hierarchical_score[pid]['para']
        sentence_score_hn = [self.hierarchical_score[hardpid]['sentence'] for hardpid in hardpids]
        para_score_hn = [self.hierarchical_score[hardpid]['para'] for hardpid in hardpids]
        passage_graph = build_graph(self.map_dict[pid])
        hardneg_graph = [build_graph(self.map_dict[hardpid]) for hardpid in hardpids]
        return query_data, passage_data, hard_passage_data, sentence_span_passage, sentence_span_hard, para_span_passage, para_span_hard, \
               sentence_score, sentence_score_hn, para_score, para_score_hn, passage_graph, hardneg_graph

class TrainInbatchWithRandDataset(TrainInbatchDataset):
    def __init__(self, rel_file, queryids_cache, 
            docids_cache, rand_num,
            max_query_length, max_doc_length):
        TrainInbatchDataset.__init__(self, 
            rel_file, queryids_cache, docids_cache, 
            max_query_length, max_doc_length)
        assert rand_num > 0
        self.rand_num = rand_num

    def __getitem__(self, item):
        qid = self.qids[item]
        pid = random.choice(self.reldict[qid])
        query_data = self.query_dataset[qid]
        passage_data = self.doc_dataset[pid]
        randpids = random.sample(range(len(self.doc_dataset)), self.rand_num)
        rand_passage_data = [self.doc_dataset[randpid] for randpid in randpids]

        return query_data, passage_data, rand_passage_data

def single_get_collate_function(max_seq_length, padding=False):
    cnt = 0
    def collate_function(batch):
        nonlocal cnt
        length = None
        if cnt < 10 or padding:
            length = max_seq_length
            cnt += 1

        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids, default=1, 
                dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0, 
                dtype=torch.int64, length=length),
        }
        ids = [x['id'] for x in batch]
        return data, ids
    return collate_function  

def dual_get_collate_function(max_query_length, max_doc_length, rel_dict, padding=False):
    query_collate_func = single_get_collate_function(max_query_length, padding)
    doc_collate_func = single_get_collate_function(max_doc_length, padding)

    def collate_function(batch):
        query_data, query_ids = query_collate_func([x[0] for x  in batch])
        doc_data, doc_ids = doc_collate_func([x[1] for x in batch])
        rel_pair_mask = [[1 if docid not in rel_dict[qid] else 0 
            for docid in doc_ids]
            for qid in query_ids]
        input_data = {
            "input_query_ids":query_data['input_ids'],
            "query_attention_mask":query_data['attention_mask'],
            "input_doc_ids":doc_data['input_ids'],
            "doc_attention_mask":doc_data['attention_mask'],
            "rel_pair_mask":torch.FloatTensor(rel_pair_mask),
            }
        return input_data
    return collate_function  

def triple_get_collate_function(max_query_length, max_doc_length, rel_dict, padding=False):
    query_collate_func = single_get_collate_function(max_query_length, padding)
    doc_collate_func = single_get_collate_function(max_doc_length, padding)

    def collate_function(batch):
        query_data, query_ids = query_collate_func([x[0] for x  in batch])
        doc_data, doc_ids = doc_collate_func([x[1] for x in batch])
        hard_doc_data, hard_doc_ids = doc_collate_func(sum([x[2] for x in batch], []))
        rel_pair_mask = [[1 if docid not in rel_dict[qid] else 0 
            for docid in doc_ids]
            for qid in query_ids]
        hard_pair_mask = [[1 if docid not in rel_dict[qid] else 0 
            for docid in hard_doc_ids ]
            for qid in query_ids]
        query_num = len(query_data['input_ids'])
        hard_num_per_query = len(batch[0][2])
        input_data = {
            "input_query_ids":query_data['input_ids'],
            "query_attention_mask":query_data['attention_mask'],
            "input_doc_ids":doc_data['input_ids'],
            "doc_attention_mask":doc_data['attention_mask'],
            "other_doc_ids":hard_doc_data['input_ids'].reshape(query_num, hard_num_per_query, -1),
            "other_doc_attention_mask":hard_doc_data['attention_mask'].reshape(query_num, hard_num_per_query, -1),
            "rel_pair_mask":torch.FloatTensor(rel_pair_mask),
            "hard_pair_mask":torch.FloatTensor(hard_pair_mask),
            }
        return input_data
    return collate_function

def build_graph(doc_tree):
    #build document trees in bfs way.
    doc_nums = 1
    para_nums = len(doc_tree['para'])
    sentence_nums = len(doc_tree['sentence'])
    all_nodes = doc_nums + para_nums + sentence_nums
    sentence_para_mask, sentence_doc_mask, para_doc_mask = [[0 for i in range(all_nodes)] for j in range(all_nodes)], [[0 for i in range(all_nodes)] for j in range(all_nodes)], \
                                                           [[0 for i in range(all_nodes)] for j in range(all_nodes)]
    for para_idx, para_span in enumerate(doc_tree['para']):
        para_doc_mask[doc_nums + para_idx][0] = 1
        para_doc_mask[0][doc_nums + para_idxs] = 1
    for sent_idx, sent_span in enumerate(doc_tree['sentence']):
        sentence_doc_mask[sent_idx + doc_nums + para_nums][0] = 1
        sentence_doc_mask[0][sent_idx + doc_nums + para_nums] = 1
        for para_idx, para_span in doc_tree:
            if (sent_span[1] - para_span[0]) * (para_span[1] - sent_span[0]) > 0:
                sentence_para_mask[sent_idx + doc_nums + para_nums][doc_nums + para_idx] = 1
                sentence_para_mask[doc_nums + para_idx][sent_idx + doc_nums + para_nums] = 1
    return sentence_doc_mask, para_doc_mask, sentence_para_mask

