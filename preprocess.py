import enum
import sys

sys.path += ['./']
import os
import torch
import gzip
import pickle
import subprocess
import csv
import multiprocessing
import numpy as np
from os import listdir
from os.path import isfile, join
import argparse
import json
from tqdm import tqdm
from star_tokenizer import RobertaTokenizer
from transformers import AutoTokenizer
import time
import math
sentvec_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-MiniLM2-L6-H768", do_lower_case=True)
paravec_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2", do_lower_case=True)
docvec_tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)

special_tokens_doc = docvec_tokenizer.encode(
        ".",
    )
special_tokens_para = paravec_tokenizer.encode(
        ".",
    )
special_tokens_sent = sentvec_tokenizer.encode(
        ".",
    )
doc_dot = special_tokens_doc[1]
para_cls = special_tokens_para[0]
para_sep = special_tokens_para[-1]
sent_cls = special_tokens_sent[0]
sent_dot = special_tokens_sent[1]
sent_sep = special_tokens_sent[-1]

def pad_input_ids(input_ids, max_length,
                  pad_on_left=False,
                  pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            input_ids = input_ids + padding_id

    return input_ids

def tokenize_to_file_query(args, in_path, output_dir, line_fn, max_length, begin_idx, end_idx, level):
    os.makedirs(output_dir, exist_ok=True)
    data_cnt = end_idx - begin_idx
    ids_array = np.memmap(
        os.path.join(output_dir, f"{level}_ids.memmap"),
        shape=(data_cnt, ), mode='w+', dtype=np.int32)
    token_ids_array = np.memmap(
        os.path.join(output_dir, f"{level}_token_ids.memmap"),
        shape=(data_cnt, max_length), mode='w+', dtype=np.int32)
    token_length_array = np.memmap(
        os.path.join(output_dir, f"{level}_lengths.memmap"),
        shape=(data_cnt, ), mode='w+', dtype=np.int32)
    pbar = tqdm(total=end_idx-begin_idx, desc=f"Tokenizing")
    for idx, line in enumerate(open(in_path, 'r')):
        if idx < begin_idx:
            continue
        if idx >= end_idx:
            break
        qid_or_pid, token_ids, length = line_fn(args, line, tokenizer)
        write_idx = idx - begin_idx
        ids_array[write_idx] = qid_or_pid
        token_ids_array[write_idx, :] = token_ids
        token_length_array[write_idx] = length
        pbar.update(1)
    pbar.close()

def tokenize_to_file(args, in_path, output_dir, line_fn, begin_idx, end_idx):
    global docvec_tokenizer, paravec_tokenizer, sentvec_tokenizer
    os.makedirs(output_dir, exist_ok=True)
    data_cnt = end_idx - begin_idx
    ids_array = np.memmap(
        os.path.join(output_dir, "ids.memmap"),
        shape=(data_cnt, ), mode='w+', dtype=np.int32)
    token_ids_array = np.memmap(
        os.path.join(output_dir, "token_ids.memmap"),
        shape=(data_cnt, 512), mode='w+', dtype=np.int32)
    token_length_array = np.memmap(
        os.path.join(output_dir, "lengths.memmap"),
        shape=(data_cnt, ), mode='w+', dtype=np.int32)
    pbar = tqdm(total=end_idx-begin_idx, desc=f"Tokenizing")
    all_para_id, all_para_tokens_ids, all_para_lens, all_sent_id, all_sent_tokens_ids, all_sent_lens = [], [], [], [], [], []
    for idx, line in enumerate(open(in_path, 'r')):
        if idx < begin_idx:
            continue
        if idx >= end_idx:
            break
        qid_or_pid, token_ids, length, para_id, para_tokens_ids, para_lens, sent_id, sent_tokens_ids, sent_lens = line_fn(args, line, docvec_tokenizer, paravec_tokenizer, sentvec_tokenizer)
        all_para_id.extend(para_id)
        all_para_tokens_ids.extend(para_tokens_ids)
        all_para_lens.extend(para_lens)
        all_sent_id.extend(sent_id)
        all_sent_tokens_ids.extend(sent_tokens_ids)
        all_sent_lens.extend(sent_lens)
        write_idx = idx - begin_idx
        ids_array[write_idx] = qid_or_pid
        token_ids_array[write_idx, :] = token_ids
        token_length_array[write_idx] = length
        pbar.update(1)
    para_token_ids_array = np.memmap(
        os.path.join(output_dir, "para_token_ids.memmap"),
        shape=(len(all_para_id), 128), mode='w+', dtype=np.int32)
    para_token_length_array = np.memmap(
        os.path.join(output_dir, "para_lengths.memmap"),
        shape=(len(all_para_id),), mode='w+', dtype=np.int32)
    para_token_ids_array[:] = all_para_tokens_ids[:]
    para_token_length_array[:] = all_para_lens[:]
    with open(os.path.join(output_dir, "para_ids.pickle"),'wb') as pp:
        pickle.dump(all_para_id,pp,protocol=4)
    sent_token_ids_array = np.memmap(
        os.path.join(output_dir, "sent_token_ids.memmap"),
        shape=(len(all_sent_id), 64), mode='w+', dtype=np.int32)
    sent_token_length_array = np.memmap(
        os.path.join(output_dir, "sent_lengths.memmap"),
        shape=(len(all_sent_id),), mode='w+', dtype=np.int32)
    with open(os.path.join(output_dir,"sent_ids.pickle"),'wb') as sp:
        pickle.dump(all_sent_id,sp,protocol=4)
    sent_token_ids_array[:] = all_sent_tokens_ids[:]
    sent_token_length_array[:] = all_sent_lens[:]
    pbar.close()
    return len(all_para_id), len(all_sent_id)

def multi_file_process(args, num_process, in_path, out_path, line_fn, max_length, isquery=False, level = ""):
    output_linecnt = subprocess.check_output(["wc", "-l", in_path]).decode("utf-8")
    print("line cnt", output_linecnt)
    all_linecnt = int(output_linecnt.split()[0])
    run_arguments = []
    for i in range(num_process):
        begin_idx = round(all_linecnt * i / num_process)
        end_idx = round(all_linecnt * (i+1) / num_process)
        output_dir = f"{out_path}_split_{i}"
        if isquery:
            run_arguments.append((
                    args, in_path, output_dir, line_fn,
                    max_length, begin_idx, end_idx, level
                ))
        else:
            run_arguments.append((
                args, in_path, output_dir, line_fn,
                begin_idx, end_idx
            ))
    pool = multiprocessing.Pool(processes=num_process)
    if isquery:
        pool.starmap(tokenize_to_file_query, run_arguments)
    else:
        data_cnt = pool.starmap(tokenize_to_file, run_arguments)
        data_cnt = list(zip(*data_cnt))
    pool.close()
    pool.join()
    splits_dir = [a[2] for a in run_arguments]
    if isquery:
        return splits_dir, all_linecnt
    else:
        return splits_dir, all_linecnt, sum(data_cnt[0]), sum(data_cnt[1])

def write_query_rel(args, pid2offset, qid2offset_file, query_file, positive_id_file, out_query_file, standard_qrel_file):

    print( "Writing query files " + str(out_query_file) +
        " and " + str(standard_qrel_file))
    query_collection_path = os.path.join(args.data_dir,query_file)
    if positive_id_file is None:
        query_positive_id = None
        valid_query_num = int(subprocess.check_output(
            ["wc", "-l", query_collection_path]).decode("utf-8").split()[0])
    else:
        query_positive_id = set()
        query_positive_id_path = os.path.join(
            args.data_dir,
            positive_id_file,
        )

        print("Loading query_2_pos_docid")
        for line in open(query_positive_id_path, 'r', encoding='utf8'):
            query_positive_id.add(int(line.split()[0]))
        valid_query_num = len(query_positive_id)

    out_query_path = os.path.join(args.out_data_dir,out_query_file,)

    qid2offset = {}

    print('start query file split processing')
    splits_dir_lst, _ = multi_file_process(
        args, args.threads, query_collection_path,
        out_query_path, QueryPreprocessingFn,
        args.max_query_length, isquery=True
        )

    print('start merging splits')

    token_ids_array = np.memmap(
        out_query_path+".memmap",
        shape=(valid_query_num, args.max_query_length), mode='w+', dtype=np.int32)
    token_length_array = []

    idx = 0
    for split_dir in splits_dir_lst:
        ids_array = np.memmap(
            os.path.join(split_dir, "ids.memmap"), mode='r', dtype=np.int32)
        split_token_ids_array = np.memmap(
            os.path.join(split_dir, "token_ids.memmap"), mode='r', dtype=np.int32)
        split_token_ids_array = split_token_ids_array.reshape(len(ids_array), -1)
        split_token_length_array = np.memmap(
            os.path.join(split_dir, "lengths.memmap"), mode='r', dtype=np.int32)
        for q_id, token_ids, length in zip(ids_array, split_token_ids_array, split_token_length_array):
            if query_positive_id is not None and q_id not in query_positive_id:
                # exclude the query as it is not in label set
                continue
            token_ids_array[idx, :] = token_ids
            token_length_array.append(length) 
            qid2offset[q_id] = idx
            idx += 1
            if idx < 3:
                print(str(idx) + " " + str(q_id))
    assert len(token_length_array) == len(token_ids_array) == idx
    np.save(out_query_path+"_length.npy", np.array(token_length_array))

    qid2offset_path = os.path.join(
        args.out_data_dir,
        qid2offset_file,
    )
    with open(qid2offset_path, 'wb') as handle:
        pickle.dump(qid2offset, handle, protocol=4)
    print("done saving qid2offset")

    print("Total lines written: " + str(idx))
    meta = {'type': 'int32', 'total_number': idx,
            'embedding_size': args.max_query_length}
    with open(out_query_path + "_meta", 'w') as f:
        json.dump(meta, f)

    if positive_id_file is None:
        print("No qrels file provided")
        return
    print("Writing qrels")
    with open(os.path.join(args.out_data_dir, standard_qrel_file), "w", encoding='utf-8') as qrel_output: 
        out_line_count = 0
        for line in open(query_positive_id_path, 'r', encoding='utf8'):
            topicid, _, docid, rel = line.split()
            topicid = int(topicid)
            if args.data_type == 0:
                docid = int(docid[1:])
            else:
                docid = int(docid)
            qrel_output.write(str(qid2offset[topicid]) +
                         "\t0\t" + str(pid2offset[docid]) +
                         "\t" + rel + "\n")
            out_line_count += 1
        print("Total lines written: " + str(out_line_count))

def preprocess(args):
    
    pid2offset = {}
    if args.data_type == 0:
        in_passage_path = os.path.join(
            args.data_dir,
            "msmarco-docs.tsv",
        )
    else:
        in_passage_path = os.path.join(
            args.data_dir,
            "collection.tsv",
        )

    out_passage_path = os.path.join(
        args.out_data_dir,
        "passages",
    )

    out_para_path = os.path.join(
        args.out_data_dir,
        "para",
    )

    out_sent_path = os.path.join(
        args.out_data_dir,
        "sent",
    )

    if os.path.exists(out_passage_path):
        print("preprocessed data already exist, exit preprocessing")
        return

    print('start passage file split processing')
    splits_dir_lst, all_linecnt, all_paracnt, all_sentcnt = multi_file_process(
        args, args.threads, in_passage_path,
        out_passage_path, PassagePreprocessingFn,
        args.max_seq_length, isquery=False
        )
    print(all_sentcnt)
    print(all_paracnt)
    token_ids_array = np.memmap(
        out_passage_path+".memmap",
        shape=(all_linecnt, 512), mode='w+', dtype=np.int32)

    para_token_ids_array = np.memmap(
        out_para_path + ".memmap",
        shape=(all_paracnt, 128), mode='w+', dtype=np.int32)

    sent_token_ids_array = np.memmap(
        out_sent_path + ".memmap",
        shape=(all_sentcnt, 64), mode='w+', dtype=np.int32)

    token_length_array = []
    para_token_length_array = []
    sent_token_length_array = []
    paraid2offset = {}
    sentid2offset = {}
    idx = 0
    pidx = 0
    sidx = 0
    out_line_count = 0
    print('start merging splits')
    for split_dir in splits_dir_lst:
        ids_array = np.memmap(
            os.path.join(split_dir, "ids.memmap"), mode='r', dtype=np.int32)
        split_token_ids_array = np.memmap(
            os.path.join(split_dir, "token_ids.memmap"), mode='r', dtype=np.int32)
        split_token_ids_array = split_token_ids_array.reshape(len(ids_array), -1)
        split_token_length_array = np.memmap(
            os.path.join(split_dir, "lengths.memmap"), mode='r', dtype=np.int32)
        with open(os.path.join(split_dir, "para_ids.pickle"),'rb') as f:
            para_ids_array = pickle.load(f)
    
        para_split_token_ids_array = np.memmap(os.path.join(split_dir, "para_token_ids.memmap"),mode='r', dtype=np.int32)
        para_split_token_ids_array = para_split_token_ids_array.reshape(len(para_ids_array), -1)
        para_split_token_length_array = np.memmap(
            os.path.join(split_dir, "para_lengths.memmap"), mode='r', dtype=np.int32)

        with open(os.path.join(split_dir, "sent_ids.pickle"), 'rb') as f:
            sent_ids_array = pickle.load(f)
        sent_split_token_ids_array = np.memmap(
            os.path.join(split_dir, "sent_token_ids.memmap"), mode='r', dtype=np.int32)
        sent_split_token_ids_array = sent_split_token_ids_array.reshape(len(sent_ids_array), -1)
        sent_split_token_length_array = np.memmap(
            os.path.join(split_dir, "sent_lengths.memmap"), mode='r', dtype=np.int32)

        for p_id, token_ids, length in zip(ids_array, split_token_ids_array, split_token_length_array):
            token_ids_array[idx, :] = token_ids
            token_length_array.append(length) 
            pid2offset[p_id] = idx
            idx += 1
            if idx < 3:
                print(str(idx) + " " + str(p_id))
            out_line_count += 1

        for para_id, para_token_ids, length in zip(para_ids_array, para_split_token_ids_array, para_split_token_length_array):
            para_token_ids_array[pidx, :] = para_token_ids
            para_token_length_array.append(length)
            paraid2offset[para_id] = pidx
            pidx += 1
            if pidx < 3:
                print(str(pidx) + " " + str(para_id))

        for sent_id, sent_token_ids, length in zip(sent_ids_array, sent_split_token_ids_array, sent_split_token_length_array):
            sent_token_ids_array[sidx, :] = sent_token_ids
            sent_token_length_array.append(length)
            sentid2offset[sent_id] = sidx
            sidx += 1
            if sidx < 3:
                print(str(pidx) + " " + str(sent_id))

    assert len(token_length_array) == len(token_ids_array) == idx
    np.save(out_passage_path+"_length.npy", np.array(token_length_array))
    np.save(out_para_path+"_length.npy", np.array(para_token_length_array))
    np.save(out_sent_path + "_length.npy", np.array(sent_token_length_array))

    print("Total lines written: " + str(out_line_count))

    def save_meta(metapath, offset_path, line_count, embedding_size, id2offset):
        meta = {
            'type': 'int32',
            'total_number': line_count,
            'embedding_size': embedding_size}
        with open(metapath + "_meta", 'w') as f:
            json.dump(meta, f)

        with open(offset_path, 'wb') as handle:
            pickle.dump(id2offset, handle, protocol=4)

        print("done saving id2offset")

    pid2offset_path = os.path.join(
        args.out_data_dir,
        "pid2offset.pickle",
    )

    paraid2offset_path = os.path.join(
        args.out_data_dir,
        "paraid2offset.pickle",
    )

    sentid2offset_path = os.path.join(
        args.out_data_dir,
        "sentid2offset.pickle",
    )
    save_meta(out_passage_path, pid2offset_path, out_line_count, args.max_seq_length, pid2offset)
    save_meta(out_para_path, paraid2offset_path, all_paracnt, 128, paraid2offset)
    save_meta(out_sent_path, sentid2offset_path, all_sentcnt, 64, sentid2offset)

    if args.data_type == 0:
        
        write_query_rel(
            args,
            pid2offset,
            "train-qid2offset.pickle",
            "msmarco-doctrain-queries.tsv",
            "msmarco-doctrain-qrels.tsv",
            "train-query",
            "train-qrel.tsv")
        
        write_query_rel(
            args,
            pid2offset,
            "test-qid2offset.pickle",
            "msmarco-test2019-queries.tsv",
            "2019qrels-docs.txt",
            "test-query",
            "test-qrel.tsv")
        write_query_rel(
            args,
            pid2offset,
            "dev-qid2offset.pickle",
            "msmarco-docdev-queries.tsv",
            "msmarco-docdev-qrels.tsv",
            "dev-query",
            "dev-qrel.tsv")
        write_query_rel(
            args,
            pid2offset,
            "lead-qid2offset.pickle",
            "docleaderboard-queries.tsv",
            None,
            "lead-query",
            None)
    else:
        write_query_rel(
            args,
            pid2offset,
            "train-qid2offset.pickle",
            "queries.train.tsv",
            "qrels.train.tsv",
            "train-query",
            "train-qrel.tsv")
        
        write_query_rel(
            args,
            pid2offset,
            "dev-qid2offset.pickle",
            "queries.dev.small.tsv",
            "qrels.dev.small.tsv",
            "dev-query",
            "dev-qrel.tsv")
    
        write_query_rel(
            args,
            pid2offset,
            "test-qid2offset.pickle",
            "msmarco-test2019-queries.tsv",
            "2019qrels-pass.txt",
            "test-query",
            "test-qrel.tsv")
        write_query_rel(
            args,
            pid2offset,
            "lead-qid2offset.pickle",
            "queries.eval.small.tsv",
            None,
            "lead-query",
            None)

def MatchEndpoints(DocTokenizer, ParaTokenizer, text):
    doc_span = []
    doc_tokens = DocTokenizer.tokenize(text)
    para_tokens = ParaTokenizer.tokenize(text)
    para_endpoints = [0] * (math.ceil(len(para_tokens) / 64))
    doc_res = ""
    para_res = ""
    para_idx = 0
    doc_idx = 0

    while doc_tokens or para_tokens:
        if not doc_res:
            doc_res += doc_tokens[0]
            doc_tokens = doc_tokens[1:]
            doc_idx += 1
        if not para_res:
            para_res += para_tokens[0]
            para_tokens = para_tokens[1:]
            para_idx += 1
        if para_idx % 64 == 0:
            para_endpoints[para_idx // 64] = doc_idx
        while doc_res or para_res:
            if para_res[0] == "#":
                para_res = para_res[1:]
            else:
                if doc_res[0] == para_res[0]:
                    doc_res = doc_res[1:]
                    para_res = para_res[1:]
                else:
                    raise Exception(f"Unmatched tokens {doc_res[0]} and {para_res[0]}")
    if para_endpoints[-1] != len(doc_tokens):
        para_endpoints.append(doc_tokens)
    for i in range(len(para_endpoints)-1):
        doc_span.append((para_endpoints[i], para_endpoints[i + 1]))
    return doc_span

def PassagePreprocessingFn(args, line, doc_tokenizer, para_tokenizer, sent_tokenizer):
    if args.data_type == 0:
        line_arr = line.split('\t')
        p_id = int(line_arr[0][1:])  # remove "D"

        url = line_arr[1].rstrip()
        title = line_arr[2].rstrip()
        p_text = line_arr[3].rstrip()
        # NOTE: This linke is copied from ANCE, 
        # but I think it's better to use <s> as the separator, 
        full_text = url + "<sep>" + title + "<sep>" + p_text
        # keep only first 10000 characters, should be sufficient for any
        # experiment that uses less than 500 - 1k tokens
        full_text = full_text[:args.max_doc_character]
    else:
        line = line.strip()
        line_arr = line.split('\t')
        p_id = int(line_arr[0])
        p_text = line_arr[1].rstrip()
        full_text = p_text[:args.max_doc_character]
    passage = doc_tokenizer.encode(
        full_text,
        add_special_tokens=True,
        max_length=512,
        truncation=True
    )
    passage_len = min(len(passage), 512)
    input_id_b = pad_input_ids(passage, 512)

    doc_span = MatchEndpoints(docvec_tokenizer, paravec_tokenizer, full_text)
    passage_cont_para = para_tokenizer.encode(
        full_text,
        add_special_tokens=True,
        max_length=args.max_seq_length,
        truncation=True
    )

    passage_cont_sent = sent_tokenizer.encode(
        full_text,
        add_special_tokens=True,
        max_length=args.max_seq_length,
        truncation=True
    )
    passage_cont_para = passage_cont_para[1:-1]
    passage_cont_sent = passage_cont_sent[1:-1]
    paragraphs = []
    sentences = []
    sent_ids = []
    sent_lens = []
    para_ids = []
    para_lens = []
    for i in range(0, len(passage_cont_para), 64):
        para_enc = [para_cls] + passage_cont_para[i:i + 126] + [para_sep]
        para_enc = pad_input_ids(para_enc, 128)
        paragraphs.append(para_enc)
        para_ids.append(str(p_id) + "_p"+str(len(paragraphs)))
        para_lens += [len(para_enc)]
    sent_nums = passage_cont_sent.count(tok_dot)
    while(len(sentences) < sent_nums):
        dot_idx = passage_cont_sent.index(sent_dot)
        sent_ids.append(str(p_id) + "_s"+str(len(sentences)))
        sent_enc = [sent_cls] + passage_cont_sent[:min(61, dot_idx + 1)] + [sent_dot, sent_sep]
        sent_enc = pad_input_ids(sent_enc, 64)
        sentences.append(sent_enc)
        sent_lens += [len(sent_enc)]
        passage_cont_sent = passage_cont_sent[dot_idx + 1:]
    if passage_cont_:
        sent_ids.append(str(p_id) + "_s"+str(len(sentences)))
        sent_enc = [sent_cls] + passage_cont_sent[:61] + [sent_dot, sent_sep]
        sent_enc = pad_input_ids(sent_enc, 64)
        sentences.append(sent_enc)
        sent_lens += [len(sent_enc)]
        sent_enc += doc_span
    return p_id, input_id_b, passage_len, para_ids, paragraphs, para_lens, sent_ids, sentences, sent_lens

def QueryPreprocessingFn(args, line, tokenizer):
    line_arr = line.split('\t')
    q_id = int(line_arr[0])
    passage = tokenizer.encode(
        line_arr[1].rstrip(),
        add_special_tokens=True,
        max_length=args.max_query_length,
        truncation=True
        )
    passage_len = min(len(passage), args.max_query_length)
    input_id_b = pad_input_ids(passage, args.max_query_length)
    return q_id, input_id_b, passage_len

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="roberta-base",
        type=str,
    )
    parser.add_argument(
        "--document_model",
        default="roberta-base",
        type=str,
    )
    parser.add_argument(
        "--paragraph_model",
        default="roberta-base",
        type=str,
    )
    parser.add_argument(
        "--sentence_model",
        default="roberta-base",
        type=str,
    )
    parser.add_argument(
        "--max_seq_length",
        default=4096,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_doc_character",
        default=10000,
        type=int,
        help="used before tokenizer to save tokenizer latency",
    )
    parser.add_argument(
        "--data_type",
        default=1,
        type=int,
        help="0 for doc, 1 for passage",
    )
    parser.add_argument(
            "--out_data_dir", type =str)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    if args.data_type == 0:
        args.data_dir = "./data/doc/dataset"
       # args.out_data_dir = "./data/doc/ropreprocess_hierarchical"
    else:
        args.data_dir = "./data/passage/dataset"
        args.out_data_dir = "./data/passage/test"

    if not os.path.exists(args.out_data_dir):
        os.makedirs(args.out_data_dir)
    preprocess(args)


if __name__ == '__main__':
    main()
