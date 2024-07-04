import os
import json
import time
import copy
import sqlite3
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from collections import defaultdict
from prompt_utils import load_json, save_json, load_cache_files, adjust_case, get_prompt_template, create_schema, create_index, run_engine, create_retrievers
from sentence_transformers import SentenceTransformer

FEASIBLE_LABEL = 'answerable'
INFEASIBLE_LABEL = 'not_answerable'

def parse_args():
    parser = argparse.ArgumentParser(description="Script for filtering out infeasible input questions.")
    parser.add_argument('--api_key_path', default='OPENAI_API_KEY.json', type=str, help='Path for OpenAI API key')
    parser.add_argument('--model', default='gpt-4o', type=str, help='Model name')
    parser.add_argument('--prompt_path', default='', type=str, help='Path for prompt')
    parser.add_argument('--data_pool_dir', required=True, type=str, help='Directory for data pool')
    parser.add_argument('--test_data_path', required=True, type=str, help='Evaluation data path')
    parser.add_argument('--pos_num_sample', default=8, type=int, help='Number of positive samples for the prompt')
    parser.add_argument('--neg_num_sample', default=0, type=int, help='Number of negative samples for the prompt')
    parser.add_argument('--temp', default=0.0, type=float, help='Temperature')
    parser.add_argument('--n', default=1, type=int, help='Number of completions')
    parser.add_argument('--inference_result_path', default='./', type=str, help='Path for inference results')
    parser.add_argument('--output_file', default='prediction.json', type=str, help='Output file')
    parser.add_argument('--input_prompt_file', default='prompt_input.json', type=str, help='Input prompt file')
    parser.add_argument('--wait_time', default=1, type=int, help='API wait time in seconds')
    parser.add_argument('--random_seed', default=0, type=int, help='Random seed')
    parser.add_argument('--load_cache', default=None, type=str, help='Path to cache for loading prompts and predictions')
    parser.add_argument('--prompt_type', default='create_col', type=str, choices=['create_col', 'col_fk'], help='Prompt type')
    parser.add_argument('--enable_batchapi', action='store_true', help='Use batch API')
    parser.add_argument('--get_length_only', action='store_true', help='Get token lengths only')
    return parser.parse_args()

def prepare_data_pool(data_pool_dir):
    data_pool_files = [f for f in os.listdir(data_pool_dir) if '_train_combined_subset.json' in f or '_valid_combined_subset.json' in f]
    data_pool = [item for sublist in [load_json(os.path.join(data_pool_dir, file)) for file in data_pool_files] for item in sublist]
    return data_pool

def create_indices(data_pool, retriever):
    feasible_pool = [sample['question'] for sample in data_pool if sample['query'] != 'null']
    feasible_index = create_index(feasible_pool, retriever)
    infeasible_pool = [sample['question'].capitalize() for sample in data_pool if sample['query'] == 'null']
    infeasible_index = create_index(infeasible_pool, retriever)
    return feasible_pool, feasible_index, infeasible_pool, infeasible_index

if __name__ == '__main__':
    args = parse_args()

    if os.path.exists(args.inference_result_path) and not args.get_length_only:
        raise Exception(f"Directory already exists ({args.inference_result_path})")

    assert args.pos_num_sample % 2 == 0 and args.neg_num_sample % 2 == 0 and args.neg_num_sample in [0, args.pos_num_sample]
    
    client = OpenAI(api_key = load_json(args.api_key_path)["API_KEY"])
    inference_data = load_json(args.test_data_path)
    data_pool = prepare_data_pool(args.data_pool_dir)
    cache_prompt_input, cache_prompt_output = load_cache_files(args.load_cache)

    if args.get_length_only:
        import tiktoken
        enc = tiktoken.get_encoding("p50k_base")
        input_toks, output_toks = [], []

    db_schema, primary_key, foreign_key = create_schema(os.path.join(args.data_pool_dir, 'tables.json'))
    if 'spider' not in str(args.test_data_path):
        db_schema = adjust_case(db_schema)
        primary_key = adjust_case(primary_key)
        foreign_key = adjust_case(foreign_key)

    create_schema_prompt = get_prompt_template(args)
    task_prompt = 'Please classify the following sentence into either answerable or unanswerable. Answer me with "answerable" or "not_answerable".\n\n'

    retriever = create_retrievers('all-mpnet-base-v2')

    feasible_pool = [sample['question'] for sample in data_pool if sample['query'] != 'null']
    feasible_pool_db = [sample['db_id'] for sample in data_pool if sample['query'] != 'null']
    feasible_index = create_index(feasible_pool, retriever)
    infeasible_pool = [sample['question'].capitalize() for sample in data_pool if sample['query'] == 'null']
    assert len(infeasible_pool) > 0
    infeasible_index = create_index(infeasible_pool, retriever)
    
    # create prompts
    prompts = {}
    for line in tqdm(inference_data):
        id_, db_id, question = line['id'], line['db_id'], line['question']
        if id_ not in cache_prompt_input:
            prompt_to_run = ''

            query_embedding = retriever.encode(question).reshape(1, -1)
            _, feasible_idx = feasible_index.search(query_embedding, feasible_index.ntotal)
            feasible_idx = feasible_idx[0].tolist()
            infeasible_idx = []
            if args.neg_num_sample > 0:
                _, infeasible_idx = infeasible_index.search(query_embedding, infeasible_index.ntotal)
                infeasible_idx = infeasible_idx[0].tolist()

            # single-domain
            if db_id in ['atis', 'advising', 'mimic_iv']:

                prompt_to_run += create_schema_prompt(db_id, db_schema, primary_key, foreign_key, args.data_pool_dir)
                prompt_to_run += task_prompt

                np.random.seed(args.random_seed)
                index_order = np.random.permutation(['pos'] * args.pos_num_sample + ['neg'] * args.neg_num_sample)
                for flag in index_order:
                    if flag == 'pos':
                        idx = feasible_idx.pop(0)
                        q, s = feasible_pool[idx], FEASIBLE_LABEL
                    else:
                        idx = infeasible_idx.pop(0)
                        q, s = infeasible_pool[idx], INFEASIBLE_LABEL
                    prompt_to_run += f'Question: {q}\nAnswer: {s}\n\n'
                prompt_to_run += f'Question: {question}\nAnswer: '

            # cross-domain
            else:

                num_example_per_db = 2
                num_db = args.pos_num_sample // 2
                similar_db_id = [feasible_pool_db[idx] for idx in feasible_idx]
                processed_db_id = {}

                np.random.seed(args.random_seed)
                for sim_db_id in similar_db_id:
                    if sim_db_id not in processed_db_id:
                        processed_db_id[sim_db_id] = ([], [])

                        prompt_to_run += create_schema_prompt(sim_db_id, db_schema, primary_key, foreign_key, os.path.join(args.data_pool_dir, 'database'))
                        prompt_to_run += task_prompt
                        feasible_idx_copy = copy.deepcopy(feasible_idx)

                        index_order = np.random.permutation(['pos'] * num_example_per_db + ['neg'] * num_example_per_db)
                        for flag in index_order:
                            if flag == 'pos':
                                while True:
                                    idx = feasible_idx_copy.pop(0)
                                    if feasible_pool_db[idx] == sim_db_id:
                                        break
                                q, s = feasible_pool[idx], FEASIBLE_LABEL
                                processed_db_id[sim_db_id][0].append(idx)
                            else:
                                idx = infeasible_idx.pop(0)
                                q, s = infeasible_pool[idx], INFEASIBLE_LABEL
                                processed_db_id[sim_db_id][1].append(idx)
                            prompt_to_run += f'Question: {q}\nAnswer: {s}\n\n'
                        prompt_to_run += '\n'

                    if len(processed_db_id) == num_db:
                        break

                prompt_to_run += create_schema_prompt(db_id, db_schema, primary_key, foreign_key, os.path.join(args.data_pool_dir, 'database'))
                prompt_to_run += task_prompt
                prompt_to_run += f'Question: {question}\nAnswer: '

        else:
            prompt_to_run = cache_prompt_input[id_]

        if args.get_length_only:
            input_toks.append(len(enc.encode(prompt_to_run)))
            output_toks.append(len(enc.encode(FEASIBLE_LABEL if '-sql' in line['type'] else INFEASIBLE_LABEL)) * args.n) # this is an approximation

        prompts[id_] = prompt_to_run

    # inference
    if args.get_length_only:
        print(f'# Samples: {len(input_toks)} (MinTokLen: {np.min(input_toks)} / MeanTokLen: {round(np.mean(input_toks), 1)} / MaxTokLen: {np.max(input_toks)})')
        print('Total Input Tokens:', sum(input_toks))
        print('Total Output Tokens:', sum(output_toks))
    else:
        os.makedirs(args.inference_result_path, exist_ok=True)
        save_json(prompts, os.path.join(args.inference_result_path, args.input_prompt_file))
        if args.enable_batchapi:
            from prompt_utils import handle_batch_api
            handle_batch_api(client, args, prompts)
        else:
            from prompt_utils import handle_individual_api
            handle_individual_api(client, args, prompts, cache_prompt_output)
