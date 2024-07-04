import os
import json
import time
import sqlite3
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from prompt_utils import load_json, save_json, load_cache_files, adjust_case, create_schema, find_fields_mysql_like, find_foreign_keys_mysql_like, find_primary_keys_mysql_like, create_index, run_engine

def parse_args():
    parser = argparse.ArgumentParser(description="Script for filtering out error in model outputs.")
    parser.add_argument('--api_key_path', default='OPENAI_API_KEY.json', type=str, help='Path for OpenAI API key')
    parser.add_argument('--model', default='gpt-4o', type=str, help='Model name')
    parser.add_argument('--data_pool_dir', required=True, type=str, help='Directory for data pool')
    parser.add_argument('--test_data_path', required=True, type=str, help='Evaluation data path')
    parser.add_argument('--target_path', required=True, type=str, help='Path to the target file for examining abstention')
    parser.add_argument('--temp', default=0.0, type=float, help='Temperature')
    parser.add_argument('--n', default=1, type=int, help='Number of completions')
    parser.add_argument('--inference_result_path', default='./', type=str, help='Path for inference results')
    parser.add_argument('--output_file', default='prediction.json', type=str, help='Output file')
    parser.add_argument('--input_prompt_file', default='prompt_input.json', type=str, help='Prompt for the output')
    parser.add_argument('--wait_time', default=1, type=int, help='API wait time in seconds')
    parser.add_argument('--random_seed', default=0, type=int, help='Random seed')
    parser.add_argument('--load_cache', default=None, type=str, help='Path to cache for loading prompts and predictions')
    parser.add_argument('--prompt_type', default='col_fk', type=str, choices=['create_col', 'col_fk'], help='Prompt type')
    parser.add_argument('--enable_batchapi', action='store_true', help='Use batch API')
    parser.add_argument('--get_length_only', action='store_true', help='Get token lengths only')
    return parser.parse_args()

def generate_error_prompt(test_sample_text, db_id, sql, db_schema, primary_key, foreign_key):
    instruction = "#### Based on the question and predicted SQL, are you sure the SQL below is correct? If you consider the SQL is correct, answer me with 'correct'. If not, answer me with 'incorrect'. Only output your response without explanation.\n"
    fields = find_fields_mysql_like(db_schema, db_id)
    fields += f"Foreign_keys = {find_foreign_keys_mysql_like(foreign_key, db_id)}\n"
    fields += f"Primary_keys = {find_primary_keys_mysql_like(primary_key, db_id)}"
    prompt = f"{instruction}{fields}\nQuestion: {test_sample_text}\nPredicted SQL: {sql}\nAnswer: "
    return prompt

if __name__ == '__main__':
    args = parse_args()

    if os.path.exists(args.inference_result_path) and not args.get_length_only:
        raise Exception(f"Directory already exists ({args.inference_result_path})")
    
    client = OpenAI(api_key = load_json(args.api_key_path)["API_KEY"])
    inference_data = load_json(args.test_data_path)
    target_dict = load_json(args.target_path)
    cache_prompt_input, cache_prompt_output = load_cache_files(args.load_cache)

    if args.get_length_only:
        import tiktoken
        enc = tiktoken.get_encoding("p50k_base")
        input_toks, output_toks = [], []

    # Check if predictions contain all test samples
    ids = {item['id'] for item in inference_data}
    missing_ids = [key for key in target_dict if key not in ids]
    if missing_ids:
        raise Exception(f"Target data is not complete ({missing_ids})")

    if args.get_length_only:
        import tiktoken
        enc = tiktoken.get_encoding("p50k_base")
        input_toks, output_toks = [], []

    db_schema, primary_key, foreign_key = create_schema(os.path.join(args.data_pool_dir, 'tables.json'))
    if 'spider' not in str(args.test_data_path):
        db_schema = adjust_case(db_schema)
        primary_key = adjust_case(primary_key)
        foreign_key = adjust_case(foreign_key)

    task_prompt = """#### Based on the question and predicted SQL, are you sure the SQL below is correct? If you consider the SQL is correct, answer me with 'correct'. If not, answer me with 'incorrect'. Only output your response without explanation.\n"""

    # create prompts
    prompts = {}
    for line in tqdm(inference_data):
        id_, db_id, question = line['id'], line['db_id'], line['question']
        if id_ not in cache_prompt_input:
            SQL = target_dict[id_]
            prompt_to_run = generate_error_prompt(question, db_id, SQL, db_schema, primary_key, foreign_key)        

        else:
            prompt_to_run = cache_prompt_input[id_]

        if args.get_length_only:
            input_toks.append(len(enc.encode(prompt_to_run)))
            output_toks.append(len(enc.encode(np.random.choice(['correct', 'incorrect']))) * args.n) # this is an approximation

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
