import json
import argparse
import numpy as np
import pandas as pd
from scoring_utils import *
import multiprocessing as mp
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions with penalty.")
    parser.add_argument('--data_file', metavar='data.json', required=True, help='Input data JSON file')
    parser.add_argument('--pred_file', metavar='pred.json', required=True, help='Model predictions')
    parser.add_argument('--db_path', required=True, type=str, help='Path to database')
    parser.add_argument("--num_workers", type=int, default=-1, help='Number of workers for multiprocessing')
    parser.add_argument("--timeout", type=int, default=60, help='Execution time limit in seconds')
    parser.add_argument("--ndigits", type=int, default=1, help='Number of digits to round scores to')
    parser.add_argument("--save_output", action='store_true', help='Save output for error analysis')
    parser.add_argument("--save_path", type=str, default='outputs_analysis', help='Path for saving error analysis files')
    return parser.parse_args()

def main(args):

    num_workers = mp.cpu_count() if args.num_workers == -1 else args.num_workers

    # Infer dataset name
    dataset = infer_dataset(args.db_path)
    
    # Load data
    data = load_json(args.data_file)
    prediction = load_json(args.pred_file)

    # Execute SQL
    real_dict, pred_dict, real_result, pred_result, db_dict, type_dict, nlq_dict, temp_dict = execute_queries(data, prediction, args.db_path, num_workers, args.timeout)

    # Evaluate
    _, score_dict = reliability_score(real_result, pred_result, dataset=dataset, return_dict=True, label=real_dict)
    result_dict = initialize_result_dicts()

    # Collect results
    data_id, db_ids, types, questions, templates = [], [], [], [], []
    query_real, query_pred, exec_real, exec_pred = [], [], [], []
    query_correct_list, exec_correct_list = [], []
    for key, score in score_dict.items():
        q_real, q_pred = real_dict[key], pred_dict[key]
        ans_real, ans_pred = real_result[key], pred_result[key]
        exec_acc = (score == 1)
        data_id.append(key)
        db_ids.append(db_dict[key])
        types.append(type_dict[key])
        questions.append(nlq_dict[key])
        templates.append(temp_dict[key])
        query_real.append(q_real)
        query_pred.append(q_pred)
        exec_real.append(ans_real)
        exec_pred.append(ans_pred)
        query_correct_list.append(q_real == q_pred)
        exec_correct_list.append(exec_acc)
        update_result_dicts(result_dict, key, q_real, q_pred, ans_real, ans_pred, exec_acc, type_dict[key])
    
    # Print results
    print_results(result_dict, len(data), args.ndigits)

    # Save results
    if args.save_output:
        save_error_analysis(args, data_id, db_ids, types, questions, templates, query_real, query_pred, exec_real, exec_pred, query_correct_list, exec_correct_list)

if __name__ == '__main__':
    args = parse_args()
    main(args)
