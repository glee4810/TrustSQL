'''
Official Evaluation Script for TrustSQL v1.0
'''

import argparse
from scoring_utils import *
import multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions with penalty.")
    parser.add_argument('--data_file', required=True, help='Input data JSON file')
    parser.add_argument('--pred_file', required=True, help='Model predictions')
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
    db_id = get_db_id(args.db_path)
    
    # Load data
    data = load_json(args.data_file)
    prediction = load_json(args.pred_file)

    # Execute SQL
    real_sql_dict, pred_sql_dict, real_ans_dict, pred_ans_dict, db_dict, type_dict, nlq_dict, temp_dict = execute_queries(data, prediction, args.db_path, num_workers, args.timeout)

    # Evaluate
    _, score_dict = reliability_score(real_ans_dict, pred_ans_dict, gt_sql_dict=real_sql_dict, db_id=db_id, return_sample_dict=True)
    result_dict = initialize_result_dicts()

    # Collect results
    data_id, db_ids, types, questions, templates = [], [], [], [], []
    query_real, query_pred, exec_real, exec_pred = [], [], [], []
    query_correct_list, exec_correct_list = [], []
    for key, score in score_dict.items():
        q_real, q_pred = real_sql_dict[key], pred_sql_dict[key]
        ans_real, ans_pred = real_ans_dict[key], pred_ans_dict[key]
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
