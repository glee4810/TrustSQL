'''
Postprocessing for predicted SQL. Modify if necessary.
'''

import sys
sys.path.insert(0, 'spider_eval')
from parse import remove_distinct

import os
import re
import argparse
import numpy as np
from scoring_utils import load_json, save_json, get_db_id

CURRENT_TIME = "2100-12-31 23:59:00"
PRECOMPUTED_DICT = {
    'temperature': (35.5, 38.1),
    'sao2': (95.0, 100.0),
    'heart rate': (60.0, 100.0),
    'respiration': (12.0, 18.0),
    'systolic bp': (90.0, 120.0),
    'diastolic bp': (60.0, 90.0),
    'mean bp': (60.0, 110.0)
}
POSTPROCESS_VAL_DICT = {'advising': {'Organogenesis: Stem Cells to Regenerative Biology': 'Organogenesis:  Stem Cells to Regenerative Biology'}}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions with penalty.")
    parser.add_argument('--pred_file', type=str, required=True, help='Model predictions')
    parser.add_argument('--save_file', type=str, required=True, help='Output file for modified predictions')
    parser.add_argument('--db_path', type=str, required=True, help='Path to database')
    parser.add_argument('--pre_abst_path', type=str, default=None, help='Pre-abstention file path')
    parser.add_argument('--post_abst_path', type=str, default=None, help='Post-abstention file path')
    return parser.parse_args()

def apply_pre_abstention(prediction, pre_abst_path):    
    print('Pre-abstention applied!')
    abstention_dict = load_json(pre_abst_path)
    assert len(set(abstention_dict).intersection(set(prediction))) == len(prediction), 'Pre-abstention ID mismatch'
    for key in abstention_dict:
        if isinstance(abstention_dict[key], list):
            abstention_dict[key] = abstention_dict[key][0]
        if abstention_dict[key].lower() != 'answerable':
            prediction[key] = 'null'
    return prediction

def apply_post_abstention(prediction, post_abst_path):
    print('Post-abstention applied!')
    abstention_dict = load_json(post_abst_path)
    assert len(set(abstention_dict).intersection(set(prediction))) == len(prediction), 'Post-abstention ID mismatch'
    for key in abstention_dict:
        if isinstance(abstention_dict[key], list):
            abstention_dict[key] = abstention_dict[key][0]
        if abstention_dict[key].lower() == 'incorrect':
            prediction[key] = 'null'
    return prediction

def apply_abstention_within_sql_demo(prediction):
    print('Abstention demo applied!')
    for key, pred in prediction.items():
        if isinstance(pred, str) and 'not_answerable' in pred.lower():
            prediction[key] = 'null'
    return prediction

def apply_abstention_within_sql_voting(prediction, consistency_ratio=1.0):
    print('Abstention voting applied!')
    for key, pred in prediction.items():
        if isinstance(pred, list):
            items, cnts = np.unique(pred, return_counts=True)
            num_consistency = int(len(pred) * consistency_ratio)
            prediction[key] = next((item for item, cnt in zip(items, cnts) if cnt >= num_consistency), 'null')
    return prediction

def postprocess_pred(query, db_id):
    '''
    Postprocessing for predicted SQL. Modify if necessary.
    '''

    if 'select' not in query.lower(): # remove non-select queries
        return 'null'
    query = query.replace('```sql', '').replace('```', '') # function calling filtering
    query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=') # tokenization adjustment for open-source models
    query = re.sub('[ ]+', ' ', query.replace('\n', ' ')).strip()

    if db_id in POSTPROCESS_VAL_DICT:
        for before, after in POSTPROCESS_VAL_DICT[db_id].items():
            query = query.replace(before, after)

    if db_id not in ['atis', 'advising', 'mimic_iv'] and query != 'null': # spider adjustment
        query = remove_distinct(query)

    if db_id == 'mimic_iv':
        if "current_time" in query: # strftime('%J',current_time) => strftime('%J','2100-12-31 23:59:00')
            query = query.replace("current_time", f"'{CURRENT_TIME}'")
        if "'now'" in query: # 'now' => "2100-12-31 23:59:00"
            query = query.replace("'now'", f"'{CURRENT_TIME}'")
        if re.search('[ \n]+([a-zA-Z0-9_]+_lower)', query) and re.search('[ \n]+([a-zA-Z0-9_]+_upper)', query): # systolic_bp_lower => 90.0
            vital_lower_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_lower)', query)[0]
            vital_upper_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_upper)', query)[0]
            vital_name_list = list(set(re.findall('([a-zA-Z0-9_]+)_lower', vital_lower_expr) + re.findall('([a-zA-Z0-9_]+)_upper', vital_upper_expr)))
            if len(vital_name_list) == 1:
                processed_vital_name = vital_name_list[0].replace('_', ' ')
                if processed_vital_name in PRECOMPUTED_DICT:
                    vital_range = PRECOMPUTED_DICT[processed_vital_name]
                    query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr, f"{vital_range[1]}")
        query = query.replace("%y", "%Y").replace('%j', '%J') # strftime('%y-%m',outputevents.charttime) => strftime('%Y-%m',outputevents.charttime)

    return query

def main(args):
    db_id = get_db_id(args.db_path)

    prediction = load_json(args.pred_file)

    # postprocessing sql
    for key, pred in prediction.items():
        if isinstance(pred, list) and len(pred)==1:
            pred = pred[0]
        if isinstance(pred, str):
            pred = postprocess_pred(pred, db_id=db_id)
        else:
            pred = [postprocess_pred(p, db_id=db_id) for p in pred]
        prediction[key] = pred

    # process abstention
    if args.pre_abst_path:
        assert os.path.exists(args.pre_abst_path)
        prediction = apply_pre_abstention(prediction, args.pre_abst_path)

    if 'demo' in args.pred_file:
        prediction = apply_abstention_within_sql_demo(prediction)

    if 'voting' in args.pred_file:
        prediction = apply_abstention_within_sql_voting(prediction)

    if args.post_abst_path:
        assert os.path.exists(args.post_abst_path)
        prediction = apply_post_abstention(prediction, args.post_abst_path)

    save_json(prediction, args.save_file)

if __name__ == '__main__':
    args = parse_args()
    main(args)
