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

CURRENT_DATE = "2100-12-31"
CURRENT_TIME = "23:59:00"
NOW = f"{CURRENT_DATE} {CURRENT_TIME}"
PRECOMPUTED_DICT = {
    'temperature': (35.5, 38.1),
    'sao2': (95.0, 100.0),
    'heart rate': (60.0, 100.0),
    'respiration': (12.0, 18.0),
    'systolic bp': (90.0, 120.0),
    'diastolic bp': (60.0, 90.0),
    'mean bp': (60.0, 110.0)
}
TIME_PATTERN = r"(DATE_SUB|DATE_ADD)\((\w+\(\)|'[^']+')[, ]+ INTERVAL (\d+) (MONTH|YEAR|DAY)\)"
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
            # items, cnts = np.unique([p.lower() for p in pred], return_counts=True) # not used due to lower performance when the penality increases in RS            
            num_consistency = int(len(pred) * consistency_ratio)
            pred = 'null'
            for item, cnt in zip(items, cnts):
                if cnt >= num_consistency:
                    pred = item
                    break
            prediction[key] = pred
    return prediction

def convert_date_function(match):
    function = match.group(1)
    date = match.group(2)
    number = match.group(3)
    unit = match.group(4).lower()
    
    # Use singular form when number is 1
    if number == '1':
        unit = unit.rstrip('s')
    else:
        unit += 's' if not unit.endswith('s') else ''
    
    # Determine the sign based on the function (DATE_SUB or DATE_ADD)
    sign = '-' if function == 'DATE_SUB' else '+'
    
    return f"datetime({date}, '{sign}{number} {unit}')"

def extract_sql_strings(sql):
    pattern = r"'([^']*(?:''[^']*)*)'|\"([^\"]*(?:\"\"[^\"]*)*)\""
    matches = re.findall(pattern, sql)
    extracted = [item for match in matches for item in match if item]    
    return extracted

def normalize_sql_spacing(query):
    
    values = extract_sql_strings(query)
    for idx, val in enumerate(values):
        query = query.replace(val, f'__PLACEHOLDER{idx}__')
    
    # postprocess remove spaces around brackets
    query = re.sub(r'\s*\(\s*', '(', query)
    query = re.sub(r'\s*\)\s*', ')', query)
    query = re.sub(r'\s*,\s*', ', ', query)
    
    for idx, val in enumerate(values):
        query = query.replace(f'__PLACEHOLDER{idx}__', val)
    
    return query.strip()

def postprocess_pred(query, db_id):
    '''
    Postprocessing for predicted SQL. Modify if necessary.
    '''
    if 'select' not in query.lower(): # remove non-select queries
        return 'null'
    
    query = query.replace('```sql', '').replace('```', '') # function calling filtering
    query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=') # tokenization adjustment for open-source models
    query = re.sub('[ ]+', ' ', query.replace('\n', ' ')).strip()

    # postprocess string literals
    if db_id in ['atis', 'advising']: # => "
        pattern = r"'([^']*)'"
        query = re.sub(pattern, r'"\1"', query)
    else: # => '
        pattern = r'"([^\']*)"'
        query = re.sub(pattern, r"'\1'", query)

    query = normalize_sql_spacing(query)

    if db_id in POSTPROCESS_VAL_DICT:
        for before, after in POSTPROCESS_VAL_DICT[db_id].items():
            query = query.replace(before, after)

    if db_id not in ['atis', 'advising', 'mimic_iv'] and query != 'null': # spider adjustment
        query = remove_distinct(query)

    if db_id == 'mimic_iv':

        # Convert MySQL to SQLite functions
        query = re.sub(TIME_PATTERN, convert_date_function, query)

        if "current_time" in query: # strftime('%J',current_time) => strftime('%J','2100-12-31 23:59:00')
            query = query.replace("current_time", f"'{NOW}'")
        if "'now'" in query: # 'now' => '2100-12-31 23:59:00'
            query = query.replace("'now'", f"'{NOW}'")
        if "NOW()" in query: # NOW() => '2100-12-31 23:59:00'
            query = query.replace("NOW()", f"'{NOW}'")
        if "CURDATE()" in query: # CURDATE() => '2100-12-31'
            query = query.replace("CURDATE()", f"'{CURRENT_DATE}'")
        if "CURTIME()" in query: # CURTIME() => '23:59:00'
            query = query.replace("CURTIME()", f"'{CURRENT_TIME}'")

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
