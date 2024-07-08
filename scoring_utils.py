import sys
sys.path.insert(0, 'spider_eval')

import os
import re
import sys
import json
import sqlite3
import sqlparse
import numpy as np
import pandas as pd
from func_timeout import func_timeout, FunctionTimedOut
import multiprocessing as mp
from parse import remove_distinct
from exec_eval import result_eq

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

def add_distinct(pred, real):
    if pred == 'null' or real == 'null':
        return pred    
    pred_toks = [t.value for t in list(sqlparse.parse(pred)[0].flatten())]
    real_toks = [t.value for t in list(sqlparse.parse(real)[0].flatten())]
    if len(pred_toks) >= 3 and real_toks[2].lower() == 'distinct' and pred_toks[2].lower() != 'distinct':
        pred_toks = pred_toks[:2] + ['DISTINCT', ' '] + pred_toks[2:]
    return ''.join(pred_toks)

def post_process_sql(query, db_id):
    query = query.replace('```sql', '').replace('```', '') # function calling filtering
    query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=') # tokenization adjustment for T5
    query = re.sub('[ ]+', ' ', query.replace('\n', ' ')).strip() 

    if 'select' not in query.lower(): # remove non-select queries
        return 'null'
    
    if db_id not in ['atis', 'advising', 'mimic_iv'] and query != 'null': # spider adjustment
        query = remove_distinct(query)

    if db_id == 'mimic_iv':
        if "current_time" in query:
            query = query.replace("current_time", f"'{CURRENT_TIME}'")
        if "'now'" in query:
            query = query.replace("'now'", f"'{__current_time}'")
        if re.search('[ \n]+([a-zA-Z0-9_]+_lower)', query) and re.search('[ \n]+([a-zA-Z0-9_]+_upper)', query):
            vital_lower_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_lower)', query)[0]
            vital_upper_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_upper)', query)[0]
            vital_name_list = list(set(re.findall('([a-zA-Z0-9_]+)_lower', vital_lower_expr) + re.findall('([a-zA-Z0-9_]+)_upper', vital_upper_expr)))
            if len(vital_name_list) == 1:
                processed_vital_name = vital_name_list[0].replace('_', ' ')
                if processed_vital_name in PRECOMPUTED_DICT:
                    vital_range = PRECOMPUTED_DICT[processed_vital_name]
                    query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr, f"{vital_range[1]}")
        query = query.replace("%y", "%Y").replace('%j', '%J')

    return query

def prettify_scores(score, decimals=1, use_thousands_suffix=True):
    if abs(score) >= 1000 and use_thousands_suffix:
        return f"{score / 1000:.{decimals}f}K"
    return f"{score:.{decimals}f}"

def load_json(pred_file):
    with open(pred_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def process_item(item, db_id):
    if db_id == 'advising':
        if item == 'N' or item == 'No': # return nothing if 'N' (value exists in advising db) or 'No' (GPT zero-shot case operation)
            item = '0'
        if item == 'Y' or item == 'Yes':
            item = '1'
    try:
        item = round(float(item),3)
    except:
        pass
    return str(item)

def process_answer(ans, db_id=None):
    if type(ans)==str:
        return ans
    else:
        if db_id in ['atis', 'advising', 'mimic_iv']:
            return str(sorted([[process_item(c, db_id) for c in row] for row in ans])[:100]) # check only up to 100th record
        else:
            return ans

def execute_sql(sql, db_path):
    con = sqlite3.connect(db_path)
    con.text_factory = lambda b: b.decode(errors="ignore")
    cur = con.cursor()
    result = cur.execute(sql).fetchall()
    con.close()
    return result

def execute_sql_wrapper(key, sql, db_path, tag, db_id, skip_indicator='null'):
    assert tag in ['real', 'pred']
    if sql != skip_indicator:
        try:
            result = execute_sql(sql, db_path)
        except:
            result = 'error_'+tag
        result = process_answer(result, db_id)
        return (key, result)
    else:
        return (key, skip_indicator)    

def execute_sql_wrapper_with_timeout(key, sql, db_path, timeout, tag, db_id, skip_indicator='null'):
    assert tag in ['real', 'pred']
    if sql != skip_indicator:
        try:
            result = func_timeout(timeout, execute_sql, args=(sql, db_path))
        except KeyboardInterrupt:
            sys.exit(0)
        except FunctionTimedOut:
            result = [(f'timeout_{tag}',)]
        except:
            result = [(f'error_{tag}',)]
        result = process_answer(result, db_id)
        return (key, result)
    else:
        return (key, skip_indicator)
    
def execute_all(dict, db_path, db_dict, tag, timeout=None):
    exec_result = {}
    for key in dict:
        sql = dict[key]
        db_id = db_dict[key]
        if os.path.isdir(db_path):
            db_path_ = os.path.join(db_path, db_id, f'{db_id}.sqlite')
        else:
            db_path_ = db_path
        if not os.path.exists(db_path_):
            raise Exception('File does not exist: %s' % db_path_)
        if timeout:
            exec_result[key] = execute_sql_wrapper_with_timeout(key, sql, db_path_, timeout, tag, db_id)[-1]
        else:
            exec_result[key] = execute_sql_wrapper(key, sql, db_path_, tag, db_id)[-1]
    return exec_result

def execute_all_distributed(dict, db_path, db_dict, tag, num_workers, timeout=None):
    exec_result = {}
    def result_tracker(result):
        exec_result[result[0]] = result[-1]
    pool = mp.Pool(processes=num_workers)
    for key in dict:
        sql = dict[key]
        db_id = db_dict[key]
        if os.path.isdir(db_path):
            db_path_ = os.path.join(db_path, db_id, f'{db_id}.sqlite')
        else:
            db_path_ = db_path
        if not os.path.exists(db_path_):
            raise Exception('File does not exist: %s' % db_path_)
        if timeout:
            pool.apply_async(execute_sql_wrapper_with_timeout, args=(key, sql, db_path_, timeout, tag, db_id), callback = result_tracker)
        else:
            pool.apply_async(execute_sql_wrapper, args=(key, sql, db_path_, tag, db_id), callback = result_tracker)        
    pool.close()
    pool.join()
    return exec_result

def reliability_score(real_result, pred_result, dataset, return_dict=False, label=False):

    assert dataset in ['atis', 'advising', 'ehrsql', 'spider']

    reliablity_score = []
    reliablity_score_dict = {}
    for key in real_result:
        ans_real = real_result[key]
        ans_pred = pred_result[key]

        if ans_pred != "[['error_pred']]":
            if ans_pred != 'null' and dataset in ['atis', 'advising', 'ehrsql'] and label:
                is_count = 'count' in label[key].lower() # count( * )
                if is_count and ans_pred=='[]':
                    ans_pred = str([['0.0']])
                is_count = re.search(r'\bcount\s*\([^)]*\)\s*>\s*0\s*from\b', label[key].lower()) # count( * ) > 0 
                if is_count:
                    ans_pred = eval(ans_pred)
                    ans_pred = [[r] for r in np.unique(ans_pred)]
                    if ans_pred == [['None']]:
                        ans_pred = [['0.0']]
                    elif ans_pred not in [[['0.0']], [['1.0']]]:
                        ans_pred = [[f'{len(ans_pred)}.0']]
                    ans_pred = str(ans_pred)
            if dataset == 'spider' and label:
                order_matters = 'order by' in label[key].lower()
                exec_acc = result_eq(ans_real, ans_pred, order_matters=order_matters)
            else:
                exec_acc = (ans_real == ans_pred)
        else:
            exec_acc = False

        # x in Feasible Question; g(x)=1; Acc(x)=1
        if ans_real != 'null' and exec_acc == True:
            score = 1
        # x in Feasible Question; g(x)=0; Acc(x)={0,1}
        elif ans_real != 'null' and ans_pred == 'null':
            score = 0
        # x in Feasible Question; g(x)=1; Acc(x)=0
        elif ans_real != 'null' and exec_acc == False:
            score = -1
        # x in Infeasible Question; g(x)=1
        elif ans_real == 'null' and ans_pred != 'null':
            score = -1
        # x in Infeasible Question; g(x)=0
        elif ans_real == 'null' and ans_pred == 'null':
            score = 1
        else:
            NotImplementedError
        reliablity_score.append(score)
        reliablity_score_dict[key] = score

    if return_dict:
        return reliablity_score, reliablity_score_dict
    else:
        return reliablity_score

def penalize(scores, penalty=1):
    return np.mean([score*penalty if score == -1 else score for score in scores])


def initialize_result_dicts():
    fea_keys = ['seen-sql:easy', 'seen-sql:medium', 'seen-sql:hard',
                'unseen-sql:easy', 'unseen-sql:medium', 'unseen-sql:hard']
    inf_keys = ['inf:column-surface', 'inf:column-related', 'inf:column-unrelated', 'inf:non-sql', 'inf:ext-know']
    keys = ['total'] + fea_keys + inf_keys
    reliablity_dict = {key: [] for key in keys}

    return reliablity_dict

def infer_dataset(db_path):
    if 'spider' in db_path:
        return 'spider'
    elif 'atis' in db_path:
        return 'atis'
    elif 'advising' in db_path:
        return 'advising'
    elif 'ehrsql' in db_path or 'mimic_iv' in db_path:
        return 'ehrsql'
    else:
        NotImplementedError

def update_result_dicts(result_dict, key, q_real, q_pred, ans_real, ans_pred, exec_acc, type_key):
    
    if type_key in result_dict:
        if ans_real != 'null' and exec_acc:
            score = 1
        elif ans_real != 'null' and ans_pred == 'null':
            score = 0
        elif ans_real != 'null' and not exec_acc:
            score = -1
        elif ans_real == 'null' and ans_pred != 'null':
            score = -1
        elif ans_real == 'null' and ans_pred == 'null':
            score = 1
        else:
            raise NotImplementedError
        
        result_dict['total'].append(score)
        result_dict[type_key].append(score)

def prepare_query_dicts(data, prediction):
    real_dict, pred_dict, db_dict, type_dict, nlq_dict, temp_dict = {}, {}, {}, {}, {}, {}
    for line in data:
        real = post_process_sql(line['query'], db_id=line['db_id'])
        pred = post_process_sql(prediction[line['id']], db_id=line['db_id'])
        if line['db_id'] in ['atis', 'advising', 'mimic_iv']: # adjustment for models with no in-domain examples
            pred = add_distinct(pred, real)

        real_dict[line['id']] = real
        pred_dict[line['id']] = pred
        db_dict[line['id']] = line['db_id']
        type_dict[line['id']] = f"{line['type']}:{line['note']}"
        nlq_dict[line['id']] = line['question']
        temp_dict[line['id']] = line['template']
    
    return real_dict, pred_dict, db_dict, type_dict, nlq_dict, temp_dict

def execute_queries(data, prediction, db_path, num_workers, timeout):
    real_dict, pred_dict, db_dict, type_dict, nlq_dict, temp_dict = prepare_query_dicts(data, prediction)
    
    if num_workers > 1:
        real_result = execute_all_distributed(real_dict, db_path, db_dict, tag='real', num_workers=num_workers, timeout=timeout)
        pred_result = execute_all_distributed(pred_dict, db_path, db_dict, tag='pred', num_workers=num_workers, timeout=timeout)
    else:
        real_result = execute_all(real_dict, db_path, db_dict, tag='real', timeout=timeout)
        pred_result = execute_all(pred_dict, db_path, db_dict, tag='pred', timeout=timeout)
    
    return real_dict, pred_dict, real_result, pred_result, db_dict, type_dict, nlq_dict, temp_dict

def print_results(result_dict, data_length, ndigits):

    levels = ['total', 'feasible', 'infeasible']
    penalties = [0, 10, data_length]
    print("{:>20} {:>20} {:>20} {:>20}".format("", *levels))
    print("{:>20} {:>20} {:>20} {:>20}".format('count', len(result_dict['total']), sum([len(value) for key, value in result_dict.items() if '-sql' in key]), sum([len(value) for key, value in result_dict.items() if 'inf' in key])))    
    print('======================================    RS    =====================================')
    for c in penalties:
        print("{:>20} {:>20} {:>20} {:>20}".format(f'RS({c})', 
                prettify_scores(100.0 * np.mean([v if v>0 else v*c for v in result_dict['total']])), 
                prettify_scores(100.0 * np.mean([v if v>0 else v*c for key, value in result_dict.items() if '-sql' in key for v in value])),
                prettify_scores(100.0 * np.mean([v if v>0 else v*c for key, value in result_dict.items() if 'inf' in key for v in value]))))
    print()

def save_error_analysis(args, data_id, db_ids, types, questions, templates, query_real, query_pred, exec_real, exec_pred, query_correct_list, exec_correct_list):
    if args.save_output:
        os.makedirs(args.save_path, exist_ok=True)
        file_path = os.path.split(os.path.split(args.pred_file)[0])[-1]
        df = pd.DataFrame({
            'db_id': db_ids,
            'id': data_id,
            'type': types,
            'template': templates,
            'question': questions,
            'gt_sql': query_real,
            'pred_sql': query_pred,
            'query_acc': query_correct_list,
            'gt_exe': exec_real,
            'pred_exe': exec_pred,
            'exec_acc': exec_correct_list
        })
        df.to_excel(f'{args.save_path}/{file_path}_error_analysis.xlsx', index=False)
