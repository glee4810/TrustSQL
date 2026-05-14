import re
import os
import json
import argparse
import warnings
import numpy as np

import sqlite3
import sys
sys.path.insert(0,'spider_eval')
from parse import remove_distinct
from exec_eval import result_eq
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut

# __current_time = "2105-12-31 23:59:00"
# __precomputed_dict = {
#                     'temperature': (35.5, 38.1),
#                     'sao2': (95.0, 100.0),
#                     'heart rate': (60.0, 100.0),
#                     'respiration': (12.0, 18.0),
#                     'systolic bp': (90.0, 120.0),
#                     'diastolic bp':(60.0, 90.0),
#                     'mean bp': (60.0, 110.0)
#                                 }

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--inference_result_path', required=True, type=str, help='path for inference')
    args.add_argument('--input_file', default='prediction_raw.json', type=str, help='path for inference')
    args.add_argument('--output_file', default='prediction.json', type=str, help='path for inference')
    args.add_argument("--field", type=str, required=True, choices=['maxent', 'minmaxprob', 'meanmaxprob'], help='field used for OOD detection')
    args.add_argument("--manual", type=float, default=-1, help='manual thresholding')
    args.add_argument("--clustering", action='store_true', help='k-means')
    args.add_argument("--logistic", action='store_true', help='build logistic regression')
    args.add_argument("--precentile", action='store_true', help='percentile value')
    args.add_argument("--automatic", action='store_true', help='automatic search')
    args.add_argument('--db_path', default=None, type=str, help='db_path')
    args.add_argument("--num_workers", type=int, default=-1)
    args.add_argument("--timeout", type=int, default=60.0, help='execution time limit in sec')    
    return args.parse_args()

# def post_process_sql(query, db_id):

#     query = query.replace('```sql', '').replace('```', '') # function calling filtering
#     query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=') # tokenization adjustment
#     query = re.sub('[ ]+', ' ', query.replace('\n', ' ')).strip()    

#     if 'select' not in query.lower(): # invalid sql
#         query = 'null'
    
#     if 'not_answerable' in query.lower(): # gpt - postprocessing
#         query = 'null'
    
#     if db_id not in ['atis', 'advising', 'ehrsql'] and query != 'null': # spider adjustment
#         query = remove_distinct(query)

#     if db_id == 'mimic_iv':
#         if "current_time" in query:
#             query = query.replace("current_time", f"'{__current_time}'")
#         if re.search('[ \n]+([a-zA-Z0-9_]+_lower)', query) and re.search('[ \n]+([a-zA-Z0-9_]+_upper)', query):
#             vital_lower_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_lower)', query)[0]
#             vital_upper_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_upper)', query)[0]
#             vital_name_list = list(set(re.findall('([a-zA-Z0-9_]+)_lower', vital_lower_expr) + re.findall('([a-zA-Z0-9_]+)_upper', vital_upper_expr)))
#             if len(vital_name_list)==1:
#                 processed_vital_name = vital_name_list[0].replace('_', ' ')
#                 if processed_vital_name in __precomputed_dict:
#                     vital_range = __precomputed_dict[processed_vital_name]
#                     query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr, f"{vital_range[1]}")
#         query = query.replace("%y", "%Y").replace('%j', '%J')

#     return query


# exec_result = []
# def result_tracker(result):
#     exec_result.append(result)

# def process_answer(ans):
#     return str(sorted([str(ret) for ret in ans[:100]])) # check only up to 100th record

# def execute(sql, db_path):
#     con = sqlite3.connect(db_path)
#     con.text_factory = lambda b: b.decode(errors="ignore")
#     cur = con.cursor()
#     result = cur.execute(sql).fetchall()
#     con.close()
#     return result

# def execute_wrapper(sql, db_path, timeout, tag, skip_indicator='null', post_process=True):
#     if sql != skip_indicator:
#         try:
#             result = func_timeout(timeout, execute, args=(sql, db_path))
#         except KeyboardInterrupt:
#             sys.exit(0)
#         except FunctionTimedOut:
#             result = [(f'timeout_{tag}',)]
#         except:
#             result = [(f'error_{tag}',)] # possibly len(query) > 512 or not executable
#         if post_process:
#             result = process_answer(result)
#     else:
#         result = skip_indicator
#     return result

# def execute_query(sql1, sql2, db, db_path, timeout, data_idx=None):
#     '''
#     Execute the query. Time out if it exceeds {args.timeout} seconds
#     '''
#     post_process = True
#     if os.path.isdir(db_path): # this indicates spider or ehrsql
#         db_path = os.path.join(db_path, db, f'{db}.sqlite')
#     if 'spider' in str(args.db_path):
#         post_process = False
#     result1 = execute_wrapper(sql1, db_path, timeout, tag='real', post_process=post_process)
#     result2 = execute_wrapper(sql2, db_path, timeout, tag='pred', post_process=post_process)
#     result = {'data_idx': data_idx, 'real': result1, 'pred': result2}
#     return result

# def execute_query_distributed(real, pred, db_id, db_path, timeout, num_workers):
#     pool = mp.Pool(processes=num_workers)
#     for data_idx, (sql1, sql2, db) in enumerate(zip(real, pred, db_id)):
#         pool.apply_async(execute_query, args=(sql1, sql2, db, db_path, timeout, data_idx), callback = result_tracker)
#     pool.close()
#     pool.join()


if __name__ == '__main__':
    args = parse_args()

    if args.manual == -1 and args.clustering is None and args.logistic is None and args.precentile is None:
        warnings.warn("Threshold value is not set! All predictions are sent to the database.")

    if args.automatic and args.db_path is None:
        warnings.warn("Must specify the database path for 'automatic'")
        
    input_file = os.path.join(args.inference_result_path, args.input_file)
    with open(input_file, 'r') as f:
        test_data = json.load(f)

    test_path = args.inference_result_path
    valid_path = test_path.replace('test', 'valid')
    if args.clustering or args.logistic or args.precentile:
        with open(os.path.join(valid_path, args.input_file), 'r') as f:
            valid_pred = json.load(f)
        with open(f'{valid_path}/acc.json', 'r') as f:
            acc_list_dict = json.load(f)

        values = []
        label = []
        for _, line in valid_pred.items():
            if args.field == 'maxent':
                if len(line['entropy'])==1:
                    values.append(max(line['entropy'][0]))
                else:
                    values.append(max(line['entropy']))
                # assert len(line['entropy'])==1
                # values.append(max(line['entropy'][0]))
            elif args.field == 'minmaxprob':
                if len(line['maxprob'])==1:
                    values.append(np.min(line['maxprob'][0]))
                else:
                    values.append(np.min(line['maxprob']))
                # assert len(line['maxprob'])==1
                # values.append(np.min(line['maxprob'][0]))
            elif args.field == 'meanmaxprob':
                if len(line['maxprob'])==1:
                    values.append(np.mean(line['maxprob'][0]))
                else:
                    values.append(np.mean(line['maxprob']))
                # assert len(line['maxprob'])==1 
            label.append(acc_list_dict[line['id']])
    else:

        num_workers = mp.cpu_count() if args.num_workers==-1 else args.num_workers
        with open(os.path.join(valid_path, args.input_file), 'r') as f:
            valid_pred = json.load(f)
        with open(f'{valid_path}/acc.json', 'r') as f:
            acc_list_dict = json.load(f)

        # db_id = []
        # query_real = []
        # query_pred = []
        acc_list = []
        values = []
        for key in valid_pred:
            line = valid_pred[key]
            # db_id.append(line['db_id'])
            # real = post_process_sql(line['real'], db_id=line['db_id'])
            # pred = post_process_sql(line['pred'][0], db_id=line['db_id'])
            # query_real.append(real)
            # query_pred.append(pred)
            acc_list.append(acc_list_dict[key])
            if args.field == 'maxent':
                assert len(line['entropy'])==1
                values.append(max(line['entropy'][0]))
            elif args.field == 'minmaxprob':
                assert len(line['maxprob'])==1
                values.append(np.min(line['maxprob'][0]))
            elif args.field == 'meanmaxprob':
                assert len(line['maxprob'])==1
                values.append(np.mean(line['maxprob'][0]))

        # exec_real = []
        # exec_pred = []
        # if num_workers>1 and 'spider' not in str(args.db_path):
        #     execute_query_distributed(query_real, query_pred, db_id, args.db_path, args.timeout, num_workers)
        #     indices = []
        #     for ret in exec_result:
        #         exec_real.append(ret['real'])
        #         exec_pred.append(ret['pred'])
        #         indices.append(ret['data_idx'])
        #     exec_real = np.array(exec_real)[np.argsort(indices)].tolist()
        #     exec_pred = np.array(exec_pred)[np.argsort(indices)].tolist()
        # else:
        #     for sql1, sql2, db in zip(query_real, query_pred, db_id):
        #         ret = execute_query(sql1, sql2, db, args.db_path, args.timeout)
        #         exec_real.append(ret['real'])
        #         exec_pred.append(ret['pred'])

        # acc_list = []
        # for idx in range(len(exec_real)):
        #     q_real, q_pred = query_real[idx], query_pred[idx]
        #     ans_real, ans_pred = exec_real[idx], exec_pred[idx]
        #     if 'spider' in str(args.db_path):
        #         order_matters = 'order by' in q_real.lower()
        #         exec_acc = result_eq(ans_real, ans_pred, order_matters=order_matters)            
        #     else:
        #         exec_acc = (ans_real == ans_pred)
        #     acc_list.append(exec_acc)

    if args.manual != -1 and not args.clustering and not args.logistic and not args.precentile and not args.automatic:
        threshold = args.manual
    elif args.manual == -1 and args.clustering and not args.logistic and not args.precentile and not args.automatic:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=0).fit(np.expand_dims(values, axis=1))
        zero_low = min(np.array(values)[kmeans.labels_==0])
        zero_high = max(np.array(values)[kmeans.labels_==0])
        one_low = min(np.array(values)[kmeans.labels_==1])
        one_high = max(np.array(values)[kmeans.labels_==1])
        if one_high > zero_high:
            threshold = (zero_high + one_low)/2
        else:
            threshold = (zero_low + one_high)/2
    elif args.manual == -1 and not args.clustering and args.logistic and not args.precentile and not args.automatic:
        from sklearn.linear_model import LogisticRegression
        log = LogisticRegression(random_state=0).fit(np.expand_dims(values, axis=1), label)
    elif args.manual == -1 and not args.clustering and not args.logistic and args.precentile and not args.automatic:
        if args.field == 'maxent':
            threshold = np.percentile(values, q=(1.0 - np.mean(label))*100)
        else:
            threshold = np.percentile(values, q=np.mean(label)*100)
    elif args.manual == -1 and not args.clustering and not args.logistic and not args.precentile and args.automatic:
        sorted_indices = np.argsort(values)
        if args.field != 'maxent':
            sorted_indices = sorted_indices[::-1]
        sorted_values = np.array(values)[sorted_indices]
        sorted_acc_list = np.array(acc_list)[sorted_indices]

        # penalties = [0, 10, len(data)]
        penalty = 1
        sorted_acc_list = [1 if score > 0 else -penalty for score in sorted_acc_list]
        max_score = 0
        threshold = -1
        for idx in range(len(sorted_acc_list)):
            cum_score = sum(sorted_acc_list[:idx+1])
            if cum_score > max_score:
                max_score = cum_score
                threshold = sorted_values[idx-1]
    else:
        if args.field == 'maxent':
            threshold = float('inf')
        else:
            threshold = float('-inf')
    
    print(f'Threshold set to {threshold} for {args.field}!')
    result = {}
    for id_, line in test_data.items():
        if args.field == 'maxent':
            if len(line['entropy'])==1:
                val = max(line['entropy'][0])
            else:
            # assert len(line['entropy'])==1
                val = max(line['entropy'])
        elif args.field == 'minmaxprob':
            if len(line['maxprob'])==1:
                val = min(line['maxprob'][0])
            else:
                val = min(line['maxprob'])
            # assert len(line['maxprob'])==1
        elif args.field == 'meanmaxprob':
            if len(line['maxprob'])==1:
                val = np.mean(line['maxprob'][0])
            else:
                val = np.mean(line['maxprob'])
            # assert len(line['maxprob'])==1

        if type(line['pred']) == list:
            line['pred'] = line['pred'][0]
        if '|' in line['pred']:
            line['pred'] = line['pred'].split(' | ')[-1]
        if args.logistic:
            if log.predict(np.expand_dims([val], axis=1))[0]:
                pred = 'null'
            else:
                pred = line['pred']
        else:
            if args.field == 'maxent':
                if val > threshold:
                    pred = 'null'
                else:
                    pred = line['pred']
            elif args.field == 'minmaxprob':
                if val < threshold:
                    pred = 'null'
                else:
                    pred = line['pred']
            elif args.field == 'meanmaxprob':
                if val < threshold:
                    pred = 'null'
                else:
                    pred = line['pred']

        result[id_] = pred

    out_file = os.path.join(args.inference_result_path, args.output_file)
    with open(out_file, 'w') as f:
        json.dump(result, f)