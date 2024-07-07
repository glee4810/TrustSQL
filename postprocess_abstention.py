import argparse
import numpy as np
from scoring_utils import load_json, save_json

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions with penalty.")
    parser.add_argument('--pred_file', metavar='pred.json', required=True, help='Model predictions')
    parser.add_argument('--save_file', metavar='pred.json', required=True, help='Output file for modified predictions')
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

def main(args):

    prediction = load_json(args.pred_file)

    if args.pre_abst_path:
        prediction = apply_pre_abstention(prediction, args.pre_abst_path)

    for key, pred in prediction.items():
        if isinstance(pred, list) and len(pred)==1:
            prediction[key] = prediction[key][0]

    if 'demo' in args.pred_file:
        prediction = apply_abstention_within_sql_demo(prediction)

    if 'voting' in args.pred_file:
        prediction = apply_abstention_within_sql_voting(prediction)

    if args.post_abst_path:
        prediction = apply_post_abstention(prediction, args.post_abst_path)

    save_json(prediction, args.save_file)

if __name__ == '__main__':
    args = parse_args()
    main(args)
