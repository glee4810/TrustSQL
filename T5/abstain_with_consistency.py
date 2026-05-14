# import os
# import json
# import argparse
# import warnings
# import numpy as np

# def parse_args():
#     args = argparse.ArgumentParser()
#     args.add_argument('--inference_result_path', required=True, type=str, help='path for inference')
#     args.add_argument("--num_consistency_ratio", type=int, default=1.0)    
#     args.add_argument('--input_file', default='prediction_raw.json', type=str, help='path for inference')
#     args.add_argument('--output_file', default='prediction.json', type=str, help='path for inference')
#     return args.parse_args()

# if __name__ == '__main__':
#     args = parse_args()

#     input_file = os.path.join(args.inference_result_path, args.input_file)
#     with open(input_file, 'r') as f:
#         data = json.load(f)

#     result = {}
#     for id_, line in data.items():
#         items, cnts = np.unique(line['pred'], return_counts=True)
#         num_consistency = int(len(line['pred']) * args.num_consistency_ratio)
#         pred = 'null'
#         for item, cnt in zip(items, cnts):
#             if cnt >= num_consistency:
#                 pred = item
#                 break
#         result[id_] = pred

#     out_file = os.path.join(args.inference_result_path, args.output_file)
#     with open(out_file, 'w') as f:
#         json.dump(result, f)