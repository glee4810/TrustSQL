# import os
# import sys
# import yaml
# import json
# import torch
# import pickle
# import argparse
# import numpy as np
# from tqdm import tqdm
# from scipy.spatial.distance import mahalanobis
# from transformers import T5Tokenizer, T5EncoderModel
# from torch.utils.data import Dataset, DataLoader, SequentialSampler

# sys.path.insert(0, os.getcwd())
# from T5.model import load_model, load_tokenizer
# from config import Config

# from random import Random
# from typing import Optional, Dict
# from dataclasses import dataclass
# @dataclass
# class AnnotatedSQL:
#     question: str
#     query: str
#     db_id: Optional[str] = None
#     id: Optional[str] = None


# # python T5/calculate_mahalanobis.py --data_path ./dataset/atis --model_path ./outputs/atis_t5_base_lr0.00001 --background_data_path 'TriageSQL v1.3' --CUDA_VISIBLE_DEVICES 0
# class SP_Dataset(Dataset):
#     def __init__(self, path, tokenizer, args): # , include_impossible=False):

#         self.db_id = args.db_id
#         self.tokenizer = tokenizer
#         self.add_schema = args.add_schema
#         self.shuffle_schema = args.shuffle_schema
#         self.random = Random(args.random_seed)
#         self.add_column_type = args.add_column_type
#         self.tables_path = args.tables_path
#         # self.include_impossible = include_impossible

#         if type(path)==str:
#             with open(path) as json_file:
#                 data = json.load(json_file)
#         else:
#             data = path

#         if self.add_schema:
#             if self.tables_path is None:
#                 raise "tables_path must be provided for add_schema=True"
#             with open(self.tables_path) as f:
#                 self.db_json = json.load(f)

#         self.data = []
#         idx = 0
#         for line in tqdm(data):
#             # if self.include_impossible==False and line["query"] == 'null':
#             #         continue
#             if type(line)==dict:
#                 annotated_sql = AnnotatedSQL(
#                     question=line["question"],
#                     query=line["query"],
#                     db_id=line["db_id"],
#                     id=line["id"] if 'id' in line else idx
#                 )
#             else:
#                 annotated_sql = AnnotatedSQL(
#                     question=line,
#                     query='null',
#                     db_id='background',
#                     id='-'
#                 )
#             instance = self.preprocess_sample(annotated_sql)
#             self.data.append(instance)
#             idx += 1

#     def preprocess_sample(self, annotated_sql: AnnotatedSQL) -> AnnotatedSQL:

#         question = annotated_sql.question

#         if self.add_schema:
#             tables_json = [db for db in self.db_json if db["db_id"] == annotated_sql.db_id][0]
#             schema_description = self.get_schema_description(tables_json, self.shuffle_schema, self.random)
#             question += f" {schema_description}"

#         processed_annotated_sql: AnnotatedSQL = AnnotatedSQL(
#             question=question,
#             query=annotated_sql.query,
#             db_id=annotated_sql.db_id,
#             id=annotated_sql.id
#         )

#         return processed_annotated_sql


#     def get_schema_description(self, tables_json: Dict, shuffle_schema: bool, random: Random):
#         table_names = tables_json["table_names_original"]
#         if shuffle_schema:
#             random.shuffle(table_names)

#         columns = [
#             (column_name[0], column_name[1], column_type)
#             for column_name, column_type in zip(tables_json["column_names_original"], tables_json["column_types"])
#         ]

#         schema_description = ""
#         for table_index, table_name in enumerate(table_names):
#             schema_description += f" | {table_name} : "
            
#             table_columns = [column[1] for column in columns if column[0] == table_index]
#             if shuffle_schema:
#                 random.shuffle(table_columns)

#             schema_description += " , ".join(table_columns)

#         return schema_description.lower().lstrip()


#     def __getitem__(self, index):
#         fields = {
#             "inputs": self.data[index].question,
#             "labels": self.data[index].query,
#             "db_id": self.data[index].db_id,
#             "id": self.data[index].id,
#         }
#         return fields


#     def __len__(self):
#         return len(self.data)

        
# class DataCollator(object):
#     def __init__(self, tokenizer, return_tensors='pt', padding=True, truncation=True, max_length=512):
#         self.tokenizer = tokenizer
#         self.return_tensors = return_tensors
#         self.padding = padding
#         self.truncation = truncation
#         self.max_length = max_length

#     def __call__(self, batch):

#         input_ids, labels = [], []
#         data_ids = []
#         db_id = []
#         for instance in batch:
#             input_ids.append(instance['inputs'])
#             labels.append(instance['labels'])
#             db_id.append(instance['db_id'])
#             data_ids.append(instance['id'])

#         inputs = self.tokenizer(input_ids, return_tensors=self.return_tensors, padding=self.padding, truncation=self.truncation, max_length=self.max_length)
#         outputs = self.tokenizer(labels, return_tensors=self.return_tensors, padding=self.padding, truncation=self.truncation, max_length=self.max_length)

#         fields = {
#             "inputs": inputs.input_ids,
#             "labels": outputs.input_ids,
#             "db_id": db_id,
#             "id": data_ids
#         }

#         return fields


# if __name__ == '__main__':

#     args = Config()
#     args.get_param(use_model_param=True,
#                    use_eval_param=True)
#     args.parser.add_argument('--training_data_path', type=str, default=None)
#     args.parser.add_argument('--eval_data_path', required=True, type=str)
#     args.parser.add_argument('--inference_config_path', required=True, type=str, help='model to calculate embeddings')
#     args.parser.add_argument('--save_path', required=True, type=str)    
#     args.parser.add_argument('--background_data_path', type=str, default=None)
#     args = args.parse()

#     # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#     # os.environ["CUDA_VISIBLE_DEVICES"]=args.CUDA_VISIBLE_DEVICES

#     with open(args.inference_config_path, "r") as stream:
#         config = yaml.safe_load(stream)
#     for k, v in config.items():
#         if config[k]:
#             setattr(args, k, config.get(k))

#     setattr(args, 'device', 'cuda')
#     setattr(args, 'add_schema', False)    
#     setattr(args, 'eval_batch_size', args.eval_batch_size*4*4*4) ####################################################
    
#     tokenizer = load_tokenizer(args.model_name)
#     data_collator = DataCollator(tokenizer=tokenizer, return_tensors='pt')
#     model = load_model(model_name=args.model_name, encoder_only=True)
#     model.resize_token_embeddings(len(tokenizer))

#     if args.bf16:
#         model = model.to(torch.bfloat16)
#     checkpoint = torch.load(args.load_model_path)    
#     model.load_state_dict(checkpoint['model_state_dict'], strict=False)
#     model = model.to(args.device)

#     if args.training_data_path:
#         train_dataset = SP_Dataset(path=args.training_data_path, tokenizer=tokenizer, args=args)
#         train_sampler = SequentialSampler(train_dataset)
#         train_dataloader = DataLoader(
#                                     train_dataset,
#                                     sampler=train_sampler,
#                                     batch_size=args.eval_batch_size,
#                                     drop_last=False,
#                                     num_workers=args.num_workers,
#                                     collate_fn=data_collator
#                                 )
#     eval_dataset = SP_Dataset(path=args.eval_data_path, tokenizer=tokenizer, args=args)
#     eval_sampler = SequentialSampler(eval_dataset)
#     eval_dataloader = DataLoader(
#                                 eval_dataset,
#                                 sampler=eval_sampler, 
#                                 batch_size=args.eval_batch_size, 
#                                 drop_last=False,
#                                 num_workers=args.num_workers,
#                                 collate_fn=data_collator
#                                 )

#     split = 'test' if 'test' in str(args.eval_data_path) else 'valid'
#     with torch.no_grad():
#         if args.training_data_path:
#             train_embedding_file = f'./outputs/{args.save_path}/train_embedding.pickle'
#             # if not os.path.isfile(train_embedding_file):
#             train_id = []
#             train_embedding = []
#             cnt = 0
#             for batch in train_dataloader:
#                 input_ids = batch['inputs'].to(args.device)
#                 last_hidden_states = model(input_ids=input_ids).last_hidden_state.cpu().detach().float().numpy()
#                 train_embedding.extend(np.mean(last_hidden_states, axis=1))
#                 train_id.extend(batch['id'])
#                 cnt += len(input_ids)
#             id2train_embedding = {id_: vec for id_, vec in zip(train_id, train_embedding)}

#             with open(train_embedding_file, 'wb') as handle:
#                 pickle.dump(id2train_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

#         test_embedding_file = f'./outputs/{args.save_path}/{split}_embedding.pickle'
#         # if not os.path.isfile(test_embedding_file):
#         test_id = []
#         test_embedding = []
#         for batch in eval_dataloader:
#             input_ids = batch['inputs'].to(args.device)
#             last_hidden_states = model(input_ids=input_ids).last_hidden_state.cpu().detach().float().numpy()
#             test_embedding.extend(np.mean(last_hidden_states, axis=1))
#             test_id.extend(batch['id'])
#         id2test_embedding = {id_: vec for id_, vec in zip(test_id, test_embedding)}

#         with open(test_embedding_file, 'wb') as handle:
#             pickle.dump(id2test_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     # MD
#     # dist_list = []
#     # for idx in range(len(test_embedding)):
#     #     vec = test_embedding[idx]
#     #     dist = mahalanobis(vec, train_embedding_mean, train_embedding_iv)
#     #     dist_list.append(dist)


#     # RMD
#     with torch.no_grad():
#         if args.background_data_path: # if specified
#             background_embedding_file = f'./outputs/{args.save_path}/background_embedding.pickle'

#             flag = True
#             if os.path.isfile(background_embedding_file):
#                 with open(background_embedding_file, 'rb') as handle:
#                     background_embedding = pickle.load(handle)
#                 if len(background_embedding) >= 344000:
#                     flag = False

#             if flag:

#                 with open(os.path.join(args.background_data_path, 'trainset.json')) as in_fp:
#                     trainset = json.load(in_fp)
#                 with open(os.path.join(args.background_data_path, 'devset.json')) as in_fp:
#                     devset = json.load(in_fp)
#                 with open(os.path.join(args.background_data_path, 'testset.json')) as in_fp:
#                     testset = json.load(in_fp)
#                 background_data = list(set([q['question'] for q in trainset + devset + testset]))

#                 background_dataset = SP_Dataset(background_data, tokenizer=tokenizer, args=args)
#                 background_sampler = SequentialSampler(background_dataset)
#                 background_dataloader = DataLoader(
#                                             background_dataset,
#                                             sampler=background_sampler,
#                                             batch_size=args.eval_batch_size,
#                                             drop_last=False,
#                                             num_workers=args.num_workers,
#                                             collate_fn=data_collator
#                                         )

#                 if os.path.exists(background_embedding_file):
#                     with open(background_embedding_file, 'rb') as handle:
#                         background_embedding = pickle.load(handle)
#                 else:
#                     background_embedding = {}
#                 batch_idx = 0
#                 for batch in tqdm(background_dataloader):
#                     batch_size = len(batch['inputs'])
#                     in_cnt = []
#                     not_in_cnt = []
#                     for ii in range((batch_idx*batch_size), (batch_idx+1)*batch_size):
#                         if ii in background_embedding:
#                             in_cnt.append(ii)
#                         else:
#                             not_in_cnt.append(ii)
#                     if len(in_cnt) != batch_size:
#                         input_ids = batch['inputs'].to(args.device)
#                         last_hidden_states = model(input_ids=input_ids).last_hidden_state.cpu().detach().float().numpy()
#                         for feature in np.mean(last_hidden_states, axis=1):
#                             if len(background_embedding) in not_in_cnt:
#                                 background_embedding[len(background_embedding)] = feature
#                         if batch_idx % 1000 ==0:
#                             with open(background_embedding_file, 'wb') as handle:
#                                 pickle.dump(background_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)
#                             with open(background_embedding_file, 'rb') as handle:
#                                 background_embedding = pickle.load(handle)
#                     batch_idx += 1

#         # background_embedding_mean = np.mean(background_embedding, axis=0)
#         # background_embedding_iv = np.linalg.inv(np.cov(np.transpose(background_embedding)))

#         # dist1_list = []
#         # dist2_list = []
#         # rel_dist_list = []
#         # for idx in tqdm(range(len(test_embedding))):
#         #     vec = test_embedding[idx]
#         #     dist1 = mahalanobis(vec, train_embedding_mean, train_embedding_iv)
#         #     dist2 = mahalanobis(vec, background_embedding_mean, background_embedding_iv)
#         #     dist1_list.append(dist1)
#         #     dist2_list.append(dist2)
#         #     dist1_list.append()
#         #     rel_dist = dist1 - dist2
#         #     rel_dist_list.append(rel_dist)

