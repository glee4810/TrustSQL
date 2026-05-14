import sys, os
sys.path.append(os.getcwd())
import json
import time
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM

sql_prompt = """### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{db_schema}

### Answer
Given the database schema, here is the SQL query that [QUESTION]{question}[/QUESTION]
[SQL]
"""

def adjust_case(df):
    return df.applymap(lambda x: x.lower() if pd.notnull(x) and isinstance(x, str) else x)

def create_schema(dataset_json):
    schema_df = pd.read_json(dataset_json)
    schema_df.drop(columns=['column_names', 'table_names'], inplace=True)
    schema, f_keys, p_keys = [], [], []

    for _, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']

        for col, col_type in zip(col_names, col_types):
            idx, col_name = col
            if idx > -1:
                schema.append([row['db_id'], tables[idx], col_name, col_type])

        for pk in primary_keys:
            idx, col_name = col_names[pk]
            p_keys.append([row['db_id'], tables[idx], col_name])

        for fk in foreign_keys:
            first, second = fk
            first_idx, first_col = col_names[first]
            second_idx, second_col = col_names[second]
            f_keys.append([row['db_id'], tables[first_idx], tables[second_idx], first_col, second_col])

    return (
        pd.DataFrame(schema, columns=['Database name', 'Table Name', 'Field Name', 'Type']),
        pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key']),
        pd.DataFrame(f_keys, columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key', 'Second Table Foreign Key'])
    )

def apply_tab(row):
    print('\t'.join(map(str, row.values)))

def create_schema_prompt_col_fk(db_id, schema, primary, foreign, db_path):
    prompt = find_fields_mysql_like(schema, db_id)
    prompt += "Foreign_keys = " + find_foreign_keys_mysql_like(foreign, db_id) + '\n\n'
    return prompt

def create_schema_prompt_create_col(db_id, schema, primary, foreign, db_path):
    schema = schema[schema['Database name']==db_id]
    primary = primary[primary['Database name']==db_id]
    foreign = foreign[foreign['Database name']==db_id]
    prompt = ''
    tab_names = []
    for tab_name in schema['Table Name'].values:
        if tab_name not in tab_names:
            tab_names.append(tab_name)
    for tab_name in tab_names:
        cols = schema['Field Name'][schema['Table Name']==tab_name].values
        types = schema['Type'][schema['Table Name']==tab_name].values
        prompt += f'CREATE TABLE {tab_name.lower()} ('
        for idx, (col, type_) in enumerate(zip(cols, types)):
            col, type_ = col.lower(), type_.lower()            
            if idx > 0:
                comma = ' ,'
            else:
                comma = ''
            prompt += f'{comma}\n{col} {type_}'
        primary_cols = primary['Primary Key'][primary['Table Name']==tab_name].values
        if len(primary_cols)>0:
            for col in primary_cols:
                col = col.lower()
                prompt += f' ,\nprimary key ( {col} )'
        foreign_cols = foreign[['Second Table Name', 'First Table Foreign Key', 'Second Table Foreign Key']][foreign['First Table Name']==tab_name].values
        if len(foreign_cols)>0:
            for tab2, col1, col2 in foreign_cols:
                col1, tab2, col2 = col1.lower(), tab2.lower(), col2.lower()
                prompt += f' ,\nforeign key ( {col1} ) references {tab2} ( {col2} )'
        prompt += '\n)'
        if db_id in ['atis', 'advising', 'mimic_iv']:
            con = sqlite3.connect(f'{db_path}/{db_id}.sqlite')
        else:
            con = sqlite3.connect(f'{db_path}/{db_id}/{db_id}.sqlite')
        sql = f'SELECT * FROM {tab_name} LIMIT 3'
        prompt += '\n/*\n'
        tab = pd.read_sql_query(sql, con)

        # SELECT ROW
        # prompt += '\t'.join([l.lower() for l in tab.keys()]) + '\n'
        # for row in tab.values:
        #     prompt += '\t'.join([str(l) for l in row]) + '\n'

        # SELECT COL
        prompt += f'Columns in {tab_name} and 3 examples in each column:\n'
        for col in tab.keys():
            prompt += f"{col}: "
            prompt += ', '.join([str(l) for l in tab[col]]) + '\n'
        prompt +=  '*/\n\n'

    prompt += "Foreign_keys = " + find_foreign_keys_mysql_like(foreign, db_id) # added
    return prompt

def find_fields_mysql_like(db_schema, db_name):
    df = db_schema[db_schema['Database name'] == db_name]
    grouped = df.groupby('Table Name')
    output = ""
    for name, group in grouped:
        output += f"Table {name}, columns = [{', '.join(group['Field Name'])}]\n"
    return output

def find_primary_keys_mysql_like(primary_key, db_name):
  df = primary_key[primary_key['Database name'] == db_name]
  output = "["
  for index, row in df.iterrows():
    output += row['Table Name'] + '.' + row['Primary Key'] +','
  output = output[:-1]
  output += "]\n"
  return output

def find_foreign_keys_mysql_like(foreign_key, db_name):
    df = foreign_key[foreign_key['Database name'] == db_name]
    output = ', '.join(f"{row['First Table Name']}.{row['First Table Foreign Key']} = {row['Second Table Name']}.{row['Second Table Foreign Key']}" for _, row in df.iterrows())
    return f"[{output}]" if output else '[]'

class TrustSQLDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.tokenizer = tokenizer
        self.dataset_path = os.path.split(data_path)[0]

        db_id = data[0]['db_id']
        db_schema, primary_key, foreign_key = create_schema(os.path.join(self.dataset_path, 'tables.json'))
        db_schema = adjust_case(db_schema)
        primary_key = adjust_case(primary_key)
        foreign_key = adjust_case(foreign_key)
        schema = create_schema_prompt_col_fk(db_id, db_schema, primary_key, foreign_key, self.dataset_path)[:-2]

        self.data = []
        for sample in data:
            prompt = sql_prompt.format(db_schema=schema, question=sample['question'])
            self.data.append({'inputs': prompt, 'labels': sample['query'], 'id': sample['id'], 'db_id': sample['db_id']})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataCollator(object):
    def __init__(self, tokenizer, return_tensors='pt', padding=True, truncation=True, src_max_length=2000, trg_max_length=1000):
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors
        self.padding = padding
        self.truncation = truncation
        self.src_max_length = src_max_length
        self.trg_max_length = trg_max_length

    def __call__(self, batch):
        input_ids, labels, ids, db_ids = [], [], [], []
        for instance in batch:
            input_ids.append(instance['inputs'])
            labels.append(instance['labels'])
            ids.append(instance['id'])
            db_ids.append(instance['db_id'])

        inputs = self.tokenizer(input_ids, return_tensors=self.return_tensors, padding=self.padding, truncation=self.truncation, max_length=self.src_max_length)
        outputs = self.tokenizer(labels, return_tensors=self.return_tensors, padding=self.padding, truncation=self.truncation, max_length=self.trg_max_length)

        fields = {
            "inputs": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": outputs.input_ids,
            "id": ids,
            "db_id": db_ids
        }
        return fields

def generate_sql(model, eval_dataset, collator, trg_max_length, verbose=0):

    start_time = time.time()
    eval_sampler = SequentialSampler(eval_dataset)
    dataloader = DataLoader(
                                eval_dataset, 
                                sampler=eval_sampler, 
                                batch_size=args.eval_batch_size, 
                                drop_last=False,
                                collate_fn=collator
                                )
    tokenizer = eval_dataset.tokenizer
    model.eval()

    with torch.no_grad():

        out_eval = {}
        for batch in tqdm(dataloader):
            input_ids = batch['inputs'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')
            db_ids = batch['db_id']
            data_ids = batch['id']
            generation_output = model.generate(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    max_new_tokens=trg_max_length,
                                    do_sample=False,
                                    return_dict_in_generate=True,
                                    output_scores=True
                                    )
            preds = generation_output['sequences'].cpu()
            logits = torch.stack(generation_output['scores'], dim=1)
            logits = logits.cpu()
            probs = torch.softmax(logits, dim=2).float()
            log_probs = torch.log_softmax(logits, dim=2).float()
            max_prob, _ = torch.max(probs, axis=2)
            max_prob = max_prob.numpy()
            entropies = ( torch.sum(probs * log_probs, axis=2) * (-1) ).numpy()

            pred_list = []
            maxprob_list = []
            entropy_list = []
            eos_token_id = tokenizer.eos_token_id
            for i in range(len(preds)):
                pred_tensor = preds[i][len(input_ids[i]):]
                pred = tokenizer.decode(pred_tensor, skip_special_tokens=True)
                maxprob_truncated = max_prob[i].tolist()
                entropy_truncated = entropies[i].tolist()
                if eos_token_id in pred_tensor:
                    pred_eos_idx = torch.nonzero(pred_tensor==eos_token_id)[0].item()
                    maxprob_truncated = maxprob_truncated[:pred_eos_idx]
                    entropy_truncated = entropy_truncated[:pred_eos_idx]

                pred_list.append(pred)
                maxprob_list.append(max_prob[i].tolist())
                entropy_list.append(entropies[i].tolist())

                result = {}
                result['db_id'] = db_ids[i]
                result['real'] = tokenizer.decode(labels[i], skip_special_tokens=True)
                result['pred'] = pred_list[i]
                result['maxprob'] = maxprob_list[i]
                result['entropy'] = entropy_list[i]
                out_eval[data_ids[i]] = result

    if verbose>0:
        print(f"inference took {round(time.time() - start_time, 6)} secs")

    return out_eval

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions with penalty.")
    parser.add_argument('--test_data_path', required=True, type=str, help='Test data')    
    parser.add_argument('--data_pool_dir', required=True, type=str, help='Path to database')
    parser.add_argument('--model_path', required=True, type=str, help='Path to model')
    parser.add_argument('--eval_batch_size', required=True, type=int, help='Eval batch size')
    parser.add_argument("--output_dir", required=True, type=str, help='Save output for error analysis')
    parser.add_argument('--src_max_length', default=2000, type=int, help='Source length')
    parser.add_argument('--trg_max_length', default=1000, type=int, help='Target length')
    return parser.parse_args()

def main(args):

    if os.path.exists(os.path.join(args.output_dir, 'prediction_raw.json')):
        raise Exception(f"directory already exists ({os.path.join(args.output_dir, 'prediction_raw.json')})")

    tokenizer = AutoTokenizer.from_pretrained(
        'defog/sqlcoder-7b-2',
        padding_side="left", # must do left
        use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
        use_auth_token=True
    )

    test_dataset = TrustSQLDataset(args.test_data_path, tokenizer=tokenizer)
    data_collator = DataCollator(tokenizer=tokenizer, return_tensors='pt', src_max_length=args.src_max_length, trg_max_length=args.trg_max_length)
    print(f"loaded {len(test_dataset)} test examples from {args.test_data_path}")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        model = model.module

    print("start inference")
    out_eval = generate_sql(model=model, eval_dataset=test_dataset, collator=data_collator, trg_max_length=args.trg_max_length, verbose=1)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'prediction_raw.json'), 'w') as f:
        json.dump(out_eval, f)

if __name__ == '__main__':
    args = parse_args()
    main(args)

