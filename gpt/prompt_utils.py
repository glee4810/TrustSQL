import os
import re
import json
import time
import sqlite3
import pandas as pd

def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

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

def get_prompt_template(args):
    if args.prompt_type == 'col_fk':
        from prompt_utils import create_schema_prompt_col_fk
        return create_schema_prompt_col_fk
    elif args.prompt_type == 'create_col':
        from prompt_utils import create_schema_prompt_create_col        
        return create_schema_prompt_create_col
    else:
        raise Exception("Invalid prompt type")

def create_index(question_pool, retriever):
    import faiss
    question_embeddings = retriever.encode(question_pool)
    index = faiss.IndexFlatIP(question_embeddings.shape[1])
    index.add(question_embeddings)
    return index

def load_cache_files(cache_path):
    cache_prompt_input = load_json(os.path.join(cache_path, 'prompt_input.json')) if cache_path else {}
    cache_prompt_output = load_json(os.path.join(cache_path, 'prediction.json')) if cache_path else {}
    return cache_prompt_input, cache_prompt_output

def adjust_case(df):
    return df.applymap(lambda x: x.lower() if pd.notnull(x) and isinstance(x, str) else x)

def prepare_data_pool(data_pool_dir):
    data_pool_files = [f for f in os.listdir(data_pool_dir) if '_train.json' in f or '_valid_combined_subset.json' in f]
    data_pool = [item for sublist in [load_json(os.path.join(data_pool_dir, file)) for file in data_pool_files] for item in sublist]
    return data_pool

def create_retrievers(model):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model)

def run_engine(client, model, prompt, temperature, n):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature if n > 1 else 0.0,
        n=n,
        max_tokens=512,
        stop=["#", ";", "Question:"]
    )
    return [choice.message.content.strip() for choice in response.choices] if n > 1 else response.choices[0].message.content.strip()


def handle_batch_api(client, args, prompts):
    batches = []
    for id_ in prompts:
        instance = {
            "custom_id": id_,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": args.model,
                "messages": [{"role": "user", "content": prompts[id_]}],
                "max_tokens": 512,
                "stop": ["#", ";", "Question:"],
                "temperature": args.temp,
                "n": args.n
            }
        }
        batches.append(instance)

    with open(os.path.join(args.inference_result_path, "batchapi_input.jsonl"), "w", encoding="utf-8") as f:
        for sample in batches:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

    input_file = client.files.create(
        file=open(os.path.join(args.inference_result_path, "batchapi_input.jsonl"), "rb"),
        purpose="batch"
    )

    batch_info = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "nightly eval job"}
    )

    while True:
        time.sleep(1)
        batchapi_status = client.batches.retrieve(batch_info.id)
        if batchapi_status.status == 'completed':
            break

    content = client.files.content(batchapi_status.output_file_id)
    byte_string = content.content
    decoded_string = byte_string.decode('utf-8')
    json_strings = decoded_string.strip().split('\n')
    parsed_objects = [json.loads(json_str) for json_str in json_strings]
    
    output = {}
    for item in parsed_objects:
        output[item['custom_id']] = [pred['message']['content'] for pred in item['response']['body']['choices']]

    save_json(output, os.path.join(args.inference_result_path, args.output_file))

def handle_individual_api(client, args, prompts, cache_prompt_output={}):
    from tqdm import tqdm
    results = {}
    for id_ in tqdm(prompts):
        if id_ not in cache_prompt_output:
            while True:
                try:
                    pred = run_engine(client, args.model, prompts[id_], temperature=args.temp, n=args.n)
                    break
                except KeyboardInterrupt:
                    exit()
                except Exception as e:
                    print(f"Retrying due to exception: {e}")
                    time.sleep(args.wait_time)
        else:
            pred = cache_prompt_output[id_]

        results[id_] = pred
        
        save_json(results, os.path.join(args.inference_result_path, args.output_file))

