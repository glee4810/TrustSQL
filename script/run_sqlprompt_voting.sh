export CUDA_VISIBLE_DEVICES=7

echo "ATIS"
python gpt/sql_prompt.py \
    --api_key_path gpt/api.json \
    --data_pool_dir dataset/atis \
    --test_data_path dataset/atis/atis_test.json \
    --inference_result_path outputs/eval_atis_test_gpt4o_colfk_pos8_voting5 \
    --pos_num_sample 8 \
    --n 5

echo "Advising"
python gpt/sql_prompt.py \
    --api_key_path gpt/api.json \
    --data_pool_dir dataset/advising \
    --test_data_path dataset/advising/advising_test.json \
    --inference_result_path outputs/eval_advising_test_gpt4o_colfk_pos8_voting5 \
    --pos_num_sample 8 \
    --n 5

echo "EHRSQL"
python gpt/sql_prompt.py \
    --api_key_path gpt/api.json \
    --data_pool_dir dataset/ehrsql \
    --test_data_path dataset/ehrsql/ehrsql_test.json \
    --inference_result_path outputs/eval_ehrsql_test_gpt4o_colfk_pos8_voting5 \
    --pos_num_sample 8 \
    --n 5

echo "Spider"
python gpt/sql_prompt.py \
    --api_key_path gpt/api.json \
    --data_pool_dir dataset/spider \
    --test_data_path dataset/spider/spider_test.json \
    --inference_result_path outputs/eval_spider_test_gpt4o_colfk_pos8_voting5 \
    --pos_num_sample 8 \
    --n 5