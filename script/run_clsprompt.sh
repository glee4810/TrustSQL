export CUDA_VISIBLE_DEVICES=0

echo "ATIS"
python gpt/cls_prompt.py \
    --model gpt-4o \
    --api_key_path gpt/api.json \
    --data_pool_dir dataset/atis \
    --test_data_path dataset/atis/atis_test.json \
    --inference_result_path outputs/eval_atis_test_gpt4o_create_select_pos16neg16_clsprompt \
    --pos_num_sample 16 \
    --neg_num_sample 16

echo "Advising"
python gpt/cls_prompt.py \
    --model gpt-4o \
    --api_key_path gpt/api.json \
    --data_pool_dir dataset/advising \
    --test_data_path dataset/advising/advising_test.json \
    --inference_result_path outputs/eval_advising_test_gpt4o_create_select_pos16neg16_clsprompt \
    --pos_num_sample 16 \
    --neg_num_sample 16

echo "EHRSQL"
python gpt/cls_prompt.py \
    --model gpt-4o \
    --api_key_path gpt/api.json \
    --data_pool_dir dataset/ehrsql \
    --test_data_path dataset/ehrsql/ehrsql_test.json \
    --inference_result_path outputs/eval_ehrsql_test_gpt4o_create_select_pos16neg16_clsprompt \
    --pos_num_sample 16 \
    --neg_num_sample 16

echo "Spider"
python gpt/cls_prompt.py \
    --model gpt-4o \
    --api_key_path gpt/api.json \
    --data_pool_dir dataset/spider \
    --test_data_path dataset/spider/spider_test.json \
    --inference_result_path outputs/eval_spider_test_gpt4o_create_select_pos16neg16_clsprompt \
    --pos_num_sample 16 \
    --neg_num_sample 16
