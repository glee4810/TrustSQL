export CUDA_VISIBLE_DEVICES=0

echo "ATIS"
python gpt/error_prompt.py \
    --model gpt-4o \
    --api_key_path gpt/api.json \
    --data_pool_dir dataset/atis \
    --test_data_path dataset/atis/atis_test.json \
    --target_path outputs/eval_atis_test_gpt3.5turbo_colfk_pos8/prediction.json \
    --inference_result_path outputs/eval_atis_test_gpt3.5turbo_colfk_pos8__gpt4o_colfk_errorprompt

echo "Advising"
python gpt/error_prompt.py \
    --model gpt-4o \
    --api_key_path gpt/api.json \
    --data_pool_dir dataset/advising \
    --test_data_path dataset/advising/advising_test.json \
    --target_path outputs/eval_advising_test_gpt3.5turbo_colfk_pos8/prediction.json \
    --inference_result_path outputs/eval_advising_test_gpt3.5turbo_colfk_pos8__gpt4o_colfk_errorprompt

echo "EHRSQL"
python gpt/error_prompt.py \
    --model gpt-4o \
    --api_key_path gpt/api.json \
    --data_pool_dir dataset/ehrsql \
    --test_data_path dataset/ehrsql/ehrsql_test.json \
    --target_path outputs/eval_ehrsql_test_gpt3.5turbo_colfk_pos8/prediction.json \
    --inference_result_path outputs/eval_ehrsql_test_gpt3.5turbo_colfk_pos8__gpt4o_colfk_errorprompt

echo "Spider"
python gpt/error_prompt.py \
    --model gpt-4o \
    --api_key_path gpt/api.json \
    --data_pool_dir dataset/spider \
    --test_data_path dataset/spider/spider_test.json \
    --target_path outputs/eval_spider_test_gpt3.5turbo_colfk_pos8/prediction.json \
    --inference_result_path outputs/eval_spider_test_gpt3.5turbo_colfk_pos8__gpt4o_colfk_errorprompt
