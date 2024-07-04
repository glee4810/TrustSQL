export CUDA_VISIBLE_DEVICES=7

echo "ATIS"
python gpt/error_prompt.py \
    --api_key_path gpt/api.json \
    --data_pool_dir dataset/atis \
    --test_data_path dataset/atis/atis_test.json \
    --target_path outputs/eval_atis_test_chatgpt_colfk_pos8/prediction.json \
    --inference_result_path outputs/eval_atis_test_gpt4o_create_select3_pos16neg16_errorprompt

echo "Advising"
python gpt/error_prompt.py \
    --api_key_path gpt/api.json \
    --data_pool_dir dataset/advising \
    --test_data_path dataset/advising/advising_test.json \
    --target_path outputs/eval_advising_test_chatgpt_colfk_pos8/prediction.json \
    --inference_result_path outputs/eval_advising_test_gpt4o_create_select3_pos16neg16_errorprompt

echo "EHRSQL"
python gpt/error_prompt.py \
    --api_key_path gpt/api.json \
    --data_pool_dir dataset/ehrsql \
    --test_data_path dataset/ehrsql/ehrsql_test.json \
    --target_path outputs/eval_ehrsql_test_chatgpt_colfk_pos8/prediction.json \
    --inference_result_path outputs/eval_ehrsql_test_gpt4o_create_select3_pos16neg16_errorprompt

echo "Spider"
python gpt/error_prompt.py \
    --api_key_path gpt/api.json \
    --data_pool_dir dataset/spider \
    --test_data_path dataset/spider/spider_test.json \
    --target_path outputs/eval_spider_test_chatgpt_colfk_pos8/prediction.json \
    --inference_result_path outputs/eval_spider_test_gpt4o_create_select3_pos16neg16_errorprompt
