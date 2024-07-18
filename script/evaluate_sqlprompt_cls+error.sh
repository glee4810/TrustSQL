# ATIS
echo "ATIS"
python postprocess_abstention.py \
    --db_path ./dataset/atis/atis.sqlite \
    --pred_file ./outputs/eval_atis_test_gpt3.5turbo_colfk_pos8/prediction.json \
    --save_file ./outputs/eval_atis_test_gpt3.5turbo_colfk_pos8/model_decision.json \
    --pre_abst_path ./outputs/eval_atis_test_gpt4o_create_select_pos16neg16_clsprompt/prediction.json \
    --post_abst_path ./outputs/eval_atis_test_gpt3.5turbo_colfk_pos8__gpt4o_colfk_errorprompt/prediction.json
python evaluate.py \
    --db_path ./dataset/atis/atis.sqlite \
    --data_file ./dataset/atis/atis_test.json \
    --pred_file ./outputs/eval_atis_test_gpt3.5turbo_colfk_pos8/model_decision.json

# Advising
echo "Advising"
python postprocess_abstention.py \
    --db_path ./dataset/advising/advising.sqlite \
    --pred_file ./outputs/eval_advising_test_gpt3.5turbo_colfk_pos8/prediction.json \
    --save_file ./outputs/eval_advising_test_gpt3.5turbo_colfk_pos8/model_decision.json \
    --pre_abst_path ./outputs/eval_advising_test_gpt4o_create_select_pos16neg16_clsprompt/prediction.json \
    --post_abst_path ./outputs/eval_advising_test_gpt3.5turbo_colfk_pos8__gpt4o_colfk_errorprompt/prediction.json
python evaluate.py \
    --db_path ./dataset/advising/advising.sqlite \
    --data_file ./dataset/advising/advising_test.json \
    --pred_file ./outputs/eval_advising_test_gpt3.5turbo_colfk_pos8/model_decision.json

# EHRSQL
echo "EHRSQL"
python postprocess_abstention.py \
    --db_path ./dataset/ehrsql/mimic_iv.sqlite \
    --pred_file ./outputs/eval_ehrsql_test_gpt3.5turbo_colfk_pos8/prediction.json \
    --save_file ./outputs/eval_ehrsql_test_gpt3.5turbo_colfk_pos8/model_decision.json \
    --pre_abst_path ./outputs/eval_ehrsql_test_gpt4o_create_select_pos16neg16_clsprompt/prediction.json \
    --post_abst_path ./outputs/eval_ehrsql_test_gpt3.5turbo_colfk_pos8__gpt4o_colfk_errorprompt/prediction.json
python evaluate.py \
    --db_path ./dataset/ehrsql/mimic_iv.sqlite \
    --data_file ./dataset/ehrsql/ehrsql_test.json \
    --pred_file ./outputs/eval_ehrsql_test_gpt3.5turbo_colfk_pos8/model_decision.json

# Spider
echo "Spider"
python postprocess_abstention.py \
    --db_path ./dataset/spider/database \
    --pred_file ./outputs/eval_spider_test_gpt3.5turbo_colfk_pos8/prediction.json \
    --save_file ./outputs/eval_spider_test_gpt3.5turbo_colfk_pos8/model_decision.json \
    --pre_abst_path ./outputs/eval_spider_test_gpt4o_create_select_pos16neg16_clsprompt/prediction.json \
    --post_abst_path ./outputs/eval_spider_test_gpt3.5turbo_colfk_pos8__gpt4o_colfk_errorprompt/prediction.json
python evaluate.py \
    --db_path ./dataset/spider/database \
    --data_file ./dataset/spider/spider_test.json \
    --pred_file ./outputs/eval_spider_test_gpt3.5turbo_colfk_pos8/model_decision.json
