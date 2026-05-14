# 🛡️TrustSQL🛡️: Benchmarking Text-to-SQL Reliability with Penalty-Based Scoring

TrustSQL is a benchmark for reliable text-to-SQL modeling. It evaluates whether a model can generate correct SQL for feasible questions and abstain from infeasible questions.

## Dataset

Place the TrustSQL dataset under `dataset/` in the repository root:

```
TrustSQL/
  dataset/
    atis/
    advising/
    ehrsql/
    spider/
```

Download the dataset from Google Drive:

- TrustSQL dataset: https://drive.google.com/file/d/19IpLSc2QncO2273E8z-lvU2ME9wEHIan/view?usp=sharing

For example:

```bash
python3 -m pip install gdown
python3 -m gdown -O trustsql_dataset.zip --fuzzy "https://drive.google.com/file/d/19IpLSc2QncO2273E8z-lvU2ME9wEHIan/view?usp=sharing"
unzip trustsql_dataset.zip -x "__MACOSX/*"
```

If the unzipped directory has a different name, rename or move it so that `dataset/atis`, `dataset/advising`, `dataset/ehrsql`, and `dataset/spider` exist directly under the repository root.

Each dataset directory contains schema files, SQLite databases, and split files such as `{dataset}_train.json`, `{dataset}_valid.json`, and `{dataset}_test.json`.

The expected split sizes are:

| Dataset | Train | Valid feasible | Valid infeasible | Test feasible | Test infeasible |
|:--|--:|--:|--:|--:|--:|
| ATIS | 1,114 | 489 | 489 | 476 | 476 |
| Advising | 1,170 | 533 | 533 | 533 | 533 |
| EHRSQL | 4,674 | 931 | 931 | 934 | 934 |
| Spider | 7,000 | 507 | 507 | 527 | 527 |

`dataset/` is intentionally ignored by git because it contains large database files and benchmark data.

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For OpenAI-based experiments, create `gpt/api.json`:

```json
{
  "API_KEY": "YOUR_API_KEY"
}
```

`gpt/api.json` is ignored by git.

## Running OpenAI Baselines

Run SQLPrompt:

```bash
bash script/run_sqlprompt.sh
```

Run the pipeline baseline:

```bash
bash script/run_clsprompt.sh
bash script/run_errorprompt.sh
```

Run SQLPrompt with demonstrations or voting:

```bash
bash script/run_sqlprompt_demo.sh
bash script/run_sqlprompt_voting.sh
```

Run the T5 evaluation example after placing checkpoints at the paths specified in the T5 config files:

```bash
bash script/run_t5_eval_example.sh
```

## Evaluation

Evaluate generated outputs with:

```bash
bash script/evaluate_sqlprompt_cls+error.sh
bash script/evaluate_sqlprompt_demo.sh
bash script/evaluate_sqlprompt_voting.sh
```

## Additional Model Code

This repository also includes model code for T5, SQLCoder, and Llama variants. These model implementations may require external checkpoints or generated outputs that are not committed to git.

`TriageSQL v1.3/` is only needed for some T5 Mahalanobis or FeatRMD-style background-distribution experiments. It is a large external artifact and is ignored by git.

To obtain it, download the official TriageSQL dataset from the TriageSQL repository:

- TriageSQL GitHub: https://github.com/yszh8/TriageSQL
- Raw TriageSQL dataset: https://drive.google.com/file/d/1w55CaVEuimUlP-jerOCrVHF1iF0FZYKe/view?usp=sharing
- Processed TriageSQL dataset: https://drive.google.com/file/d/1ol1xFpGuH0BdLw26MvQoeCHLOtTqQ60i/view?usp=sharing

For the TrustSQL T5 background-data scripts, use the raw TriageSQL dataset and place it as:

```
TrustSQL/
  TriageSQL v1.3/
    trainset.json
    devset.json
    testset.json
```

For example:

```bash
python3 -m pip install gdown
python3 -m gdown --fuzzy "https://drive.google.com/file/d/1w55CaVEuimUlP-jerOCrVHF1iF0FZYKe/view?usp=sharing"
unzip TriageSQL.zip
mv TriageSQL "TriageSQL v1.3"
```

If the unzipped directory has a different name, rename it so that `TriageSQL v1.3/trainset.json`, `TriageSQL v1.3/devset.json`, and `TriageSQL v1.3/testset.json` exist directly under the repository root.

## Citation

```
@article{lee2024trustsql,
  title={TrustSQL: Benchmarking Text-to-SQL Reliability with Penalty-Based Scoring}, 
  author={Lee, Gyubok and Chay, Woosog and Cho, Seonhee and Choi, Edward},
  journal={arXiv preprint arXiv:2403.15879},
  year={2024}
}
```
