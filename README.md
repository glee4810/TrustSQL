# 🛡️TrustSQL🛡️: Benchmarking Text-to-SQL Reliability with Penalty-Based Scoring

## Overview

Text-to-SQL allows users to interact with databases using natural language, simplifying information retrieval without requiring special knowledge of databases or SQL. Despite the success of large language models in translating natural language to SQL, widespread deployment remains limited due to two main challenges: (1) users' lack of understanding of the models' capabilities (e.g., individual models' text-to-SQL generation capabilities, the exact scope of database contents, and SQL functionalities) and (2) the absence of abstention mechanisms, which can result in incorrect SQL generation going unnoticed.

We believe that building trust in model outputs is crucial for the broader adoption of text-to-SQL models. As a step towards this goal, we introduce TrustSQL, a new benchmark designed to evaluate text-to-SQL reliability, defined as a model's ability to accurately handle any type of input question (i.e., correct SQL generation for feasible questions and abstention for infeasible questions). TrustSQL contains four text-to-SQL datasets (ATIS, Advising, EHRSQL, and Spider), augmented with diverse infeasible questions regarding text-to-SQL tasks (e.g., questions that go beyond database contents and SQL functionalities).

With these diverse questions, we allow models to have the option to abstain while generating SQL (termed reliable text-to-SQL modeling) and evaluate their performance using a novel penalty-based scoring metric, the Reliability Score (RS), tailored to measure text-to-SQL reliability. Due to the penalty term, the challenge in TrustSQL is not only to maximize correct SQL generation for feasible questions and abstention for infeasible questions but also to minimize mistakes in SQL generation (i.e., incorrect SQL generation for feasible questions and failure to abstain from generating SQL for infeasible questions).

The benchmark data is released along with our paper [TrustSQL: Benchmarking Text-to-SQL Reliability with Penalty-Based Scoring](https://arxiv.org/abs/2403.15879) (Preprint). For further details, please refer to the paper.


## Interactive Demo

<p align="middle" float="middle">
  <img src="image/demo.png" height="300" />
</p>

Please check out our [demo page](https://trustsqlapp-2gmvmktqkar8fzcyvwyj9k.streamlit.app/) for an interactive experience of the TrustSQL benchmark. It is important to note that the model used for this demo still makes mistakes in SQL generation. The goal of the TrustSQL benchmark is to remove these mistakes by allowing models to selectively generate only correct SQL among any input questions.


## Dataset

TrustSQL can be downloaded from this [Google Drive link](https://drive.google.com/file/d/1j0Otma8aC-e8-7ej-nwObpB-S-pxPpZX/view?usp=sharing). Locate the unzipped file in the main directory.

Text-to-SQL data is licensed under CC-BY-4.0. One of the four databases in TrustSQL is based on the MIMIC-IV demo. Please review the corresponding license terms on the [MIMIC-IV Demo website](https://physionet.org/content/mimic-iv-demo/2.2/) before use.

Each database consists of schema information (`tables.json`), actual database (`{database}.sqlite`), and text-to-SQL samples for each data split (`{database}_{split}.json`). Files ending with "combined_subset" contain TrustSQL samples combined with data sampled from question answering datasets (i.e., MS Marco, CoQA, and QuAC) to augment infeasible samples.


## Models

The TrustSQL benchmark can be addressed using two approaches: pipeline-based and unified. Below are three example methods utilizing OpenAI models.

### Pipeline-based Approach

1) CLS<sub>P</sub> → SQLPROMPT → ERROR<sub>P</sub> (referred to as SQLPipeline)

<p align="left" float="middle">
  <img src="image/pipeline.png" height="120" />
</p>

```
bash ./script/run_clsprompt.sh # CLS_Prompt - Pre-abstention before SQL generation (i.e., filtering out infeasible questions)
bash ./script/run_sqlprompt.sh # SQLPrompt - SQL generation
bash ./script/run_errorprompt.sh # Error_Prompt - Post-abstention after SQL generation (i.e., filtering out incorrect SQL outputs)
```

### Unified Approach

2) SQLPROMPT[Demo]

<p align="left" float="middle">
  <img src="image/sqlprompt_demo.png" height="70" />
</p>

```
bash ./script/run_sqlprompt_demo.sh # SQLPrompt[Demo] - SQL generation and abstention through demonstrations
```

3) SQLPROMPT[Voting]

<p align="left" float="middle">
  <img src="image/sqlprompt_voting.png" height="150" />
</p>

```
bash ./script/run_sqlprompt_voting.sh # SQLPrompt[Voting] - SQL generation and abstention through voting 5 sampled outputs
```


## Evaluation

To evaluate the model performance, run the following code:

```
# 1) SQLPipeline
bash script/evaluate_sqlprompt_cls+error.sh
```

```
# 2) SQLPROMPT[Demo]
bash script/evaluate_sqlprompt_demo.sh
```

```
# 3) SQLPROMPT[Voting]
bash script/evaluate_sqlprompt_voting.sh
```

## Result

<table>
  <tr>
    <th style="text-align: center;">Model</th>
    <th colspan="4" style="text-align: center;">RS(0)</th>
    <th colspan="4" style="text-align: center;">RS(10)</th>
    <th colspan="4" style="text-align: center;">RS(N)</th>
  </tr>
  <tr>
    <td></td>
    <td style="text-align: center;">ATIS</td>
    <td style="text-align: center;">Advising</td>
    <td style="text-align: center;">EHRSQL</td>
    <td style="text-align: center;">Spider</td>
    <td style="text-align: center;">ATIS</td>
    <td style="text-align: center;">Advising</td>
    <td style="text-align: center;">EHRSQL</td>
    <td style="text-align: center;">Spider</td>
    <td style="text-align: center;">ATIS</td>
    <td style="text-align: center;">Advising</td>
    <td style="text-align: center;">EHRSQL</td>
    <td style="text-align: center;">Spider</td>
  </tr>
  <tr>
    <td style="text-align: center;">SQLPipeline</td>
    <td style="text-align: center;">54.5</td>
    <td style="text-align: center;">58.9</td>
    <td style="text-align: center;">57.4</td>
    <td style="text-align: center;">67.6</td>
    <td style="text-align: center;">51.4</td>
    <td style="text-align: center;">45.8</td>
    <td style="text-align: center;">47.3</td>
    <td style="text-align: center;">-41.5</td>
    <td style="text-align: center;">-245.5</td>
    <td style="text-align: center;">-1.3K</td>
    <td style="text-align: center;">-1.8K</td>
    <td style="text-align: center;">-11.4K</td>
  </tr>
  <tr>
    <td style="text-align: center;">SQLPROMPT[Demo]</td>
    <td style="text-align: center;">73.4</td>
    <td style="text-align: center;">70.4</td>
    <td style="text-align: center;">69.5</td>
    <td style="text-align: center;">66.8</td>
    <td style="text-align: center;">-190.2</td>
    <td style="text-align: center;">-226.1</td>
    <td style="text-align: center;">-231.9</td>
    <td style="text-align: center;">-265.3</td>
    <td style="text-align: center;">-25.0K</td>
    <td style="text-align: center;">-31.5K</td>
    <td style="text-align: center;">-56.2K</td>
    <td style="text-align: center;">-34.9K</td>
  </tr>
  <tr>
    <td style="text-align: center;">SQLPROMPT[Voting]*</td>
    <td style="text-align: center;">76.5</td>
    <td style="text-align: center;">73.7</td>
    <td style="text-align: center;">78.8</td>
    <td style="text-align: center;">51.0</td>
    <td style="text-align: center;">42.9</td>
    <td style="text-align: center;">39.0</td>
    <td style="text-align: center;">37.0</td>
    <td style="text-align: center;">23.5</td>
    <td style="text-align: center;">-3.1K</td>
    <td style="text-align: center;">-3.6K</td>
    <td style="text-align: center;">-7.7K</td>
    <td style="text-align: center;">-2.8K</td>
  </tr>
</table>

- RS(0): No penalty for incorrect SQL generation
- RS(10): A penalty of 10 is applied; 1 incorrect SQL weighs the same as 10 correct model decisions (i.e., correct SQL for feasible questions and abstention for infeasible questions)
- RS(N): A penalty of N (size of eval data) is applied; 1 incorrect SQL outweighs the rest of model decisions being all correct

\*New results after postprocessing updates (postprocessing after unanimous voting -> postprocessing before unanimous voting)



## Have Questions?

Ask us questions on our Github issues page or contact gyubok.lee@kaist.ac.kr.


## Citation

```
@article{lee2024trustsql,
  title={TrustSQL: Benchmarking Text-to-SQL Reliability with Penalty-Based Scoring}, 
  author={Lee, Gyubok and Chay, Woosog and Cho, Seonhee and Choi, Edward},
  journal={arXiv preprint arXiv:2403.15879},
  year={2024}
}
```
