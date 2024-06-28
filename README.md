# TrustSQL: Benchmarking Text-to-SQL Reliability with Penalty-Based Scoring

## Overview

Text-to-SQL allows users to interact with databases using natural language, simplifying information retrieval without requiring special knowledge of databases or SQL. Despite the success of large language models in translating natural language to SQL, widespread deployment remains limited due to two main challenges: (1) users' lack of understanding of the models' capabilities (such as individual models' text-to-SQL generation capabilities, the scope of the database contents, and SQL functionalities) and (2) the absence of abstention mechanisms, which can result in incorrect SQL generation going unnoticed.

We believe that building trust in model outputs is crucial for the broader adoption of text-to-SQL models, thereby unlocking the full potential of big data stored in databases, especially for domain experts. As a step towards this goal, we introduce TrustSQL, a new benchmark designed to evaluate text-to-SQL reliability, defined as a model's ability to accurately handle any type of input question, whether feasible or infeasible, by allowing models to selectively generate correct SQL and abstain from generating incorrect SQL.

The benchmark data is released along with our paper [TrustSQL: Benchmarking Text-to-SQL Reliability with Penalty-Based Scoring](https://arxiv.org/abs/2403.15879) (Preprint). For further details, please refer to the paper.


## Interactive Demo

<p align="middle" float="middle">
  <img src="image/demo.png" height="300" />
</p>

Please check out our [demo page](http://3.35.140.88:8501) for an interactive experience of the TrustSQL benchmark. It's important to note that the model used for this demo still makes mistakes in SQL generation without abstention.


## Dataset

TrustSQL can be downloaded from this [Google Drive link](https://drive.google.com/file/d/1j0Otma8aC-e8-7ej-nwObpB-S-pxPpZX/view?usp=sharing). Locate the unzipped file in the main directory.

Text-to-SQL data is licensed under CC-BY-4.0. One of the four databases in TrustSQL is based on the MIMIC-IV demo. Please review the corresponding license terms on the [MIMIC-IV Demo website](https://physionet.org/content/mimic-iv-demo/2.2/) before use.

Each database consists of schema information (`tables.json`), actual database (`{database}.sqlite`), and text-to-SQL samples for each data split (`{database}_{split}.json`). Files ending with "combined_subset" contain TrustSQL samples combined with data sampled from question answering datasets (i.e., MS Marco, CoQA, and QuAC) to augment infeasible samples in the training and validation splits.


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
