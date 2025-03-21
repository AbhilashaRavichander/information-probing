# Information-Guided Identification of Training Data Imprint in (Proprietary) Large Language Models

This repository provides the code and data of [Information-Guided Identification of Training Data Imprint in
(Proprietary) Large Language Models](https://arxiv.org/abs/2503.12072) by Abhilasha Ravichander, Jillian Fisher, Taylor Sorensen, Ximing Lu , Yuchen Lin , Maria Antoniak , Niloofar Mireshghallah , Chandra Bhagavatula, and Yejin Choi.


## Overview

High-quality training data has proven crucial for developing performant large language models (LLMs). However, commercial LLM providers disclose few, if any, details about the data used for training.  This lack of transparency creates multiple challenges: it limits external oversight and inspection of LLMs for issues such as copyright infringement, it undermines the agency of data authors, and it hinders scientific research on critical issues such as data contamination and data selection. How can we recover what training data is known to LLMs? 

We propose a new method, **information-guided probing**, to identify training data memorized by even *completely black-box proprietary LLMs* (like GPT-4) i.e. without requiring any access to model weights or token probabilities. Information-guided probes rely on a simple observation: text contains tokens that are challenging to predict based on context alone--- and if a model can predict such tokens successfully, the remaining mechanism for prediction must be based on memorized training data.

This repository contains resources related to 

👉 information-guided probing, +

👉 a new dataset of New York Times articles that were allegedly recovered from GPT-4, which can be used to evaluate the performance of methods to identify training data.

:star:  If you find our work useful, please consider citing our work  :star::

```bibtex
@misc{ravichander2025informationguidedidentificationtrainingdata,
      title={Information-Guided Identification of Training Data Imprint in (Proprietary) Large Language Models}, 
      author={Abhilasha Ravichander and Jillian Fisher and Taylor Sorensen and Ximing Lu and Yuchen Lin and Maria Antoniak and Niloofar Mireshghallah and Chandra Bhagavatula and Yejin Choi},
      year={2025},
      eprint={2503.12072},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.12072}, 
}
```


If you have any questions, please feel free to send an email to **aravicha[AT]cs.washington.edu**.


## Installation

Please run the following commands:

```
export PYTHONPATH="${PYTHONPATH}:./src/"
pip install -r requirements.txt
python setup.py
```



## 🚀 Run our method to find high-surprisal tokens


Our codebase currently supports `bert-base-uncased` as the reference model to identify high-surprisal tokens, but you can easily modify it to use other models.


To extract high-surprisal tokens from a dataset of text samples, you will need to follow the following procedure:
1. Run

```
python information-probing/score_distribution.py --output_file [PATH_TO_DISTRIBUTION_FILE]
```

This is set to run on [BookMIA](https://huggingface.co/datasets/swj0419/BookMIA). If you wish to run this on a different dataset, please modify `L 68` accordingly.

2. Run

```
python information-probing/get_surprise_tokens.py --distribution_file [PATH_TO_DISTRIBUTION_FILE] --output_filestring [OUTPUT_FILE_PREFIX] --probability_threshold -12 --rank_threshold 2000
```

This will compute surprising tokens using both probability and rank measures. You can additionally set the thresholds for these measures, for probability this will extract lower probability tokens than the value specified, and for rank this will extract higher-rank tokens than the value specified.

## Changing the reference model

To change the reference model, modify L 72 of `information-probing/score_distribution.py`. We use the HF implementation of `bert-base-uncased`, so your reference model will have to be compatible with the `AutoModelForMaskedLM, AutoTokenizer` libraries for this to work directly.

## 📰 Data

You can find the dataset of New York Times articles [here](https://huggingface.co/datasets/lasha-nlp/NYT_Memorization). 

The 📰**NYT datasets** serve as a benchmark designed to evaluate methods to identify memorized training data or to infer membership, specifically from OpenAI models that are released before 2023. 

🔧 Memorized Training Data

This dataset contains examples provided in [Exhibit-J](https://nytco-assets.nytimes.com/2023/12/Lawsuit-Document-dkt-1-68-Ex-J.pdf) of the New York Times Lawsuit (label=1). 

- `Snippet` is the article.
- `Prefix` is the prefix provided to the model according to the evidence in Exhibit-J.
- `Completion` is the original article content that follows the prefix.

🔧 Non-Member Data
  
This dataset also contains excerpts of CNN articles scraped in 2023 (label=0).
