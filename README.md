# Language Modelling - CBOW Word Embeddings Project

This repository contains all the code, models, data, and documentation for the **Language Modelling** assignment, which involved training CBOW-based word embeddings and evaluating them on a downstream text classification task (Reuters dataset). The experiments were conducted as part of the Master in Artificial Intelligence at UDC (Academic Year 2024-2025).

## Structure

- `main.ipynb`: Main Jupyter notebook with all stages of the project, from preprocessing to training and evaluation.
- `build_model.py`: Python script to define and compile the classification model.
- `tokenizer.ipynb`: Notebook for tokenizing and preprocessing the raw corpus.
- `utils.ipynb`: Utility functions used throughout the project.
- `eng_news_2024_30k_sentences.txt`: Raw corpus used to train the embeddings (30,000 English news sentences).
- `target_words.txt`: List of sample words used for similarity analysis.
- `lab_assignment.pdf`: Original assignment instructions.
- `lm_guide.pdf`: Official guide provided for the lab.
- `LM_finalReport.pdf`: Final 3-page report summarizing methodology, results, and conclusions.

## Embedding Files

Pretrained word embeddings stored in `.npy` format:
- `embeddings_w2_d50.npy`, `embeddings_w2_d100.npy`, `embeddings_w2_d200.npy`
- `embeddings_w5_d50.npy`, `embeddings_w5_d100.npy`, `embeddings_w5_d200.npy`

These correspond to different configurations of:
- Window size: 2 or 5
- Embedding dimension: 50, 100, or 200

Additionally:
- `trained_embeddings.npy`: One consolidated embedding matrix used for final classifier runs.

## Trained Classifier Models

Keras H5 model files:
- `model_baseline.h5`: CNN trained with randomly initialized embeddings.
- `model_pretrained_finetuned.h5`: CNN with pretrained embeddings that were fine-tuned.
- `model_pretrained_frozen.h5`: CNN with pretrained embeddings kept frozen during training.

## How to Run

1. Open `main.ipynb` in JupyterLab or Google Colab.
2. Run all cells in order to train the CBOW model, generate embeddings, and evaluate classifier performance.
3. To skip embedding training, use the saved `.npy` files and go directly to classification.
4. GPU usage is recommended for faster training.

## Summary of Experiments

Experiments include:
- Varying context window size (2 vs 5)
- Varying embedding dimension (50, 100, 200)
- Comparing baseline (random) vs pretrained embeddings (frozen and fine-tuned)
- Testing classifier capacity (number of filters)
- Evaluating different batch sizes (32, 64, 128)

For results and analysis, see `LM_finalReport.pdf`.

## Authors

- Pablo Fuentes Chemes  
- Paula Biderman Mato

Master in Artificial Intelligence – Universidad de A Coruña  
