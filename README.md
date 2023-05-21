# transformer-model (BERT)

## Introduction
BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art transformer-based language model introduced by Google AI in 2018. It revolutionized natural language processing (NLP) tasks by significantly improving performance on various benchmarks, including sentence classification, named entity recognition, and question answering.

The key innovation of BERT lies in its ability to capture bidirectional context, enabling it to understand the meaning of a word based on its surrounding words on both sides. This is achieved through a transformer architecture, which is a deep neural network model consisting of multiple layers of self-attention and feed-forward neural networks.

Here's a step-by-step description of the BERT model:

1. Tokenization: The input text is first split into individual tokens. These tokens can be as short as one character or as long as one word. Additionally, special tokens are added to mark the beginning and end of the text, separate sentences, and indicate padding.

2. WordPiece Embeddings: BERT uses WordPiece embeddings, where the input tokens are mapped to fixed-length vectors. Each token is represented by a combination of subword units, allowing the model to handle out-of-vocabulary words and capture more fine-grained information.

3. Input Representation: BERT takes as input a sequence of token embeddings, position embeddings, and segment embeddings. The position embeddings convey the positional information of tokens in the input sequence, while the segment embeddings differentiate between different sentences in the input.

4. Transformer Layers: BERT consists of multiple stacked transformer layers. Each layer has two sub-layers: a multi-head self-attention mechanism and a position-wise feed-forward neural network.

   a. Self-Attention: The self-attention mechanism allows the model to capture contextual relationships between tokens. It computes attention weights for each token based on its interactions with other tokens in the sequence. This enables BERT to weigh the importance of different tokens during encoding.

   b. Feed-Forward Networks: The feed-forward networks apply a non-linear transformation to the outputs of the self-attention layer. It processes each token independently and projects it into a higher-dimensional space, allowing for more complex interactions between tokens.

5. Pre-training and Fine-tuning: BERT is pre-trained on large amounts of unlabeled text data using two unsupervised learning tasks: masked language modeling (MLM) and next sentence prediction (NSP). MLM randomly masks some of the input tokens, and the model is trained to predict the original masked tokens. NSP involves predicting whether two sentences appear consecutively in the original text.

6. Fine-tuning: After pre-training, BERT can be fine-tuned on specific downstream NLP tasks. Additional task-specific layers are added on top of the pre-trained BERT model, and the entire network is fine-tuned using labeled task-specific data. The fine-tuning process adapts BERT to the specific task, allowing it to achieve state-of-the-art performance on various NLP benchmarks.

BERT's major advantages are its ability to capture bidirectional context, handle out-of-vocabulary words, and transfer knowledge through fine-tuning. It has been widely adopted and has led to significant advancements in a wide range of NLP applications.

## Code Description


This is a clone of transformers model taken from - "https://www.kaggle.com/code/samarthsarin/bert-with-transformers".

Dataset available here - https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
