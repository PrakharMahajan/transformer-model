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
1. Importing the necessary libraries: The code starts by importing the required libraries, including pandas, scikit-learn, tqdm, and transformers.

2. Loading and preprocessing the dataset: The script reads the IMDB movie review dataset from a CSV file and converts the sentiment labels from "positive" and "negative" to binary values (1 and 0, respectively). It then splits the dataset into training and testing sets.

3. Tokenization: The script uses the BERT tokenizer from the transformers library to tokenize the movie review sentences. It iterates over the sentences and encodes them using the tokenizer, adding special tokens such as "[CLS]" and "[SEP]". The encoded sentences are stored in the `input_ids` list.

4. Padding: The script pads the input tokens with zeros to ensure they have the same length. It uses the `pad_sequences` function from the Keras library to perform padding.

5. Creating attention masks: The script creates attention masks for the input sentences. The attention mask is a binary tensor that indicates which tokens should be attended to (1) and which should be ignored (0). It iterates over the padded input tokens and sets the attention mask value to 0 for padding tokens and 1 for real tokens. The attention masks are stored in the `attention_masks` list.

6. Splitting the data: The script splits the input tokens, attention masks, and labels into training and validation sets using the `train_test_split` function from scikit-learn.

7. Converting data to tensors: The script converts the training and validation inputs, masks, and labels to PyTorch tensors.

8. Creating data loaders: The script creates data loaders to load the training and validation data in batches during training. It uses the `TensorDataset`, `DataLoader`, `RandomSampler`, and `SequentialSampler` classes from the PyTorch library.

9. Setting up the BERT model: The script loads the BERT model for sequence classification (`BertForSequenceClassification`) from the transformers library. It configures the model to use the "bert-base-uncased" pre-trained model and sets the number of output labels to 2 (positive and negative). The model is moved to the GPU if available.

10. Setting up the optimizer and scheduler: The script sets up the AdamW optimizer and a linear learning rate scheduler using the `get_linear_schedule_with_warmup` function from transformers.

11. Training the model: The script enters a training loop where it trains the BERT model on the training data. It iterates over the training data batches, performs a forward pass through the model, calculates the loss, computes gradients, updates model parameters, and adjusts the learning rate. It also measures the average training loss and reports it after each epoch.

12. Evaluating the model: After each training epoch, the script evaluates the model on the validation data. It calculates the accuracy of the model's predictions and reports it.

13. Testing the model: Finally, the script loads the test dataset, tokenizes the test sentences, and evaluates the model on the test data, reporting the accuracy.

Overall, the script demonstrates the process of using the BERT model for sentiment analysis on the IMDB movie review dataset, including data preprocessing, tokenization, model training, and evaluation.

### Remarks
This is a clone of transformers model taken from - "https://www.kaggle.com/code/samarthsarin/bert-with-transformers".

Dataset available here - https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
