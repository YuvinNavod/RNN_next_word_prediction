# Next Word Prediction Using RNN

This repository contains an implementation of a Recurrent Neural Network (RNN) for predicting the next word in a sequence. The model is built using TensorFlow and Keras and demonstrates various techniques in data preprocessing, model training, and evaluation.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Recurrent Neural Networks (RNNs) are powerful tools for sequence prediction problems. This project aims to build an RNN model to predict the next word in a given sequence of words. The model is trained on a dataset that contains sequences of text, and the goal is to achieve accurate predictions to assist in text generation tasks.

## Dataset

The dataset used for training the model consists of sequences of text extracted from a variety of sources. The text data is preprocessed to create sequences of a fixed length, where each sequence is used to predict the subsequent word.

## Model Architecture

The RNN model is built using TensorFlow and Keras. The architecture consists of:
- An embedding layer to convert words into dense vectors.
- One or more LSTM (Long Short-Term Memory) layers to capture temporal dependencies.
- A dense layer with softmax activation to predict the next word.

## Usage

### Requirements

To run the code in this repository, you need the following dependencies:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Jupyter Notebook (optional, for running the notebook)

### Running the Model

1. Clone the repository:
    ```bash
    git clone https://github.com/YuvinNavod/RNN_next_word_prediction
.git
    cd next_word_prediction
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Open the Jupyter Notebook:
    ```bash
    jupyter notebook next_word_prediction.ipynb
    ```

4. Follow the steps in the notebook to preprocess the data, train the model, and evaluate the results.

## Results

The trained RNN model achieves promising results in predicting the next word in a sequence. Detailed results and evaluation metrics are provided in the notebook.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create an issue or submit a pull request.
