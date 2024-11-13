# Simple Neural Network for Flower Detection

This project implements a simple neural network to classify flowers as either red or blue based on their features. The neural network is built from scratch using NumPy.

## Dataset

The dataset consists of flower features represented by two attributes:
- Petal length
- Petal width

The target variable is:
- 1 for red flowers
- 0 for blue flowers

## Code Explanation

### Data Preparation

The data is normalized and split into training and testing sets.

### Neural Network Structure

The neural network has:
- 2 input neurons
- 3 hidden neurons
- 1 output neuron

### Training

The network is trained using forward and backward propagation. The sigmoid function is used as the activation function.

### Prediction

After training, the network predicts the class of a flower based on its features.

## Usage

To run the neural network, execute the `neural_network_flower.py` script. The script will train the network and output the predicted class for the test data.

## Requirements

- Python 3.x
- NumPy

## How to Run

1. Ensure you have Python and NumPy installed.
2. Run the script:
    ```bash
    python neural_network_flower.py
    ```

## Output

The script will print the training progress and the final prediction for the test data.
