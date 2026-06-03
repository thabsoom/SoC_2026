
# Deep Learning & CNN

## Resources
- playlist : https://www.youtube.com/playlist?list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO

## Video 3: What is a Neuron?

### Artificial Neuron

- Input --> Processing --> Output
- Inputs: x1, x2, x3, ...
-  Weights: w1, w2, w3, ...
- Output depends on: Inputs × Weights

#### Neuron Formula
 z = w1*x1 + w2*x2 + ... + wn*xn + b

where:
- x = input
- w = weight
- b = bias
- z = weighted sum

### Activation Function

- Output = Activation(z)
- Converts weighted sum into useful output.
---

## Video 4: Neural Network Simply Explained

### What is a Neural Network?

- Collection of interconnected neurons.

### Layers

- Input Layer
- Hidden Layer(s)
- Output Layer

### Deep Neural Network

- Neural Network with multiple hidden layers.
- More hidden layers = deeper network = Deep Learning

### Learning Process

- Compare prediction with actual value.
- Calculate error.
- Adjust weights and bias.
- Repeat until error becomes small.
---
## Video 5: TensorFlow

### Why TensorFlow?

 - Build neural networks
 - Train deep learning models
 - GPU support
 - Production deployment
---
## Video 6: TensorFlow vs PyTorch vs Keras

### TensorFlow
- Developed by Google
- Library:
```python
import tensorflow as tf
```
### PyTorch
- Developed by Meta (Facebook)
- Library:
```python
import torch
```
### Keras
- Simplifies Deep Learning code.
- integrated into TensorFlow.
```python
from tensorflow import keras
```
---