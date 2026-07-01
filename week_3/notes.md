
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
## Video 7 : Neural Network for handwritten digits classification
### loading datasets in keras of tensorflow:
```python
(X_train, y_train) , (X_test,y_test) = keras.datasets.mnist.load_data()
```
### data visualization using matplotlib  :
```python
plt.matshow(X_train[1])
```
### flattening a 2d array into 1d :
```python
plt.matshow(X_train[1])
```
### Training a model using keras :
```python
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #flattening using keras itself
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train,y_train,epochs=5)
model.evaluate(X_test_flattened,y_test)
np.argmax(y_pred[0]) #for getting target value in digits
```
### Confusion matrix in Keras :
```python
cm=tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
```
---
## Video 8 : Activation functions

### WHY ACTIVATION FUNCTIONS?
 Without activation functions:
-  Output = weighted sum of inputs
-  Entire neural network becomes a linear equation
-  Hidden layers become useless
-  Activation functions introduce NON-LINEARITY.
- Non-linearity helps neural networks learn complex patterns.
### STEP FUNCTION

- Output: 1 if x > threshold, 0 otherwise

### SIGMOID FUNCTION
-  σ(x) = 1 / (1 + e^(-x))
- Output Range: (0, 1)
- vanishing gradient problem when drivatives close to 0

### TANH (Hyperbolic Tangent)
- tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
- Output Range: (-1, 1)
- Zero-centered data
- Usually better than sigmoid in hidden layers
- Vanishing Gradient Problem

### ReLU (Rectified Linear Unit)
- f(x) = max(0, x)
- Most popular hidden-layer activation
- Default choice for hidden layers

### LEAKY ReLU
- f(x) = x          if x > 0
- f(x) = 0.1*x     if x <= 0
- Reduces ReLU's dying neuron problem
---
## Video 9 : Derivatives
- Basic derivative concepts
---
## Video 10 : Matrix  Basics
- Basic matrix operations
```python
revenue=np.array([[100,200,220],[24,36,40],[12,18,20]])
profit=expenses-revenue
# if u use * for matrix multiplication, broadcasting happens if matrix shapes does not match
# dot product:
np.dot(price,unit)
```
---
## Video 11 : Loss or Cost Function/ binary cross entropy
- learned mae,mse and log loss function
- implementation of mae and log loss in codes directory
---
## Video 12 : Gradient Descent for Neural Network
- Learned gradient descent implementation for neural networks
- implementation in codes directory
---
## Video 13 : Implementing neural network in python
- made a class customModel for the neural network just created in python
- while making a class:
```python
class myNN:
    def __init__(self):
        self.w1 = 1 
        self.w2 = 1
        self.bias = 0
```
---
## Video 14 : Stochastic Gradient Descent vs Batch Gradient Descent vs Mini Batch Gradient Descent
- Implemented all descents from plain python
---
## Video 15 : Chain rule
- Chain rule of differential calculus
---
## Video 16 : TensorBoard Introduction
- TensorBoard is TensorFlow's visualization toolkit.
```python
import tensorflow as tf
# Create callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
log_dir="logs",
histogram_freq=1
)
```
- histogram_freq=1 -> logs weight distributions after every epoch

```python
model.fit(X_train, y_train, epochs=5,callbacks=[tensorboard_callback])
```

### STARTING TENSORBOARD
- Open CMD / PowerShell
- Move to project directory:
- tensorboard --logdir logs
- Alternative:
- python -m tensorboard.main --logdir logs

### Properties
- Shows metrics over epochs:
- Histograms : Displays weight distributions. Helps identify exploding or vanishing weights.
---
## Video 17 : GPU bench-marking with image classification
## Useful code syntaxes
### One hot encoding in tensorflow
```python
y_train_categorical=keras.utils.to_categorical(y_train,num_classes=10, dtype='float')
```
### Model training using keras :
```python
model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32,32,3)),
        keras.layers.Dense(3000, activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')    
    ])

model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train_categorical, epochs=1)
```
### timing calc with cpu or gpu :
```python
%%timeit -n1 -r1 
with tf.device('/CPU:0'):
    cpu_model = get_model()
    cpu_model.fit(X_train_scaled, y_train_categorical, epochs=1)
```
---
## Video 18 : Customer churn prediction with ANN
- An exercise with lots of pre processing, data exploration etc.. - check codes directory
---
## Video 19: perfomance metrics 

### Why Accuracy Alone Is Not Enough

- Accuracy works well only when classes are balanced.

### Confusion Matrix

For Binary Classification:

|                | Predicted Positive | Predicted Negative |
|----------------|-------------------|-------------------|
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

### Precision

Measures:

Out of all positive predictions,
how many were actually positive?

Formula:

Precision = TP / (TP + FP)

### Recall

Measures:

Out of all actual positives,
how many did we identify correctly?

Formula:

Recall = TP / (TP + FN)

### F1 Score

Balances Precision and Recall.

Formula:

F1 = 2 * (Precision * Recall) / (Precision + Recall)

- High only when both precision and recall are high
- Useful for imbalanced datasets

Range:
0 to 1

Higher = Better

### Sklearn Implementation

Import:

```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
```

Confusion Matrix:

```python
confusion_matrix(y_test, y_pred)
```

Classification Report:

```python
print(classification_report(y_test, y_pred))
```
---
## Video 20 : Dropout regularization

### Overfitting Symptoms

Training Accuracy >> Test Accuracy

Model memorizes training data instead of learning patterns.

### Dropout Regularization

- Reduces Overfitting
- Randomly disable some neurons during training.

### ANN With Dropout

```python
keras.layers.Dropout(0.5)
```

0.5 means:

```text
50% neurons are randomly turned off
during each training iteration
```

```python
modeld = keras.Sequential([
    keras.layers.Dense(60, input_dim=60, activation='relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(1, activation='sigmoid')
])
```
---
## Video 21 : Handling imbalanced dataset in machine learning

A dataset where one class has significantly more samples than the other.

Do NOT rely only on Accuracy.

Use:

```python
Precision
Recall
F1 Score
Confusion Matrix
```

### Checking Class Distribution

```python
df['Class'].value_counts()
```

### Handling Imbalanced Dataset

1. Undersampling
2. Oversampling
3. SMOTE
4. Ensemble Techniques

### 1. Undersampling

Reduce majority class samples.
Separate classes:

```python
df_class0 = df[df.Class == 0]
df_class1 = df[df.Class == 1]

df_class0_under = df_class0.sample(
    len(df_class1)
)

df_test_under = pd.concat(
    [df_class0_under, df_class1],
    axis=0
)
```
### Oversampling

Increase minority class samples.

```python
df_class1_over = df_class1.sample(
    len(df_class0),
    replace=True
)
df_test_over = pd.concat(
    [df_class0, df_class1_over],
    axis=0
)
```
```python
replace=True
```

Allows duplicate rows to be generated.

### SMOTE

Synthetic Minority Oversampling Technique

```bash
pip install imbalanced-learn
```

```python
from imblearn.over_sampling import SMOTE
```

```python
smote = SMOTE(
    sampling_strategy='minority'
)
X_sm, y_sm = smote.fit_resample(X, y)
```
### Ensemble Techniques

stratification to preserve class proportions.
you take batches of majority sample concat and train and take the majority vote

---
## Video 22 : Applications of Computer Vision
So cool !
---
## Video 23 : Simple explanation of convolutional neural network 

### Why CNN?

Regular ANN works well for:

- Structured data
- Tabular data
- Numerical features

CNN is specifically designed for image processing.

In ANN image processing:
- Huge computation
- Overfitting
- Loss of spatial information

---

### CNN Idea

Instead of looking at entire image: CNN looks at small portions of image.
Example:
```text
3x3 filter
5x5 filter
```
Main operation in CNN.
A filter slides over the image.
Extract important features like:
- Edges
- Corners
- Shapes
- Textures

### Filter / Kernel

Example:

```python
[
 [1,0,1],
 [0,1,0],
 [1,0,1]
]
```

Filter scans image and produces feature maps.
Feature map contains useful patterns from image.
Different filters learn different features.
CNN automatically learns these filters.

### Activation Function

Usually:

```python
ReLU
```

Formula:

```python
f(x) = max(0,x)
```

Purpose:

- Introduce non-linearity
- Remove negative values

### Pooling Layer

Purpose:

Reduce image size.

Most common:

```text
Max Pooling
```
Take maximum value from each window.

Example:

```text
1 5
3 2
```

Output:

```text
5
```

### CNN Architecture

```text
Input Image
      ↓
Convolution
      ↓
ReLU
      ↓
Pooling
      ↓
Convolution
      ↓
ReLU
      ↓
Pooling
      ↓
Flatten
      ↓
Dense Layer
      ↓
Output
```
---
## Video 24 : Image Classification using CNN (CIFAR10 dataset)
In codes directory  
---
## Video 25 : Convolution padding and stride

Formula for feature extraction :

```text
Output Size =
((Input Size - Kernel Size) / Stride) + 1
```

After multiple convolutions:

```text
32x32
↓
30x30
```
Image size rapidly decreases.
Important information near edges may be lost.

### Padding
Padding means adding extra pixels around image boundaries.
Typically:
```text
0s are added
```
around the image.

```text
0 0 0 0 0 0 0
0 x x x x x 0
0 x x x x x 0
0 x x x x x 0
0 x x x x x 0
0 x x x x x 0
0 0 0 0 0 0 0
```
Boundary pixels are used more effectively.
With Padding:

```text
5 x 5
↓
5 x 5
```
### Valid convolution

No padding added.
Image shrinks after convolution.
```python
padding='valid'
```
### Same Padding

Adds padding automatically.
Output size remains approximately same as input size.
```python
padding='same'
```
### Stride
Stride determines how many pixels the filter moves each step.
```python
    strides=(1,1),
```
---
## Video 26 : Data augmentation to address overfitting
```python 
import cv2
import os
import PIL
import pathlib
data_dir=pathlib.Path(data_dir)/'flower_photos'
flowers_images_dict = {
    'roses': list(data_dir.glob('roses/*.jpg')),
    'daisy': list(data_dir.glob('daisy/*.jpg')),
    'dandelion': list(data_dir.glob('dandelion/*.jpg')),
    'sunflowers': list(data_dir.glob('sunflowers/*.jpg')),
    'tulips': list(data_dir.glob('tulips/*.jpg')),
}
flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}
# to create numpy arrays from imgs of same dimensions
for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(180,180))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])
```
### data augmentation 
```python
data_augmentation = keras.Sequential([
    layers.Input(shape=(180,180, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
model = Sequential([
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
```
---
## Video 27 : Transfer learning
Transfer Learning involves using a pre-trained model (trained on a large dataset) as the starting point for a new task. Instead of training a deep neural network from scratch, we reuse the learned features and fine-tune the model on our target dataset, reducing training time and improving performance.
```python
# to use a model to classify something for you : 
classifier = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=True,
    input_shape=(224,224,3)
)
# to use it on another dataset and retrain : 
pretrained_model_without_top_layer = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3),
)
model = tf.keras.Sequential([
  pretrained_model_without_top_layer,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(num_of_flowers)
])
#compile, fit and evaluate
```
---
## Video 28 : Image Classification vs Object detection vs Image Segmentation

- **Image Classification** assigns a single label to an entire image (e.g., "cat" or "dog") without identifying the object's location.
- **Object Detection** identifies and localizes multiple objects in an image using bounding boxes and class labels.
- **Image Segmentation** performs pixel-level classification, assigning a label to every pixel to precisely outline object boundaries.

**Hierarchy:** Classification → What is in the image?
Detection → What is in the image and where is it?
Segmentation → What is in the image, where is it, and what exact pixels belong to it?
---
## Video 29 : Popular datasets for computer vision
#### ImageNet

- One of the largest image datasets used for image classification tasks.
- Widely used for pre-training deep learning models and transfer learning.

#### COCO (Common Objects in Context)

- A large-scale dataset designed for object detection, segmentation, and image captioning.
- Contains images with multiple objects, bounding boxes, and segmentation masks.
- Common benchmark for modern object detection models such as YOLO and Faster R-CNN.

#### Google Open Images

- A massive dataset released by Google containing millions of images.
- Supports image classification, object detection, visual relationship detection, and segmentation tasks.
- Includes a large variety of object categories and annotations.

ImageNet is mainly used for classification and pre-training, while COCO and Open Images are commonly used for object detection and segmentation.
---
## Video 30 : Sliding Window Object detection

- Sliding Window is an object detection technique where a fixed-size window moves across different regions of an image.
- Each window is passed through a classifier to determine whether it contains the target object.
- To detect objects of different sizes, the image is processed at multiple scales using an image pyramid.
- Although conceptually simple, Sliding Window is computationally expensive because the classifier must evaluate a large number of overlapping regions.
---
## Video 31 : What is YOLO Algorithm ?
- You Only Look Once!
- YOLO is a real-time object detection algorithm that detects and classifies objects in a single pass through the neural network.
- Unlike Sliding Window methods, YOLO processes the entire image at once, making it significantly faster.
- The image is divided into a grid, and each grid cell predicts bounding boxes, confidence scores, and object classes.
- YOLO performs object localization and classification simultaneously in one network pass, enabling efficient real-time object detection.
- say you want to detect two objects, the 16 grids will each have a 7*1 vector with informations like probability,x coordinate, y coordinate,width and height of the bounding boxes which class is 0 and which class is 1 etc..
- if multiple bounding boxes are created, we use IOU, Ingtersection over union method. ie = intersection area/union area > 0.65 means its the same object having multiple bounding box so remove the one with less probability score
- when the centre of multiple objects fall into a single grid there is anchoring which outputs say 14*1 output and so on.
---
## Video 32 : Object detection using YOLO v4 and pre trained model

- YOLOv4 is an improved version of the YOLO object detection algorithm that provides better accuracy and speed compared to earlier versions.
- Instead of training a model from scratch, a pre-trained YOLOv4 model can be used to detect common objects such as people, cars, animals, and traffic signs.
- The model predicts bounding boxes, confidence scores, and class labels for detected objects in an image.
- Using pre-trained weights significantly reduces training time and allows object detection with minimal setup.
---
