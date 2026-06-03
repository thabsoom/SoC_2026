# Week 2

## Resources

Playlist:
https://www.youtube.com/playlist?list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw

## Video 1: What is Machine Learning?

- ML = learning patterns from data to make predictions or decisions.
- Deep Learning is one technique within ML.
- Common applications: spam detection, recommendations, voice assistants, autonomous vehicles.

## Video 2: Linear regression single variable

- Used to predict continuous values.
- Fits the best-fit line through data points by minimizing squared error.

### Equation

y = mx + b

- x: independent variable
- y: dependent variable
- m: coefficient/slope
- b: intercept

### Scikit-Learn

- `fit(X, y)` → train model
- `predict(X)` → make predictions
- `coef_` → slope
- `intercept_` → intercept

### Notes

- `X` must be a 2D DataFrame/array because Scikit-Learn expects input features in tabular form.
- Example:

  ```python
  df[['area']]   

## Video 3: Linear regression Multiple variables

### Equation

y = m₁x₁ + m₂x₂ + ... + b

- Multiple independent variables can be used to predict one dependent variable.

### Handling Missing Values

Replace missing values with median:

```python
import math

median_score = math.floor(df['test_score(out of 10)'].median())

df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_score)
```

### Converting Number Words to Integers

```python
from word2number import w2n

df['experience'] = df['experience'].fillna('zero')
df['experience'] = df['experience'].apply(w2n.word_to_num)
```

### Multiple Linear Regression Syntax

```python
from sklearn import linear_model

lr = linear_model.LinearRegression()

lr.fit(
    df[['experience',
        'test_score(out of 10)',
        'interview_score(out of 10)']],
    df['salary($)']
)
```

## Video 4: Gradient Descent and Cost Function

### Mean Squared Error (Cost Function)

Cost = average squared difference between actual and predicted values.

\[
\text{Cost} = \frac{1}{n}\sum (y - y_{pred})^2
\]

- Lower cost = better fit.
- Goal: minimize cost.

### Gradient Descent

Used to find the best values of:
- m (slope)
- b (intercept)

Start with:
```python
m_curr = 0
b_curr = 0
```

Then repeatedly update them:

```python
m_curr = m_curr - learning_rate * md
b_curr = b_curr - learning_rate * bd
```

### Partial Derivatives

```python
md = -(2/n) * sum(x * (y - y_pred))
bd = -(2/n) * sum(y - y_pred)
```

### Learning Rate

Controls step size.

```python
learning_rate = 0.0002
```

- Too small → slow convergence.
- Too large → overshoots minimum and may diverge.

### Cost Calculation

```python
cost = (1/n) * sum(val**2 for val in (y - y_pred))
```

### Stopping Condition

```python
if math.isclose(cost, previous_cost, rel_tol=1e-20):
    break
```

- Stop when cost is no longer changing significantly.

### Useful Functions

```python
math.isclose(a, b)
```

Checks if two floating-point numbers are approximately equal.

Parameters:
```python
rel_tol  # relative tolerance
abs_tol  # absolute tolerance
```

## Video 5: Save Model using Joblib and Pickle

### Why?

save the trained model and load it later.

### Using Pickle

Import:

```python
import pickle
```

Save model:

```python
with open('model_pickle', 'wb') as file:
    pickle.dump(model, file)
```

Load model:

```python
with open('model_pickle', 'rb') as file:
    mp = pickle.load(file)
```

### Using Joblib

Import:

```python
import joblib
```

Save:

```python
joblib.dump(model, 'model_joblib')
```

Load:

```python
mj = joblib.load('model_joblib')
```

### Note

- `'wb'` → write binary
- `'rb'` → read binary
---
## Video 6: Dummy variables & One Hot Encoding

ML models cannot directly use text → we convert it.



### 2. Types of Categorical Variables

#### Nominal
No order (e.g. town, color, gender)

#### Ordinal
Has order (e.g. low < medium < high)


### Why not Label Encoding?

Model assumes:
2 > 1 > 0 (wrong meaning)



### 4. One Hot Encoding (Dummy Variables)

Convert categories into binary columns:

| town | Monro | Westminster | Robbinsville |
|------|------|------------|-------------|
| Monro | 1 | 0 | 0 |
| West  | 0 | 1 | 0 |
| Rob   | 0 | 0 | 1 |



### 5. Pandas Method

```python
dummies = pd.get_dummies(df.town)
df = pd.concat([df, dummies], axis=1)
```

### 6. Dummy Variable Trap

Avoid multicollinearity:
Drop one dummy column


### 11. Model Score

```python
model.score(X, y)
```
Returns R² score (accuracy of model)


### 12. Alternative: Label Encoding + OneHotEncoder

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
```

#### Label Encoding:
```python
le = LabelEncoder()
df.town = le.fit_transform(df.town)
```

#### OneHotEncoder:
```python
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)

X = ct.fit_transform(X)
```

---
## Video 7: Train Test Split

In machine learning, we want to:
- Train a model on data
- Test it on unseen data

 Goal: measure how well the model generalizes

---

## 4. sklearn Train Test Split

```python
from sklearn.model_selection import train_test_split
X_train, y_train, X_test, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
```
---
## Video 8: Logistic Regression (Binary Classification)


### Preprocessing: Get Dummies (Optimized)

final_df = pd.get_dummies(
    final_df,
    columns=['salary'],
    prefix='salary',
    dtype=int,
    drop_first=True   # avoids dummy variable trap
)


### EDA: Department vs Employee Attrition

pd.crosstab(df.Department, df.left).plot(kind='bar')

"""
Purpose:
- Understand how employee leaving (left=1) varies across departments
- Helps identify high-risk departments
"""


### EDA: Groupby Mean Analysis

df_num = df.select_dtypes(include='number')

df_num.groupby('left').mean()

"""
Purpose:
- Compare average feature values
- left = 0 → stayed employees
- left = 1 → left employees
"""


### Core Theory: Logistic Regression

"""
Used for:
- Binary classification problems

Goal:
- Predict probability that y = 1 (e.g., employee leaves)
"""


### Linear Combination (Logit)

"""
z = w0 + w1*x1 + w2*x2 + ... + wn*xn

- z = linear score
- w = weights learned by model
- x = input features
"""


### Sigmoid Function

"""
p = 1 / (1 + e^(-z))

Converts linear output into probability (0 to 1)
"""


### Probability Interpretation

"""
p = probability that y = 1
Example:
- p = 0.8 → high chance employee will leave
"""


### Decision Rule

"""
if p >= 0.5:
    predict 1
else:
    predict 0
"""


### Loss Function (Log Loss)

"""
Loss = -[ y log(p) + (1 - y) log(1 - p) ]

- Penalizes wrong confident predictions
- Goal: minimize loss
"""
---