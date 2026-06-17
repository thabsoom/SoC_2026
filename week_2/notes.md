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
```python
final_df = pd.get_dummies(
    final_df,
    columns=['salary'],
    prefix='salary',
    dtype=int,
    drop_first=True   # avoids dummy variable trap
)
```

### EDA: Department vs Employee Attrition
```python
pd.crosstab(df.Department, df.left).plot(kind='bar')
```
Purpose:
- Understand how employee leaving (left=1) varies across departments
- Helps identify high-risk departments



### EDA: Groupby Mean Analysis
```python
df_num = df.select_dtypes(include='number')
df_num.groupby('left').mean()
```
Purpose:
- Compare average feature values
- left = 0 → stayed employees
- left = 1 → left employees

### Core Theory: Logistic Regression
Used for:
- Binary classification problems

Goal:
- Predict probability that y = 1 (e.g., employee leaves)

### Linear Combination (Logit)
z = w0 + w1*x1 + w2*x2 + ... + wn*xn

- z = linear score
- w = weights learned by model
- x = input features

### Sigmoid Function

p = 1 / (1 + e^(-z))

Converts linear output into probability (0 to 1)

### Probability Interpretation
p = probability that y = 1
Example:
- p = 0.8 → high chance employee will leave

### Decision Rule
if p >= 0.5:
    predict 1
else:
    predict 0

### Loss Function (Log Loss)
Loss = -[ y log(p) + (1 - y) log(1 - p) ]

- Penalizes wrong confident predictions
- Goal: minimize loss
---
## Video 9: Decision Tree

### What is a Decision Tree?

- Works by repeatedly splitting data into smaller groups
- based on feature values.

### Example

 Is Sex = Male?
        /      \
      Yes      No
      /         \
    Survive=0  Survive=1

### How Does It Choose Splits?

- Decision Trees use measures of impurity.

 Common criteria:
 - Gini Impurity
 - Entropy (Information Gain)


### Gini Impurity

- Measures how mixed the classes are.
- Gini = 0
-> Pure node (all samples belong to one class)
- Lower Gini is better.

### Entropy

 Measures randomness/disorder in a node.

### Information Gain:
 Reduction in entropy after a split.


### Training a Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

### Important Parameters
```python
DecisionTreeClassifier(
    criterion='gini',    # or 'entropy'
    max_depth=None,
    random_state=42
)
```
#### criterion:
- 'gini'
- 'entropy'

#### max_depth:
- Limits tree depth
- Helps prevent overfitting
---
## Video 10: Support Vector Machine (SVM)

### What is SVM?

- classification problems

### Margin

- Margin = distance between the decision boundary and the nearest data points.
- Larger margin = better classification.
- SVM tries to maximize the margin.


### Support Vectors

- The data points closest to the decision boundary
- These points determine the position of the boundary.

### Hyperplane

- In 2D: Decision boundary = Line
- In 3D: Decision boundary = Plane
- In n-dimensions:Decision boundary = Hyperplane

### Gamma

 Gamma controls how far the influence of a training example reaches.

#### High Gamma:
 - Considers nearby points only
 - Can lead to overfitting

#### Low Gamma:
 - Considers far away points also
 - Smoother decision boundary
 - May reduce accuracy slightly

### Regularization (C)

- C controls regularization.

#### High C:
 - Tries to classify all training points correctly
 - Less tolerance for errors
 - Can overfit

#### Low C:
 - Allows some misclassification
 - Simpler boundary
 - Better generalization

### Kernel Trick
- Some datasets cannot be separated using a straight line.
- SVM can transform features into a higher dimension where separation becomes easier.
- This transformation is called a Kernel.
- Kernel helps create complex decision boundaries.

#### Common Kernels

##### Linear Kernel
- Straight decision boundary

##### RBF Kernel (default)
- Handles non-linear data well

##### Polynomial Kernel
- Creates polynomial boundaries



### Add Flower Names
```python
df['flower_name'] = df.target.apply(
    lambda x: iris.target_names[x]
)
```

### Training SVM Model
```python
from sklearn.svm import SVC
model = SVC()
```
---
## Video 11: Random Forest

It is called "Random Forest" because instead of using one decision tree, Random Forest uses multiple decision trees.

### How Random Forest Works
- Create multiple random subsets of the dataset.
- Train a separate decision tree on each subset.
- Make predictions using all trees.
- Take a majority vote (classification) or average (regression).
- Final prediction = combined result of all trees.

### Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
```
### n_estimators
- Number of decision trees in the forest.
```python
model = RandomForestClassifier(
    n_estimators=10
)
```
- More trees generally improve performance.

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(
    y_test,
    y_pred
)
```
### Visualizing Confusion Matrix
```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
```
---
## Video 12: K-Fold Cross Validation

## 1. Why K-Fold Cross Validation?

- Using a single train-test split may give different accuracies depending on how the data is split.
- K-Fold Cross Validation provides a more reliable estimate by training and testing multiple times on different splits.

### KFold

Import:

```python
from sklearn.model_selection import KFold
```

Create K folds:

```python
kf = KFold(n_splits=3)
```

Example:

```python
for train_index, test_index in kf.split(data):
    print(train_index, test_index)
```

### StratifiedKFold

Import:

```python
from sklearn.model_selection import StratifiedKFold
```
- Maintains approximately the same class distribution in every fold.

### cross_val_score (Shortcut)

Import:

```python
from sklearn.model_selection import cross_val_score
```

Syntax:

```python
cross_val_score(model, X, y, cv=3)
```
---
## Video 13: K-Means Clustering
- Unsupervised Learning algorithm
- Used to group similar data points into **K clusters**

### Import Libraries

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
```
### Train Model & Predict Clusters

```python
y_predicted = km.fit_predict(X)
```
### Cluster Centers (Centroids)

```python
km.cluster_centers_
```
### Plot Clusters

```python
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(df1.Age, df1['Income($)'])
plt.scatter(df2.Age, df2['Income($)'])
plt.scatter(df3.Age, df3['Income($)'])

plt.scatter(
    km.cluster_centers_[:,0],
    km.cluster_centers_[:,1],
    marker='*'
)

plt.legend()
```

---

### Feature Scaling 

Different feature ranges can distort clustering.

Use **MinMaxScaler**:

```python
scaler = MinMaxScaler()
```

Scale Income:

```python
scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
```

### Elbow Method

Used to find the **optimal value of K**.

```python
sse = []

for k in range(1,10):
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)
```
---
## Video 14 and 15 : Naive Bayes Classification
```python 
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
```
```python
df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
```
-to convert long texts to numbers :
```python
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
```
### Sklearn Pipeline
```python
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)
```
---
## Video 16 : Hyperparameter tuning (GridSearchCV)
- to get best hyperparemeters for a model,tries all permutations
```python
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C': [1,10,20],
    'kernel': ['rbf','linear']
}, cv=5, return_train_score=False)
clf.fit(iris.data, iris.target)
clf.cv_results_
```
- dir(clf), .best_params_, .best_score_
- Use RandomizedSearchCV to reduce number of iterations and with random combination of parameters. This is useful when you have too many parameters to try and your training time is longer. It helps reduce the cost of computation
```python
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(svm.SVC(gamma='auto'), {
        'C': [1,10,20],
        'kernel': ['rbf','linear']
    }, 
    cv=5, 
    return_train_score=False, 
    n_iter=2
)
rs.fit(iris.data, iris.target)
pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]
```
- n_iter is no of random combinations
### To find best model and best params 
```python
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df
```
---
## Video 17 : L1 and L2 regularization

- Lasso Regression (L1 regularization)
- Ridge Regression (L2 regularization)
- Reduce overfitting by penalizing large coefficients.

### preprocessing
```python
dataset[zero_fill_cols] = dataset[zero_fill_cols].fillna(0)
dataset['Landsize'] = dataset['Landsize'].fillna(dataset['Landsize'].mean())
dataset.dropna(inplace=True)
dataset = pd.get_dummies(dataset, drop_first=True)
```
### LASSO REGRESSION (L1)
```python
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(train_X, train_y)
```
L1 Regularization (Lasso):
Loss = MSE + alpha * sum(|weights|)

Effect:
- Shrinks some coefficients to EXACT 0
- Performs feature selection
###  RIDGE REGRESSION (L2)
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(train_X, train_y)

L2 Regularization (Ridge):
Loss = MSE + alpha * sum(weights^2)

Effect:
- Shrinks coefficients smoothly (NOT zero)
- Reduces overfitting


WHEN TO USE:
- Lasso → when many features irrelevant
- Ridge → when all features somewhat useful
---
## Video 18 : K Nearest Neighbors Classifciation
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
# visualization of confusion matrix:
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# classification report for precision recall and f1 score for each classes:
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```
---
## Video 19 and 20 : Principal Component Analysis (PCA) + Bagging
```python
#better to make a function and do an & by doing all filtering on the main df..this is sequential
df1 = df[df.Cholesterol<=(df.Cholesterol.mean()+3*df.Cholesterol.std())]
df1.shape
```
- Label Encoding for ordinal values
```python
df4.ExerciseAngina=
df4.ExerciseAngina.replace(
    {
        'N': 0,
        'Y': 1
    },
    inplace=True)
```
- PCA:
```python
from sklearn.decomposition import PCA
pca = PCA(0.95)
X_pca = pca.fit_transform(X)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=30)
```
- Bagging
```python
from sklearn.ensemble import BaggingClassifier
bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=0), 
    n_estimators=100, 
    max_samples=0.9, 
    oob_score=True,
    random_state=0
)
scores = cross_val_score(bag_model, X, y, cv=5)
scores.mean()
```
- Random forest classifier is bagging with n  decision trees with additional feature split like both column and rows gets bagged :
```python
from sklearn.ensemble import RandomForestClassifier
```
---