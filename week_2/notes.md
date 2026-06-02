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