# Multivariate Linear Regression from Scratch 📈

A Python implementation of Linear Regression using Gradient Descent without external ML libraries.
Supports multiple features, tracks training cost, and visualizes convergence.

---

## 📌 Features
- Implements gradient descent manually
- Supports multivariate regression (multiple input features)
- Uses Mean Squared Error (MSE) as the cost function
- Prints training progress (weights, bias, cost)
- Stops when the cost stabilizes or max epochs are reached
- Plots Cost vs Epochs for convergence visualization

---

## 📐 Mathematical Background

### Hypothesis Function
For *n* features, the hypothesis $h_{\theta}(x)$ is the model's prediction, calculated as:

$$h_{\theta}(x) = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$

* **w**: The vector of weights.
* **b**: The bias term.
* **x**: The vector of input features.

### Cost Function (MSE)
We measure the model's error using the Mean Squared Error ($J$). Our goal is to find the values for **w** and **b** that minimize this cost.

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2$$

* `m`: The number of training examples.
* $h_{\theta}(x^{(i)})$: The predicted value for the i-th example.
* $y^{(i)})$: The actual target value for the i-th example.

### Gradient Descent Update Rules
To minimize the cost, we iteratively update the weights and bias by taking small steps in the direction of the steepest descent. The update rules are:

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$
$$b := b - \alpha \frac{\partial J}{\partial b}$$

* `α` is the **learning rate**, controlling the step size.
* $\frac{\partial J}{\partial w_j}$ is the partial derivative of the cost with respect to a weight $w_j$.

The partial derivatives are calculated as:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})$$

---

## 💻 Example

### Input Data:
```python
data = [
    [1, 2, 8],
    [2, 3, 13],
    [3, 4, 18],
    [4, 5, 23],
    [5, 6, 28]
]
alpha = 0.001
lr(data, alpha)
Sample Output:
TRIAL NO:1
FUNC: [0 0]*x + 0
COST: 474.0
...
FINAL FUNC: [5.0003 5.0003]*x + -2.0001
RUNTIME: 0.45s
📊 The cost decreases over epochs until the model converges.

▶️ Usage
Run the script
Bash

python linear_regression.py
Initialize a new repository (Git Bash)
Bash

# initialize git repo
git init

# add all files
git add .

# commit
git commit -m "Initial commit - Linear Regression from scratch"

# connect to GitHub (replace with your repo link)
git remote add origin [https://github.com/your-username/linear-regression-scratch.git](https://github.com/your-username/linear-regression-scratch.git)

# push to GitHub
git push -u origin main

