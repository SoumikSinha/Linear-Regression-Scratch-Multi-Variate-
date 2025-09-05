# Multivariate Linear Regression from Scratch 📈  

A Python implementation of Linear Regression using Gradient Descent without external ML libraries.  
Supports multiple features, tracks training cost, and visualizes convergence.  

---

## 📌 Features
- Implements gradient descent manually  
- Supports multivariate regression (multiple input features)  
- Uses Mean Squared Error (MSE) as cost function  
- Prints training progress (weights, bias, cost)  
- Stops when cost stabilizes or max epochs reached  
- Plots Cost vs Epochs for convergence visualization  

---

## 📐 Mathematical Background  

### Hypothesis Function  
For n features, the hypothesis is:  

hθ(x) = w1x1 + w2x2 + ... + wn*xn + b

shell
Copy code

### Cost Function (MSE)  
We minimize the Mean Squared Error:  

J(w, b) = (1 / (2m)) * Σ (hθ(x(i)) - y(i))²

markdown
Copy code

where:  
- `m` = number of training examples  
- `x(i)` = input features  
- `y(i)` = actual target values  

### Gradient Descent Update Rules  
To minimize cost, we iteratively update weights and bias:  

wj := wj - α * (∂J/∂wj)
b := b - α * (∂J/∂b)

csharp
Copy code

where `α` = learning rate.  

The partial derivatives are:  

∂J/∂wj = (1/m) * Σ (hθ(x(i)) - y(i)) * xj(i)
∂J/∂b = (1/m) * Σ (hθ(x(i)) - y(i))

yaml
Copy code

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
yaml
Copy code
TRIAL NO:1  
FUNC: [0 0]*x + 0  
COST: 474.0  
...  
FINAL FUNC: [5.0003 5.0003]*x + -2.0001  
RUNTIME: 0.45s
📊 Cost decreases over epochs until convergence.

▶️ Usage
Run the script
bash
Copy code
python linear_regression.py
Initialize a new repository (Git Bash)
bash
Copy code
# initialize git repo
git init

# add all files
git add .

# commit
git commit -m "Initial commit - Linear Regression from scratch"

# connect to GitHub (replace with your repo link)
git remote add origin https://github.com/your-username/linear-regression-scratch.git

# push to GitHub
git push -u origin main

