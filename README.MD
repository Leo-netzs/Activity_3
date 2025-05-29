# Derivative Estimation with Neural Networks

This project uses an `MLPRegressor` neural network from `scikit-learn` to estimate the derivative of a function based solely on its values. The goal is to train the model to learn numerical differentiation from examples generated using polynomials, and then test its generalization ability with functions like sine, cosine, and quadratic functions.

## 🧠 Model

- Type: `MLPRegressor`
- Hidden Layers: 10
- Neurons per Layer: 10
- Activation: `tanh`
- Optimizer: `adam`
- Learning Rate: adaptive

## 📊 Data Generation

The training functions are combinations of polynomials with random coefficients and added noise. Derivatives are computed analytically and normalized. The data is split into training and testing sets (80%/20%).

Examples of training functions:

- \( f_1(x) = (x^2 - a^2)(x^2 - b^2) \)
- \( f_2(x) = x^5 + ax^3 - b^2x^3 - ab^2x \)

And their corresponding derivatives:

- \( f_1'(x) = 2x(2x^2 - (a^2 + b^2)) \)
- \( f_2'(x) = 5x^4 + 3ax^2 - 3b^2x^2 - ab^2 \)

## 🧪 Tests

The network is tested with functions **not seen during training**, such as:

- \( f(x) = \sin(2\pi x) \), with derivative \( f'(x) = 2\pi \cos(2\pi x) \)
- \( f(x) = \cos(2\pi x) \), with derivative \( f'(x) = -2\pi \sin(2\pi x) \)
- \( f(x) = x^2 \), with derivative \( f'(x) = 2x \)

Plots compare the true derivative with the predicted one.

## 📈 Output

The mean squared error (MSE) is printed after training, and plots visually demonstrate how well the network generalizes.

## 🛠️ Requirements

- Python 3
- numpy
- matplotlib
- scikit-learn

Install dependencies with:

```bash
pip install numpy matplotlib scikit-learn
```
## 🚀 Running

 run the main script:
 ```Bash
 python derivative_mlp.py
```