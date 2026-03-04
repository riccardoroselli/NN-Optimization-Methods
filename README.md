# Extreme Learning Neural Network Optimization Methods

### Model (M)

**(M)** is a so-called *extreme learning* model, i.e., a neural network with one hidden layer $z = W_2 \sigma(W_1 x)$

where:

* $W_1$ is the weight matrix of the hidden layer and is a **fixed random matrix**,
* $\sigma(\cdot)$ is an **elementwise activation function** of your choice,
* $W_2$ is the **output weight matrix**, which is determined by minimizing the **Mean Squared Error (MSE)** of the produced output.

The objective function is $f(z) = \|W_2 \sigma(W_1 x) - y \|_2^2$

with additional $L_1$ (**lasso**) regularization terms.

The project must involve solving **at least medium-sized instances**, i.e.:
* **at least 10,000 weights**
* **at least 1,000 inputs**

### Algorithms

**Momentum Descent (Heavy Ball)** : A standard **momentum descent** (also known as the **heavy ball method**) optimization approach.

**Smoothed Gradient Methods** : An algorithm from the class of **smoothed gradient methods**.

### Constraints

No off-the-shelf solvers allowed.

### Credits

Chiara Capodagli - c.capodagli@studenti.unipi.it

Roselli Riccardo - r.roselli1@studenti.unipi.it

<p align="center"> Master Degree in Data Science and Business Informatics​ </p>

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/it/e/e2/Stemma_unipi.svg" width="70"/>
</p>