# Gradient Descent

> The engine behind almost every model you'll ever train. If a model "learns", it's probably using this.

---

## The Intuition

Imagine you're blindfolded and dropped somewhere on a hilly landscape. Your goal: find the lowest point in the valley.

You can't see anything, but you *can* feel the slope under your feet. So you do the sensible thing — you take a small step in whichever direction feels most downhill, then stop, feel the slope again, and take another step. Repeat until the ground feels flat.

That's gradient descent. The "hill" is your loss function (how wrong your model is). The "lowest point" is the set of parameters where your model is as right as it can be. The "slope under your feet" is the gradient — a mathematical measure of how steeply the loss is changing with respect to each parameter.

You're not jumping to the answer. You're *iterating your way there*, one careful step at a time.

**Why not just solve for the minimum directly?**

For simple models like linear regression, you can — there's a closed-form solution. But for a neural network with millions of parameters, solving directly is computationally impossible. Gradient descent sidesteps that by being cheap, iterative, and surprisingly effective.

---

## The Math

You have a model with parameters `θ` (weights and biases) and a loss function `L(θ)` that measures how wrong your predictions are.

The gradient `∇L(θ)` is a vector of partial derivatives — one per parameter — telling you how much the loss would change if you nudged each parameter slightly.

The update rule is:

```
θ = θ - α · ∇L(θ)
```

Where:
- `θ` — the current parameters
- `α` (alpha) — the **learning rate**: how big a step to take
- `∇L(θ)` — the gradient of the loss with respect to `θ`

The minus sign is the key: you move *opposite* to the gradient. If the loss is increasing in a direction, you go the other way.

**What is the gradient, concretely?**

For a single weight `w`, the partial derivative `∂L/∂w` answers: *"If I increased `w` by a tiny amount, how much would the loss go up?"*

If `∂L/∂w = 3.5`, the loss is rising steeply with `w` — so you decrease `w`.  
If `∂L/∂w = -0.1`, the loss is barely affected — small update.  
If `∂L/∂w = 0`, you're at a flat point — no update. You might be at the minimum.

**Learning rate intuition:**

| Learning Rate | Behaviour |
|---|---|
| Too large | Overshoot the minimum, loss bounces or explodes |
| Too small | Correct direction but takes forever, can get stuck |
| Just right | Converges smoothly to a minimum |

There's no formula for the "right" learning rate. It's one of the most fiddled-with hyperparameters in ML.

---

## The Three Flavours

This is where it gets practical. "Gradient descent" is actually a family of approaches depending on *how much data* you use to compute each gradient.

### Batch Gradient Descent

Use the entire dataset to compute the gradient before taking a single step.

```
for each epoch:
    compute gradient over ALL training examples
    update θ
```

**Pros:** The gradient is accurate. The loss curve is smooth.  
**Cons:** One update per pass through the data. If you have 10 million examples, you're waiting a long time for each step. Completely impractical for large datasets.

---

### Stochastic Gradient Descent (SGD)

Use *one random training example* to compute the gradient, then immediately update.

```
for each epoch:
    shuffle the data
    for each single example:
        compute gradient on just this one example
        update θ
```

**Pros:** Extremely fast updates. The noise in the gradient actually helps escape shallow local minima.  
**Cons:** The gradient is very noisy. The loss curve bounces around violently. Can overshoot constantly and never truly settle.

The word "stochastic" just means random — you're using a random sample to estimate the true gradient.

---

### Mini-Batch Gradient Descent

The one actually used in practice. Split the dataset into small batches (typically 32, 64, or 128 examples), compute the gradient on one batch, update, move to the next.

```
for each epoch:
    shuffle the data
    for each mini-batch of size B:
        compute gradient on this batch
        update θ
```

**Pros:** Gets the best of both worlds — more stable than pure SGD, far faster than full batch. GPU hardware is specifically designed to crunch these batch computations in parallel.  
**Cons:** Introduces a new hyperparameter (batch size) you have to tune.

When people say "SGD" in deep learning, they almost always mean mini-batch SGD. The naming is confusing. Live with it.

---

## The Code

```python
import numpy as np
import matplotlib.pyplot as plt

# ── Simple demonstration: fitting y = 2x + 1 using gradient descent ──

np.random.seed(42)
X = np.random.uniform(0, 5, 100)
y_true = 2 * X + 1 + np.random.normal(0, 0.5, size=X.shape)

# Parameters to learn
w = 0.0  # weight
b = 0.0  # bias
lr = 0.01
epochs = 200
loss_history = []

for epoch in range(epochs):
    # Predictions
    y_pred = w * X + b

    # MSE Loss
    loss = np.mean((y_pred - y_true) ** 2)
    loss_history.append(loss)

    # Gradients (partial derivatives of MSE w.r.t w and b)
    dL_dw = (2 / len(X)) * np.sum((y_pred - y_true) * X)
    dL_db = (2 / len(X)) * np.sum(y_pred - y_true)

    # Update
    w -= lr * dL_dw
    b -= lr * dL_db

print(f"Learned: w = {w:.4f}, b = {b:.4f}")
print(f"True:    w = 2.0000, b = 1.0000")

# ── Mini-batch version ──

def mini_batch_gd(X, y, lr=0.01, epochs=200, batch_size=16):
    w, b = 0.0, 0.0
    n = len(X)
    loss_history = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled, y_shuffled = X[indices], y[indices]
        epoch_loss = []

        for start in range(0, n, batch_size):
            Xb = X_shuffled[start:start + batch_size]
            yb = y_shuffled[start:start + batch_size]

            y_pred = w * Xb + b
            loss = np.mean((y_pred - yb) ** 2)
            epoch_loss.append(loss)

            dL_dw = (2 / len(Xb)) * np.sum((y_pred - yb) * Xb)
            dL_db = (2 / len(Xb)) * np.sum(y_pred - yb)

            w -= lr * dL_dw
            b -= lr * dL_db

        loss_history.append(np.mean(epoch_loss))

    return w, b, loss_history

w_mb, b_mb, loss_mb = mini_batch_gd(X, y_true, batch_size=16)

# ── Plot ──

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(loss_history, label='Batch GD (full dataset)')
axes[0].plot(loss_mb, label='Mini-batch GD (size=16)', alpha=0.8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Loss Curves: Batch vs Mini-Batch')
axes[0].legend()

axes[1].scatter(X, y_true, alpha=0.4, label='Data')
X_plot = np.linspace(0, 5, 100)
axes[1].plot(X_plot, 2 * X_plot + 1, 'k--', label='True line')
axes[1].plot(X_plot, w * X_plot + b, 'r-', label=f'Batch GD: {w:.2f}x + {b:.2f}')
axes[1].set_title('Fitted Line')
axes[1].legend()

plt.tight_layout()
plt.savefig('gradient_descent.png', dpi=150)
plt.show()
```

**What to observe:**
- Batch GD produces a perfectly smooth, monotonically decreasing loss curve.
- Mini-batch GD is noisier but arrives at the same answer faster per wall-clock time.
- Both converge to nearly `w = 2, b = 1` — close to the true values.

---

## The Complications

This is the part textbooks skip over. Real gradient descent is messier than the valley analogy suggests.

### Local Minima

The loss landscape isn't a single smooth bowl. It's a high-dimensional surface with dips, ridges, and flat plains everywhere.

A **local minimum** is a point where the gradient is zero but it isn't the global best — you're in a little dip, not the deepest valley. In theory, gradient descent could get stuck there.

In practice, this is less of a problem for neural networks than people used to think. Most local minima in high-dimensional spaces turn out to have very similar loss values to the global minimum. The thing that actually kills you more often is:

### Saddle Points

A saddle point is where the gradient is zero but you're not at a minimum in any direction — you're at a minimum in some dimensions and a maximum in others. The surface curves up in some directions and down in others, like a horse saddle.

Gradient descent can stall around saddle points, especially with standard SGD. The noisiness of mini-batch gradients actually helps here — the random fluctuations can kick the optimizer off the flat region.

### Vanishing and Exploding Gradients

This one is a big deal for deep networks.

When you compute gradients through many layers via backpropagation, you're multiplying many numbers together. If those numbers are consistently less than 1, the gradients shrink exponentially as they travel back through the network — they **vanish**. The early layers learn nothing.

If the numbers are consistently greater than 1, gradients **explode** — you get numerical overflow, NaN losses, and a dead training run.

**Fixes:**
- Vanishing: use ReLU activations instead of sigmoid/tanh; use residual connections (skip connections like in ResNets)
- Exploding: gradient clipping (cap the gradient norm to a maximum value)

This is exactly why your CNN used ReLU — sigmoid would have killed the gradients in early layers.

### The Learning Rate Problem (and How Optimizers Solve It)

A fixed learning rate is fragile. Modern training uses **adaptive optimizers** that automatically adjust the step size per parameter.

| Optimizer | What it does | When to use |
|---|---|---|
| SGD + Momentum | Accumulates velocity in persistent directions; dampens oscillations | When you understand your landscape; often best final accuracy |
| RMSprop | Scales learning rate by recent gradient magnitudes | Recurrent networks, non-stationary problems |
| Adam | Combines momentum + RMSprop; adaptive per-parameter learning rate | Default choice for most tasks; converges fast |
| AdamW | Adam + weight decay decoupled properly | Transformers, large language models |

**Adam** is the safe default. You almost certainly used it (or something close to it) in your CNN training. But there's a known phenomenon where Adam converges fast but slightly worse than well-tuned SGD + momentum — so for final production models, many practitioners switch back.

### Batch Size Effects

Larger batch sizes give more accurate gradient estimates but have a subtle problem: they find **sharper minima** that generalise worse. Smaller batch sizes, with their noisy gradients, tend to find **flatter minima** that generalise better.

This is counterintuitive — you'd expect cleaner gradients to be better. But the noise is a feature, not a bug. It acts as an implicit regulariser.

General rule of thumb: if you double the batch size to speed up training, also scale the learning rate up proportionally (linear scaling rule). It's imperfect but helps maintain similar dynamics.

---

## Common Mistakes

**1. Not normalising your input features**  
If one feature ranges from 0–1 and another ranges from 0–10,000, the loss landscape becomes elongated and gradient descent takes a tortured, zig-zagging path to the minimum. Always normalise or standardise inputs.

**2. Using a learning rate that's too large and declaring the algorithm "broken"**  
If your loss is NaN or oscillating wildly after a few steps, halve the learning rate before doing anything else. This fixes it 90% of the time.

**3. Not shuffling data between epochs**  
If you feed data in the same order every epoch, mini-batch gradients see the same sequence of batches. The model starts adapting to the order of the data, not the data itself.

**4. Stopping too early out of impatience**  
Loss curves often plateau for a while before descending again. What looks like convergence is sometimes the optimizer navigating a flat region. Use proper early stopping criteria with patience — not eyeballing the curve.

**5. Forgetting that gradient descent only finds a local or saddle solution**  
It's not guaranteed to find the global minimum. It finds *a* solution. That's usually fine for deep networks, but worth remembering for non-convex problems where it might matter.

---

## When to Think About This

| Situation | Diagnosis | Fix |
|---|---|---|
| Loss is NaN after a few steps | Learning rate too high, or exploding gradients | Lower `lr`; add gradient clipping |
| Loss decreases then suddenly spikes | Learning rate still too high for later training | Learning rate schedule (decay over time) |
| Training loss drops but validation loss doesn't | Overfitting; not a GD problem per se | Regularisation, dropout, more data |
| Loss barely moves for many epochs | Learning rate too low, or vanishing gradients | Increase `lr`; check activations; try Adam |
| Model trains fine but final accuracy disappoints | Sharp minima from large batch size | Reduce batch size; try SGD + momentum |
| Early layers not learning (in deep nets) | Vanishing gradients | ReLU, batch norm, residual connections |

---

## Connection to Models I've Used

| Model | How Gradient Descent is Used | Complications |
|---|---|---|
| Linear / Logistic Regression | Convex loss — one global minimum, GD always finds it | Basically none; this is the easy case |
| Decision Trees | Doesn't use gradient descent at all — uses greedy splitting | N/A |
| Random Forest | Same — no gradient descent | N/A |
| SVM | Uses a convex optimisation problem, not standard GD | Quadratic programming; GD variants used at scale |
| Neural Network (your CNN) | Mini-batch GD via backpropagation; Adam or similar | Vanishing gradients, learning rate tuning, saddle points |
| Transformer / LLM | Same as NN, but at massive scale | Gradient clipping mandatory; AdamW with careful warmup schedules |

The pattern: the more expressive and complex the model, the more you depend on getting gradient descent right. Simple models are forgiving. Deep networks are not.

---

## A Note on Backpropagation

Gradient descent tells you *how to update* given a gradient. **Backpropagation** tells you *how to compute* that gradient efficiently for a neural network.

They're not the same thing, but they're inseparable in practice. Backprop is just the chain rule applied systematically through every layer of the network, from output back to input. Every modern deep learning framework (PyTorch, TensorFlow, JAX) computes this automatically with autograd — you define the forward pass and the gradient computation is handled for you.

This will be covered in depth in the neural networks chapter, but it's worth knowing they're two distinct ideas that work together.

---
