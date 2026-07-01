# Loss Functions — When to Use Which and Why

> The loss function is the only thing your model actually listens to. Pick the wrong one and gradient descent will optimise perfectly for the wrong goal.

---

## The Intuition

A loss function is how you tell a model *what "wrong" means*.

Everything else — the architecture, the optimiser, the learning rate — only matters in service of minimising this one number. Two models can have identical architectures and training loops, but if you swap the loss function, you get a completely different model, because you've changed what it's being punished for.

Think of it like grading a student. If you only grade based on "did you get the final number right," the student optimises for guessing the right number, even by luck. If you grade based on "how far off was your reasoning," the student optimises for a careful process. Same student, same exam — different grading rubric produces a different kind of learner.

**Loss function vs. cost function vs. objective function** — people use these almost interchangeably, but technically:
- **Loss** — error on a *single* training example
- **Cost** — the average loss across the *whole* dataset (or a batch)
- **Objective** — the general term for whatever you're optimising (may include the cost plus regularisation terms)

In practice, most people say "loss" for all three. Don't lose sleep over it.

---

## The Math

At the core, every loss function is a distance measure between what the model predicted (`ŷ`) and what actually happened (`y`). The differences are in *how* that distance is measured, and that choice encodes assumptions about your data and your goals.

**General form:**

```
Cost(θ) = (1/n) · Σ L(yᵢ, ŷᵢ)
```

Where `L` is the per-example loss, `n` is the number of examples, and `θ` are the model parameters gradient descent is adjusting.

The shape of `L` determines:
1. **How much big errors are punished relative to small ones** (quadratic vs. linear penalties)
2. **How sensitive the model is to outliers**
3. **What kind of prediction the model is even structurally capable of making** (a number, a probability, a ranking)

---

## Regression Losses

### Mean Squared Error (MSE / L2 Loss)

```
L(y, ŷ) = (y - ŷ)²
```

Squares the error. A prediction that's off by 10 contributes 100x more loss than one off by 1. This makes MSE **extremely sensitive to outliers** — a single bad data point can dominate the entire loss.

**Use when:** errors should be penalised quadratically, your target is roughly Gaussian-distributed, and you don't have significant outliers you want the model to ignore.

**Why it's the default:** it's differentiable everywhere (smooth gradient, easy for gradient descent), and it has a clean probabilistic interpretation — minimising MSE is equivalent to maximum likelihood estimation under the assumption that your errors are normally distributed.

### Mean Absolute Error (MAE / L1 Loss)

```
L(y, ŷ) = |y - ŷ|
```

Penalises errors linearly. A prediction off by 10 contributes 10x the loss of one off by 1 — not 100x. This makes MAE **robust to outliers**: a few wild data points won't drag the whole model toward them.

**Use when:** your data has outliers you don't want to dominate training, or when the median (not the mean) is a more meaningful summary of your target.

**The cost:** MAE's gradient is constant everywhere except at zero, where it's undefined (a sharp corner, not a smooth curve). This makes optimisation slightly less stable near the minimum compared to MSE's smooth bowl.

### Huber Loss

```
L(y, ŷ) = 0.5(y - ŷ)²           if |y - ŷ| ≤ δ
        = δ|y - ŷ| - 0.5δ²      otherwise
```

The compromise: quadratic (like MSE) for small errors, linear (like MAE) for large ones. `δ` is a threshold you choose.

**Use when:** you want MSE's smooth gradient near the minimum but MAE's outlier robustness for large errors. Common in reinforcement learning (e.g. DQN) and robotics, where reward/error spikes are common but you still want stable optimisation most of the time.

| Loss | Outlier Sensitivity | Gradient Smoothness | Typical Use |
|---|---|---|---|
| MSE | High | Smooth everywhere | Clean regression data, default choice |
| MAE | Low | Sharp at zero | Data with outliers, median-based targets |
| Huber | Medium (tunable via δ) | Smooth, tunable | RL, robotics, noisy real-world regression |

---

## Classification Losses

### Binary Cross-Entropy (Log Loss)

```
L(y, ŷ) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

For binary classification where `ŷ` is a predicted probability (0 to 1) and `y` is the true label (0 or 1).

**The intuition:** it punishes confident wrong answers *severely*. If the true label is 1 and the model predicted `ŷ = 0.01` (very confident it's 0), `-log(0.01) ≈ 4.6` — a large penalty. If the model had predicted `ŷ = 0.5` (unsure), the penalty is `-log(0.5) ≈ 0.69` — much smaller. Confidently wrong is punished harder than honestly unsure.

**Use when:** the output is a probability from a sigmoid activation and you have two classes.

### Categorical Cross-Entropy

```
L(y, ŷ) = -Σ yᵢ · log(ŷᵢ)
```

The multi-class generalisation of binary cross-entropy. `y` is a one-hot vector, `ŷ` is a probability distribution over all classes (from softmax).

**Use when:** multi-class classification where each example belongs to exactly one class. This is the standard loss paired with a softmax output layer — they're almost always used together.

**Sparse categorical cross-entropy** is the same thing, just with integer labels instead of one-hot vectors — a memory-saving implementation detail, not a different loss mathematically.

### Hinge Loss

```
L(y, ŷ) = max(0, 1 - y·ŷ)
```

Where `y ∈ {-1, +1}` and `ŷ` is the raw model output (not a probability).

**Use when:** training Support Vector Machines. The loss is zero once a prediction is not just correct but correct *with margin* — it doesn't just want the right side of the boundary, it wants comfortable distance from it. This is what gives SVMs their "maximum margin" property.

**Not commonly used outside SVMs** — cross-entropy dominates for neural network classifiers because it produces well-calibrated probabilities, which hinge loss doesn't.

| Loss | Output Type | Typical Use |
|---|---|---|
| Binary Cross-Entropy | Sigmoid probability | Binary classification |
| Categorical Cross-Entropy | Softmax distribution | Multi-class, single-label |
| Hinge Loss | Raw margin score | SVMs |

---

## Specialised Losses Worth Knowing

**KL Divergence** — measures how one probability distribution diverges from another. Used in variational autoencoders and whenever you're matching a learned distribution to a target distribution, not just a single label.

**Focal Loss** — a modification of cross-entropy that down-weights easy, well-classified examples and focuses training on hard ones. Built for extreme class imbalance (e.g. object detection, where background pixels vastly outnumber object pixels).

**Contrastive / Triplet Loss** — instead of predicting a label, the model learns to pull similar examples together and push dissimilar ones apart in embedding space. Used in face recognition and metric learning, where the goal is a good embedding, not a class prediction.

**Dice Loss** — common in medical image segmentation (like your CT scan work). Measures overlap between predicted and true segmentation masks. Handles class imbalance between "tumour pixels" and "background pixels" far better than plain cross-entropy would, since cross-entropy would happily predict "background everywhere" and still get a deceptively low loss.

---

## The Code

```python
import numpy as np
import matplotlib.pyplot as plt

# ── Comparing MSE, MAE, and Huber under an outlier ──

y_true = 3.0
y_pred_range = np.linspace(-5, 10, 200)

def mse(y, yp):
    return (y - yp) ** 2

def mae(y, yp):
    return np.abs(y - yp)

def huber(y, yp, delta=1.0):
    err = np.abs(y - yp)
    return np.where(err <= delta, 0.5 * err**2, delta * err - 0.5 * delta**2)

plt.figure(figsize=(7, 5))
plt.plot(y_pred_range, mse(y_true, y_pred_range), label='MSE')
plt.plot(y_pred_range, mae(y_true, y_pred_range), label='MAE')
plt.plot(y_pred_range, huber(y_true, y_pred_range), label='Huber (δ=1)')
plt.axvline(y_true, color='gray', linestyle='--', alpha=0.5, label='true value')
plt.xlabel('Predicted value')
plt.ylabel('Loss')
plt.title('Loss Curves: Sensitivity to Error Magnitude')
plt.legend()
plt.tight_layout()
plt.savefig('regression_losses.png', dpi=150)
plt.show()

# ── Binary cross-entropy: cost of confident wrong predictions ──

y_true_label = 1
p_range = np.linspace(0.001, 0.999, 200)

bce = -(y_true_label * np.log(p_range) + (1 - y_true_label) * np.log(1 - p_range))

plt.figure(figsize=(7, 5))
plt.plot(p_range, bce)
plt.xlabel('Predicted probability that y=1')
plt.ylabel('Loss')
plt.title('Binary Cross-Entropy — true label is 1')
plt.annotate('Confidently wrong\n(huge penalty)', xy=(0.05, 3), fontsize=9)
plt.annotate('Confidently right\n(near-zero penalty)', xy=(0.75, 0.6), fontsize=9)
plt.tight_layout()
plt.savefig('bce_loss.png', dpi=150)
plt.show()
```

**What to observe:**
- The MSE curve is a steep parabola — it punishes the outlier region far more than MAE does.
- The MAE curve is a straight V — constant penalty per unit of error, regardless of how large the error already is.
- The Huber curve matches MSE near the true value but flattens into a line further out, exactly as designed.
- The binary cross-entropy curve shoots up sharply as the predicted probability approaches the *wrong* extreme — this is what "confidently wrong is punished hardest" looks like as a graph.

---

## Common Mistakes

**1. Using MSE for classification**
MSE technically "works" as a loss for classification, but its gradients are poorly suited to sigmoid/softmax outputs — they vanish when predictions are confidently wrong, which is the opposite of what you want. Cross-entropy's gradient stays strong exactly when the model needs the biggest push. Always use cross-entropy family losses for classification.

**2. Using accuracy as if it were the loss**
Accuracy is not differentiable — you can't compute a gradient from it, so gradient descent can't optimise it directly. The loss function is a *differentiable proxy* for the metric you actually care about. This is why you train on cross-entropy but report accuracy — they're related but not the same thing.

**3. Ignoring class imbalance in the loss choice**
Plain cross-entropy on a dataset that's 99% negative class will happily converge to "always predict negative" and still look like it has low loss. This is exactly the trap in medical imaging (tumour vs. no-tumour pixels). Solutions: class-weighted cross-entropy, focal loss, or Dice loss for segmentation tasks.

**4. Forgetting that the loss defines what "good" means**
If you optimise MAE, you're implicitly training the model to predict the *median*. If you optimise MSE, you're implicitly training it to predict the *mean*. These are different targets that happen to coincide only for symmetric distributions. Choosing a loss is choosing a statistical target, not just a training detail.

**5. Mismatching loss and final-layer activation**
Cross-entropy expects probabilities (post-sigmoid/softmax). Feeding it raw logits without the matching activation — or double-applying softmax — silently produces broken gradients. Most frameworks now offer numerically stable combined implementations (e.g. `BCEWithLogitsLoss` in PyTorch) specifically to avoid this bug.

---

## When to Think About This

| Situation | Diagnosis | Fix |
|---|---|---|
| Regression model wildly thrown off by a few extreme values | MSE amplifying outliers | Switch to MAE or Huber |
| Classifier's loss decreasing but predictions all one class | Class imbalance not accounted for | Class-weighted loss, focal loss, or resampling |
| Segmentation model predicts "background everywhere" | Cross-entropy dominated by majority class | Dice loss or IoU-based loss |
| Training loss is NaN early in classification training | Log(0) from an unclamped probability, or logits/probabilities mismatch | Use framework's combined logits+loss function |
| Model's loss looks great but real-world errors feel "off" | Loss doesn't match the actual business objective | Re-derive which statistic (mean/median/margin) you actually want |
| Face recognition / similarity model not separating classes well | Using classification loss instead of embedding loss | Switch to contrastive or triplet loss |

---

## Connection to Models I've Used

| Model / Task | Typical Loss | Notes |
|---|---|---|
| Linear Regression | MSE | Closed-form solution assumes Gaussian errors |
| Logistic Regression | Binary Cross-Entropy | Paired naturally with sigmoid output |
| Random Forest / Decision Trees | Gini impurity or entropy (a loss for *splitting*, not gradient-based) | Not optimised via gradient descent at all |
| SVM | Hinge Loss | Encodes the max-margin objective directly |
| CNN for lung cancer detection (multi-class) | Categorical Cross-Entropy | Paired with softmax; watch for class imbalance between malignant/benign slices |
| CNN for tumour segmentation | Dice Loss (often combined with cross-entropy) | Handles the small-tumour-region-vs-huge-background imbalance |
| Transformer / LLM (next-token prediction) | Cross-Entropy over vocabulary | Same core idea as categorical cross-entropy, just an enormous number of classes |

The pattern: **the loss function is a translation of your real-world goal into something gradient descent can act on.** Get the translation wrong, and no amount of architecture tuning or hyperparameter search will fix it — you're aiming the whole training process at the wrong target.

---
