# Bias-Variance Tradeoff

> The single most important concept in ML. Every algorithm design decision is secretly a bias-variance decision.

---

## The Intuition

Imagine you're trying to hit a bullseye on a dartboard.

- **High Bias** — your throws are consistently off-centre in the same direction. You're missing *systematically*. The model has wrong assumptions baked in.
- **High Variance** — your throws are all over the place. Sometimes close, sometimes way off. The model is too sensitive to the specific training data it saw.
- **The goal** — low bias *and* low variance. Consistently close to the bullseye.

The cruel reality: **reducing bias tends to increase variance, and vice versa.** This is the tradeoff.

```
Total Error = Bias² + Variance + Irreducible Noise
```

You can never eliminate irreducible noise (it's in the data itself). But you can tune the bias-variance balance.

---

## The Math

For a model `f̂(x)` trained on dataset `D`, predicting target `y`:

```
E[(y - f̂(x))²] = Bias[f̂(x)]² + Var[f̂(x)] + σ²
```

Where:
- **Bias** = `E[f̂(x)] - f(x)` — how far the average prediction is from the truth
- **Variance** = `E[(f̂(x) - E[f̂(x)])²]` — how much predictions scatter around their own average
- **σ²** = irreducible noise in the data

Key insight: Bias measures error from wrong assumptions. Variance measures error from sensitivity to training data fluctuations.

---

## How Model Complexity Connects

| Model Complexity | Bias | Variance | Likely Problem |
|---|---|---|---|
| Too simple (underfit) | High | Low | Misses real patterns |
| Just right | Low | Low | Good generalisation |
| Too complex (overfit) | Low | High | Memorises training data |

**Examples:**
- Linear Regression on non-linear data → High Bias (underfitting)
- Deep Decision Tree with no pruning → High Variance (overfitting)
- Random Forest → deliberately averages many high-variance trees to reduce overall variance

---

## The Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Generate noisy sine wave data
np.random.seed(42)
X = np.sort(np.random.uniform(0, 2 * np.pi, 60))
y = np.sin(X) + np.random.normal(0, 0.3, size=X.shape)
X = X.reshape(-1, 1)

degrees = [1, 3, 9, 15]
train_errors, val_errors = [], []

for d in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=d)),
        ('lin', LinearRegression())
    ])
    # Training error
    model.fit(X, y)
    train_pred = model.predict(X)
    train_mse = np.mean((y - train_pred) ** 2)
    train_errors.append(train_mse)

    # Validation error via cross-validation
    cv_mse = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean()
    val_errors.append(cv_mse)

# Plot
plt.figure(figsize=(10, 5))

# Left: fitted curves
plt.subplot(1, 2, 1)
X_plot = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
for d in degrees:
    model = Pipeline([('poly', PolynomialFeatures(degree=d)), ('lin', LinearRegression())])
    model.fit(X, y)
    plt.plot(X_plot, model.predict(X_plot), label=f'degree={d}')
plt.scatter(X, y, color='black', s=15, label='data', zorder=5)
plt.plot(X_plot, np.sin(X_plot), 'k--', label='true fn', linewidth=2)
plt.legend(fontsize=8)
plt.title('Model Complexity vs Fit')
plt.ylim(-2, 2)

# Right: bias-variance tradeoff curve
plt.subplot(1, 2, 2)
plt.plot(degrees, train_errors, 'o-', label='Training Error (proxy for Bias²)')
plt.plot(degrees, val_errors, 's-', label='Validation Error (Bias² + Variance)')
plt.xlabel('Polynomial Degree (Complexity)')
plt.ylabel('MSE')
plt.title('Bias-Variance Tradeoff')
plt.legend()

plt.tight_layout()
plt.savefig('bias_variance_tradeoff.png', dpi=150)
plt.show()
```

**What to observe:**
- Degree 1 → high training *and* validation error → underfitting (high bias)
- Degree 3 → low training error, low validation error → sweet spot
- Degree 15 → very low training error, high validation error → overfitting (high variance)

---

## Common Mistakes

**1. Judging a model only by training accuracy**
Training accuracy always improves with complexity. Validation accuracy is what matters.

**2. Thinking more data fixes variance**
More data *helps* with variance but doesn't fix a fundamentally wrong model (high bias). If your model is too simple, no amount of data makes it capture the true pattern.

**3. Confusing overfitting with a bad model**
An overfit model isn't broken — it's just too good at memorising. The fix is regularisation, simpler architecture, or more data — not a different algorithm.

**4. Slice-level train/test splits in medical imaging**
If you split CT scan slices randomly (not patient-level), slices from the same patient appear in both train and test sets. The model learns patient-specific noise → appears low variance but actually has data leakage. This is a variance illusion.

---

## When to Think About This

| Situation | Diagnosis | Fix |
|---|---|---|
| Model does well on training, poorly on test | High Variance | Regularise, simplify, get more data |
| Model does poorly on both training and test | High Bias | More complex model, better features |
| Validation curve improves then plateaus | Getting more data won't help | Fix the model architecture |
| Validation curve still improving with data | High Variance | Get more data |

---

## Connection to Algorithms I've Used

| Algorithm | Bias | Variance | Notes |
|---|---|---|---|
| Logistic Regression | Medium | Low | Linear decision boundary assumption |
| Random Forest | Low | Medium→Low | Bagging reduces variance of individual trees |
| KNN (k=1) | Low | High | Memorises training data |
| KNN (k=large) | High | Low | Over-smooths |
| CNN (deep, small dataset) | Low | High | Why dropout and data augmentation matter |

---

*Next: [Gradient Descent](../02-gradient-descent/gradient_descent.md)*
