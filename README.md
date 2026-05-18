# Machine Learning Notes 📚

Personal revision notes on Machine Learning concepts — written while solidifying my understanding after working on real projects during my B.E. at PRMIT&R.

Each note covers the **intuition**, **math**, and **code** together. Not a tutorial series — more like thinking out loud on paper.

---

## Why this exists

After building a CNN-based lung cancer detection system for my final year project, I realized I could train models without fully owning the *why* behind every decision. These notes are my attempt to fix that — going back to fundamentals with fresh eyes and more context.

---

## Roadmap

### Foundation
- [ ] Bias-Variance Tradeoff ← *start here*
- [ ] Gradient Descent (Batch, Stochastic, Mini-batch)
- [ ] Loss Functions — when to use which and why
- [ ] Overfitting & Regularization (L1, L2, Dropout)
- [ ] Train/Validation/Test Split — and why slice-level splits leak data

### Supervised Learning
- [ ] Linear Regression — OLS, assumptions, when it breaks
- [ ] Logistic Regression — odds, log-loss, decision boundary
- [ ] Decision Trees — entropy, information gain, pruning
- [ ] Random Forests — bagging, feature importance, OOB error
- [ ] Support Vector Machines — margin, kernel trick, soft margin
- [ ] K-Nearest Neighbours — distance metrics, curse of dimensionality

### Unsupervised Learning
- [ ] K-Means Clustering — inertia, elbow method, limitations
- [ ] PCA — variance explained, eigenvectors, when to apply

### Model Evaluation
- [ ] Confusion Matrix, Precision, Recall, F1
- [ ] ROC Curve & AUC — what it actually means
- [ ] Cross-Validation — k-fold, stratified, leave-one-out

### Neural Networks & Deep Learning
- [ ] Perceptron to MLP — forward pass, backpropagation
- [ ] Activation Functions — ReLU, Sigmoid, Softmax
- [ ] CNNs — convolution, pooling, feature maps ← *built one, now understanding it*
- [ ] Batch Normalization & Dropout

---

## Format of each note

Every note follows the same structure:

```
## Concept Name
### The Intuition        — what is it, why does it matter
### The Math             — key equations, no hand-waving
### The Code             — minimal working Python example
### Common Mistakes      — what trips people up
### When to use it       — practical decision guidance
```

---

## Stack

- Python 3
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras (for deep learning notes)

---

## Progress

| Topic | Status |
|---|---|
| Bias-Variance Tradeoff | ✅ Done |
| Everything else | 🔄 In progress |

---

*Started May 2026. Updated as I go.*
