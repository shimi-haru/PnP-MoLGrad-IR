# PnP Image Restoration with Weight-Tying Nonnegative Neural Network

This repository provides the official PyTorch implementation for the paper:

**"Plugging Weight-tying Nonnegative Neural Network into Proximal Splitting Method:
Architecture for Guaranteeing Convergence to Optimal Point"**
Haruya Shimizu, Masahiro Yukawa
(arXiv: https://arxiv.org/abs/2510.21421)

---

## 🔍 Overview

This repository contains:

- **MoL-Grad denoiser**: a multi-layer weight-tying nonnegative neural network
  that satisfies the MoL-Grad conditions (convex potential + Lipschitz gradient).
- **Denoiser training code**
  (noise addition, loss function with nonnegativity penalty, Adam optimization).
- **Deblurring training / inference code**
  using the primal–dual splitting Algorithm 1 in the paper.
- **Pretrained models**
  (optionally provided through GitHub Releases).

The goal is to provide an _explainable_ Plug-and-Play (PnP) framework whose
denoiser is mathematically guaranteed to be an s-prox operator, ensuring convergence
to the minimizer of a cost function involving an implicit weakly convex regularizer.

---

## 📚 Paper Summary

The proposed denoiser:

- is a **multi-layer autoencoder-type NN** with weight tying
  (encoder weights = decoder weightsᵀ)
- enforces **non-negativity** of weights
- satisfies **monotonicity** and **Lipschitz continuity**
- yields a **MoL-Grad denoiser**:
  \[
  D(x) = \nabla \psi(x)
  \]
  for a smooth convex potential ψ
- induces an _implicit weakly convex regularizer_
  \[
  \varphi = \psi^\* - \frac{1}{2}\|x\|^2
  \]
- **guarantees convergence** of PnP proximal splitting without restricting
  Lipschitz < 1 (unlike classical nonexpansive PnP)

Deblurring is solved by the primal–dual algorithm described in **Algorithm 1** of the paper.

---

## 📦 Requirements

## Citation

```bibtex
@article{shimizu2025molgrad,
  title={Plugging Weight-tying Nonnegative Neural Network into Proximal Splitting Method: Architecture for Guaranteeing Convergence to Optimal Point},
  author={Shimizu, Haruya and Yukawa, Masahiro},
  journal={arXiv preprint arXiv:2510.21421},
  year={2025}
}

```
