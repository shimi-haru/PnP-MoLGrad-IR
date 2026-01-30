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

## 📚 Abstract

We propose a novel multi-layer neural network architecture that gives a promising neural network empowered optimization approach to the image restoration problem. The proposed architecture is motivated by the recent study of monotone Lipschitz-gradient (MoL-Grad) denoiser (Yukawa and Yamada, 2025) which establishes an "explainable" plug-and-play (PnP) framework in the sense of disclosing the objective minimized. The architecture is derived from the gradient of a superposition of functions associated with each layer, having the weights in the encoder and decoder tied with each other. Convexity of the potential, and thus monotonicity of its gradient (denoiser), is ensured by restricting ourselves to nonnegative weights. Unlike the previous PnP approaches with theoretical guarantees, the denoiser is free from constraints on the Lipschitz constant of the denoiser. Our PnP algorithm employing the weight-tying nonnegative neural network converges to a minimizer of the objective involving an "implicit" weakly convex regularizer induced by the denoiser. The convergence analysis relies on an efficient technique to preserve the overall convexity even in the ill-conditioned case where the loss function is not strongly convex. The simulation study shows the advantages of the Lipschitz-constraint-free nature of the proposed denoiser in training time as well as deblurring performance.

---

## Citation

```bibtex
@article{shimizu2025WNNN,
  title={Plugging Weight-tying Nonnegative Neural Network into Proximal Splitting Method: Architecture for Guaranteeing Convergence to Optimal Point},
  author={Shimizu, Haruya and Yukawa, Masahiro},
  journal={arXiv preprint arXiv:2510.21421},
  year={2025}
}

```
