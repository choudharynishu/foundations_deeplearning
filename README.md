# Foundations of Deep Learning

This repository is a hands-on exploration of the **building blocks of deep learning**.  
Instead of relying on high-level frameworks, the goal here is to re-implement and experiment with the core ideas that make neural networks work.

Deep learning systems are shaped by a few fundamental components:  
- **Activation functions** (introducing non-linearity)  
- **Initialization strategies** (setting the stage for stable learning)  
- **Optimization algorithms** (navigating the loss surface)  
- **Network architectures** (defining the flow of information)  

By breaking these concepts down and testing them in isolation, this project builds intuition for how neural networks learn
and why certain design choices matter.

![Pathological Loss Surface](./Optimization/artifacts/PathologicalCurveLoss.png)

---

## What You’ll Find in This Repository

- **From-scratch implementations** of fundamental components:
  - Activation functions (`tanh`, `ReLU`, `sigmoid`, etc.)
  - Weight initialization strategies
  - Optimizers (SGD, Momentum, Adam)
- **Feed-forward neural network** experiments on FashionMNIST
- Visualizations of training dynamics and loss landscapes
- Comparisons that highlight how design choices affect performance

