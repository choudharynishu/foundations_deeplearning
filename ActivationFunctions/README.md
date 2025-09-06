This experiment investigates the impact of activation functions on neural network performance by systematically comparing 
multiple activation functions while holding all other variables constant. 

The experimental setup will utilize a fixed neural network architecture(4-layer feed forward neural network) and a single
dataset, with only the activation function varying across experimental conditions. The activation functions to be tested
will include:

1.	Sigmoid
2.	Tanh
3.	ReLU
4.	LeakyReLU
5.	ELU
6.	Swish

Each activation function will be trained using identical hyperparameters (learning rate, batch size, optimizer, number of epochs)
and the same train/validation/test data splits to ensure fair comparison. The figure illustrates common activation functions 
along with their gradients. We observe that functions such as ReLU, ELU, LeakyReLU, and Swish can produce increasingly large gradients
as the input values become highly positive (“exploding gradient”). In contrast, activations like Tanh and Sigmoid have bounded gradients
that saturate for large positive or negative inputs, potentially causing vanishing gradients. Swish is somewhat intermediate, 
since it retains smoothness but can still suffer from unbounded growth on the positive side.

![Activation-Functions](ActivationFunctions/ActivationFunctions.png)
 
Performance Metrics: The behavior of each trained network will be assessed through test set accuracy. 
Observations: All activations show similar accuracy of around 88.23 % except for Sigmoid for which it is around 10%. 
Typically, this type of performance difference could be due to several reasons: 

- Output–loss mismatch: Adding an activation before the loss can be problematic (especially Sigmoid), since CrossEntropyLoss
internally applies LogSoftmax. This can shrink gradients, but the architecture here avoids that issue.
- Initialization: PyTorch defaults to Kaiming initialization, which suits ReLU-like activations but not Sigmoid/Tanh, where Xavier is preferred.
- Activation saturation: Sigmoid is prone to vanishing gradients when inputs fall in the saturated regions, while exploding gradients are unlikely. 
- Learning rate sensitivity: Sigmoid’s smaller gradients may require a different learning rate to train effectively.
- Dead neurons: A concern with ReLU variants, but not relevant for Sigmoid in this setup.
- Regularization effects: Dropout or weight decay can interact poorly with saturating activations, though no such regularization was applied here.

The most likely issue is activation saturation, shown by the gradient distributions in case of two activation functions, 
Sigmoid and ReLU. Top (ReLU) Bottom (Sigmoid)
![ReLU-Gradient_evolution](ActivationFunctions/gradient_evolution_relu.gif)
![Sigmoid-Gradient_evolution](ActivationFunctions/gradient_evolution_sigmoid.gif)

These two gradient plots show a critical difference: 
- ReLU maintains healthy gradient flow across all layers, enabling effective network-wide learning. 
- While, Sigmoid suffers from vanishing gradients, where only deep layers receive meaningful signals while earlier layers barely update during training. 

