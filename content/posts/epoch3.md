---
title: "Epoch 3: Micrograd, Part 3"
date: 2025-12-30
draft: false
---



## Intro

Hi, this is chefcinnamon, This is Epoch 3, Micrograd Part 3, in the previous epoch we learnt how to create a neuron manually, and we made an introduction about Tensors and PyTorch, now we are going to learn how to make a simple neural network (nn) layers of neurons 

## Let's train manually with a neuron!

Let me go through a neuron first, an abstract about its mechanism with a practical example:

neuron is

<p align="center">
z = x1*w1 + x2*w2 + b
</p>

and we activate it with tanh(or other activators)

<p align="center">
y_hat = tanh(z)
</p>

Also for the Loss, let's use squared error:

<p align="center">
L = (y_hat - y_true)^2
</p>

**y_hat** is the prediction, output that comes from our model, it changes every run with a little more trained weights and bias to get closer to **y_true**: the output from our dataset, predefined, and our goal to reach!

**Step 0**

Initialize inputs and random weights & bias<br><br>
w1:	0.0259 (learnable)<br>
w2:	0.1967 (learnable)<br>
b:	0.5 (learnable)<br>
x1:  0.5 <br>
x2:  0.7 <br>
y_true = 1.0 <br>
lr = 0.1 <br>

**Step 1: First Forward pass**

Calculate *y_hat* with the inputs and the current weights and bias

z = (0.5 * 0.0259) + (0.7 * 0.1967) + 0.5 <br>
z = 0.65064 <br>
y_hat = tanh(0.65064) ≈ 0.572101

*Note:* tanh(x) splash a number into a <br>
number range: -1 < y < 1

**Step 2: Compute Loss**

L = (y_hat - y_true)^2 ≈ 0.1831

***Everytime we check this loss to go lower***

Now we need to go backward, calculate the gradients of loss (L) with respect to the inputs, then create and use new weights and bias, then repeat the process of going forward, backward, train

**Step 3: Backward Pass: compute gradients**

To compute the gradients of **Loss(L)** with respect to the parameters we need to begin from calculating the gradient of L w.r.t its parents (y_hat and y_true), <br>then the gradient of L w.r.t **tanh(y_hat)** <br>then keep going backward and calculate the gradient of **L** w.r.t its parents by multiplying *∂L/∂child * ∂child/∂parent* (Chain Rule)<br> and continue the process until we reach to the inputs.

**3.1: Gradient L w.r.t y_hat**

*Each operation calrify what formula to use for calculating local derivatives, refer to [Epoch 1](https://chefcinnamon.github.io/memo/posts/epoch1/)*

∂L/∂y_hat = 2 * (y_hat - y_true) <br>
∂L/∂y_hat = -0.855798

*note* Since y_true is not a leaf of other branches, no need to calcualte its gradient.

**3.2: Gradient y_hat w.r.t z and L w.r.t z**
<br><br>
∂y_hat/∂z = 1 - y_hat^2 <br>
∂y_hat/∂z ≈ 0.6727

from the Chain Rule, to calculate the gradient of L w.r.t z:

∂L/∂z = ∂L/∂y_hat * ∂y_hat/∂z <br>
∂L/∂z ≈ -0.5757

**3.3: Gradient L w.r.t w1 and w2 and b**
<br><br>
∂L/∂w1 = ∂L/∂z * x1 ≈ -0.288 <br>
∂L/∂w2 = ∂L/∂z * x2 = -0.403 <br>
∂L/∂b  = ∂L/∂z * 1 ≈ -0.5757 <br>

**3.4: Update parameters (gradient descent)**
$$
\theta_{n+1} = \theta_n - \eta \cdot \frac{\partial L}{\partial \theta}
$$

w1 = w1 - lr * ∂L/∂w1 ≈ 0.0259 - 0.1 * -0.288 ≈ 0.0547<br>
w2 = w2 - lr * ∂L/∂w2 ≈ 0.1967 - 0.1 * -0.403 ≈0.237<br>
b  = b  - lr * ∂L/∂b ≈ 0.5 - 0.1 * -0.5757 ≈ 0.55757<br>

As you see all numbers changed slightly! that's learning! <br>
w1: 0.0259 -> 0.0547
w2: 0.1967 -> 0.237
b: 0.5 -> 0.55757

Now if we do the forward pass again with the new weights and bias: <br>
L = (tanh(0.5 * 0.0547 + 0.7 * 0.237 + 0.55757) - 1) ^ 2 <br>
≈ 0.13276 <br><br>
0.1831 -> 0.13276
So the Loss also decreased, and going toward 0

**Step 4: Repeat**


## Content: nn

Now let's implement a neural network with layers in micrograd.

We learnt how actually a neuron works, and we were able to build a neuron from scratch in the micrograd, now we want to build a multi layer neurons in micrograd.

A neural network made of input and output layers + Wiring:

**Input layer:** This layer contains input neuron, they are not computational neurons

**Hidden Layers:** They are the computation layers, located in the middle of input and output layers.

In each hidden layer neurons follow the same formula of

<p align="center">
z = x1*w1 + x2*w2 + b
</p>

Note: in one hidden layer: neurons have same inputs but different weights each time

in hidden layer 2: neurons have different inputs than layer 1 because the inputs have changed after going through layer 1.

e.g hidden layer 1:

<p align="center">
z1 = x1*w1 + x2*w2 + b1
</p>

hidden layer 2:

<p align="center">
z2 = x3*w3 + x2*w4 + b2
</p>

**Wiring**: Clarifies the wiring and connections between layers. We have different architecture for wiring, such as: <br> <br>
***MLP:*** Fully conneced neurons between layers. <br>
***CNN:*** Not fully connected, uses local connections (kernels). <br>
***RNN:*** Has loops, has memory <br>
***Transformer:*** Use attention <br> <br> <br> 


Let's make nn manually wih micrograd:






****


Firstly we begin with<br><br>
 ## Content: Neuron

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
```


**nin** number of inputs each neuron receives <br>
**self.w** generates n random numbers between -1 and 1 for weights<br>
**self.b** generates 1 random number for bias <br>
**__init__ of Neuron** is used one time to initialize the neuron <br>
**parameters** returns a list of trainable Values in this neuron <br>
**__call__ of Neuron** forward pass <br>
**zip** pairs up ws with xs in order <br> <br><br>

## Content: Layer
```python
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
```

**nin** number of inputs each neuron receives <br>
**nout** number of neurons in this layer <br> <br>

 ```python 
 [Neuron(nin) for _ in range(nout)]
 ```
 
Creates a list of neurons <br><br>
e.g. nin = 3, nout = 4 <br>
self.neurons = [ <br>
    Neuron(3), <br>
    Neuron(3), <br>
    Neuron(3), <br>
    Neuron(3), <br>
] <br> <br>

Until here we initialize our layer with 4 neurons (calling Layer), each neuron accepts 3 inputs (calling Neuron from Layer) in our example <br> <br>

**__call__ of Layer** 

```python 
[n(x) for n in self.neurons]
```

From the list of neurons we made, we give same inputs x to each neuron<br>
folloiwng our previous example of nin = 3, nout = 4 <br> <br>
imagine we add x = [x1, x2, x3] <br><br>

So by this until here: we have: <br>
[tanh(w11*x1 + w12*x2 + w13*x3 + b1),<br>
 tanh(w14*x1 + w15*x2 + w16*x3 + b2),<br>
 tanh(w17*x1 + w18*x2 + w19*x3 + b3),<br>
 tanh(w20*x1 + w21*x2 + w22*x3 + b4)]<br>
<br><br><br>

```python 
return outs[0] if len(outs) == 1 else outs
```
This is just for the case we have 1 neuron to return 1 neuron instead of list. <br> <br>

```python 
[p for neuron in self.neurons for p in neuron.parameters()]
```

This is a nested loop, <br>(loop over all neurons in the layer, and for each neuron it loops and extract the parameters w and b)<br><br>
As a result it creates a flat array of all parameters in order <br><br>
e.g <br>
[
  w11, w12, w13, b1, <br>
  w21, w22, w23, b2, <br>
  w31, w32, w33, b3, <br>
  w41, w42, w43, b4  <br>
]<br><br> <br>


## Content: Wires (MLP)
``` python
class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
        
```

**nin** number of inputs to the first layer <br>
**nouts** number of neurons in each layer <br>
e.g [4, 4, 1] <br><br>

```python
sz = [nin] + nouts
```
This size of array helps us to show the inputs and outputs of each layer<br><br>
e.g. <br>
nin = 3 <br>
nouts = [4, 4, 1] <br>
sz = [3, 4, 4, 1] <br> <br>

<br>

```python
self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
```
This creates **list of layers**, that each layer has the property: number of inputs and number of neurons(= outputs) <br>
means: <br>
Layer 0: inputs = 3, outputs = 4 also Layer(3, 4)<br><br>
Layer 1: inputs = 4, outputs = 4 also Layer(4, 4)<br><br>
Layer 2: inputs = 4, outputs = 1 also Layer(4, 1)<br><br>

*note:* Layer(3, 4) will be expanded to <br>
    Neuron(3), <br>
    Neuron(3), <br>
    Neuron(3), <br>
    Neuron(3), <br>

<br><br>

```python
def __call__(self, x):
   for layer in self.layers:
       x = layer(x)
   return x
```
Now it's the time to give x to nn, because of our MLP arcitucture, we begin the loop for all layers, first layer gets x(one or more inputs), then it calculates the y which if you remember from **Neuron**: <br>

act = sum(wi*xi for wi, xi in zip(self.w, x), self.b) <br> <br>
y = act.tanh() <br>

because of: *x = layer(x)* , our vector y (y1 neuron 1, y2 neuron 2,...), is considered as x of the next layer <br>

> **Important note**
>
> Because we use the vector of all **y** values from layer *n* as the input **x** for layer *n + 1*,  
> Each neuron output in layer *n* can influence all the outputs of layer *n + k*.

<br>

***Example:***

> MLP: nin = 3, nouts = [4,4,1]

**Initial input:**

x = [x1, x2, x3]

**Layer 0: Layer(3,4)**

Input: [x1, x2, x3]

Neurons N1..N4 produce [y1, y2, y3, y4]

x = [y1, y2, y3, y4]

**Layer 1: Layer(4,4)**

Input: [y1, y2, y3, y4]

Neurons N5..N8 produce [y5, y6, y7, y8]

x = [y5, y6, y7, y8]

**Layer 2: Layer(4,1)**

Input: [y5, y6, y7, y8]

Neuron N9 produces single y9

x = y9

**Final return →** y9 (Value object) <br><br> <br>
Excellent! now we activate non-linearity with tanh to get the **y_hat** <br>
tanh(y) = y_hat <br><br>

Now we calculate Loss (L):
<p align="center">
L = (y_hat - y_true)^2
</p>
Then <br><br>

**Update parameters (gradient descent)** <br>
with: <br>
$$
\theta_{n+1} = \theta_n - \eta \cdot \frac{\partial L}{\partial \theta}
$$
<br>

Let's implement this last step to our micrograd too <br>
We want to make the initiator of MLP to calculate the tanh(y_hat), then calculate the loss, then do backward() to calculate the gradient, then find the new parameters (train), then find the new loss, and do these steps in loops k times. <br><br> I call it "tunning a big radio with many knobs" the better result is for being more precise (smaller learning ratio with more steps)

<br>

```python
for k in range(10): #1
    ypred = [n(x) for x in xs] #2
    loss = sum((y_hat - y_true)**2 for y_true, y_hat in zip(ys, ypred)) #3

    for p in n.parameters(): #5
        p.grad = 0.0 #6
    loss.backward() #7
    
    for p in n.parameters(): 
        p.data += -0.05 * p.grad #10

    print(k, loss.data) #12


xs = [
    [3.0, 3.0, 2.0],
    [2.5, 3.0, 1.0],
    [-2.0, -1.5, -1.0],
    [1.0, -3.0, 0.5],
]
ys = [1.0 , -1.0, -1.0, 1.0]
n = MLP(3, [4,4,1])

draw_dot(loss) #24
```

*Line 1*: This training has 10 loops (10 times going forward and backward and train) <br><br>
*Line 2*: Uses each x in xs list, to call n, which is<br>
*MLP(3, [4,4,1])*<br><br>
*Line 3*: Calulates the sigma of Losses,  <br><br>
*Line 5*: Gather all the parameters from the MLP (all layers, all neurons) <br><br>
*Line 6*: For each step (k) we need to reset the gradient, to not add gradients of each step (k)
<br><br>
*Line 7*: backward calculates all the gradients, <br><br>

> Note: We could do backward, because loss is a Value object refer to [Epoch 1](https://chefcinnamon.github.io/memo/posts/epoch1/), that from **forward pass**, it stored ._prev and .data, <br>
and from **backward pass** it stores .grad 
<br><br>

*Line 10*: After each backward, it updates the parameters with:

$$
\theta_{n+1} = \theta_n - \eta \cdot \frac{\partial L}{\partial \theta}
$$

<br><br>
*Line 12*: 
> 0 2.094403631655075 <br>
> 1 2.0751115878578568 <br>
> 2 2.058205309464937 <br>
> 3 2.0417761466655073 <br>
> 4 2.024357050402771 <br>
> 5 2.0042442666008617 <br>
> 6 1.9787204164593886 <br>
> 7 1.942384309638119 <br>
> 8 1.8821567826349068 <br>
> 9 1.7582003851194667 <br>

<br><br>
*Line 24*:
generate the latest graph
<br>
![epoch3-garph-micrograd.svg](https://chefcinnamon.github.io/memo/img/epoch3-garph-micrograd.svg)

<br><br>
You can access the code here:
https://github.com/chefcinnamon/memo/blob/main/codes/micrograd_3.ipynb

## Resources

### Micrograd - A Tiny Autograd Engine
[GitHub: karpathy/micrograd](https://github.com/karpathy/micrograd)

A tiny scalar-valued autograd engine and a neural net library on top of it with PyTorch-like API, created by Andrej Karpathy.

### The spelled-out intro to neural networks and backpropagation: building micrograd
<iframe width="560" height="315" src="https://www.youtube.com/embed/VMj-3S1tku0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
