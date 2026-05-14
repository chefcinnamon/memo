---
title: "Epoch1: Micrograd, Part 1"
date: 2025-12-25
draft: false
---



## Intro

Hi! this is chefcinnamon. I've decided to study AI, ML... almost every day with doing practical projects.

I write about them here.

This is Epoch 1: the very first step.



## Content: Gradient
Building a neural network; a tiny autograd engine (micrograd) — an automatic differentiation engine with backpropagation.

**Derivative** (single input change → single output change): the rate of change.

<div class="math-container">
<p>Example: if $f(x)=x^2$, and $x$ changes by a fixed amount, say 1:</p>
<p>if $x=3$, then $f(x)=9$</p>
<p>if $x=4$, then $f(x)=16$</p>
<p>Derivative:</p>
<p>$\frac{16-9}{4-3} = \frac{7}{1.00} = 7.00$</p>
</div>

**Gradient**: how a single output changes by adjusting multiple inputs.

<div class="math-container">
<p>Example: if $f(x,y) = x^2 + y^2$ at point (1,2):</p>
<p>Partial derivative w.r.t. x: $\frac{\partial f}{\partial x} = 2x = 2(1) = 2$</p>
<p>Partial derivative w.r.t. y: $\frac{\partial f}{\partial y} = 2y = 2(2) = 4$</p>
<p>Gradient vector: $\nabla f = \langle 2, 4 \rangle$</p>
<p>This points in the direction of steepest ascent from (1,2)</p>
</div>

**Extended description**: We have a function with two inputs $f(x, y)$ (say it as "f of x comma y"). $f(x, y) = x^2 + y^2$ (say it as "x squared plus y squared").

Partial derivative of f with respect to x: $\frac{\partial f}{\partial x}$ means: How fast does f change if I move x a tiny bit, while y stays frozen?

In this example the derivative of $x^2 \rightarrow 2x$, derivative of $y^2 \rightarrow 0$ (because y is treated as fixed).

How we calculate derivative: $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$ (say it as: "The derivative of f at x equals the limit as h goes to zero of (f of x plus h minus f of x) divided by h").

Derivatives are also written as $f'(x)$ or $\frac{df}{dx}$.

So we got $\frac{\partial f}{\partial x} = 2x$ (say it as "partial derivative of f with respect to x").

E.g. $\frac{\partial f}{\partial x}$ at (1, 2) = 2 × 1 = 2. "At the point one comma two, the partial derivative with respect to x equals two" means: If I move a tiny bit in the x direction, the height goes up by about 2 per unit.

From the same function: $f(x, y) = x^2 + y^2$.

Derivative of $y^2 \rightarrow 2y$, derivative of $x^2 \rightarrow 0$ (x is frozen now).

So: $\frac{\partial f}{\partial y} = 2y$ (say it as "partial derivative of f with respect to y").

E.g. $\frac{\partial f}{\partial y}$ at (1, 2) = 2 × 2 = 4. "At the point one comma two, the partial derivative with respect to y equals four".

**Gradient Vector**: Putting all partial derivatives together into a vector. For our function $f(x, y) = x^2 + y^2$, the gradient is $\nabla f = \langle \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \rangle = \langle 2x, 2y \rangle$.

At point (1, 2): $\nabla f = \langle 2(1), 2(2) \rangle = \langle 2, 4 \rangle$

This vector points in the direction of steepest ascent. The gradient vector points exactly in the direction that maximizes the slope. That slope is the fastest rate of increase.

In neural networks, we want to go in the opposite direction (steepest descent) to minimize the loss function.

**Weight**: input.

**Loss**: output.

**Backward**: going from the final result back to the inputs to figure out how each one contributed.

**Backpropagation**: An algorithm for computing gradients by repeatedly applying Backward with respect to all weights, in order to find directions where the Loss decreases.

Forward + Backward + Update loop = train (update parameter and weight).



## Content: Micrograd

1. Let's begin to build a Micrograd from scratch!  
Basically Micrograd is a tiny neural network library that computes gradients efficiently through a chain of operations to minimize the loss, to train.

2. These are the concepts:

2.1 Functions of numbers: Micrograd treats operations like addition, multiplication, powers as functions it can track

2.2 Gradient vector: How to change each input to increase or decrease the output

2.3 Direction of steepest ascent / descent: Gradient points uphill  
For training, we usually go opposite direction (downhill) to minimize loss

2.4 Backpropagation: A way to compute gradients efficiently through a chain of operations

<br><br><br>

**Let's go!**

Make sure you have Python 3 in your terminal:
```bash
python3 --version
```

Create a workspace (virtual env):
```bash
mkdir micrograd_tutorial
cd micrograd_tutorial
python3 -m venv venv
source venv/bin/activate
pip install jupyter
jupyter notebook
```

It opens a browser tab, File > Create a new notebook:

We are going to implement:

- **Gradient**
- **Neuron: one weighted input + bias + activation**
- **Layer: a list of neurons**
- **MLP: a sequence of layers**

Let's begin with Gradient

**1. First we create an object called "Value"**

```python
class Value:
    def __init__(self, data):
        self.data = data    # the number
        self.grad = 0.0     # gradient
```

For now it holds a number.

```python
x = Value(3.0)
y = Value(4.0)
print(x.data, x.grad)  # 3.0 0.0
```

**2. Track history for backprop**

Every time you perform an operation, you want to remember how it was computed.

```python
class Value:
    def __init__(self, data, _children=(), _op='', name=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)  # parents
        self._op = _op               # operation
        self.name = name
        self._backward = lambda: None  # will define later

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

# for better readability, if not using this, we can see the memory locations
```

<br><br>

```python
# example
a = Value(3.0)
b = Value(4.0)
c = Value(a.data + b.data, (a, b), '+')

print(a.data, a.grad)
print(c.data, c._prev)

print("Parent values:")
for parent in c._prev:
    print(parent.data)


# 3.0 0.0
# 7.0 {Value(data=3.0, grad=0.0), Value(data=4.0, grad=0.0)}
# Parent values:
# 3.0
# 4.0
```

From this we can make graph of 3 things: result, parents, operations

let's make it less manual work, by defining operations:

**addition**

```python
def __add__(self, other):
    # ensure other is a Value so constants become part of the computation graph
    other = other if isinstance(other, Value) else Value(other)
    out = Value(
        self.data + other.data,   # numeric result
        (self, other),            # parents
        '+'                        # operation
    )

    def _backward():
        self.grad += 1.0 * out.grad
        other.grad += 1.0 * out.grad

    out._backward = _backward

    return out
```

**multiplication**

```python
def __mul__(self, other):
    # ensure other is a Value so constants become part of the computation graph
    other = other if isinstance(other, Value) else Value(other)
    out = Value(
        self.data * other.data,   # numeric result
        (self, other),            # parents
        '*'
    )

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward

    return out
```

example:

```python
a = Value(2.0)
b = Value(3.0)
c = a * b + b
```

Internally, this creates:

```python
tmp._prev = {a, b}
c._prev = {tmp, b}
a._prev = {}
b._prev = {}
```

graph:

```
a ──┐
    ├─ (*) → tmp ───┐
b ──┘               ├─ (+) → c
                 b ─┘
```

`tmp = a * b`<br>
`c = tmp + b`

**important**<br>
**_prev** tells backprop where to go<br>
**_backward** tells backprop what math to do

now lets make the backward:

```python
def backward(self):
    topo = []
    visited = set()
    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)
    build(self)         # gathers nodes in topological order
    self.grad = 1.0      # seed
    for v in reversed(topo):
        v._backward()   # call stored backward function
```

same example

```python
a = Value(2.0)
b = Value(3.0)
```

<br>

```python
print(a.data) #2.0
print(b.data) #3.0

print(a._prev) #{}
print(b._prev) #{}
```

**See the graph**
<br>
```python
print("c.data:", c.data)
print("c._prev:", c._prev)

print("tmp.data:", tmp.data)
print("tmp._prev:", tmp._prev)

# c.data: 9.0
# c._prev: {tmp, b}
# tmp.data: 6.0
# tmp._prev: {a, b}
```

When we run `c.backward()`<br>
It begins to run `build(v)`

**Step 1: Topological sort**

1. Mark the current node `v` as visited
2. Recursively call itself on all parent nodes (`v._prev`)
3. Add the node `v` to `topo` **after visiting its parents**

**Effect:** topological sorting of the computation graph ensures parents always appear before children.

**Step by step example:**

* Visit `c`

  * Not visited → mark `c` visited
  * Look at `c._prev = {tmp, b}` → call `build(tmp)` and `build(b)`

* Visit `tmp`

  * Not visited → mark `tmp` visited
  * Look at `tmp._prev = {a, b}` → call `build(a)` and `build(b)`

* Visit `a`

  * Not visited → mark `a` visited
  * `a._prev = {}` → no parents
  * Append `a` to `topo`

* Visit `b` from `tmp`

  * Not visited → mark `b` visited
  * `b._prev = {}` → no parents
  * Append `b` to `topo`

* Back to `tmp`

  * Parents done → append `tmp` to `topo`

* Visit `b` from `c`

  * Already visited → skip

* Back to `c`

  * Parents done → append `c` to `topo`

**Result:**

```
topo = [a, b, tmp, c]
```

> Note: Parents always appear before children, so we reverse the topo in backprop with `reversed(topo)`.

---

**Step 2: Core logic of chain rule**

$$
\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}
$$

Where:

* `L` = final loss / output (`c` in our example)
* `x` = any earlier node or parents (`a`, `b`)
* `y` = intermediate node

<br>

---

**Step 3: Seed gradient**

Start at the final output:

```python
c.grad = 1  # always seed with 1
```

---

**Step 4: Local derivatives by operation**

| Operation                   | Local derivatives     |
| --------------------------- | --------------------- |
| Addition: `c = a + b`       | ∂c/∂a = 1, ∂c/∂b = 1  |
| Multiplication: `c = a * b` | ∂c/∂a = b, ∂c/∂b = a  |
| Division: $c = \frac{a}{b}$ | $\frac{\partial c}{\partial a} = \frac{1}{b}$, $\frac{\partial c}{\partial b} = -\frac{a}{b^2}$ |
| Power: $c = a^b$            | $\frac{\partial c}{\partial a} = b \cdot a^{b-1}$ |
| Tanh: `c = tanh(a)`         | ∂c/∂a = 1 - tanh(a)^2 |

---

**Step 5: Example graph and forward pass**

```
a ──┐
    ├─ (*) → tmp ───┐
b ──┘               ├─ (+) → c
                 b ─┘
```

```
a = 2
b = 3
tmp = a * b = 6
c   = tmp + b = 6 + 3 = 9
```

---

**Step 6: Backprop step by step**

$$
v.grad += \sum_{\text{children } c} \left( c.grad \cdot \frac{\partial c}{\partial v} \right)
$$

1. Backprop from c

* `c.grad = 1` (seed)
* `c` depends on `tmp` and `b`
* Local derivatives:

```
∂c/∂tmp = 1
∂c/∂b   = 1
```

* Contribution to parents:

```
tmp.grad += c.grad * ∂c/∂tmp = 1 * 1 = 1
b.grad   += c.grad * ∂c/∂b   = 1 * 1 = 1
```

2. Backprop through tmp = a * b

* Local derivatives:

```
∂tmp/∂a = b = 3
∂tmp/∂b = a = 2
```

* Contribution to parents:

```
a.grad += tmp.grad * ∂tmp/∂a = 1 * 3 = 3
b.grad += tmp.grad * ∂tmp/∂b = 1 + (1 * 2) = 3
```

---

**Step 7: Final gradients**

```
a.grad   = 3
b.grad   = 3
tmp.grad = 1
c.grad   = 1
```

---

**Step 8: Try to change inputs to see the gradient changes**

in `c = a * b + b`

example 1:

Forward pass:
```
a: 2.0
b: 3.0
c: 9.0
```

Backward pass (gradients):
```
a.grad: 3.0
b.grad: 3.0
c.grad: 1.0
```

example 2:

Forward pass:
```
a: 2.0001
b: 3.01
c: 9.030301
```

Backward pass (gradients):
```
a.grad: 3.01
b.grad: 3.0001
c.grad: 1.0
```

Example 3:

![epoch1-example](https://chefcinnamon.github.io/memo/img/epoch1-example.png)

Next epoch will be about building a tiny neural network.

## Resources

### Micrograd - A Tiny Autograd Engine
[GitHub: karpathy/micrograd](https://github.com/karpathy/micrograd)

A tiny scalar-valued autograd engine and a neural net library on top of it with PyTorch-like API, created by Andrej Karpathy.

### The spelled-out intro to neural networks and backpropagation: building micrograd
<iframe width="560" height="315" src="https://www.youtube.com/embed/VMj-3S1tku0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
