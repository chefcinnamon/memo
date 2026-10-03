---
title: "Epoch 9: Deep Learning with fastai Part 2"
date: 2026-08-29
draft: true
aliases:
  - /memo/posts/epoch10/
---

## Intro {.no-counter}

Hi, in this epoch, I am continuing to read *Deep Learning for Coders with fastai and PyTorch*.

This is the beginning of Part 2. I will add notes, code, and examples as I work through the next section.

## Classifier example: review sentiment {.no-counter}

There is a model using the IMDb Large Movie Review dataset from “Learning Word Vectors for Sentiment Analysis” by Andrew Maas et al.

It can predict a review has positive or negative sentiment.

```python
from fastai.text.all import *

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test', bs=32)
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)
```
Let's decompose it in smaller parts and understand how it works.

**1. import fastai library**
```
from fastai.text.all import *
```
**2. create DataLoaders object**
```
dls = TextDataLoaders.from_folder(
    untar_data(URLs.IMDB),
    valid='test'
)
```
`TextDataLoaders` is a class(class= a way of storing data) and <br> `from_folder` is a method of that class (method = a function inside class)

`untar_data(...)` a function to download imdb dataset url , extract and return the path like /root/.fastai/data/imdb

`URLs` is a class (class = a way of storing data), and <br> `IMDB` is a *class attribute*, a data that is stored inside a class<br>
`example:` 
```
class URLs:
    IMDB = "https://example.com/imdb.tgz"
    MNIST = "https://example.com/mnist.tgz"
```

`valid='test'`

This tells fastai:

Use the test folder as the validation dataset.

As a result `TextDataLoaders.from_folder(...)`

1. Looks through those folders and creates batches of text + labels.

*For example:*
```
"This movie was amazing..." → positive

"This movie was terrible..." → negative
```

`dls` now looks like:

```
dls = TextDataLoaders(
    train = <training DataLoader>,
    valid = <validation DataLoader>
)
```
So inside it, you can think of:
```
dls
├── train
│   ├── batch 1
│   ├── batch 2
│   ├── batch 3
│   └── ...
│
└── valid
    ├── batch 1
    ├── batch 2
    ├── batch 3
    └── ...
```
Each batch contains things like:

>text → "This movie was amazing..."<br>
>label → positive`

or numerically something like:

> text tokens → [....] <br> label       → 1

<br>

`AWD_LSTM` is a type of LSTM language model designed for text.

Very roughly, it reads a sentence sequentially:

I → really → liked → this → movie

and builds an internal representation of what the sentence means, like:
```
"I"
internal state:
[0.1, 0.0, -0.2, ...]

"I really"
internal state:
[0.2, 0.4, -0.1, ...]

"I really liked"
internal state:
[0.8, 0.7, 0.2, ...]

```
Then use ***logits*** to convert them to **scores** <br> <br>
 Afterwards use ***softmax***  for the normalization with probability-distribution to convert them to **likelihood**

 I have explained logits and softmax in previous blog posts, Highly recommended to study them first.
 

 ----
 `drop_mult=0.5`

This controls dropout.

**Dropout** is a technique used to reduce overfitting.

The model randomly disables some parts of the neural network during training.

----
`metrics=accuracy`

This tells fastai:

While evaluating the model, show me its classification accuracy.

---
`learn.fine_tune(4, 1e-2)` 
<br>
means fine tune 4 times with 1e-2(0.01) learning rate(size of weights adjustment)

---


### overall

```
Download IMDB reviews
        ↓
Prepare training/validation batches
        ↓
Create an AWD-LSTM text classifier
        ↓
Start from pretrained language knowledge
        ↓
Train it on positive/negative IMDB reviews
        ↓
Measure accuracy
```

--- 
![training result](/img/epoch10-sentiment.png)

This shows this review is recognized as positive. 
