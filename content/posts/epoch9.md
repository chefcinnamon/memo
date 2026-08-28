---
title: "Epoch 9: Deep Learning with fastai Part 1"
date: 2026-08-27
draft: false
aliases:
  - /memo/posts/epoch9/
---

## Intro {.no-counter}

Hi, in this epoch, I am reading *Deep Learning for Coders with fastai and PyTorch*.

![Cover of *Deep Learning for Coders with fastai and PyTorch*](/img/deep-learning-for-coders-with-fastai-and-pytorch.jpg)

I am taking notes as I go and trying to keep them as minimal as possible.

![Overview of the book's introduction](/img/intro-chart.png)

## Content: First steps with fastai {.no-counter}

### Jupyter notebooks

In theory, a neural network with enough neurons can approximate a wide range of functions. In practice, a useful model also needs the right architecture, data, training method, and amount of computation.

Jupyter is a popular platform for experimenting with code. A notebook can contain formatted text, executable code, images, videos, and the output of each cell in one interactive document.

The common cell types are:

- **Markdown cells** for explanations and formatting.
- **Code cells** for executable code and its output.

Google Colab and Kaggle Notebooks are hosted alternatives. In this experiment, Kaggle was useful because its accelerator option provided two NVIDIA T4 GPUs.

### Train our first model: a cat-versus-dog classifier

For this setup, an NVIDIA GPU is convenient because PyTorch and many of the surrounding libraries use CUDA. A GPU can perform many numerical operations in parallel, which makes it useful for neural-network training.

The notebook trains a model to recognize photos of cats and dogs. This is the complete first-training cell:

```python
#id first_training
#caption Results from the first training
# CLICK ME

from fastai.vision.all import *

path = untar_data(URLs.PETS) / "images"

def is_cat(x):
    return x[0].isupper()

dls = ImageDataLoaders.from_name_func(
    path,
    get_image_files(path),
    valid_pct=0.2,
    seed=42,
    label_func=is_cat,
    item_tfms=Resize(224),
)

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```

![Results from the first training](/img/run_1.png)

*Results from the first training.*

This is a **classifier** because the model predicts a category.

At a high level, the model receives an image, transforms it through several layers, and produces an output. During training, it adjusts its weights to reduce the prediction error.

![Neural-network training overview](/img/train.png)

### Test the model with an uploaded image

Jupyter widgets let us upload an image from the notebook. Running the last line displays an upload button:

```python
import ipywidgets as widgets

uploader = widgets.FileUpload()
uploader
```

![Cat image used for testing](/img/epoch9_cat.jpg)

The uploaded image is a cat, and the model produced this sample prediction:

```text
Is this a cat?: True
Probability it's a cat: 1.000000
```

### A few important ideas

The **Universal Approximation Theorem** says:

A neural network with enough neurons can approximate (mimic) any continuous function as closely as you want.

**Stochastic Gradient Descent (SGD)** is a method for training a neural network:

1. Make a prediction.
2. Measure how wrong the prediction is.
3. Adjust the parameters slightly to reduce the error.
4. Repeat this process many times.

We explored gradient descent in the previous epochs, so I recommend reading those posts first if the process is unfamiliar.

A separate issue is a data feedback loop: if a model's predictions influence the data collected for future training, the model can reinforce the same bias over time.

### Load and label the data

The first two lines of the training cell set up the dataset:

- `untar_data(URLs.PETS)` downloads the Oxford-IIIT Pet Dataset if it is not already available, then extracts it.
- `/ "images"` points to the images directory inside the extracted dataset.

`URLs` is a fastai class containing predefined dataset URLs, so `URLs.PETS` means the URL for fastai's Oxford-IIIT Pets dataset.

The path will look something like:

```text
~/.fastai/data/oxford-iiit-pet/images
```

The dataset's filenames use a convention that lets us distinguish cats from dogs. The `is_cat` function turns that convention into a label: if the first character of the filename is uppercase, the image is labeled as a cat; otherwise, it is labeled as a dog.

`ImageDataLoaders.from_name_func` builds data loaders using that labeling function.

- `path` is the base directory containing the images.
- `get_image_files(path)` finds the image files.
- `valid_pct=0.2` reserves 20% of the images for validation.
- `seed=42` makes the random split repeatable.
- `label_func=is_cat` assigns the cat-or-dog label.
- `item_tfms=Resize(224)` resizes each image to 224 × 224 pixels.

The validation set is important because it tests how well the model works on images that were not used to update its weights.

### Classification and regression

- **Classification** predicts a category, such as `cat` or `dog`.
- **Regression** predicts a numerical value, such as a price or temperature.

### Overfitting

**Overfitting** happens when a model memorizes the training examples instead of learning patterns that generalize to new data.

![Example of overfitting](/img/overfitting.png)

If the validation performance starts getting worse while the training performance keeps improving, the model may be overfitting.

### ResNet and metrics

**ResNet-34** is a 34-layer convolutional neural-network architecture used for image recognition. ResNet means *Residual Network*.

1. **Regular training:** forward pass → calculate loss → backpropagation → update weights with SGD/another optimizer.
2. **ResNet:** changes how layers are connected by adding a skip connection:

   `output = F(x) + x`

A **metric** is a function that measures the quality of a model's predictions, usually on the validation set. `error_rate`, for example, measures the proportion of incorrect predictions.

### Pretrained models and transfer learning

A model whose weights were learned from an earlier dataset is called a **pretrained model**.

When `vision_learner` uses a pretrained model, it replaces the final layer, or **head**, because the original head was designed for the original training task.

Using a pretrained model can help us train a more accurate model with less data, time, and computation. Reusing a model for a different task is called **transfer learning**.

**Fine-tuning** is the process of training a pretrained model further on the new task. The later layers are usually adapted more to the new task, while earlier layers can retain general features such as edges, shapes, and textures.

The images below show how the visual representation changes through the layers of a vision model. The illustration is courtesy of Matthew D. Zeiler and Rob Fergus.

![vision learner layer 1](/img/epoch9-visionlearner-1.png)
![vision learner layer 2](/img/epoch9-visionlearner-2.png)
![vision learner layer 3](/img/epoch9-visionlearner-3.png)
![vision learner layer 4 & 5](/img/epoch9-visionlearner-4-5.png)

When we fine-tune a pretrained model for cats versus dogs, we adapt the model's later features in later layers (more detailed) to this specific challenge. The same idea can be used to specialize a pretrained model for many other tasks.

## Image recognizers can tackle non-image tasks {.no-counter}

Other types of data can sometimes be transformed into images. A computer-vision model can then learn patterns in the resulting representation.

This approach can be used for data such as:


The representation needs to preserve useful information.

![Sound represented as spectrograms](/img/epoch9-visionlearner-6.png)
![Time-series events represented as an image](/img/epoch9-visionlearner-7.png)
![Mouse behavior represented as an image](/img/epoch9-visionlearner-8.png)
![Malware represented as an image](/img/epoch9-visionlearner-9.png)

## Image segmentation {.no-counter}

Classification gives one label to an image. **Segmentation** predicts a label for every pixel in an image.

This is useful in applications such as autonomous driving, where the model needs to distinguish roads, cars, people, and other parts of a scene.

```python
from fastai.vision.all import *

path = untar_data(URLs.CAMVID_TINY)
dls = SegmentationDataLoaders.from_label_func(
    path,
    bs=8,
    fnames=get_image_files(path / "images"),
    label_func=lambda o: path / "labels" / f"{o.stem}_P{o.suffix}",
    codes=np.loadtxt(path / "codes.txt", dtype=str),
)

learn = unet_learner(dls, resnet34)
learn.fine_tune(20)
```

The CamVid dataset includes both the original images and pre-labeled segmentation masks. The `label_func` connects each image to its matching mask:

```text
image: .../images/0016E5_07959.png
mask:  .../labels/0016E5_07959_P.png
```

Each pixel in a mask contains a numeric class ID. The `codes` array maps those IDs to class names.

```python
codes = np.loadtxt(path / "codes.txt", dtype=str)
```

For example, `codes[class_id]` returns the class name associated with that pixel value.

We can inspect a mask as a two-dimensional array:

```python
mask_path = path / "labels" / "0016E5_07959_P.png"
mask = PILMask.create(mask_path)
print(np.array(mask))
```

The output contains class IDs, for example:

```text
[[21 21 21 ...]
 [21 21 21 ...]
 [17 17 17 ...]]
```

`bs=8` sets the batch size to eight images at a time.

U-Net is a neural-network architecture commonly used for image segmentation. It has two main parts:

- **Encoder**: reads the image and detects patterns such as edges, shapes, textures, and object parts.
- **Decoder**: uses those features to predict a class for every pixel.

In this example, `resnet34` is the encoder and U-Net provides the decoder:

```python
learn = unet_learner(dls, resnet34)
```

To inspect a few predictions:

```python
learn.show_results(max_n=6, figsize=(7, 8))
```

- `max_n=6` shows up to six examples.
- `figsize=(7, 8)` sets the figure size to 7 × 8 inches.


![Segmentation targets and predictions](/img/epoch9-segmentation.png)

Example training output:

| epoch | train_loss | valid_loss | time |
| ----- | ---------- | ---------- | ---- |
| 0     | 1.658884   | 1.644030   | 00:01 |
| 1     | 1.479218   | 1.343398   | 00:01 |
| 2     | 1.348353   | 1.284355   | 00:01 |
| 3     | 1.241375   | 1.121249   | 00:01 |
| 4     | 1.159236   | 1.188174   | 00:01 |
| 5     | 1.095831   | 1.034621   | 00:01 |
| 6     | 1.016362   | 0.873740   | 00:01 |
| 7     | 0.932146   | 0.796478   | 00:01 |
| 8     | 0.856378   | 0.841596   | 00:01 |
| 9     | 0.795792   | 0.775887   | 00:01 |
| 10    | 0.739415   | 0.744660   | 00:01 |
| 11    | 0.688554   | 0.730134   | 00:01 |
| 12    | 0.643139   | 0.728994   | 00:01 |
| 13    | 0.603084   | 0.738516   | 00:01 |
| 14    | 0.569019   | 0.709449   | 00:01 |
| 15    | 0.535957   | 0.691849   | 00:01 |
| 16    | 0.506096   | 0.701629   | 00:01 |
| 17    | 0.479810   | 0.709441   | 00:01 |
| 18    | 0.457027   | 0.705072   | 00:01 |
| 19    | 0.438452   | 0.704024   | 00:01 |
