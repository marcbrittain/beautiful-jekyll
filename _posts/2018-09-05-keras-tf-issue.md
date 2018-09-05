---
layout: post
title: Keras+Tensorflow GPU Training
tags: [Keras, TensorFlow, python, Ubuntu]
permalink: /2018/09/05/keras-tf-issue/
---

I was attempting to train a neural network model on a single GPU in a multi-GPU Linux machine. I followed the advice that others had posted online to add the following lines at the beginning of my python script:

```python
import os
os.environ['CUDA_VISIBLE_DEVICES']=0  #(or whichever GPU you want to use)
```

BUT, this did not help the problem. Tensorflow was still finding the other GPU and preallocating the entire memory of all GPUs. I work in virtual environments, but I don't think this should be the problem here. I was able to solve the problem by removing the code above from my script and running the following code in the terminal after activating my virtual environment, but before running the python script:

```python
export CUDA_VISIBLE_DEVICES=0
```
After adding this, I was able to run the code properly on whichever GPU specified.


