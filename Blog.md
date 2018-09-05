---
layout: page
title: Blog
---


----
The purpose of this blog is to share and document errors I run into with descriptions on how I was able to fix them. In addition, I will also share things that I find interesting :)


#### Keras+Tensorflow GPU Training (09/05/2018)
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

