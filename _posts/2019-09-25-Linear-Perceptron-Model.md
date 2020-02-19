---
title: "Linear Perceptron Model"
date: 2019-09-25
header:
  image: "/images/anhui.jpg"
excerpt: "This is an assignment I worked on in my Machine Learning class."
mathjax: "true"
---

The purpose of question 1 of this homework assignment was to run a linear perceptron algorithm 
on three training samples x1, x2, and x3. In this scenario, samples x1 belongs to class 1 while 
samples x2 and x3 belong to class 2. In part (i), I simply plotted the samples, the starting weight 
line and its direction. I also defined two functions, one for plotting these three things and another 
for computing dot product for part (ii). In part (ii), I first wrote the data points and weight as vectors 
and then computed the inner product between the weight and each class to see which samples were misclassified. 
In this case, it was sample x2, so in part (iii) I used the weight update rule to calculate a new weight based 
on sample x2 and graphed this new weight line and its direction. After computing the inner products again, 
now x1 was the only misclassified sample. I used the weight update rule one more time, this time basing it on 
sample x1 to get a new weight and I plotted this line. Finally after calculating these inner products, I could 
verify that all of the samples were correctly classified with this new weight line. 

## Question 1 (5 points)
### Linear Perceptron Algorithm:
The goal of this problem is to run a linear perceptron algorithm. Assume that you have three training samples in the 2D space:

1. Sample $x_{1}$ with coordinates (1, 3) belonging to Class 1 ($y_{1}$ = 1)
2. Sample $x_{2}$ with coordinates (3, 2) belonging to Class 2 ($y_{2}$ = −1)
3. Sample $x_{3}$ with coordinates (4, 1) belonging to Class 2 ($y_{3}$ = −1)

The linear perceptron is initialized with a line with corresponding weight w(0) = $[2, -1, 1]^{T}$, or
else the line 2 − x + y = 0.
In contrast to the example that we have done in class, in this problem **we will include the
intercept $w_{0}$ or $x_{0}$.**

**(i) (0.5 points)** Plot $x_{1}$, $x_{2}$, and $x_{3}$ in the given 2D space. Plot the line corresponding to weight w(0), as well as the direction of the weight w(0) on the line.


```python
import numpy as np
from pandas import *
from math import *

%matplotlib inline
import matplotlib.pyplot as plt
from operator import itemgetter
```


```python
# function for plotting the sample points, weight line and the direction of the weight
def plot_graph(x, y, sample_names, colors, x0, y0, u, v, label_name):
    # plot points
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=colors)
    for i, txt in enumerate(sample_names):
        ax.annotate(txt, (x[i] * (1.03), y[i]* (1.03)))
    
    # plot weight line
    plt.plot(x0, y0, '-k', label=label_name)
    plt.legend()
    
    # plot weight direction
    plt.quiver(x0[25], y0[25], u, v, scale=5)
    ax.set_aspect(1./ax.get_data_ratio())
    
```


```python
# function for computing dot product and printing results
def inner_product(line, samples, line_name):
    for i in np.arange(3):
        print("The inner product of " + line_name + " and x" + str(i+1) + ":", np.dot(line, samples[i]))
```


```python
# data for samples x1, x2, x3 
x = [1, 3, 4]
y = [3, 2, 1]

sample_names = ['x1', 'x2', 'x3']
colors = ['red' if i == 1 else 'blue' for i in x]
```


```python
# data for line corresponding to weight w(0)
x0 = np.linspace(0,5)
y0 = x0-1

# plot data points and weight line w(0) with direction
plot_graph(x, y, sample_names, colors, x0, y0, -1, 1, 'w(0)')
```


<img src="{{ site.url }}{{ site.baseurl }}/images/p2/output_7_0.png" alt="">


**(ii) (0.5 points)** Using the rule $sign(w(t)^{T}x_{n}$), please indicate the class in which samples $x_{1}$, $x_{2}$, and $x_{3}$ are classified using the weight w(0). Which samples are not correctly classified based on this rule?


**Note:** You have to compute the inner product $w(0)^{T}x_{n}$, n = 1, 2, 3, and see if it is greater or less than 0.


```python
# data points as vectors
x1 = np.array([1, 1, 3])
x2 = np.array([1, 3, 2])
x3 = np.array([1, 4, 1])
samples = [x1, x2, x3]

# weight line as vector
w0 = np.array([2, -1, 1])

# inner product of weight line w(0) and data samples
inner_product(w0, samples, 'w0')
```

    The inner product of w0 and x1: 4
    The inner product of w0 and x2: 1
    The inner product of w0 and x3: -1
    

*The inner product of w(0) and x1 should be greater than zero because x1 is part of class 1 and the inner products of w(0) and x2 as well as w(0) and x3 should both be negative because they're in class 2. Going off of this, x1 and x3 are correctly classified while x2 is not correctly classified.*

**(iii) (1.5 points)** Using the weight update rule from the linear perceptron algorithm, please
find the value of the new weight w(1) based on the misclassified sample from question (ii). Find
and plot the new line corresponding to weight w(1) in the 2D space, as well as the direction of
the weight w(0) on the line. Indicate which samples are correctly classified and which samples
are not correctly classified.


**Note:** The update rule is $w(t + 1) = w(t) + y_{s}x_{s}$, where $x_{s}$ and $y_{s} ∈ {−1, 1}$ is the feature
and class label of misclassified sample s.

**Hint:** The line corresponding to a vector $w = [w_{0}, w_{1}, w_{2}]$ can be written as $w_{0}+w_{1}x+w_{2}y = 0$.
Make sure that you get the direction of the vector w correctly based on the sign of $w_{1}$ and $w_{2}$.



```python
# w(1) = w(0) + ys * xs
# where:
# w(0) = [2, -1, 1]
# ys = -1
# xs = x2

# weight update rule to get w(1)
w1 = np.array(w0 - x2)
print("New weight w(1) =", w1)
```

    New weight w(1) = [ 1 -4 -1]
    


```python
# data for line corresponding to weight w(1)
x0 = np.linspace(-6,6)
y0 = 1-4*x0

# plot data points and weight line w(1) with direction
plot_graph(x, y, sample_names, colors, x0, y0, -1, -1, 'w(1)')
```


<img src="{{ site.url }}{{ site.baseurl }}/images/p2/output_13_0.png" alt="">


```python
# inner product of weight line w(1) and data samples
inner_product(w1, samples, 'w1')
```

    The inner product of w1 and x1: -6
    The inner product of w1 and x2: -13
    The inner product of w1 and x3: -16
    

*The inner product of w(1) and x1 should be greater than zero because x1 is part of class 1 and the inner products of w(1) and x2 as well as w(1) and x3 should both be negative because they're in class 2. Going off of this, x2 and x3 are correctly classified while x1 is not correctly classified.*

**(iv) (2.5 points)** Using the rule $sign(w(t)^{T}x_{n})$, run the linear perceptron algorithm until it converges, find and plot the weights w(2) and the corresponding lines in each iteration. For each iteration, please indicate which samples are classified correctly and which samples are not classified correctly.

**Hint:** In order to make the linear perceptron algorithm converge as fast as possible, you can
always update the weight based on sample $x_{1}$. Why?



```python
# weight update rule to get w(2)
w2 = np.array(w1 + x1)
print("New weight w(2) =", w2)
```

    New weight w(2) = [ 2 -3  2]
    


```python
# data for line corresponding to weight w(2)
x0 = np.linspace(0,5)
y0 = -1+(3/2)*x0

# plot data points and weight line w(2) with direction
plot_graph(x, y, sample_names, colors, x0, y0, -1, 1, 'w(2)')
```


<img src="{{ site.url }}{{ site.baseurl }}/images/p2/output_18_0.png" alt="">


```python
# inner product of weight line w(2) and data samples
inner_product(w2, samples, 'w2')
```

    The inner product of w2 and x1: 5
    The inner product of w2 and x2: -3
    The inner product of w2 and x3: -8
    

*The inner product of w(1) and x1 should be greater than zero because x1 is part of class 1 and the inner products of w(1) and x2 as well as w(1) and x3 should both be negative because they're in class 2. Going off of this, all of the samples are correctly classified.*
