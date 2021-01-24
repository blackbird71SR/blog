---
title: WhatWhyHow Part 3 - Linear Regression
excerpt: This is not just another ML tutorial! Learn What, Why and How of simplest regression algorithm!
categories:
  - Data Science
tags:
  - [ML, Regression]
published: true
comments: true
classes: wide
header:
  teaser: /assets/img/5.jpg
  image: /assets/img/5.jpg
---
<small>*<span>Photo by <a href="https://unsplash.com/@meric"> Meriç Dağlı</a> on <a href="https://unsplash.com/photos/rvegUP-pNYU">Unsplash</a></span>*</small>

In the last two articles, we talked about some commonly used classification algorithms like [Logistic Regression](https://blackbird71sr.github.io/blog/data%20science/WhatWhyHow1-Logistic-Regression/) and [Naive Bayes](https://blackbird71sr.github.io/blog/data%20science/WhatWhyHow2-NaiveBayes/). While classification algorithms work when target variable is discrete, regression algorithms work for continuous target variable. While there are tons of regression algorithms, let's start with most simple of them: **Simple Linear Regression**.

## What?

### Situation

![Situation](https://media.giphy.com/media/5xtDarqCp0eomZaFJW8/giphy.gif)

Let's consider a problem to work with. We have a dataset of houses in the Mumbai area. For each house, we have Living Area (in feet<sup>2</sup>), No. of bedrooms, No. of bathrooms and the price. We want to create a model that will predict a price of house given first 3 parameters.

**[Here](https://www.kaggle.com/c/home-data-for-ml-course/data)** is one such dataset on Kaggle with around 80 features, but we will consider a simple example of just 3 features.

### Assumptions

![Assumptions](https://media.giphy.com/media/iGA00NM1EtaJTTiCmc/giphy.gif)

Before moving forward, let's see some assumptions made by the linear regression:

- The independent variable(y) is linearly dependent on each feature(X<sub>i</sub>). In our case this means, price of house changes linearly acc. to each feature like no. of bedrooms, bathrooms and even area.

Now remember, this is not the case really! We know that there are many different ways `y` can be dependent on `x` like exponential, polynomial or even logarithmic. But as this is **Simple linear** regression, we assume that everything in linear.

- Linear regression also requires that all variables have normal distribution. We can check this with histograms. If some variable is not normally distributes, we can use something like log transformation to fix that variable.
  
- Linear regression also assumes that there is no co-relation between the independent variables. This can be tested with the corelation matrix or Variance Inflation Factor (VIF). 

Foe our current example, we will assume that our data matches all this assumptions, but in real-life when dealing with data, remember to check all this before moving forward. You can learn more about them **[here](https://www.statisticssolutions.com/assumptions-of-linear-regression/)**

## Why Then?

If there are so may assumptions, why to use linear regression at all? 

The reason is same as I gave in the naive bayes classification model. This provides a quick and easy baseline before trying for all your complex models. Let's say you try a deep learning model after this and accuracy is even lower than a linear regression, you know that you have missed something really big!

## How?

### Define a Problem

Now we have have features(X) using which we want predict price of house(y). Let's define a equation of line (as everything is linear), to define the problem

$$
y' = h(x) = w_0 + w_1x_1 + w_2x_2 + w_3x_3
$$

Here,

$y'$ - Predicted value of house price

$x_1, x_2, x_3$ - Features like no. of bedrooms, bathrooms and area

$w_0, w_1, w_2, w_3$ - Weights or parameters associated with the features

To simplify this equation, let's assume $x_0 = 1$. This is nothing but the intercept term in the line equation. Now we can write the equation as

$$
y' = h(x) = w_0x_0 + w_1x_1 + w_2x_2 + w_3x_3
$$

$$
y' = h(x) = \sum_{i=0}^{n} w_ix_i
$$

That's it! We can predict the price of new house if we have all the features(X) needed to calculate above term, but we need one more thing. We need parameters(all the $w$'s). 

### Getting Weights

We will initialize all the weights randomly and then try to optimize them so that they represent the correct structure of our equation of line to predict the price of house.

#### What to optimize?

For this, we need to define a error function which we will calculate every time our equation predicts the house price. This is also known as **cost function**. 

For us, this will be least squares. That is

$$ 
J(w) = \frac{1}{2}\sum_{i=1}^{m} (h(x^{i})-y^{i})^{2}
$$

Here,

$m$ - No.of training examples

$h(x^{i})$ - Prediction for i<sup>th</sup> example using our equation

$y^{i}$  - Actual price of house for ith example

In short, 

- we will calculate the price of house for a single example.
- Then we will take difference between predicted value and actual value. 
- Square that difference. 
- Keep adding this term for all examples. 
- And at last multiply with 0.5 (which is just for ease of differentiation). 

You got the error. We are going to use this to make the weights perfect that is optimize.

#### How to optimize?

We will use concept of **Gradient Descent** for the optimization. This is very simple iterative approach and is used even in almost every deep learning algorithms. Let' see how to do this:

- Randomly initialize the weights $w$
- Until the weights are converged(that is you are satisfied this can't get any better), Update every $w_j$ in $w$ using:
  $$
    w_j := w_j - \alpha \frac{\partial}{\partial w_j} J(w)
  $$

Wait a second...what is this? Everything was crystal clear till now!

![Clear](https://media.giphy.com/media/10QSPD57AKjCXm/giphy.gif)

No, it's still simple! 

Here,

$\alpha$ - Learning rate which determines how fast or slow we want to optimize. It's a constant which we decide. For now, let say it is 0.001.

$\frac{\partial}{\partial w_j}$ -  This is partial derivative of cost/ error function $J(w)$ with respect to $w_j$.

We need to calculate this. Our cost function includes summation over all training examples. Let's kep it aside for a second and try to calculate for just a single example.

For one single training examples:

$$
J(w) = \frac{1}{2} (h(x)-y)^{2}
$$

$$
\frac{\partial}{\partial w_j} J(w) = \frac{\partial}{\partial w_j} \frac{1}{2} (h(x)-y)^{2}
$$

$$
\qquad\qquad = 2.\frac{1}{2} (h(x)-y) \frac{\partial}{\partial w_j}(h(x)-y)
$$

$$
\qquad\qquad = (h(x)-y) \frac{\partial}{\partial w_j}(\sum_{j=0}^{n} w_jx_j-y)
$$

$$
\qquad = (h(x)-y) x_j
$$

Replacing this in our gradient descent equation, for a single training example

$$
    w_j := w_j - \alpha (h(x)-y) x_j
$$

That is for i<sup>th</sup> training example,

$$
    w_j := w_j - \alpha (h(x^{i})-y^{i}) x_j
$$

Remember , this is just for one single example. For all examples, we can put use the summation over all the training examples to get our final optimization rule:

$$
    w_j := w_j - \alpha \sum_{i=1}^{m} (h(x)-y) x_j \space \space (for \space every \space j)
$$

Here,

$m$ - Total training examples in our dataset.

## The Big Picture

![Big Picture](https://media.giphy.com/media/l2YWuU2n6lYHtLHEI/giphy.gif)

We are done with all pieces of puzzle. Let's review them:

- We have dataset with some features $(x_1, x_2, ...)$ and target variable $y$ which we want to predict. This variable is continuous and not discrete.
- We initiate no. of parameters($w$'s) based on no. of features randomly.
- We calculate the prediction for each example.

$$
y' = h(x) = w_0 + w_1x_1 + w_2x_2 + w_3x_3
$$

- Compare $y'$ with actual value $y$ and calculate the total error or cost value using

$$ 
J(w) = \frac{1}{2}\sum_{i=1}^{m} (h(x^{i})-y^{i})^{2}
$$

- Of course, this will be too large as all the parameters are random values. Now to change parameters using gradient descent do this:

$$
    w_j := w_j - \alpha \sum_{i=1}^{m} (h(x)-y) x_j \space \space (for \space every \space j)
$$

- Now the parameters are changed, calculate the prediction again and compare with actual values. Keep doing this until your predictions are closer to actual values.
- You now have optimal values of parameters. Say after 30 iterations of this, you get 

$$
w_o = 0.34, \space w_1 = 23 \space w_2 = 1.45 \space w_3 = 3.21
$$

So the equation of line(linear regression) is:

$$
y' = h(x) = 0.34 + 23x_1 + 1.45x_2 + 3.21x_2
$$

where,

$x_1$ - Area of house

$x_2$ - No. of bedrooms

$x_3$ - No. of bathrooms


That's it, if you have these 3 values for any house, you can predict the price of house using the above equation.

![Done](https://media.giphy.com/media/YnBntKOgnUSBkV7bQH/giphy.gif)

## Some Important Points

 - Gradient descent is the backbone of this algorithm and actually most of the you will encounter. Read **[this](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)** and **[this](https://towardsdatascience.com/understanding-the-mathematics-behind-gradient-descent-dde5dc9be06e)** before moving forward. You will get much deeper understanding of what's going on here! You will also get idea about learning rate $\alpha$.

![Read Now!](https://media.giphy.com/media/L3K5LhWkGBGqTVNxZM/giphy.gif)

- While coding, you don't have to do this all steps manually, we it is always better to know all the maths behind it. To get in-depth understanding of both, read the following two articles:

- **[Linear Regression in Python](https://realpython.com/linear-regression-in-python/)**
- **[In Depth: Linear Regression](https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html)**

---
That's all for today. Let me know in comments, your suggestions, doubts and what do you think. 

See you next week!  

![](https://media.giphy.com/media/Q79Xp6bkWLSmvAuPUa/giphy.gif)