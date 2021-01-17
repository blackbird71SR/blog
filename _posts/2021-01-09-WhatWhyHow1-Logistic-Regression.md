---
title: WhatWhyHow Part 1 - Classify with Logistic Regression
excerpt: This is not just another ML tutorial! Learn What, Why and How of most used machine learning classification algorithm!
categories:
  - Data Science
tags:
  - [ML, Classification]
published: true
comments: true
classes: wide
header:
  teaser: /assets/img/3.jpg
  image: /assets/img/3.jpg
---
<small>*<span>Photo by <a href="https://unsplash.com/@paucasals">Pau Casals</a> on <a href="https://unsplash.com/photos/1Gvog1VdtDA">Unsplash</a></span>*</small>

The two most widely used terms in machine learning are the **Classification** and **Regression**. While both are techniques to predict a dependent variable using one or more independent variables, regression works for continuous dependent variable and classification is for discrete/ categorical dependent variable.

But important point to remember here is that even if **Logistic Regression** has word *regression* in it, it is a classification algorithm. It is used when dependent variable is categorical.

![What?](https://media.giphy.com/media/91fEJqgdsnu4E/giphy.gif)

So if it is a classification algorithm, why it has  a 'regression' in its name? This is because the technique used in logistic regression is similar to linear regression. The term 'Logistic' is taken from the *logit* function that we will see later in the article.

Logistic regression can be used for for various classification tasks such as:
- To predict whether email is spam(1) or not(0)
- To check whether tweet has positive(1) or negative(0) sentiment

## Situation

So consider a situation where you have to predict whether this email is spam or not. You have some features for each email such as sender email address, time of email which we call as X. The variable we want to predict that is spam or not is Y.

## Why Logistic Regression?

How can we try to solve the above situation using say just linear regression? Our model will learn from features X and will try to predict a number for Y, but we need it to be just 0 or 1. One solution is to use *threshold*. 

Say the threshold is 0.5, so for all the values predicted by model below 0.5, we will consider them as not spam and for the values above 0.5 we will consider spam. The question arises, why 0.5? It can be 0.4 or even 10, because linear regression is an unbounded technique. It can predict any real number as its prediction and that is clearly not suitable in our case.
 
## Overview of Logistic Regression

The general idea behind any machine learning model remain the same. We pass features X and parameters $\theta$ to our model, which predicts the output $\hat{Y}$. Then we check the difference between output $\hat{Y}$ and actual labels Y, which is called the cost and change parameters $\theta$ accordingly. This process continues till we think that that our predictions $\hat{Y}$ are sufficiently similar to Y, that is our model is sufficiently accurate.

So now to understand the logistic regression, we need to concentrate on things that are different, namely
- Function predicting $\hat{Y}$
- Cost function identifying difference between $\hat{Y}$ and Y
- How to update the parameters $\theta$

## Function predicting $\hat{Y}$

If you have studied the simple linear regression, in which say we have N feature/independent variable X and dependent variable Y, the function to predict is

$$
z = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + ... \theta_N x_N
$$

where $\theta$ = [$\theta$<sub>0</sub>, $\theta$<sub>1</sub>, ...$\theta$<sub>N</sub>] are set of parameters. We can also call them as *weights*. 

Logistic regression takes this regular linear regression and applies a **sigmoid function** to the output of the linear regression.

---
Now what is sigmoid function?
$$ h(z) = \frac{1}{1+\exp^{-z}} $$

This is a simple function which you see again and again in both machine and deep learning. This function takes the input `z` and output the value between 0 and 1. This is how it's plot looks like:
>![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1200px-Logistic-curve.svg.png)
<small>*Credits: [Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)* </small>

As you can see, 
- For all the values of input z from -$\infty$ to 0, sigmoid function will output the value less than 0.5. 
- For z = 0, the value is 0.5  
- For all the values of input z above 0 until $\infty$, the value will be above 0.5.

![Big Deal!](https://media.giphy.com/media/9LFBOD8a1Ip2M/giphy.gif)

Do you remember why we can't use linear regression in our classification situation? Yeah, beacuse it is unbounded. It can predict anything in the range of -$\infty$ to $\infty$. This is not the case for sigmoid function. Whatever you throw at it, it will always output a value between 0 and 1.

This is very important, like really, really important! 

What else always ranges from 0 to 1? ...Right, probability! As sigmoid output also ranges from 0 to 1, we can also treat it as a probability.

---

So, now for our 

- input: X = [X<sub>0</sub>, X<sub>1</sub>, ..., X<sub>N</sub>]

- parameters:  $\theta$ = [$\theta$<sub>0</sub>, $\theta$<sub>1</sub>, ..., $\theta$<sub>N</sub>]

First, we calculate 

$$
z = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + ... \theta_N x_N
$$

and then we apply sigmoid function to the output of linear regression

$$ h(z) = \frac{1}{1+\exp^{-z}}$$

This $z$ is known as **logits**.

Now $h(z)$ is a value between 0 and 1 and as we discussed can be considered as a probability. If this,
- $probability >= 0.5$ means $prediction = 1$ means **Spam**
- $probability < 0.5$ means $prediction = 0$ means **Not Spam**

This solves our first question of prediction. We can now easily complete first step to predict the $\hat{Y}$. Now let's move on to next question.

## Cost function identifying difference between $\hat{Y}$ and Y

First and foremost, what is a cost function? 
> Cost function is a function that measures the performance of a Machine Learning model for given data. It quantifies error between label Y and prediction $\hat{Y}$ in the form of single number. We try to minimize this cost also known as loss.

For linear regression, the concept of loss is very simple. We mostly use one of the **Mean Absolute Error** which is just average of absolute difference between target and prediction

>![MAE](https://miro.medium.com/proxy/0*zX9jlpZ8k0CuEpFE.jpg)
<small>*Credits: [Towards Data Science](https://towardsdatascience.com/regression-an-explanation-of-regression-metrics-and-what-can-go-wrong-a39a9793d914)*</small>

or **Mean Squared Error** which is average of squared difference between target and prediction.

>![MSE](https://miro.medium.com/max/875/0*aTUPK_ILg7-n0znw.jpg)
<small>*Credits: [Towards Data Science](https://towardsdatascience.com/regression-an-explanation-of-regression-metrics-and-what-can-go-wrong-a39a9793d914)*.</small>

We can't use this in our case of logistic regression because we use gradient-based techniques(explained later in this article) to find our optimal values of parameters that is $\theta$. This means that our loss function is not convex and can have multiple minimas. Using MAE or MSE can lead our model to stuck in some local minima and can never reach global minima.
Please read **[this](https://towardsdatascience.com/why-not-mse-as-a-loss-function-for-logistic-regression-589816b5e03c)** great article why we can't use MAE or MSE as loss function in logistic regression.

So what we can use?

We use the **average of log loss** as our loss function.

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^m y^{(i)}\log (h(z(\theta)^{(i)})) + (1-y^{(i)})\log (1-h(z(\theta)^{(i)})) $$

* $m$ =  Number of training examples
* $y^{(i)}$ = Actual label of the i-th training example.
* $h(z(\theta)^{(i)})$ = Model prediction using function we saw earlier for the i-th training example.

Scary, right?

Really, not at all! Let's break it down...

- First, there is $-\frac{1}{m}$ outside the $\sum$ sign. This just means that we are we are summing something for all the training examples and dividing it by no. of training examples. Ohh! It's just a average as we do in MAE or MSE. Nothing to worry about! We will see about that negative sign later!
- Now what we are averaging? This is what inside term looks like for a single training example:

$$ -1 \times \left( y^{(i)}\log (h(z(\theta)^{(i)})) + (1-y^{(i)})\log (1-h(z(\theta)^{(i)})) \right)$$

- Now in this, $y^{(i)}$ is actual label for the training example. It can be only 0 or 1, right! So let's break it down in both cases:

  - For $y^{(i)}$ = 1 
  
  $Loss = - \log (h(z(\theta)^{(i)})$
  
  - For $y^{(i)}$ = 0
  
  $Loss = - \log (1 - h(z(\theta)^{(i)})$ 

- First of look, we have taken that -ve sign inside for each term. As we know, that output of $h(z(\theta)^{(i)})$ is always between 0 and 1, log of that will always be negative. That is why we multiply by -1.
- When our model predicts 1, that is ($h(z(\theta)) = 1$) and actual label is also 1, see loss is 0.
- Similarly, when model predicts 0 that is ($h(z(\theta)) = 0$) and actual label is also 0, loss is 0. This is behavior we expect.
- But when prediction is close to 1 like $h(z(\theta)) = 0.99$) and actual label is 0,

$$ 
Loss = - \log (1 - h(z(\theta)^{(i)}) 
$$

$$
Loss = - \log (1 - 0.99)
$$

$$
Loss = - (-2)
$$

$$
Loss = 2
$$

- Similarly when prediction is close to 0 like $h(z(\theta)) = 0.01$) and actual label is 1,

$$ 
Loss = - \log (h(z(\theta)^{(i)}) 
$$

$$
Loss = - \log (0.01)
$$

$$
Loss = - (-2)
$$

$$
Loss = 2
$$

>![Loss](https://miro.medium.com/max/625/0*JRbkNpnepqQCtL7X)
<small>*Credits: [Towards Data Science](https://becominghuman.ai/machine-learning-series-day-2-logistic-regression-144af00f6ff5)*.</small>

That means loss increases as prediction gets closer to 1 and actual label is 0 or prediction gets closer to 0 and actual label is 1. This is exactly the work of loss function. Loss should increase if we are predicting wrong and should be less if are predicting correctly. 

This solves our second piece of puzzle. Now we can predict using logistic regression and then also calculate the loss to see how accurate our predictions are. Now of-course, as we have initialized our parameters $\theta$, to random numbers, our predictions are not anywhere near what we want. We want to update $\theta$ using our loss. How we can do that?

## How to update the parameters $\theta$

We need to minimize our cost value. To do this, we need to change parameters $\theta$. For this, we will use something called **Gradient Descent**.  

Now, what is gradient descent?

> According to [Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent), Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. The idea is to take repeated steps in the opposite direction of the gradient (or approximate gradient) of the function at the current point, because this is the direction of steepest descent.

You can learn more about gradient decent [Here](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html). Read it before moving forward.

To update the weights/ parameters $\theta$, we will use this gradient descent concept:

$$\theta_j = \theta_j - \alpha \times \nabla_{\theta_j}J(\theta) $$

Here,
* $\alpha$ = Learning rate. It determines how fast or slow we need to update the weights. 
* $\theta_j$ = jth index in $\theta$. This is associated with jth index in $x$ that is $x_j$.
* $\nabla_{\theta_j}J(\theta)$ = Gradient of cost function $J$ with respect to $\theta_j$

Remember, 

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^m y^{(i)}\log (h(z(\theta)^{(i)})) + (1-y^{(i)})\log (1-h(z(\theta)^{(i)}) $$

Differentiating this with respect to $\theta_j$, gives us

$$\nabla_{\theta_j}J(\theta) = \frac{1}{m} \sum_{i=1}^m(h^{(i)}-y^{(i)})x_j$$

where
* $m$ = No. of training examples
* $i$ = Current index of training example
* $j$ = Index of weight $\theta$ that is $\theta_j$

So, substituting this $\nabla_{\theta_j}J(\theta)$ in the update equation, we get final equation to update the weights:

$$\theta_j = \theta_j - \alpha \times \frac{1}{m} \sum_{i=1}^m(h^{(i)}-y^{(i)})x_j $$

Great! This solves the third question of updating the weight. Now, we have solved our all 3 questions, let's look at complete picture!

![Let's Finish](https://media1.tenor.com/images/ba745c2701c772beaccd2fa689e253c6/tenor.gif?itemid=16172875)

## The Complete Picture

### What we have

- $m$ - No. of training examples
- $n$ - No. of features
- $x$ - Independent Variables for $m$ examples
  - The shape of this will be ($m$, $N+1$)   
- $y$ - Dependent Variable (Spam or Not Spam) for $m$ examples
  - The shape of this will be ($m$, 1)

### Assume

- $\theta$ - Parameters initialized randomly with size same as no. of independent variables
  - The shape of this will be ($n+1$, 1) as there are $n$ features and there is one more term for bias $\theta_0$. Corresponding value $x_0$ is 1.
- $num\_epohcs$ - No. of times we want to train our model

### Do the following

- For $num\_epohcs$ times
  - Calculate Prediction for each of $m$ training examples
  
    $$
    z = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + ... \theta_N x_N
    $$

    $$ 
    h = \frac{1}{1+\exp^{-z}}
    $$

  - Calculate Loss
  
    $$
    J(\theta) = -\frac{1}{m} \sum_{i=1}^m y^{(i)}\log (h(z(\theta)^{(i)})) + (1-y^{(i)})\log (1-h(z(\theta)^{(i)})) 
    $$

  - Update Weights
  
    $$
    \theta_j = \theta_j - \alpha \times \frac{1}{m} \sum_{i=1}^m(h^{(i)}-y^{(i)})x_j 
    $$

- Now we have weights/ parameters $\theta$ that give use minimum loss. We use this to predict on our test data using the same way we did for training data
  
  $$
  z = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + ... \theta_N x_N
  $$

  $$ h = \frac{1}{1+\exp^{-z}}$$

  - If the $h >= 0.5$  : prediction = 1. This is a spam mail.
  - If $h < 0.5$ : prediction = 0. This is not the spam mail.

## Conclusion

![You did it!](https://media.giphy.com/media/3og0IHYTAkTARJMDpC/giphy.gif)

That' all! Congratulations! You now know to classify using logistic regression. You can use this now for any  binary classification. For multi-classification, where more than two categories exist, we need to use something called **Softmax Function** instead of Sigmoid Function so that we get probabilities for all the classes.

---
Do you want to code this in Python? Here are some articles to get you started:

- **[Example of Logistic Regression in Python](https://datatofish.com/logistic-regression-python/)** - Very sipmple example to predict whether student will get admitted to the university
- **[Building A Logistic Regression in Python, Step by Step](https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8)** - Predict whether client will subscribe to a term deposit or not.


---
That's all for today. This post is inspired from Prof. Andrew Ng's deep learning course. Let me know in comments, your suggestions and what do you think. 

See you next week!  
