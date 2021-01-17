---
title: WhatWhyHow Part 2 - Naive Bayes
excerpt: This is not just another ML tutorial! Learn What, Why and How of most naive machine learning classification algorithm!
categories:
  - Data Science
tags:
  - [ML, Classification]
published: true
comments: true
classes: wide
header:
  teaser: /assets/img/4.jpg
  image: /assets/img/4.jpg
---
<small>*<span>Photo by <a href="https://unsplash.com/@freegraphictoday"> AbsolutVision</a> on <a href="https://unsplash.com/photos/bSlHKWxxXak">Unsplash</a></span>*</small>

Last week I wrote a blog about [Logistic Regression](https://blackbird71sr.github.io/blog/data%20science/WhatWhyHow1-Logistic-Regression/), which is one of the most widely used classification algorithm. 

There is another algorithm which so simple and easy that there is **naive** in it's name. Yeah....talking about Naive Bayes...This algorithm is based on very fundamental concept of **Probability**. Specifically we use **Bayes theorem** to understand naive bayes. So let's look at it first...

![Probably!](https://media.giphy.com/media/WUsh2fu9dAFHn86bM1/giphy.gif)

## Bayes Theorem

Let's say there are two independent events A and B. No we can define four terms here:

$$
P(A) = Probability \space of \space A \space happening
$$

$$
P(B) = Probability \space of \space B \space happening
$$

$$
P(A|B) = Probability \space of \space A \space happening \space given \space that \space B \space has \space already \space happened
$$

$$
P(B|A) = Probability \space of \space B \space happening \space given \space that \space A \space has \space already \space happened
$$

Now according to Bayes Theorem:

$$
P(A|B) = \frac{P(B|A). P(A)}{P(B)}
$$

Now this statement comes at a cost of one big assumption that

**Features are independent**.
This means presence of one particular feature does not affect the other feature. This is not at all true in real-life. *That's why this is called a naive algorithm!*

If this is not all true in real-life examples and datasets, why bother about it?

Because it is so easy to build Naive Bayes algorithm for classification and provides us with a quick baseline before moving to more complex and time-consuming algorithms.

## Naive Bayes

Let's understand this with simple example.

### Situation

![Situation!](https://media.giphy.com/media/W0VtJNnBAtyEuxyE4g/giphy.gif)

So consider a situation where you have to predict whether this email is spam or not. You have some features for each email such as sender email address, time of email which we call as X. The variable we want to predict that is spam or not is Y. 

This is exact same problem we tried to solve using Logistic Regression last week.

### Assumptions

![Assume!](https://media.giphy.com/media/GHc1i70ZAUBcA/giphy.gif)

1. **All the features are independent.**

That means there is no effect of one feature or another. Say, if sender email address and time of sending does not have any co-relation. Actually, in real life we see the corelation between this two due to use of bots in such spam email, but according to naive bayes we consider it so. 


2. **All the features have same effect on the prediction Y.**

That means sender email address has same effect on spam or not spam as time of sending according to naive bayes. We surely know this is not possible, but still!


### Algorithm

According to Bayes theorem, we saw earlier, we can write this:

$$
P(y|X) = \frac{P(X|y). P(y)}{P(X)}
$$

Here,

$$
P(y) = Probability \space of \space email \space is \space spam \space or \space not
$$

$$
P(X) = Probability \space of \space features \space X
$$

$$
P(y|X) = Probability \space of \space email \space spam \space or \space not \space given \space features \space X
$$

$$
P(X|y) = Probability \space of \space features \space X \space given \space email \space is \space spam \space or \space not
$$


Now as we have more than 1 features, we can write

$$
X = (x_1, x_2, x_3, ...., x_n)
$$

Here $x_1, x_2, x_3, ...., x_n$ represent features like time of email, sender address, etc.

Now we can substitute this in the bayes theorem formula and expand it using chain rule:

$$
P(y|x_1,x_2,...,x_n) = \frac{P(x_1|y) P(x_2|y)...P(x_n|y)P(y)}{P(x_1)P(x_2)...P(x_n)}
$$

Now we can easily find this all values using our dataset. For example, set $x_1$ is time of email in hours. Say for our text example $x_1=5$. Then $P(x_1|y)$ for $y=1$ means probability of spam email arriving at hour 5. 

Assume we have 100 training examples out of which 40 are spam and rest 60 are not spam. There are total 20 emails with $x_1 = 5$. Out of this 6 are when email is spam are rest 14 when email is not spam.

So for probability of spam email arriving at hour 5 is $\frac{6}{40}$. You will find this for all features. This will give all the terms except last one which is nothing but probability of email being spam. This is nothing but total spam emails divided by total email that is  $\frac{40}{100}$.

This completes the numerator of the equation. Now, we don't calculate the denominator because it does not change if we change y. So we can remove the denominator and can say that numerator is directly proportional to LHS.

$$
P(y|x_1,x_2,...,x_n) \propto {P(x_1|y) P(x_2|y)...P(x_n|y) \space P(y)}
$$

$$
P(y|x_1,x_2,...,x_n) \propto P(y) \space  { \Pi_{i=1}^{n} P(x_i|y) }
$$

As there are only two outcomes possible, spam or not spam we can easily choose the y with max probability. That means we will find two probabilities:

$$
P(spam|x_1,x_2,...,x_n) \space and \space P(notspam|x_1,x_2,...,x_n)
$$

and choose the one which is more giving us the prediction for our test example.

$$
y = max(P(spam|x_1,x_2,...,x_n), \space P(notspam|x_1,x_2,...,x_n))
$$

Yeah, and we are done!

![Simple](https://media.giphy.com/media/dWy2WwcB3wvX8QA1Iu/giphy.gif)

Although there are many other types of Naive Bayes, the fundamental concept remains the same.

## Pros

- Easy and Simple to implement! Gives quick baseline for experiments...
- Doesn't require much training data
- Highly Scalable

## Cons

- Assumptions attributes independent. No possible in real life!
- Treats all features equally! This can actually solved by adding weights but default Naive Bayes doesn't care!

---
Do you want to code this in Python? Here are some articles to get you started:
  
- **[Naive Bayes Classifier From Scratch in Python](https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/)** - Predict flower species on Iris Flower Species Dataset.
- **[In Depth: Naive Bayes Classification](https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html)** - In-depth explannation of what I just covered with great visualizations

---
That's all for today. Let me know in comments, your suggestions and what do you think. 

See you next week!  