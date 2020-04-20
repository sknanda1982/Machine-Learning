# Heart Risk Prediction using Machine Learning Algorithm
Today the heart disease is one of the most important causes of death in the world. So, its early prediction and diagnosis 
is important in medical field, which could help in on time treatment, decreasing health costs and decreasing death caused by 
it. In fact, the main goal of using data mining algorithms in medicine by using patients’ data is better utilizing the database
and discovering tacit knowledge to help doctors in better decision making. Therefore, using data mining and discovering knowledge 
in cardiovascular centres could create a valuable knowledge, which improves the quality of service provided by managers, and could 
be used by doctors to predict the future behaviour of heart diseases using past records. Also, some of the most important applications 
of data mining and knowledge discovery in heart patient’s system includes: diagnosing heart attack from various signs and properties, 
evaluating the risk factors which increases the heart attack. Here various type of Machine Learning Model were used. 

# Logistic Regression
Logistic Regression is a Machine Learning algorithm which is used for the classification problems, it is a predictive analysis algorithm and based on the concept of probability.

![](Images/lr1.png)

We can call a Logistic Regression a Linear Regression model but the Logistic Regression uses a
more complex cost function, this cost function can be defined as the Sigmoid Function knwon as
`logistic function' instead of a linear function. The hypothesis of logistic regression tends it to limit the cost function between 0 and 1. Therefore linear functions fail to represent it as it can have a value greater than 1 or less than 0 which
is not possible as per the hypothesis of logistic regression.  Refere 
##### Model/Log_regression.ipynb

# K-Neareat-Neighbor ML
K-Nearest Neighbor is one of the simplest Machine Learning algorithms based on Supervised
Learning technique-KNN algorithm assumes the similarity between the new case/data and avail-
able cases and put the new case into the category that is most similar to the available categories-
KNN algorithm stores all the available data and classies a new data point based on the sim-
ilarity. This means when new data appears then it can be easily classied into a well suite
category by using K- NN algorithm. K-NN algorithm can be used for Regression as well as for
Classification but mostly it is used for the Classification problems.
Suppose there are two categories, i.e., Category A and Category B, and we have a new data
point x1, so this data point will lie in which of these categories. To solve this type of problem,
we need a K-NN algorithm. With the help of K-NN, we can easily identify the category or class
of a particular dataset. Consider the below diagram:
![](Images/knn.png)

* Step-1: Select the number K of the neighbors
* Step-2: Calculate the Euclidean distance of K number of neighbors
* Step-3: Take the K nearest neighbors as per the calculated Euclidean distance.
* Step-4: Among these k neighbors, count the number of the data points in each category.
* Step-5: Assign the new data points to that category for which the number of the neighbor is
maximum.
* Step-6: Our model is ready.

# Support Vector Machine 
The objective of the support vector machine algorithm is to nd a hyperplane in an N-dimensional
space(N | the number of features) that distinctly classies the data points.
To separate the two classes of data points, there are many possible hyperplanes that could be
chosen. Our objective is to nd a plane that has the maximum margin, i.e the maximum dis-
tance between data points of both classes.

![](Images/svm1.png)

In the SVM algorithm, we are looking to maximize the margin between the data points and the
hyperplane. The loss function that helps maximize the margin is hinge loss.

![](Images/svm2.png)

The cost is 0 if the predicted value and the actual value are of the same sign. If they are not,
we then calculate the loss value. We also add a regularization parameter the cost function. The
objective of the regularization parameter is to balance the margin maximization and loss. After
adding the regularization parameter, the cost functions looks as below.

![](Images/svm3.png)

Now that we have the loss function, we take partial derivatives with respect to the weights to
and the gradients. Using the gradients, we can update our weights.

# Multiple Perceptron (MLP)

 

