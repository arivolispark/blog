# blog

# Tools
https://explaineverything.com/

# Github Markdown cheatsheet
https://docs.github.com/en/get-started/writing-on-github

https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet

# SQL
Querying:
1. Subquery
2. Common Table Expression
3. View

# Numpy
'Numpy' is the short form for 'numeric python'.

Operations:

#SAPLEMM
1. *S*quare root
2. *A*bsolute
3. *P*ower
4. *L*ogarithm
5. *E*xponential
6. *M*inimum
7. *M*aximum


# inner 
import numpy as np

1-D array example:

A = np.array([[1,2,3],[4,5,6]])

B = np.array([[2,2,2],[2,2,2]])

C = np.inner(A, B)

print(C)



A:
[[1 2 3]
 [4 5 6]]


B:
[[2 2 2]
 [2 2 2]]


C:
[[12 12]
 [30 30]]


Reference:  https://numpy.org/doc/stable/reference/generated/numpy.inner.html


# Pandas
'Pandas' is the short form for 'Panel data'.


Common Table Expressions (CTE)

Perhaps its more meaningful to think of a CTE as a substitute for a view used for a single query. But doesn't require the overhead, metadata, or persistence of a formal view. Very useful when you need to:

Create a recursive query.
Use the CTE's resultset more than once in your query.
Promote clarity in your query by reducing large chunks of identical subqueries.
Enable grouping by a column derived in the CTE's resultset.

Reeference:  https://stackoverflow.com/questions/4740748/when-to-use-common-table-expression-cte


# Set vs JOIN

| SET | JOIN
| --- | ---
| Vertical stacking | Horizontal stacking 
| The number of columns and datatypes must match.  The number of rows can differ. | The number of columns and datatypes may mismatch. 


# Matrix operations
| Matrix operation | Output
| --- | ---
| Dot product | Scalar 
| Cross product | Vector 


# Cross product

```
import numpy as np


def cross_product(a, b):
    i_component = a[1] * b[2] - a[2] * b[1]
    j_component = a[2] * b[0] - a[0] * b[2]
    k_component = a[0] * b[1] - a[1] * b[0]
    return [i_component, j_component, k_component]


def numpy_cross_product(a, b):
    return np.cross(a, b)


if __name__ == "__main__":
    a = [1, 2, 3]
    print(" a: ", a)

    b = [4, 5, 6]
    print(" b: ", b)

    cross_product = cross_product(a, b)
    print("\n cross_product: ", cross_product)

    cross_product_using_numpy = numpy_cross_product(a, b)
    print(" cross_product_using_numpy: ", cross_product_using_numpy)
```

## Output

```
a:  [1, 2, 3]
b:  [4, 5, 6]

cross_product:  [-3, 6, -3]
cross_product_using_numpy:  [-3  6 -3]
```

# Hadamard product

https://en.wikipedia.org/wiki/Hadamard_matrix

https://stackoverflow.com/questions/30437418/how-can-i-find-out-if-a-b-is-a-hadamard-or-dot-product-in-numpy


# Database benchmarking

The TPC is a non-profit corporation focused on developing data-centric benchmark standards and disseminating objective, verifiable data to the industry.

https://tpc.org/

The database performance is directly related to I/O.


# Database stored procedures vs functions

| Stored procedures | Functions
| --- | ---
| perform actions | return a value 
| may or may not return values | can be used in SQL queries 


# Database optimizers

Up until the 4th normalized form, the optimizers can be impactful.  Beyond 4th normal form, the optimizers would be challenged.  For practical purposes, designing up until 3rd or 4th normal form and Boyce-Codd normalization would suffice.

# Vector databases
The following are some of the leading Vector databases:

1. [Pinecone](https://www.pinecone.io/ "Pinecone's Homepage")
2. [Milvus](https://milvus.io/ "Milvus's Homepage")
3. [Weaviate](https://weaviate.io/ "Weaviate's Homepage")

Qdrant (https://qdrant.tech/) is a vector database implemented using Rust.

# Wolfram Computational Intelligence
https://www.wolframalpha.com/

Compute answers using Wolfram's breakthrough technology & knowledgebase, relied on by millions of students & professionals.

# Calculus formulae
https://byjus.com/calculus-formulas/

https://sac.edu/AcademicProgs/ScienceMathHealth/MathCenter/Documents/calculus%20cheat%20sheet.pdf


# Types of Gradient descent learning algorithms
1. Batch gradient descent
2. Stochastic gradient descent
3. Mini-batch gradient descent

# Mathematics
https://math.stackexchange.com/

# D-Tale
https://github.com/man-group/dtale

# Realtime Streaming analytics
https://github.com/madderle/Capstone-Realtime-Streaming-Analytics

# Best 16 Vector Databases for 2024
https://lakefs.io/blog/12-vector-databases-2023/

# Animated Math
https://www.youtube.com/c/3blue1brown

https://www.3blue1brown.com/topics/linear-algebra

# Linear Algebra
Multiplication can be only between matrices or between a matrix and a vector.  It can not be between vectors.

Dot product of vectors tells how similar in direction the two vectors are.

a.b = |a||b|cos(o)

| Dot product | Cross product
| --- | ---
| Returns a scalar | Returns a vector
| Inner product | Outer product 


Linear Independence = Matrix rank

Matrix multiplication is nothing but a bunch of Inner (or Dot) products of all rows of matrix A with all columns of matrix B.

Conceptually, 'matrix multiplication' is a 'linear transformation of spaces'.

Shear mapping = if we shift vertically

Application of 'determinant' of Matrix:  Determinant tells whether the matrix can be inverted or not.

Determinant tells the change of areas when a region in one space is transformed to another region in a different space.

Application of determinant:  If determinant of a matrix is 0, then there is no inverse of that matrix.  So we are compressing the dimension which helps in dimensionality reduction.  So, looking for determinant = 0 for highly correlated features can help reduce that feature / dimension.

Symmetric matrix: A = Transpose of A

If A is an orthogonal matrix or orthonormal matrix, then (A) into (Transpose of A) = I.


Principle Component Analysis (PCA) is nothing but Identity decomposition.  
1. PCA is meant for dimension reduction.
2. PCA is not meant for Feature selection.

Note:  If you are going to go the PCA route, then you dont have to do Feature selection.  For example, if you are using Linear regression, then you typically dont do PCA.

Eigen decomposition of vectors:
Eigen values and Eigen vectors exist in 'complex' space.  It exists only sometimes 'real' space.

Eigen vector and Eigen values:  If the value and direction are unaffected, then they are independent (from a transformation perspective).

Determinant of a matrix = product of its eigen values

Covariance matrices are symmetric.

Median is insensitive to outliers.

https://numpy.org/doc/stable/reference/routines.linalg.html

https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity

R^2 is the Explained part of the error.  So, a high value of R^2 is recommended.

Moore-Penrose pseudo inverse:
https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse

Determinant is not the same as dot product.
The determinant of a matrix A is equal to the determinant of its transpose.

A non-singular matrix means, it has a determinant which is not zero.

Not all matrices have an inverse.

Determinant value is the Volume.

Determinant of singular matrices = 0

Eigen vectors and Inverse matrices do not exist for all matrices.

Coefficient of determination = R-squared

OLS (Ordinary Least square) can be achieved by a variety of ways like Gradient Descent, PCA, etc.

PCA can not perform scaling and normalization.

Matrix inversion method is more prone to numerical instability  than the Gradient Descent method.

The linalg.sklearn internally uses OLS
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html


In "overfitting", the line will go through "data points" and "noise or error points".  This is called "overfitting" because no room for "error" to participate in the equation.


| Eigen value decomposition | Singular value decomposition
| --- | ---
| Uses Eigen vector and Eigen values | Uses Singular vector and Singular values



The following are the methods to check if the independent variables in a linear regression model are linearly independent:
1. Correlation matrix
2. Scatter plot matrix
3. Variance inflation factor


# Notes
Submatrix = co-factor

# Probability

Benford's law:
https://en.wikipedia.org/wiki/Benford%27s_law

Pareto principle:
https://en.wikipedia.org/wiki/Pareto_principle
https://blog.hubspot.com/marketing/pareto-principle

Set = well defined collection of objects
Fruits = { Apple, Pear, Mango }

Statistics:  science of perception where to try to understand the world
Mathmatics: a way to understand all universes
Physics:  trying to understand our reality

Newton's law:
F = G M m / (r ^ 2)

Axiom:  self evident true statement

Binomial coefficient = Combination

*Law of Large Numbers:*
https://en.wikipedia.org/wiki/Law_of_large_numbers

*Central limit theorem:*
https://en.wikipedia.org/wiki/Central_limit_theorem

*Negative Binomial distribution:*
https://en.wikipedia.org/wiki/Negative_binomial_distribution

*Bayes theorem:*
https://www.freecodecamp.org/news/bayes-rule-explained/

"expected" is an integral of "likely".
"density" is an integral of "mass".

Functions:
1.  Cumulative Distribution Function (CDF):  https://en.wikipedia.org/wiki/Cumulative_distribution_function
2.  Probability Mass Function (PMF):  https://en.wikipedia.org/wiki/Probability_mass_function
3.  Probability Density Function (PDF):  https://en.wikipedia.org/wiki/Probability_density_function



Bernoulli distribution:
https://en.wikipedia.org/wiki/Bernoulli_distribution

The probability distribution of coin tosses can be explained by Bernoulli distribution.


CDF is an integral of PMF.

CDF applies to both discrete and continuous, but PDF applies to only continuous variables.

Binomial distribution is a _collection_ of _independent_ Bernoulli trials.


https://learningeconometrics.blogspot.com/2016/09/four-moments-of-distribution-mean.html

| Movement | Concept
| --- | ---
| First movement | Mean
| Second movement | Variance
| Third movement | Skew
| Fourth movement | Kurtosis


Normal distribution = Gaussian distribution

Outliers are 3-Sigma events.


The famous case where the 3-Sigma events are very rare is the LTCM collapse in the 1990s:
https://en.wikipedia.org/wiki/Long-Term_Capital_Management


The most important property of *Expectation* is *linearity*.

Standard Normal Distribution table:
https://www.math.arizona.edu/~rsims/ma464/standardnormaltable.pdf



# "Pattern recognition and Machine Learning”, by Christopher Bishop

https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738/ref=asc_df_0387310738/?tag=hyprod-20&linkCode=df0&hvadid=312125971120&hvpos=&hvnetw=g&hvrand=6330914398694388717&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9032030&hvtargid=pla-523035035000&psc=1&mcid=82bf1d9d2860386b80decce93dc3f017&tag=&ref=&adgrpid=61316180839&hvpone=&hvptwo=&hvadid=312125971120&hvpos=&hvnetw=g&hvrand=6330914398694388717&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9032030&hvtargid=pla-523035035000&gclid=CjwKCAjw_e2wBhAEEiwAyFFFo7ByO_gxYTVRuhOp1XcSMw4y5uIGG57YBjvplvFRu26HFB27Xf3V5BoCWM4QAvD_BwE

# LightGBM (Light Gradient Boosting Machine)
https://www.geeksforgeeks.org/lightgbm-light-gradient-boosting-machine/

# Black Monday 1987
https://en.wikipedia.org/wiki/Black_Monday_(1987)

# Beta distribution
https://en.wikipedia.org/wiki/Beta_distribution

# Conjugate prior
https://en.wikipedia.org/wiki/Conjugate_prior

# t-distribution
https://en.wikipedia.org/wiki/Student%27s_t-distribution

The t-distribution becomes normal distribution as the sample size increases.  When the sample size increases, the degrees of freedom increases as well.

# Memoryless property
https://www.statisticshowto.com/memoryless-property/

# Monte Carlo method
https://en.wikipedia.org/wiki/Monte_Carlo_method

# Frequentist statistics
https://reflectivedata.com/dictionary/frequentist-statistics/

# Git markdown for calculus formulae and greek letters

$\sqrt{3x + 1} + (1+x)^2$

```
&psi; | &#968; | Greek small letter psi | w |
&omega; | &#969; | Greek small letter omega | w |
```

# Posterior, Likelihood and Prior
Posterior is proportional to the product of likelihood and prior.

# Other conjugate prior examples

| Name | Description
| --- | ---
| Beta-Bernoulli | Here Beta is conjugate and Bernoulli is likelihood
| Gamma-Poisson | Here Gamma is conjugate and Bernoulli is likelihood
| Dirichlet-Multinomial | Here Dirichlet is conjugate and Multinomial is likelihood

# Prior, Posterior
| Name | Description
| --- | ---
| Prior | Hypothesis or belief, before any data has been incorporated
| Posterior | The understanding of the distribution of the data, after incorporating data

# COVID-19 data
https://github.com/owid/covid-19-data/tree/master/public/data

# t-distribution
When the sample size is 30 or less, then t-distribution is typically the choice.

https://www.tdistributiontable.com/

Beta distribution is a the conjugate prior for Binomial likelihood.

# Cumulative distribution function
https://en.wikipedia.org/wiki/Cumulative_distribution_function

# Statistics

| Name | Description
| --- | ---
| Probability | Deal with uncertainty
| Statistics | Deal with data; no uncertainty involved
| Statistical Inference | Deal with data under uncertainty


# What are the different scenarios where mean = median = mode?
The mean, median and mode could be same for any of the following scenarios:

1. If the sample size is one
2. If the data distribution has only one value:
     eg: 200, 200, 200, 200, 200
3. If the data distrbution is like this:  100, 200, 200, 200, 100
4. Uniform distribution
5. Normal distribution (in some cases)


Usually, having no outliers helps immensely.

# Interquartile range:
https://en.wikipedia.org/wiki/Interquartile_range

| Quartile range | Quartile range | Quartile range | Quartile range
| --- | --- | --- | ---
| 1st quartile | 2nd quartile | 3rd quartile | 4th quartile
| 25th percentile | 50th percentile | 75th percentile | 100th percentile


# Box plot
https://en.wikipedia.org/wiki/Box_plot

https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html

# Law of large numbers
https://en.wikipedia.org/wiki/Law_of_large_numbers

# Central limit theorem
https://en.wikipedia.org/wiki/Central_limit_theorem

# CRLB
https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound

# Dify
https://dify.ai/

# SLURM vs LSF vs Kubernetes scheduler

LSF stands for the IBM's Platform _L_oad _S_haring _F_acility.

https://www.run.ai/guides/slurm/slurm-vs-lsf-vs-kubernetes-scheduler-which-is-right-for-you#:~:text=kube%2Dscheduler%20vs%20Slurm&text=Slurm%20is%20the%20go%2Dto,to%20integrate%20with%20common%20frameworks

| SLURM | Kubernetes
| --- | ---
| SLURM is the go-to scheduler for managing distributed, batch-oriented workloads typical for HPC | Kube-schedule is the go-to scheduler for management of flexible, containerized workloads and microservices


# Homoscedasticity, Heteroscedasticity
https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity

# Overfitting
MSE = Mean Squared Error

If the training set has a _low MSE_ (ie, looks near perfect), and the testing set has a _high MSE_, then this could mean _Overfitting_.

_Using many features in a model that is trained on a small training set_ could likely lead to Overfitting.

# Underfitting
- Using few features in a model that is trained on a large training set
- Using linear features to fit a polynomial relationship
  
# Spurious Correlation
https://statisticsbyjim.com/basics/spurious-correlation/

# K-fold cross validation
https://machinelearningmastery.com/k-fold-cross-validation/

# Manage Machine Learning with Amazon SageMaker Experiments
https://docs.aws.amazon.com/sagemaker/latest/dg/experiments.html

# Quantile normalization
https://en.wikipedia.org/wiki/Quantile_normalization

# Feature scaling
1. Standard scaler
2. MinMax scaler

# AlloyDB, BigQuery
| AlloyDB | BigQuery
| --- | ---
| If the workloads require faster response time (for eg:  OLTP) | If workloads require data warehousing needs (for eg:  OLAP)


Regularization is a set of methods for reducing overfitting in machine learning models.

# ANOVA
https://www.investopedia.com/terms/a/anova.asp


# Bayesian estimation:
The following are some give-away words for 'Bayesian':
1. "Limited data"
2. "Prior knowledge"


Bayesian estimation is a technique where "prior knowledge" is used with "new" data to estimate parameters.

# Mean = Median = Mode
The following are the distributions:

1. Normal distribution
2. Uniform distribution
3. Symmetric distribution


In skewed or multimodal distributions, mean <> median <> mode.


# Mean, Median
| S.No. | Where median is better | Where mean is better
| --- | --- | ---
| 1 | Skewed data | Symmetric data
| 2 | Ordinal data | Interval or ratio data


# Point estimation
1. We want to assign a rating for a test for the students.  It is not always feasible to do the exercise on ALL students.  So, the exercise is done with a sample set of student population.  Based on that, the point estimation is calculated.
2. Clinical Trials:  Whether a medication works or not - it is not always feasible to do the exercise on ALL students.


# Unknown, determined vs Known, random

# Estimators
1. Maximum Likelihood  Estimators (MLE)
2. Method of Moments (MoM)


Bias is inversely proportional to the volume of data.

Underfitting = under parameterized

# Ways of modeling regressions:
There are several approaches.  They are:
1. Linear modeling
2. Neural networks
3. Decision trees
   

# Regression types
1. Vanilla Linear regression
2. Ridge regression (L1 regularization)
3. Lasso regression (L2 regularization)


Clustering comes under Unsupervised classification.

Logistic regression is the only Classification algorithm with _regression_ in it?

# Probability vs Likelihood

Likelihood starts with the data, and approximate the probability.  Probability is based on the distribution.

Logit function = Inverse of Sigmoid function

```
logit(p) = ln ( p / (1-p) )
```

# Cross Entropy loss
https://en.wikipedia.org/wiki/Cross-entropy

# Harmonic mean
https://www.investopedia.com/terms/h/harmonicaverage.asp

# ROC curve
https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=An%20ROC%20curve%20(receiver%20operating,False%20Positive%20Rate

AUC under ROC curve = 1

# F1 score
https://www.labelf.ai/blog/what-is-accuracy-precision-recall-and-f1-score

# Haversine_formula
https://en.wikipedia.org/wiki/Haversine_formula


PCA is meant for dimension reduction.  It is not meant for feature selection.

# MAPE
https://en.wikipedia.org/wiki/Mean_absolute_percentage_error


Logistic regression termed “regression” even though it is used for classification tasks because it was historically developed from linear regression models.

# Exotic Link functions for GLMs
Generalized linear models (GLMs) are a class of linear-based regression models developed to handle varying types of error distributions.

https://freakonometrics.hypotheses.org/56682

# KNN
K-nearest neighbor can be used for both:
1. Regression
2. Classification

KNN comes under Supervised learning.  K-means comes under Unsupervised learning.

| Supervised learning | Unsupervised learning
| --- | --- 
| KNN | K-means


KNN can be used for imputing data.

# Euclidean distance
https://en.wikipedia.org/wiki/Euclidean_distance


| Over fitting | Under fitting
| --- | --- 
|  If 'k' is very small | If 'k' is very large

# Naive Bayes
1. Features need to be independent of one another


For unbalanced data, Naive Bayes performs better than KNN.


https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf


# SVM
https://codedamn.com/news/machine-learning/what-are-support-vector-machines

| High 'c' | Low 'c'
| --- | --- 
|  No mis-classification | Mis-classification is tolerated


If data has outliers, we would have ‘c’ to deal with them.


### Maximum Margin Classifier
https://bookdown.org/mpfoley1973/data-sci/maximal-margin-classifier.html


### Hyperplane separation theorem
https://en.wikipedia.org/wiki/Hyperplane_separation_theorem

The 'c' parameter and Kernel are the 2 concepts that control the accuracy of SVMs.

# Notes
- *Regularization* technique is used to prevent overfitting in a classification model by adding a penalty term to the loss function.

- In *Cross validation* technique, dataset is split into a training set and a test set.

- *Accuracy* is an evaluation metric which is used to measure the performance of a classification model.

- | Overfitting | Underfitting
  | --- | --- 
  |  model fits the training data well | model does not fit the training data well
  |  fails to generalize to new data | fails to generalize to new data
  |  To address this, use: |           
  | - increase training dataset |
  | - use regularization technique |


- Tikhonov regularization = Ridge regression = L2 regularization,
is a regularization technique in regression 

- Bias-Variance trade offs:
  
| Bias | Variance
  | --- | --- 
  |  error due to underfitting the training data | error due to overfitting the training data

- Direction of the margin in a SVM is orthogonal to the decision boundary.

- Decision boundary in the K-NN algorithm is non-linear.

- Time complexity of training a Bayesian Classifier is linear in number of features.
  
- https://towardsdatascience.com/tree-algorithms-explained-ball-tree-algorithm-vs-kd-tree-vs-brute-force-9746debcd940
  
- https://en.wikipedia.org/wiki/Centroidal_Voronoi_tessellation#:~:text=Centroidal%20Voronoi%20tessellations%20are%20useful,according%20to%20a%20certain%20function.

# Bagging, Boosting

| Bagging | Boosting
| --- | --- 
|  this technique is used for solving overfitting | this technique is used for solving underfitting

# Ensemble techniques
The following are various ensemble techniques:
1. Neural Networks
2. SVM
3. Decision Tree
4. Linear Regression


The following are various techniques for solving overfitting/underfitting:
1. Bagging
2. Boosting
3. Stacking/blending
4. Mixture of Experts (MoE)

MoE is used in GenAI.

  
# Rho
'rho' typically refers to the correlation coefficient between the predictions of different base models (often Decision Trees or other classifiers) within the ensemble. This correlation coefficient measures how strongly the predictions of different models within the bagging ensemble are related to each other.

# Random Forest
Random Forest is a Bagging algorithm.

https://en.wikipedia.org/wiki/Random_forest

# Boosting
- Used in solving Underfitting.  
- For 'Tabular' data, Boosting is the champion, even better than Neural Networks.  (Note:  For imaging data, Neural Networks are a better choice).
- Popular applications:
    - Online websites:
        - CTR:  Click through rate
        - CVR:  Conversion rate
        - CPC:  Cost per click  

## AdaBoost
https://en.wikipedia.org/wiki/AdaBoost

## Gradient Boosting
Gradient Boosting = Gradient Descent + Boosting

### XGBoost
https://arxiv.org/pdf/1603.02754

### LightGBM
https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf

## CatBoost vs XGBoost vs LightGBM
https://www.kaggle.com/code/nholloway/catboost-v-xgboost-v-lightgbm


# Notes

- Random forest is preferred instead of Decision tree to reduce variance of the model.
- 
| Bagging | Boosting
| --- | ---
| Does not increase bias | May increase variance
| Bagging is a parallel process | Boosting is a sequential process

- Estimators = Decision trees

- Mixture of Experts (MoE) is used in Neural Networks.  It is complex to get it right.

# Eigenface
https://en.wikipedia.org/wiki/Eigenface

There are about 8 billion people in the world.  Based on the latest statistical research, all 8 billion people can be represented by about 50 features.  This is "dimensionality reduction" in "Unsupervised learning".  PCA is one of the popular dimensionality reduction techniques.

## Face recognition using Eigenfaces
https://sites.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf

- Question:  The number of hairs and weight or height is correlated as well.  Likewise, there are many other feature combinations.  How to pick which ones for dimensionality reduction?  Is it looking at more data?
  Answer:  To pick features for dimensionality reduction, you can use correlation analysis to remove highly correlated features, principal component analysis (PCA) to capture the most variance, feature importance from models to select relevant features, domain knowledge to prioritize important features, variance threshold to remove low variance features, and recursive feature elimination (RFE) to iteratively remove less important features using a model. Looking at more data can help clarify feature importance but isn't the only factor.

# Netflix competition
The winner of a Netflix competition used SVD.

https://pantelis.github.io/cs301/docs/common/lectures/recommenders/netflix/

# PCA vs vs SVD vs NMF 
https://sqlandsiva.blogspot.com/2023/02/svd-vs-pca-vs-nmf-singular-value.html

https://stats.stackexchange.com/questions/502072/what-is-the-main-difference-between-pca-and-nmf-and-why-to-choose-one-rather-tha

# Elbow method for K-means clustering
https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/

# Silhoutte score
This is the most common technique used to evaluate the k-means clustering approach.
https://en.wikipedia.org/wiki/Silhouette_(clustering)

# Dendrogram
https://en.wikipedia.org/wiki/Dendrogram

# Decision trees vs Hierarchical clustering 
Hierarchical clustering is similar to decision trees in that both create a hierarchical structure to represent data, but they differ in purpose and approach. Hierarchical clustering groups data points based on similarity to form clusters, while decision trees split data based on feature values to make predictions or classifications.

# StatQuest with Josh Starmer 
https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw

# PCA vs t-SNE
https://www.kaggle.com/code/agsam23/pca-vs-t-sne

t-SNE is very good for visulizating high dimensional data in low dimension space.  t-SNE is not very good for data analysis.

# Graph theory

## Centrality
https://en.wikipedia.org/wiki/Centrality

# Pinecone vs Milvus
https://myscale.com/blog/pinecone-vs-milvus-best-vector-database-efficiency/

# Sentence Transformer
https://sbert.net/

Sentence Transformers (a.k.a. SBERT) is the go-to Python module for accessing, using, and training state-of-the-art text and image embedding models. It can be used to compute embeddings using Sentence Transformer models (quickstart) or to calculate similarity scores using Cross-Encoder models (quickstart). This unlocks a wide range of applications, including semantic search, semantic textual similarity, and paraphrase mining.

# Geodesic distance
https://en.wikipedia.org/wiki/Geodesic

# Backward propagation
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

# Ultralytics
https://github.com/ultralytics/ultralytics

https://docs.ultralytics.com/modes/predict/#why-use-ultralytics-yolo-for-inference

# Vanishing Gradient problem
https://medium.com/@amanatulla1606/vanishing-gradient-problem-in-deep-learning-understanding-intuition-and-solutions-da90ef4ecb54#:~:text=A1%3A%20The%20vanishing%20gradient%20problem,to%20update%20the%20weights%20effectively.

https://www.youtube.com/watch?v=JIWXbzRXk1I


# RNN
https://en.wikipedia.org/wiki/Recurrent_neural_network

# GRU
https://en.wikipedia.org/wiki/Gated_recurrent_unit

# LSTM
https://en.wikipedia.org/wiki/Long_short-term_memory#:~:text=Long%20short%2Dterm%20memory%20(LSTM,and%20other%20sequence%20learning%20methods.

https://weberna.github.io/blog/2017/11/15/LSTM-Vanishing-Gradients.html

# Ada-Grad optimizer
https://medium.com/@brijesh_soni/understanding-the-adagrad-optimization-algorithm-an-adaptive-learning-rate-approach-9dfaae2077bb

# Adam optimizer
https://www.geeksforgeeks.org/adam-optimizer/

Adam optimizer = RMSProp + SGD with momentum

SGD = Stochastic Gradient Descent

# Data augmentation
Data augmentation = augmenting existing training data

# Regularization

| Name | Technique
| --- | ---
| L1 | Lasso regression
| L2 | Ridge regression
| Elastic net regularization | https://en.wikipedia.org/wiki/Elastic_net_regularization

# Internal Covariate Shift


# Theano
https://pypi.org/project/Theano/

# Renumics Spotlight
https://renumics.com/docs/getting-started

This is a tool used to identify outliers in imaging data.

# Imaging
- Weights and biases are the parameters.  The parameters impact the quality of the Deep Learning model.
- Learning rate is a hyper-parameter.  Hyper-parameters do not impact the quality of the Deep Learning model.

# Trainable parameters of Batch normalization
The following are the trainable parameters of Batch normalization:

1. Scale
2. Shift
3. EMA of mean
4. EMA of variance


https://keras.io/api/layers/normalization_layers/batch_normalization/

# Davies–Bouldin index
https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index

# Calinski–Harabasz index
https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index

# Pytorch
For "binary classification" problem, "Binary entropy" would be a great candidate for the loss function, and "Sigmoid function" would be a great candidate for the "Final" layer.

# Autograd
Autograd in PyTorch tracks the operations performed on tensors and automatically computes gradients.

# NN to LLM evolution
NN -> RNN -> LSTM -> Transformers -> LLMs

# CNN
https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks

## Stride
_Stride is how far the filter moves in every step along one direction._

https://medium.com/machine-learning-algorithms/what-is-stride-in-convolutional-neural-network-e3b4ae9baedb

Each filter corresponds to a neuron.

"Kernels" and "filters" can be used interchangeably.

kernel = filter

## Feature Map
https://medium.com/@saba99/feature-map-35ba7e6c689e#:~:text=In%20Convolutional%20Neural%20Networks%20(CNNs,a%20previous%20layer's%20feature%20map.

Regularization is a technique to take care of overfitting.
Dropout is a regularization technique.

# XAI
Explainable AI

https://ethical.institute/xai.html

# Gated Recurrent Unit (GRU)
https://en.wikipedia.org/wiki/Gated_recurrent_unit

# Keras, PyTorch

| Keras | PyTorch
| --- | ---
| Recommended for educational purposes and quick PoC | Efficient for deployment

# Gradients and Weights
Training uses Gradients.  Testing uses Weights.

# Computer Vision
https://livebook.manning.com/book/deep-learning-for-vision-systems/chapter-1/

# Data Annotator
https://trainingdata.pro/who-is-data-annotator

# Computer Vision tasks
1. Classification
2. Detection
3. Segmentation
4. Generation
5. Image Similarity

# Multiclass classification vs Multilabel classification

| Multiclass classification | Maultilabel classification
| --- | ---
| Sum of all probabilities would equate to 1 | Sum of all probabilities may equate to more than 1

# Image processing Occlusion
https://www.baeldung.com/cs/image-processing-occlusions

# Pixel density
https://en.wikipedia.org/wiki/Pixel_density#:~:text=%22PPI%22%20or%20%22pixel%20density,the%20area%20of%20the%20sensor.

# RGB and HSV color model demo
https://math.hws.edu/graphicsbook/demos/c2/rgb-hsv.html

# Kurtosis
https://www.scribbr.com/statistics/kurtosis/#:~:text=Kurtosis%20is%20a%20measure%20of,(medium%20tails)%20are%20mesokurtic.

# ResNet
ResNet primarily addresses vanishing gradient problem in deep neural networks.

# Computer Vision - Object detection
Object detection = Localization (boundary identification) + Classification (assign labels like ‘Cat’, ‘Dog’, ‘Mouse’)

# Intersection over Union (IoU)
https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

IoU = GT / P

where GT = Ground Truth
      P = Predicted

# Mean Average Precision (mAP)
https://www.v7labs.com/blog/mean-average-precision

# YOLO (You Only Look Once)
https://encord.com/blog/yolo-object-detection-guide/

# Action functions

## Scalar
The following operates on scalar values:

1. Sigmoid:  (if you want positive output values)
2. Tanh:  (if you want posistive or negative output values)
3. RELU
   
## Distribution function
**Softmax** operations on distributions to produce probabilities. 

# Loss functions

## L1/L2 loss

### L1 loss
Mean Average Error (MAE)

### L2 loss
Mean Squared Error (MSE)

## Entropy loss

### Cross Entropy Loss
### Binary Cross Entropy Loss

# Optimizers
List the optimizers here.

# 3blue1brown Neural Networks    
https://www.3blue1brown.com/topics/neural-networks

# ResNet
In deeper architecture, ResNet is for addressing the vanishing gradient problem.

## Resnet Architecture Explained
https://medium.com/@siddheshb008/resnet-architecture-explained-47309ea9283d

## YOLO notebook
https://colab.research.google.com/drive/1dC6m2S0nefw2Dv8bAV4jbuP5EDCQHqgG?usp=sharing

# FCN, CNN, RNN
FCN: for tabular data
CNN: ?
RNN:  for sequence of words

# Understanding LSTMs
https://colah.github.io/posts/2015-08-Understanding-LSTMs/

# Illustrated Guide to Transformers Neural Network: A step by step explanation
https://www.youtube.com/watch?v=4Bdc55j80l8


https://jalammar.github.io/illustrated-transformer/

# Attention is all you need
https://arxiv.org/abs/1706.03762

# Image classification demo
https://colab.research.google.com/drive/1pAhxl7GWKrbIHi5nJsLjAV-FnH1guBL9?usp=sharing

# Fourier series graph
https://www.mathsisfun.com/calculus/fourier-series-graph.html

# Time series analysis using Prophet in Python — Part 1: Math explained 
https://github.com/sophiamyang/sophiamyang.github.io/blob/master/DS/timeseries/timeseries1.md


https://www.geeksforgeeks.org/derivative-of-the-sigmoid-function/

# Invariants
Here are some examples of Invariants:
1. Scale
2. Texture
3. Brightness

**Viewing angle** is key.  

Panoptic segmentation = Semantic segmentation + Instance segmentation

# Dice loss
https://cvinvolution.medium.com/dice-loss-in-medical-image-segmentation-d0e476eb486

# Precision Agriculture
https://en.wikipedia.org/wiki/Precision_agriculture

# U-net Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/abs/1505.04597

# Oxford-IIIT Pet Dataset
https://www.robots.ox.ac.uk/~vgg/data/pets/

# Intro to Diffusion Model — Part 5
https://dzdata.medium.com/intro-to-diffusion-model-part-5-d0af8331871

# Trimap
https://github.com/lnugraha/trimap_generator

# Penn-Fudan Database for Pedestrian Detection and Segmentation
https://www.cis.upenn.edu/~jshi/ped_html/

# Siamese networks
https://www.analyticsvidhya.com/blog/2023/08/introduction-and-implementation-of-siamese-networks/

# Popular Similarity measures
https://dataaspirant.com/five-most-popular-similarity-measures-implementation-in-python/

# The Database of Faces (AT&T)
https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/

# History of Trignometry
https://en.wikipedia.org/wiki/History_of_trigonometry#:~:text=In%20the%2015th%20century%2C%20Jamsh%C4%ABd,a%20form%20suitable%20for%20triangulation.

In the 15th century, Jamshīd al-Kāshī provided the first explicit statement of the law of cosines in a form suitable for triangulation.

# Spline regression
https://en.wikipedia.org/wiki/Spline_(mathematics)

# AlexNet
https://en.wikipedia.org/wiki/AlexNet#:~:text=AlexNet%20is%20the%20name%20of,D.

# LeNet
https://en.wikipedia.org/wiki/LeNet

# Bag of Words
https://en.wikipedia.org/wiki/Bag-of-words_model

# TF-IDF
https://en.wikipedia.org/wiki/Tf%E2%80%93idf

# Stemming vs Lemmatization
https://www.analyticsvidhya.com/blog/2022/06/stemming-vs-lemmatization-in-nlp-must-know-differences/

# Vader
https://medium.com/@rslavanyageetha/vader-a-comprehensive-guide-to-sentiment-analysis-in-python-c4f1868b0d2e
