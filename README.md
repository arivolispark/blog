# blog

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

