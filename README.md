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


# Database

The TPC is a non-profit corporation focused on developing data-centric benchmark standards and disseminating objective, verifiable data to the industry.

https://tpc.org/

Performance = I/O


=====

who is the leader in vector dbs today?

Pinecone
Milvus
Weaviate



=====

Is functions here is same as stored procedures?

In databases, functions and stored procedures serve similar purposes but have differences. Functions return a value and can be used in SQL queries, while stored procedures perform actions and may or may not return values.

=====

> 4 joins will throw the Optimizer off


