--- 
description: Here we show how to solve the number partitioning problem using JijZept and JijModeling.
---

# Number Partitioning

Here we show how to solve the number partitioning problem using JijZept and [JijModeling](https://www.ref.documentation.jijzept.com/jijmodeling/). 
This problem is also mentioned in 2.1. Number Partitioning on [Lucas, 2014, "Ising formulations of many NP problems"](https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full).

## What is number partitioning? 

Number partitioning is the problem of dividing a given set of numbers into two sets such that the sum of the numbers is equal. Let us consider a simple example.

For example, we have such a set of numbers $A = \{1, 2, 3, 4\}$. 
It is easy to divide $A$ to $\{1, 4\}, \{2, 3\}$ and we can get the sum of each subset as 5. 
Thus, when the size of the set is small, the answer is relatively easy to obtain. 
However, the larger problem size is hard to solve quickly.
For this reason, we explain to solve this problem using annealing in this tutorial.

## Mathematical model

First, let us model the Hamiltonian of this problem.
Let $A$ be the set to be partitioned, and $A$ has elements $a_i \ (i = \{0, 1, \dots, N-1\})$, where $N$ is the number of elements in this set.
We consider to divide $A$ into two sets $A_0$ and $A_1$. 
We define a binary variable $x_i$ that is 0 when $a_i$ is contained in $A_0$ and 1 when $a_i$ is contained in $A_1$.
Using $x_i$, the total value of the numbers into $A_0$ can be written as $\sum_i a_i(1-x_i)$, and the sum of $A_1$ is $\sum_i a_i x_i$.
We need to find a solution that satisfies the constraint that the sum of each of the two subsets is equal.

$$
\sum_{i=0}^{N-1} a_i (1-x_i) 
= \sum_{i=0}^{N-1} a_i x_i \ \Longrightarrow \ 
\sum_{i=0}^{N-1}a_i (2 x_i - 1) 
= 0 \tag{1}
$$

Applying the penalty method to (1) yields the Hamiltonian for the number partitioning.

$$
H = \left\{ \sum_{i=0}^{N-1} a_i (2 x_i - 1) \right\}^2 \tag{2}
$$

## Modeling by JijModeling

Next, we show how to implement above equation using JijModeling. We first define variables for the mathematical model described above.


```python
import jijmodeling as jm

a = jm.Placeholder("a",ndim = 1)
N = a.shape[0]
x = jm.BinaryVar("x",shape=(N,))
i = jm.Element("i",belong_to=(0,N))
```

`a` is a one-dimensional array representing the elements in $A$. 
We can get the number of elements `N` from the length of `a`.
We define a binary variable `x`.
Finally, we define subscripts `i` used in (2).  
Then, we implement the Hamiltonian of number partitioning.


```python
problem = jm.Problem("number partition")
s_i = 2*x[i] - 1
problem += (jm.sum(i, a[i]*s_i)) ** 2
```

We can check the implementation of the mathematical model on the Jupyter Notebook.


```python
problem
```




$$\begin{array}{cccc}\text{Problem:} & \text{number partition} & & \\& & \min \quad \displaystyle \left(\left(\sum_{i = 0}^{\mathrm{len}\left(a, 0\right) - 1} a_{i} \cdot \left(2 \cdot x_{i} - 1\right)\right)^{2}\right) & \\\text{{where}} & & & \\& x & 1\text{-dim binary variable}\\\end{array}$$



## Prepare an instance

We prepare a set of numbers $A$. 
Here, we consider the problem of partitioning numbers from 1 to 40.
In the problem of partitioning consecutive numbers from $N_i$ to $N_f$ and the number of numbers is even, there are various patterns of partitioning.
However, the total value of the partitioned set can be calculated as follows: 

$$
(\mathrm{total \ value}) 
= \frac{(N_f + N_i) (N_f - N_i + 1)}{4} 
$$

In this case, the total value is expected to be 410. Let's check it.


```python
import numpy as np

N = 40
instance_data = {"a":np.arange(1,N+1)}
```

## Solve by JijZept's SA

We solve this problem using JijZept `JijSASampler`. 
In this case, we have no constraints. Thus `multipliers` dictionary is empty.


```python
import jijzept as jz

# set sampler
sampler = jz.JijSASampler(config="config.toml")
# solve problem
results = sampler.sample_model(problem, instance_data)
```

## Visualize the solution

Let's check the result obtained. 
We separate the indices classified as $A_0$ and $A_1$.
Finally, we sum over them.


```python
feasibles = results.feasible()

class_1_index = feasibles.record.solution['x'][0][0][0]
class_0_index = [i for i in range(0,N) if i not in class_1_index]

class_1 = instance_data['a'][class_1_index]
class_0 = instance_data['a'][class_0_index]

print(f"class 1 : {class_1} , total value = {np.sum(class_1)}")
print(f"class 0 : {class_0} , total value = {np.sum(class_0)}")
```

    class 1 : [ 1  3  4  7 10 11 12 14 15 16 17 18 20 21 22 23 26 27 29 35 39 40] , total value = 410
    class 0 : [ 2  5  6  8  9 13 19 24 25 28 30 31 32 33 34 36 37 38] , total value = 410


As we expected, we obtain both total values are 410.
