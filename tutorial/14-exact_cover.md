--- 
description: Here we show how to solve exact cover problem using JijZept and JijModeling.
---

# Exact Cover

Here we show how to solve the exact cover problem using JijZept and [JijModeling](https://www.ref.documentation.jijzept.com/jijmodeling/). 
This problem is also described in 4.1. Exact Cover on [Lucas, 2014, "Ising formulations of many NP problems"](https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full).

## What is Exact Cover Problem?

We consider a set $U = \{1,...,M\}$, and subsets $W_i \subseteq U (i = 1,...,N)$
such that
$$
U = \bigcup_{i} W_i
$$

The question posed is: 'Does there exist a subset, denoted as $R$, within the set of sets ${W_i}$, such that the elements of $R$ are mutually disjoint and the union of these elements constitutes the set $U$?'

### Example

Let's take a look at the following situation.

Let $U$ be the universe of elements $\{1, 2, 3, 4, 5, 6, 7\}$, and let $W_i$ be a collection of subsets of $U$, defined as follows:

$$
\begin{align}
&W_1 = \{1, 2, 3\}, \\ 
&W_2 = \{4, 5\},\\
&W_3 = \{6, 7\},\\
&W_4 = \{3, 5, 7\},\\
&W_5 = \{2, 5, 7\},\\
&W_6 = \{3, 6, 7\}.
\end{align}
$$

The exact cover problem for this example asks whether there exists a subset $R$ of $W_i$, such that the subsets in $R$ are disjoint and their union is exactly $U$. In other words, we are looking for a way to choose some of the subsets from $W_i$, such that each element in $U$ appears in exactly one subset of $R$, and no two subsets in $R$ share any elements.
In this case, one possible exact cover for this instance of the problem is: 
$$
R = \{W_1, W_2, W_3\}.
$$

### Mathematical Model
Let $x_i$ be a binary variable that takes on the value $1$ if subset $W_i$ is selected, and $0$ otherwise. 

**Constraint: each element in $U$ appears in exactly one selected subset**

Consider the following expression:

$$
\sum_{i=1}^N x_i \cdot V_{i, j} = 1 \text{ for } j = 1, \ldots, M
\tag{1}
$$

In this expression, $V_{i, j}$ represents a matrix that maps subset $i$ to a set of elements $j$. Specifically, $V_{i, j}$ is $1$ if $W_i$ contains $j$ and $0$ otherwise

For instance, the above example can be written as the following.
$$
\begin{align}
&x_1 = 1 \because 1 \text{ appears only in } W_1,
\\
&x_1 + x_5 = 1 \because 2  \text{ appears in } W_1 \text{ and } W_5,
\\
&x_1 + x_4 + x_6 = 1 \because 3  \text{ appears in } W_1, W_4, \text{ and } W_6,
\\
&x_2 = 1 \because 4  \text{ appears only in } W_2,
\\
&x_2 + x_4 + x_5 = 1 \because 5  \text{ appears in } W_2, W_4, \text{ and } W_5,
\\
&x_3 + x_6 = 1 \because 6  \text{ appears in } W_3 \text{ and } W_6,
\\
&x_3 + x_4 + x_5 + x_6 = 1 \because 7  \text{ appears in } W_3, W_4, W_5, \text{ and } W_6 .
\end{align}
$$

**Objective function: minimize the set cover**

This can be expressed as the following.
$$
\min \sum_i x_i
\tag{2}
$$

<!-- **Constraint 2: no two selected subsets overlap**

This constraint ensures that each subset $V_i$ is used at most once in the solution.
$$
\sum_{j=1}^n [j \in V_i] \cdot x_j \leq 1 \text{ for } i =  1, \ldots, N
$$

Let us consider the example mentioned above.
For $i=1$, We want to ensure that $V_1$ is used at most once in the solution. We can do this by setting a constraint that limits the number of subsets that include elements from $V_1$ to be at most one, which is shown below.
$$
x_1 + x_2 + x_3 \leq 1
$$
This equation ensures that at most one subset from the family of subsets $V_i$ that includes any of the elements in $V_1$ is selected. If $x_1 = 1$ (meaning that subset $V_1$ is selected), then $x_2$ and $x_3$ must be zero (meaning that subsets $V_4$ and $V_6$, which also include elements from $V_1$, cannot be selected). Similarly, if $x_4 = 1$ (meaning that subset $V_4$ is selected), then $x_1$, $x_2$, and $x_3$ must be zero.

We repeat this process for each subset in ${V_i}$ to ensure that each subset is used at most once in the solution. The second constraint enforces this requirement for all subsets in the family ${V_i}$. -->

## Modeling by JijModeling

Next, we show an implementation using JijModeling. We first define variables for the mathematical model described above.


```python
import jijmodeling as jm


# define variables
U = jm.Placeholder('U')
N = jm.Placeholder('N')
M = jm.Placeholder('M')
V = jm.Placeholder('V', ndim=2)
x = jm.BinaryVar('x', shape=(N,))
i = jm.Element('i', N)
j = jm.Element('j', M)
```

`U` is the universe.
`N` denotes the number of subsets.
`M` is the number of elements.
`V` defines if subset $i$ contains an element $j$.
We define a two-dimensional list of binary variables `x`. 
Finally, we set the subscripts `i` and `j` used in the mathematical model.

### Constraint

We implement a constraint Equation (1).


```python
# set problem
problem = jm.Problem('Exact Cover')
# set constraint: each element j must be in exactly one subset i
problem += jm.Constraint('onehot', jm.sum(i, x[i]*V[i, j]) == 1, forall=j)
```

### Objective function

We implement an objective function.


```python
problem += jm.sum(i, x[i])
```

Let us display the implemented mathematical model in Jupyter Notebook.


```python
problem
```




$$\begin{array}{cccc}\text{Problem:} & \text{Exact Cover} & & \\& & \min \quad \displaystyle \sum_{i = 0}^{N - 1} x_{i} & \\\text{{s.t.}} & & & \\ & \text{onehot} & \displaystyle \sum_{i = 0}^{N - 1} x_{i} \cdot V_{i, j} = 1 & \forall j \in \left\{0,\ldots,M - 1\right\} \\\text{{where}} & & & \\& x & 1\text{-dim binary variable}\\\end{array}$$



## Prepare an instance

Here, we use the same values from an example as we describe before.


```python
import numpy as np

# set a list of W
W_1 = [1, 2, 3]
W_2 = [4, 5]
W_3 = [6, 7]
W_4 = [3, 5, 7]
W_5 = [2, 5, 7]
W_6 = [3, 6, 7]

# set the number of Nodes
inst_N = 6
inst_M = 7

# Convert the list of lists into a NumPy array
inst_V = np.zeros((inst_N, inst_M))
for i, subset in enumerate([W_1, W_2, W_3, W_4, W_5, W_6]):
    for j in subset:
        inst_V[i, j-1] = 1  # -1 since element indices start from 1 in the input data

instance_data = {'V': inst_V, 'M': inst_M, 'N': inst_N}
```

## Solve by JijZept's SA

We solve this problem using JijZept `JijSASampler`. We also use the parameter search function by setting `search=True`.


```python
import jijzept as jz

# set sampler
config_path = "../../../config.toml"
sampler = jz.JijSASampler(config=config_path)
# solve problem
response = sampler.sample_model(problem, instance_data, multipliers={'onehot': 1.0}, num_reads=100, search=True)
```

##  Check the solution

In the end, we extract the solution from the feasible solutions.


```python
# get sampleset
sampleset = response.get_sampleset()
# extract feasible samples
feasible_samples = sampleset.feasibles()
# get the values of feasible objectives
feasible_objectives = [sample.eval.objective for sample in feasible_samples]
if len(feasible_objectives) == 0:
    print("No feasible sample found ...")
else:
    # get the lowest index of values
    lowest_index = np.argmin(feasible_objectives)
    # get the lowest solution
    lowest_solution = feasible_samples[lowest_index].var_values["x"].values
    # get the indices x == 1
    x_indices = [key[0] for key in lowest_solution.keys()]
    # show the result
    for i in x_indices:
        print(f"W_{i+1} = {inst_V[i, :].nonzero()[0]+1}")
```

    W_3 = [6 7]
    W_2 = [4 5]
    W_1 = [1 2 3]


With the above calculation, we obtain a the result.
