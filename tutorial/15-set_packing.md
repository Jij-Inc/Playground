--- 
description: Here we show how to solve set packing problem using JijZept and JijModeling.
---

# Set Packing

Here we show how to solve set packing problem using JijZept and [JijModeling](https://www.ref.documentation.jijzept.com/jijmodeling/). 
This problem is also mentioned in 4.2. Set Packing on [Lucas, 2014, "Ising formulations of many NP problems"](https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full).

## What is Set Packing?

Let us consider the same setup as the [Exact Cover problem](https://www.documentation.jijzept.com/docs/tutorial/exact_cover), but now ask a different question: what is the largest number of subsets $W_i$ which are all disjoint?

## Mathematical Model

Let $x_i$ be a binary variable that takes on the value 1 if subset $W_i$ is selected, and 0 otherwise.

### Constraint: each element in $U$ appears in exactly one selected subset

This can be expressed as following using $V_{i, j}$ where it represents a mapping from a subset $i$ to a set of elements $j$ that it contains.
Here we set $V_{i, j}$ to be a matrix that is 1 when $W_i$ contains $j$ and 0 otherwise.
$$
\sum_{i=1}^N x_i \cdot V_{i, j} = 1 \text{ for } j = 1, \ldots, M
\tag{1}
$$

### Objective function : maximize the number of sets

We simply counts the number of sets we include as the following.

$$
\max \sum_i x_i \tag{2}
$$

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

We use the same variables in the exact cover problem.

### Constraint

We implement a constraint Equation (1).


```python
# set problem
problem = jm.Problem('Set Packing',sense = jm.ProblemSense.MAXIMIZE)
# set constraint: each element j must be in exactly one subset i
problem += jm.Constraint('onehot', jm.sum(i, x[i]*V[i, j]) == 1, forall=j)
```

### Objective function

Next, we implement an objective function Equation (2).


```python
# set objective function: maximize the number of sets
problem += x[:].sum()
```

Let's display the implemented mathematical model in Jupyter Notebook.


```python
problem
```




$$\begin{array}{cccc}\text{Problem:} & \text{Set Packing} & & \\& & \max \quad \displaystyle \sum_{\ast_{0} = 0}^{N - 1} x_{\ast_{0}} & \\\text{{s.t.}} & & & \\ & \text{onehot} & \displaystyle \sum_{i = 0}^{N - 1} x_{i} \cdot V_{i, j} = 1 & \forall j \in \left\{0,\ldots,M - 1\right\} \\\text{{where}} & & & \\& x & 1\text{-dim binary variable}\\\end{array}$$



## Prepare an instance

We prepare as below.


```python
import numpy as np

# set a list of W
W_1 = [1, 2, 3]
W_2 = [4, 5]
W_3 = [6]
W_4 = [7]
W_5 = [2, 5, 7]
W_6 = [6, 7]

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
response = sampler.sample_model(problem, instance_data, multipliers={'onehot': 0.5}, num_reads=100, search=True)
```

##  Check the solution

In the end, we extract the solution from the feasible solutions.


```python
# get sampleset
sampleset = response.get_sampleset()
# extract feasible samples
feasible_samples = sampleset.feasibles()
# get the values of feasible objectives
feasible_objectives = [-sample.eval.objective for sample in feasible_samples]
if len(feasible_objectives) == 0:
    print("No feasible sample found ...")
else:
    # get the index of the highest objective value
    highest_index = np.argmax(feasible_objectives)
    # get the highest solution
    highest_solution = feasible_samples[highest_index].var_values["x"].values
    # get indices of x == 1
    x_indices = [key[0] for key in highest_solution.keys()]
    # show the result
    for i in x_indices:
        print(f"W_{i+1} = {inst_V[i, :].nonzero()[0]+1}")
```

    W_2 = [4 5]
    W_1 = [1 2 3]
    W_4 = [7]
    W_3 = [6]


As we expected, JijZept successfully returns the result.
