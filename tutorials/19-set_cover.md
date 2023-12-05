--- 
description: Here we show how to solve the set cover problem using JijZept and JijModeling.
---

# Set Cover

Here we show how to solve the set cover problem using JijZept and [JijModeling](https://www.ref.documentation.jijzept.com/jijmodeling/). 
This problem is also mentioned in 5.1. Set Cover on [Lucas, 2014, "Ising formulations of many NP problems"](https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full).

## What is Set Cover?

We consider a set $U = \{1,...,M\}$, and subsets $V_i \subseteq U (i = 1,...,N)$
such that
$$
U = \bigcup_{i} V_i
$$
The set covering problem is to find the smallest possible number of $V_i$ s, such that the union of them is equal to $U$.
This is a generalization of the exact covering problem, where we do not care if some $\alpha \in U$ shows up in multiple sets $V_i$

## Mathematical model

Let $x_i$ be a binary variable that takes on the value 1 if subset $V_i$ is selected, and 0 otherwise. 

**Constraint: each element in $U$ appears in at least one selected subset**

This can be expressed as following using $V$ where it represents a mapping from a subset $i$ to a set of elements $j$ that it contains.
$$
\sum_{i=1}^N x_i \cdot V_{i, j} \geq 1 \text{ for } j = 1, \ldots, M
\tag{1}
$$


## Modeling by JijModeling

Next, we show how to implement above equation using JijModeling. We first define variables for the mathematical model described above.


```python
import jijmodeling as jm

# define variables
U = jm.Placeholder('U')
N = jm.Placeholder('N')
M = jm.Placeholder('M')
V = jm.Placeholder('V', ndim=2)
x = jm.BinaryVar('x', shape=(N,))
i = jm.Element('i', belong_to=N)
j = jm.Element('j', belong_to=M)
```

We use the same variables in the exact cover problem.


```python
# set problem
problem = jm.Problem('Set Cover')
# set constraint: each element j must be in exactly one subset i
problem += jm.Constraint('onehot', jm.sum(i, x[i]*V[i, j]) >= 1, forall=j)
```

We can check the implementation of the mathematical model on the Jupyter Notebook.


```python
problem
```




$$\begin{array}{cccc}\text{Problem:} & \text{Set Cover} & & \\& & \min \quad \displaystyle 0 & \\\text{{s.t.}} & & & \\ & \text{onehot} & \displaystyle \sum_{i = 0}^{N - 1} x_{i} \cdot V_{i, j} \geq 1 & \forall j \in \left\{0,\ldots,M - 1\right\} \\\text{{where}} & & & \\& x & 1\text{-dim binary variable}\\\end{array}$$



## Prepare an instance

We prepare as below.


```python
import numpy as np

# set a list of V
V_1 = [1, 2, 3]
V_2 = [4, 5]
V_3 = [5, 6, 7]
V_4 = [3, 5, 7]
V_5 = [2, 5, 7]
V_6 = [3, 6, 7]

# set the number of Nodes
inst_N = 6
inst_M = 7

# Convert the list of lists into a NumPy array
inst_V = np.zeros((inst_N, inst_M))
for i, subset in enumerate([V_1, V_2, V_3, V_4, V_5, V_6]):
    for j in subset:
        inst_V[i, j-1] = 1  # -1 since element indices start from 1 in the input data

instance_data = {'V': inst_V, 'M': inst_M, 'N': inst_N}
```

## Solve by JijZept's SA

We solve this problem using JijZept `JijSASampler`. We also use the parameter search function by setting `search=True`.


```python
import jijzept as jz

# set sampler
config_path = "./config.toml"
sampler = jz.JijSASampler(config=config_path)
# solve problem
multipliers = {"onehot": 0.5}
results = sampler.sample_model(problem, instance_data, multipliers, num_reads=100, search=True)
```

##  Check the solution

In the end, we extract the solution from the feasible solutions.


```python
# extract feasible solution
feasibles = results.feasible()
feasibles.record.solution['x']
# get the index of the lowest objective function
objectives = np.array(feasibles.evaluation.objective)
lowest_index = np.argmin(objectives)
# # get indices of x = 1
indices, _, _ = feasibles.record.solution['x'][lowest_index]
for i in indices[0]:
    print(f"V_{i+1} = {inst_V[i, :].nonzero()[0]+1}")
```

    V_1 = [1 2 3]
    V_2 = [4 5]
    V_6 = [3 6 7]


With the above calculation, we obtain a the result.
