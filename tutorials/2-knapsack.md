--- 
description: Here we show how to solve the knapsack problem using JijZept and JijModeling.
---

# Knapsack Problem

Here we show how to solve the knapsack problem using JijZept and [JijModeling](https://www.ref.documentation.jijzept.com/jijmodeling/). 
This problem is also mentioned in 5.2. Knapsack with Integer Weights on [Lucas, 2014, "Ising formulations of many NP problems"](https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full)

## What is the knapsack problem?

The knapsack problem is the problem of finding the optimal solution in the following situation.
Also, it is known as one of the most famous NP-hard integer programming problems. 

### Example

As a concrete example of this problem, we consider the following story: 

> An explorer was exploring a cave. After walking in the cave for a while, he unexpectedly found some treasures. 

||Treasure A|Treasure B|Treasure C|Treasure D|Treasure E|Treasure F|
|-|-|-|-|-|-|-|
|Price|$5000|$7000|$2000|$1000|$4000|$3000|
|weight|800g|1000g|600g|400g|500g|300g|

> Unfortunately, the explorer only had a small knapsack to carry these treasures. This knapsack can only hold 2 kg. The explorer wants the value of the treasures in this knapsack to be as valuable as possible. Which treasures should the explorer choose to bring back most efficiently?

### The knapsack problem

We consider generalization above problem. 
Let $\{ 0, 1, \dots, i, \dots, N-1 \}$ be the set of items to put in the knapsack.
Lists of the cost $\bm{v}$ and weight $\bm{w}$ of each item $i$ allow us to represent the problem.

$$
\bm{v} = \{v_0, v_1, \dots, v_i, \dots, v_{N-1}\}
$$
$$
\bm{w} = \{w_0, w_1, \dots, w_i, \dots, w_{N-1}\}
$$

Furthermore, we define a binary variable $x_i$ that represents the selection of the $i$th item. 
This binary is 1 if we choose $i$th item to put into the knapsack, and 0 otherwise.
Finally, we denote $W$ to be the capacity of the knapsack.  
We want to maximize the total cost of item put into the knapsack.
Therefore, let us express this requirement as an objective function.
In addition, we should take into account the constraint for knapsack capacity limitation.
Finally, the mathematical model of this problem is as follows.


$$
\max \quad \sum_{i=0}^{N-1} v_i x_i \tag{1}
$$
$$
\mathrm{s.t.} \quad \sum_{i=0}^{N-1} w_i x_i \leq W \tag{2}
$$
$$
x_i \in \{0, 1\} \quad (\forall i \in \{0, 1, \dots, N-1\}) \tag{3}
$$

## Modeling by JijModeling

Next, we show an implementation of the above mathematical model in JijModeling. We first define variables for the mathematical model.


```python
import jijmodeling as jm

# define variables
v = jm.Placeholder('v', ndim=1)
N = v.len_at(0, latex="N")
w = jm.Placeholder('w', ndim=1)
W = jm.Placeholder('W')
x = jm.BinaryVar('x', shape=(N,))
i = jm.Element('i', belong_to=(0, N))
```

`v=jm.Placeholder('v', ndim=1)` represents a one-dimensional list of values of items.
The number of items `N` is obtained from the length of `v`.
Using `N`, we can define a one-dimensional list of weights of items as `w=jm.Placeholder('w', ndim=1)`.
Such a definition ensures that `v` and `w` have the same length.
`W = jm.Placeholder('W')` is a scalar $W$ representing the knapsack capacity.
We define a list of binary variables `x` of the same length as `v`, `w` by writing `x=jm.BinaryVar('x', shape=(N,))`.
Finally, `i=jm.Element('i', belong_to=(0, N))` represents the index of $v_i, w_i, x_i$. 
This denotes `i` is an integer in the range $0 \leq i < N$.
`.set_latex` allow us to set the character when it is displayed in Jupyter Notebook.

### Objective function

We implement an objective function Equation (1). 


```python
# set problem
problem = jm.Problem('Knapsack')
# set objective function
obj = - jm.sum(i, v[i]*x[i])
problem += obj
```

We create a problem `problem=jm.Problem('Knapsack')` and add an objective function.
`sum(i, formula)` represents the sum from $i=0$ to $i=N-1$ of formula.

### Constraint 

Next, we implement a constraint Equation (2).


```python
# set total weight constarint
const = jm.sum(i, w[i]*x[i])
problem += jm.Constraint('weight', const<=W)
```

`Constraint('name', formula)` allows us to set the constraint and name it.

Let's display the implemented mathematical model in Jupyter Notebook.


```python
problem
```




$$\begin{array}{cccc}\text{Problem:} & \text{Knapsack} & & \\& & \min \quad \displaystyle - \sum_{i = 0}^{N - 1} v_{i} \cdot x_{i} & \\\text{{s.t.}} & & & \\ & \text{weight} & \displaystyle \sum_{i = 0}^{N - 1} w_{i} \cdot x_{i} \leq W &  \\\text{{where}} & & & \\& x & 1\text{-dim binary variable}\\\end{array}$$



## Prepare an instance



```python
import numpy as np
# set a list of values & weights 
inst_v = np.random.randint(5,30,100)
inst_w = inst_v + np.random.randint(-2,20,100)
# set maximum weight
inst_W = 100
instance_data = {'v': inst_v, 'w': inst_w, 'W': inst_W}    
```

## Solve by JijZept's SA

We solve this problem using JijZept `JijSASampler`. We also use the parameter search function by setting `search=True`.


```python
import jijzept as jz

# set sampler
sampler = jz.JijSASampler(config='config.toml')
# solve problem
results = sampler.sample_model(problem, instance_data, num_reads=100, search=True)
```

## Visualize the solution

In the end, we extract the lowest energy solution among the feasible solutions and visualize it.


```python
feasibles = results.feasible()
objectives = np.array(feasibles.evaluation.objective)
lowest_index = np.argmin(objectives)

indices, _, _ = feasibles.record.solution['x'][lowest_index]
inst_w = instance_data['w']
sum_w = 0
for i in indices[0]:
    sum_w += inst_w[i]
print('Value of chosen items: ', inst_v[indices[0]])
print('Value of weight items: ', inst_w[indices[0]])
print('Total value of items: ', -feasibles.evaluation.objective[lowest_index])
print('Constraint violation: ', feasibles.evaluation.constraint_violations['weight'][lowest_index])
print('Total weight: ', sum_w)
```

    Value of chosen items:  [26  9  7 16 12 17  9 10]
    Value of weight items:  [26  7  6 14 10 18  9 10]
    Total value of items:  106.0
    Constraint violation:  0.0
    Total weight:  100


The kanpsack is well packed as much as possible and the items which are light and valuable are chosen to put in the knapsack.
