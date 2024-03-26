--- 
description: In this section we will show you how to model binary linear programming.
pagination_prev: null
---

# Binary Linear Programming

In this section we will show you how to model binary linear programming
$$
\max_{x} \sum_i c_i x_i\\
\mathrm{s.t.}~\sum_{i}S_{j, i}x_i = b_j,~\forall j\\
x_i \in \{0, 1\}.
$$

## Applications

Linear programming problems with discrete variables, known as 'Mixed integer programming (MIP)', have many applications.
You may be surprised at the wide range of applications even though the objective function and constraints are all linear.
Two applications are listed below, but there are too many applications to list here.

- Capital Budgeting
- Warehouse Location

A linear programming solver based on the branch-and-bound method is useful if the size is not that large. Of course, JijModeling supports linear programming solvers.
However, for consistency with other tutorials, we will solve it here using Simulated annealing in JijZept.

## Modeling by JijModeling


```python
import jijmodeling as jm

# set problem
problem = jm.Problem('binary_lp', sense=jm.ProblemSense.MAXIMIZE)

# define variables
S = jm.Placeholder('S', ndim=2)
M = S.len_at(0, latex="M")
N = S.len_at(1, latex="N")
b = jm.Placeholder('b', ndim=1)
c = jm.Placeholder('c', ndim=1)
x = jm.BinaryVar('x', shape=(N,))
i = jm.Element('i', belong_to=(0, N))
j = jm.Element('j', belong_to=(0, M))


# Objective
problem += jm.sum(i, c[i]*x[i])

# Constriants
problem += jm.Constraint("eq_const", jm.sum(i, S[j, i] * x[i]) == b[j], forall=j)

problem
```




$$\begin{array}{cccc}\text{Problem:} & \text{binary\_lp} & & \\& & \max \quad \displaystyle \sum_{i = 0}^{N - 1} c_{i} \cdot x_{i} & \\\text{{s.t.}} & & & \\ & \text{eq\_const} & \displaystyle \sum_{i = 0}^{N - 1} S_{j, i} \cdot x_{i} = b_{j} & \forall j \in \left\{0,\ldots,M - 1\right\} \\\text{{where}} & & & \\& x & 1\text{-dim binary variable}\\\end{array}$$



The meaning of `Problem(..., sense=jm.ProblemSense.MAXIMIZE)` is to explicitly state that the optimization problem is to be solved by maximizing the objective function.
If `sense` is not specified, the default is to solve the problem by minimizing the objective function.

:::info
The `len_at` method can be used to override the representation of a formula in the LaTeX display on Jupyter; overriding the `shape` often results in a clean look.

e.g. 

``` python
S = jm.Placeholder('S', ndim=2)
M = S.len_at(0, latex="M")
N = S.len_at(1, latex="N")
```
:::

## Prepare an instance


```python
# set S matrix
inst_S = [[0, 2, 0, 2, 0], [1, 0, 1, 0, 1], [1, 2, 3, 2, 1]]
# set b vector
inst_b = [2, 2, 6]
# set c vector
inst_c = [1, 2, 3, 4, 5]
instance_data = {'S': inst_S, 'b': inst_b, 'c': inst_c}
```

$$S = \left( \begin{array}{ccccc}
0 & 2 & 0 & 2 & 0 \\
1 & 0 & 1 & 0 & 1 \\
1 & 2 & 3 & 2 & 1 
\end{array}\right), \quad 
\mathbf{b} = \left( \begin{array}{c}
2 \\
2 \\
6 
\end{array}\right), \quad 
\mathbf{c} = \left( \begin{array}{c}
1 \\
2 \\
3 \\
4 \\
5 
\end{array}\right)$$

:::info  
Be careful with variable names and scopes.
Variable names such as `S`, `b`, and `c` are used when modeling with JijModeling and cannot be used when preparing instances. To avoid this problem, we use the prefix `inst_`.
:::

## Solve by JijZept's SA

JijZept's SA solves the problem using SA after converting it to a quadratic unconstrained binary optimization problem called QUBO. Therefore, the constraints are assigned to the objective function as penalty terms, and we must set their strength.
The strength of the penalty term is passed in the `multipliers` argument in dictionary form, along with the labels of the constraint conditions.
If the `search` option is set to `True`, SA will iterate through the problem and JijZept middleware will adjust the multiplier's strength.


```python
import jijzept as jz

# set sampler
sampler = jz.JijSASampler(config="config.toml")
# solve problem
response = sampler.sample_model(problem, instance_data, multipliers={"eq_const": 1}, search=True)
```

## Check the results

`response.get_sampleset()` returns the `SampleSet` object which has solutions and information obtained from a solver.
The data of `SampleSet` contains an array of `Sample`, each of which contains a solution obtained from a solver and its associated information.
`eval` has information related to the evaluation of the solution, such as the value of an objective function and a degree of constraint violation.
`eval.objective` contains a value of an objective function of a solution and `eval.constraint` contains a dictionary whose key is the name of the constraint and whose value is the `Violation` of the constraint.
First, we check the evalutation of `SampleSet`.


```python
# get sampleset
sampleset = response.get_sampleset()
# extract the values of objective function and constraint violation
objectives = [-sample.eval.objective for sample in sampleset]
violations = [sample.eval.constraints["eq_const"].total_violation for sample in sampleset]
# show these results
print("Objective values: ", objectives)
print("Constraint vilations: ", violations)
```

    Objective values:  [12.0, 15.0, 12.0, 12.0, 12.0, 12.0, 12.0, 10.0, 12.0, 12.0, 12.0, 12.0, 10.0, 10.0, 12.0]
    Constraint vilations:  [0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


### Extract feasible solutions and an index of lowest solution

Next, we obtain feasible solutions using `feasibles` method and extract the index with the lowest value of the objective function from them. 


```python
import numpy as np

# get feasible solutions
feasible_samples = sampleset.feasibles()
# get valuse of feasible objectives
feasible_objectives = [-sample.eval.objective for sample in feasible_samples]
# get highest value index
highest_index = np.argmax(feasible_objectives)
print(f"highest solution index: {highest_index}, highest objective value: {feasible_objectives[highest_index]}")
```

    highest solution index: 0, highest objective value: 12.0


## Check the solution

Finally, we get the solution from JijZept.


```python
# check solution
highest_solution = feasible_samples[highest_index].to_dense()
print(highest_solution)
```

    {'x': array([0., 0., 1., 1., 1.])}


Using `to_dense`, we obtain the decision variable as an ordinary NumPy array. 


```python
print(feasible_samples[highest_index].var_values)
```

    {'x': SparseVarValues(name="x", values={(2,): 1, (3,): 1, (4,): 1}, shape=(5,), var_type=VarType.CONTINUOUS)}


Using `var_values`, we can get a dictionary whose key is the name of decision variable and whose value is the information of the decision variable `SparseVarValues`.
`SparseVarValues.valuse` property contains the value of each index of the decision variable.


