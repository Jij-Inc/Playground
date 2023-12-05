--- 
description: In this section we will show you how to model binary linear programming.
pagination_prev: null
---

# Binary Linear Programming

In this section we will show you how to model binary linear programming
$$
\min_{x} \sum_i c_i x_i\\
\mathrm{s.t.}~\sum_{i}S_{j, i}x_i = b_j,~\forall j\\
x_i \in \{0, 1\}.
$$

## Applications

Linear programming problems with discrete variables, known as 'Mixed integer programming (MIP)', have many applications.
You may be surprised at the wide range of applications even though the objective function and constraints are all linear.
Two applications are listed below, but there are too many applications to list here.

- Capital Budeting
- Warehouse Location

A linear programming solver based on the branch-and-bound method is useful if the size is not that large. Of course, JijModeling supports linear programming solvers.
However, for consistency with other tutorials, we will solve it here using Simulated annealing in JijZept.

## Modeling by JijModeling


```python
import jijmodeling as jm

# set problem
problem = jm.Problem('binary_lp')

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




$$\begin{array}{cccc}\text{Problem:} & \text{binary\_lp} & & \\& & \min \quad \displaystyle \sum_{i = 0}^{N - 1} c_{i} \cdot x_{i} & \\\text{{s.t.}} & & & \\ & \text{eq\_const} & \displaystyle \sum_{i = 0}^{N - 1} S_{j, i} \cdot x_{i} = b_{j} & \forall j \in \left\{0,\ldots,M - 1\right\} \\\text{{where}} & & & \\& x & 1\text{-dim binary variable}\\\end{array}$$



The `len_at` method can be used to override the representation of a formula in the LaTeX display on Jupyter; overriding the `shape` often results in a clean look.

Ex.
``` python
S = jm.Placeholder('S', ndim=2)
M = S.len_at(0, latex="M")
N = S.len_at(1, latex="N")
```

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

JijZept's SA solves the problem using SA after converting it to an unconstrained optimization problem called QUBO. Therefore, the constraints are assigned to the objective function as penalty terms, and we must set their strength.

The strength of the penalty term is passed in the `multipliers` argument in dictionary form, along with the labels of the constraint conditions.

If the `search` option is set to `True`, SA will iterate through the problem and JijZept middleware will adjust the multiplier's strength.


```python
import jijzept as jz

# set sampler
sampler = jz.JijSASampler(config="config.toml")
# solve problem
result = sampler.sample_model(problem, instance_data, multipliers={"eq_const": 1}, search=True)
```

## Check the results

- `result.record`: store the value of solutions
- `result.evaluation`: store the results of evaluation of the solutions.

First, check the results of evaluation.


```python
# Show the result of evaluation of solutions
print("Energy: ", result.evaluation.energy)       # Energy is objective value of QUBO
print("Objective: ", result.evaluation.objective) # Objective values of original constrained problem
print("Constraints violation: ", result.evaluation.constraint_violations)  # violation of constraints
```

    Energy:  [-8.80000019  0.         -9.19999981 -9.19999981 -9.19999981 -9.19999981
     -9.19999981 -8.80000019 -9.19999981 -9.19999981 -9.19999981 -9.19999981
     -9.19999981 -9.19999981 -9.19999981]
    Objective:  [6. 0. 6. 6. 6. 6. 6. 8. 6. 6. 6. 6. 6. 6. 6.]
    Constraints violation:  {'eq_const': array([ 0., 10.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.])}


### Extract feasible solutions and an index of lowest solution


```python
import numpy as np
# Get feasible solution index
feasible = [i for i, violation in enumerate(result.evaluation.constraint_violations["eq_const"]) if violation == 0]

# Get feasible objective
objective = np.array(result.evaluation.objective)
feas_obj = {i: obj_value for i, obj_value in zip(feasible, objective[feasible])}

lowest_index = min(feas_obj, key=feas_obj.get)

print(f"Lowest solution index: {lowest_index}, Lowest objective value: {feas_obj[lowest_index]}")
```

    Lowest solution index: 0, Lowest objective value: 6.0


## Check the solution

Finally, we get the solution from JijZept.


```python
# check solution
nonzero_indices, nonzero_values, shape = result.record.solution["x"][lowest_index]
print("indices: ", nonzero_indices)
print("values: ", nonzero_values)
```

    indices:  ([0, 1, 2],)
    values:  [1.0, 1.0, 1.0]

