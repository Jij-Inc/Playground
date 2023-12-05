--- 
description: The Travelling Salesman Problem (TSP) is the problem of finding the shortest tour salesman to visit every city.
---

# Traveling Salesman Problem

The Travelling Salesman Problem (TSP) is the problem of finding the shortest tour salesman to visit every city. 
It is known as an NP-hard problem. 
There are several well-known linear formulations for TSP.
However, here we show a quadratic formulation for TSP because it is more suitable for Annealing.

## Mathematical model

Let's consider $n$-city TSP.

### Decision variable

A binary variables $x_{i,t}$ are defined as:
$$
x_{i,t} = \begin{cases}
1~\text{the salesman visits $t$-th city $i$,}\\
0~\text{otherwise}\\
\end{cases}
$$
for all $i \in \{0, ..., n-1\}$ and $t \in \{0, ..., n-1\}$.
$x_{i,t}$ represents the salesman visits city $i$ at $t$ when it is 1.

### Constraints

We have to consider two constraints;

1. A city is visited only once.
$$
\sum_t x_{i, t} = 1, ~\forall i
$$

2. The salesman visits only one city at a time.
$$
\sum_i x_{i, t} = 1, ~\forall t 
$$

The following figure illustrates why these two constraints are needed for $x_{i,t}$ to represent a tour.

![](./assets/tsp_variable.png)

### Objective function

The total distance of the tour, which is an objective function to be minimized, can be written as follows using the product of $x$ representing the edges used between $t$ and $t+1$.

$$
\sum_{i,j,t} d_{i, j}x_{i, t}x_{j, (t+1)\mod n}
$$
where $d_{i, j} \geq 0$ is a distance between city $i$ and city $j$.
"$\mod n$" is used to include in the sum the distance from the last city visited back to the first city visited.


## $n$-city TSP.
$$
\begin{aligned}
\min_x \sum_{i, j} &d_{i,j} \sum_t x_{i,t} x_{j, (t+1) \mod n}\\
\mathrm{s.t.}~&\sum_i x_{i, t} = 1,~\forall t\\
&\sum_t x_{i, t} = 1, ~\forall i\\
&x_{i, t} \in \{0, 1\}
\end{aligned}
$$
where $d_{i,j}$ is distance between city $i$ and city $j$.

## Modeling by JijModeling

Next, we show an implementation using JijModeling. We first define variables for the mathematical model described above.


```python
import jijmodeling as jm

# define variables
d = jm.Placeholder('d', ndim=2)
N = d.len_at(0, latex="N")
i = jm.Element('i', belong_to=(0, N))
j = jm.Element('j', belong_to=(0, N))
t = jm.Element('t', belong_to=(0, N))
x = jm.BinaryVar('x', shape=(N, N))
```

`d` is a two-dimensional array representing the distance between each city.
`N` denotes the number of cities. This can be obtained from the number of rows in `d`.
`i`, `j` and `t` are subscripts used in the mathematical model described above.
Finally, we define the binary variable `x` to be used in this optimization problem.


<!-- `d`は2次元の距離行列で、都市間の距離を表すものです。`N`は都市数で、これは`d`の行数から得ることができます。`i, j, t`は上述の数理モデルで使われている添字です。
最後に、この最適化問題で用いるバイナリ変数`x`を定義しています。 -->


```python
# set problem
problem = jm.Problem('TSP')
problem += jm.sum([i, j], d[i, j] * jm.sum(t, x[i, t]*x[j, (t+1) % N]))
problem += jm.Constraint("one-city", x[:, t].sum() == 1, forall=t)
problem += jm.Constraint("one-time", x[i, :].sum() == 1, forall=i)
```

`jm.sum([i, j], d[i, j] * jm.sum(t, x[i, t]*x[j, (t+1) % N]))` represents objective function $\sum_{i,j,t} d_{i, j}x_{i, t}x_{j, (t+1)\mod n}$. 
The following `jm.Constraint("one-city", x[:, t] == 1, forall=t)` represents a constraint to visit one city at one time.
`x[:, t]` allows us to express $\sum_i x_{i, t}$ concisely.


<!-- `jm.Sum([i, j], d[i, j] * jm.Sum(t, x[i, t]*x[j, (t+1) % N]))`は目的関数$\sum_{i,j,t} d_{i, j}x_{i, t}x_{j, (t+1)\mod n}$を表します。
次の`jm.Constraint("one-city", x[:, t] == 1, forall=t)`は一つの時刻に一つの都市を訪れる制約を表します。`x[:, t]`とすることで$\sum_i x_{i, t}$を簡潔に表現することができます。 -->

We can check the implementation of the mathematical model on the Jupyter Notebook.


```python
problem
```




$$\begin{array}{cccc}\text{Problem:} & \text{TSP} & & \\& & \min \quad \displaystyle \sum_{i = 0}^{N - 1} \sum_{j = 0}^{N - 1} d_{i, j} \cdot \sum_{t = 0}^{N - 1} x_{i, t} \cdot x_{j, \left(t + 1\right) \bmod N} & \\\text{{s.t.}} & & & \\ & \text{one-city} & \displaystyle \sum_{\ast_{0} = 0}^{N - 1} x_{\ast_{0}, t} = 1 & \forall t \in \left\{0,\ldots,N - 1\right\} \\ & \text{one-time} & \displaystyle \sum_{\ast_{1} = 0}^{N - 1} x_{i, \ast_{1}} = 1 & \forall i \in \left\{0,\ldots,N - 1\right\} \\\text{{where}} & & & \\& x & 2\text{-dim binary variable}\\\end{array}$$



## Prepare an instance

We prepare the number of cities and their coordinates. 
Here we select a metropolitan area in Japan using `geocoder`.

<!-- 都市の数と座標を準備しましょう。ここでは日本の首都圏を`geocoder`を用いて選び出します。 -->


```python
import geocoder as gc
import numpy as np

# set the name list of traveling points
points = ['茨城県', '栃木県', '群馬県', '埼玉県', '千葉県', '東京都', '神奈川県']
# get the latitude and longitude
latlng_list = []
for point in points:
    location = gc.osm(point)
    latlng_list.append(location.latlng)
# make distance matrix
num_points = len(points)
inst_d = np.zeros((num_points, num_points))
for i in range(num_points):
    for j in range(num_points):
        a = np.array(latlng_list[i])
        b = np.array(latlng_list[j])
        inst_d[i][j] = np.linalg.norm(a-b)
# normalize distance matrix
inst_d = (inst_d-inst_d.min()) / (inst_d.max()-inst_d.min())

geo_data = {'points': points, 'latlng_list': latlng_list}
instance_data = {'d': inst_d}
```

## Solve by JijZept's SA

We solve this problem using JijZept `JijSASampler`. We also use the parameter search function by setting `search=True`.


```python
import jijzept as jz

# set sampler
sampler = jz.JijSASampler(config="config.toml")
# solve problem
multipliers = {"one-city": 0.5, "one-time": 0.5}
results = sampler.sample_model(problem, instance_data, multipliers, num_reads=100, search=True)
```

## Check the results

- `results.record`: store the value of solutions
- `results.evaluation`: store the results of the evaluation of the solutions.

First, check the results of the evaluation.


```python
# Show the result of evaluation of solutions
max_show_num = 5
print("Energy: ", results.evaluation.energy[:max_show_num])       # Energy is objective value of QUBO
print("Objective: ", results.evaluation.objective[:max_show_num]) # Objective values of original constrained problem
print("one-city violation: ", results.evaluation.constraint_violations["one-city"][:max_show_num])  # violation of constraints
print("one-time violation: ", results.evaluation.constraint_violations["one-time"][:max_show_num])  # violation of constraints
```

    Energy:  [-4.11652422 -4.12254763 -4.12254763 -4.12254763 -4.12254763]
    Objective:  [1.88347592 2.87745219 2.87745219 2.87745219 2.87745219]
    one-city violation:  [1. 0. 0. 0. 0.]
    one-time violation:  [1. 0. 0. 0. 0.]


### Extract feasible solutions and an index of the lowest solution

From the results, we extract the feasible solutions and show the lowest value of the objective function among them.


```python
import numpy as np

feasibles = results.feasible()
objectives = np.array(feasibles.evaluation.objective)
lowest_index = np.argmin(objectives)
print(f"Lowest solution index: {lowest_index}, Lowest objective value: {objectives[lowest_index]}")
```

    Lowest solution index: 0, Lowest objective value: 2.8774521894834977


## Check the solution

Finally, we get the solution from JijZept.


```python
# check solution
nonzero_indices, nonzero_values, shape = feasibles.record.solution["x"][lowest_index]
print("indices: ", nonzero_indices)
print("values: ", nonzero_values)
```

    indices:  ([0, 1, 2, 3, 4, 5, 6], [0, 6, 5, 4, 1, 2, 3])
    values:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


## Draw tour


```python
N_value = len(geo_data['latlng_list'])
tour = np.zeros(N_value, dtype=int)

i_value, t_value = nonzero_indices
tour[t_value] = i_value

tour = np.append(tour, [tour[0]])
tour
```




    array([0, 4, 5, 6, 3, 2, 1, 0])



We check the salesman’s route.


```python
import matplotlib.pyplot as plt

position = np.array(geo_data['latlng_list']).T[[1, 0]]

print(position)

plt.axes().set_aspect('equal')
plt.plot(*position, "o")
plt.plot(position[0][tour], position[1][tour], "-")
plt.show()

print(np.array(geo_data['points'])[tour])
```

    [[140.4703384 139.8096549 139.033483  139.4160114 140.2647303 139.7744912
      139.374753 ]
     [ 36.2869536  36.6782167  36.52198    35.9754168  35.549399   35.6840574
       35.4342935]]



    
![png](3-tsp_files/3-tsp_22_1.png)
    


    ['茨城県' '千葉県' '東京都' '神奈川県' '埼玉県' '群馬県' '栃木県' '茨城県']

