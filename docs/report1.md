### Sampling

#### IDS (Inverse Density Sampling)

- Code

```python
def inverse_density_sampling(points, k, sample_num):
    D = batch_distance_matrix(points)
    distances, _ = tf.nn.top_k(-D, k=k, sorted=False)  # 为P个点中的每个点选出k个距离最近的点 (N, P, K)
    distances_avg = tf.abs(tf.reduce_mean(distances, axis=-1)) + 1e-8  # 计算出每个点 到 为其所选出最近K个点的平均距离 (N, P)
    prob_matrix = distances_avg / tf.reduce_sum(distances_avg, axis=-1, keep_dims=True)  # (N, P)
    point_indices = tf.py_func(random_choice_2d, [sample_num, prob_matrix], tf.int32)  # (N, S) 元素值为0-P
    point_indices.set_shape([points.get_shape()[0], sample_num])  # (N, S)

    batch_size = tf.shape(points)[0]
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1)), (1, sample_num, 1))  # (N, S, 1)
    indices = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=2)], axis=2)  # (N, S, 2)
    return indices
```
```python
def random_choice_2d(size, prob_matrix):
    n_row = prob_matrix.shape[0]  # N
    n_col = prob_matrix.shape[1]  # P
    choices = np.ones((n_row, size), dtype=np.int32)  # (N, P) 全1
    for idx_row in range(n_row):
        choices[idx_row] = np.random.choice(n_col, size=size, replace=False, p=prob_matrix[idx_row])
    return choices  # (N, P) 元素值为0-P
```

- Algorithm
此处$P$为上一层的点数, $S$为从$P$中抽样出来的该层的点数
$$
Prob
\leftarrow
\left[
\begin{matrix}
\frac{\sum_{i=1}^{K}d_{1i}}{K}/\sum_{j=1}^P{\frac{\sum_{i=1}^{K}d_{ji}}{K}}\\\\
\frac{\sum_{i=1}^{K}d_{2i}}{K}/\sum_{j=1}^P{\frac{\sum_{i=1}^{K}d_{ji}}{K}}\\\\
...\\\\
\frac{\sum_{i=1}^{K}d_{Pi}}{K}/\sum_{j=1}^P{\frac{\sum_{i=1}^{K}d_{ji}}{K}}
\end{matrix}
\right]
\stackrel{normalize}{\longleftarrow}
\left[
\begin{matrix}
\frac{\sum_{i=1}^{K}d_{1i}}{K}\\\\
\frac{\sum_{i=1}^{K}d_{2i}}{K}\\\\
...\\\\
\frac{\sum_{i=1}^{K}d_{Pi}}{K}
\end{matrix}
\right]
\stackrel{mean}{\longleftarrow}
\left[
\begin{matrix}
d_{11} & d_{12} & ... & d_{1K}\\
d_{21} & d_{22} & ... & d_{2K}\\
... & ... & & ... \\
d_{P1} & d_{P2} & ... & d_{PK}
\end{matrix}
\right] \tag{1}
$$

令$X$为服从分布律$Prob$的随机变量, $\{X(t), t=1,2,...\}$为随机过程

$$
PointIndices
\leftarrow
\left[
\begin{matrix}
X(1)\\
X(2)\\
...\\
X(S)\\
\end{matrix}
\right]
$$
