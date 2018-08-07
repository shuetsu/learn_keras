import numpy as np

def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z

def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z

def naive_matrix_vector_dot2(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y)
    return z

def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            col_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, col_y)
    return z

x1 = np.array([1, 2])
y1 = np.array([3, 4])
print(naive_vector_dot(x1, y1))

x2 = np.array([[1, 2], [3, 4]])
y2 = np.array([10, 100])
print(naive_matrix_vector_dot(x2, y2))

x2 = np.array([[1, 2], [3, 4]])
y2 = np.array([10, 100])
print(naive_matrix_vector_dot2(x2, y2))

x3 = np.array([[1, 2, 3],[4, 5, 6]])
y3 = np.array([[1, 2], [3, 4], [5, 6]])
print(naive_matrix_dot(x3, y3))

x4 = np.array([[1, 2], [3, 4], [5, 6]])
y4 = np.array([[1, 2, 3],[4, 5, 6]])
print(naive_matrix_dot(x4, y4))
