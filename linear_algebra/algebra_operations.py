import tensorflow as tf
import numpy as np

# x_1 = tf.constant([[2, 4, 5], [56, 7, 24], [34, 76, 45]])
# x_2 = tf.constant([[21, 64, 95], [52, 47, 64], [34, 76, 45]])
# tensor_matmul = tf.linalg.matmul(
#     x_1,
#     x_2,
#     transpose_a=False,
#     transpose_b=False,
#     adjoint_a=False,
#     adjoint_b=False,
#     a_is_sparse=False,
#     b_is_sparse=False,
#     output_type=None,
#     name=None
# )
# print(tensor_matmul)
# print(x_1@x_2)
'''
tf.Tensor(
[[  420   696   671]
 [ 2356  5737  6848]
 [ 6196  9168 10119]], shape=(3, 3), dtype=int32)'''

# tensor_band_part = tf.linalg.band_part(
#     [[45, 1, 5], [7, 3, 9], [1, 3, 5]], 0, 1, name=None
# )
# each number indicate how many lines we want to show from centre as diagonal
# 0 0 - diagonal part
# 0 1 upper part
# 1 0 lower part
# print(tensor_band_part)


# det_input = tf.constant([[1.0, 5, 3], [4, 5, 6], [7, 8, 9]])
# DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128
# tensor_det = tf.linalg.det(
#     det_input, name=None
# )
# print(tensor_det)  # tf.Tensor(17.999992, shape=(), dtype=float32)


# tensor_inverse = tf.linalg.inv(
#     det_input, adjoint=False, name=None
# )
# input should be a square matrix to be an inverse
# print(tensor_inverse)
'''
tf.Tensor(
[[-0.16666667 -1.1666673   0.83333373]
 [ 0.3333333  -0.66666687  0.33333346]
 [-0.16666666  1.5000006  -0.83333373]], shape=(3, 3), dtype=float32)'''

# singular value
# s, v, d = tf.linalg.svd(
#     det_input, full_matrices=False, compute_uv=True, name=None
# )
# print(s, end='\n')  # represents singular values
# print(v, end='\n')  # tensor of left singular vector
# print(d, end='\n')  # tensor of right singular vector
'''
tf.Tensor([17.315311   2.4494896  0.424392 ], shape=(3,), dtype=float32)
tf.Tensor(
[[ 0.3142598  -0.94775826 -0.05472769]
 [ 0.5060649   0.11846987  0.8543204 ]
 [ 0.8032058   0.29617438 -0.5168575 ]], shape=(3, 3), dtype=float32)
tf.Tensor(
[[ 0.45976427  0.65292865 -0.6019145 ]
 [ 0.6079746  -0.72547626 -0.32256952]
 [ 0.6472896   0.21764284  0.73051214]], shape=(3, 3), dtype=float32)'''

# a = tf.constant([[1, 2], [3, 4]])
# b = tf.constant([[5, 6], [7, 8]])
# tensor_einsum = tf.einsum(
#     "ij,jk -> ik", a, b
# )
# print(tensor_einsum)

# kinda like lamda function for matrix
# first param describe the equation following up with inputs

'''
tf.Tensor(
[[19 22]
 [43 50]], shape=(2, 2), dtype=int32)'''

input_a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(input_a.T)
# transpose the matrix using numpy not tf
'''
[[ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]
 [ 4  8 12]]'''
