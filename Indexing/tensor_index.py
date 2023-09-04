import tensorflow as tf

# tensor_indexed = tf.constant([2, 6, 12, 78, 4, 12])
# print(tensor_indexed)
# tf.Tensor([ 2  6 12 78  4 12], shape=(6,), dtype=int32)
# print(tensor_indexed[:3])
# tf.Tensor([ 2  6 12], shape=(3,), dtype=int32)

# tensor_two_d = tf.constant([[1, 2, 0], [4, 2, 9], [5, 79, 54], [5, 90, 43]])
# print(tensor_two_d[0:3, :])
'''
tf.Tensor(
[[ 1  2  0]
 [ 4  2  9]
 [ 5 79 54]], shape=(3, 3), dtype=int32)'''

# three_d_input = [[[1, 2, 0], [3, 5, -1]],
#                  [[10, 2, 0], [1, 0, 2]],
#                  [[5, 8, 0], [2, 7, 0]]]
# tensor_three_d = tf.constant(three_d_input)
# print(tensor_three_d[0:2, 0:2, 2])
'''
tf.Tensor(
[[ 0 -1]
 [ 0  2]], shape=(2, 2), dtype=int32)'''

# x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 45, 10000, float("inf")])
# print(tf.math.tan(x))
'''tf.Tensor(
[        nan  0.45231566 -0.5463025   1.5574077   2.572152    1.6197752 
  0.32097113         nan], shape=(8,), dtype=float32)'''

# x = tf.constant([1, 2, 3, 4])
# y = tf.constant([[7], [5], [4]])
# result = tf.math.multiply(x, y)
# print(result)
'''tf.Tensor(
[[ 7 14 21 28]
 [ 5 10 15 20]
 [ 4  8 12 16]], shape=(3, 4), dtype=int32)'''

# x = tf.constant([0., 0., 0., 0.])
# y = tf.constant([-5., -2., 0., 3.])
# result = tf.math.minimum(x, y)
# print(result)
# tf.Tensor([-5. -2.  0.  0.], shape=(4,), dtype=float32)

# A = tf.constant([2, 20, 30, 3, 6])
# print(tf.math.argmax(A))  # A[2] is maximum in tensor A
# tf.Tensor(2, shape=(), dtype=int64) with 2 as indices

# tensor_power = tf.math.pow(
#     [2, 3], [6, 7], name=None
# )
# print(tensor_power)
# tf.Tensor([  64 2187], shape=(2,), dtype=int32)

# x = tf.constant([[1, 1, 1], [1, 1, 1]])
# reducing_sum = tf.reduce_sum(x)
# print(reducing_sum)  # tf.Tensor(6, shape=(), dtype=int32)
# print(reducing_sum.numpy())  # 6

tensor_top_k = tf.math.top_k(
    [[1, 4, 434], [90, 78, 23]],
    k=1,
    sorted=True,
    index_type=tf.dtypes.int32,
    name=None
)
print(tensor_top_k)
# returns two output 1. output number 2. number's index
'''
TopKV2(values=<tf.Tensor: shape=(2, 1), dtype=int32, numpy=
array([[434],
       [ 90]])>, indices=<tf.Tensor: shape=(2, 1), dtype=int32, numpy=  
array([[2],
       [0]])>)'''
