import tensorflow as tf

# tensor_indexed = tf.constant([2, 6, 12, 78, 4, 12])
# print(tensor_indexed)
# tf.Tensor([ 2  6 12 78  4 12], shape=(6,), dtype=int32)
# print(tensor_indexed[:3])
# tf.Tensor([ 2  6 12], shape=(3,), dtype=int32)

tensor_two_d = tf.constant([[1, 2, 0], [4, 2, 9], [5, 79, 54], [5, 90, 43]])
print(tensor_two_d[0:3, :])
'''
tf.Tensor(
[[ 1  2  0]
 [ 4  2  9]
 [ 5 79 54]], shape=(3, 3), dtype=int32)'''
