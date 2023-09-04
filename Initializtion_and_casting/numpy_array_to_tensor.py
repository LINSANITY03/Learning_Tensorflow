import numpy as np
import tensorflow as tf

# np_array = np.array([1, 2, 4])
# print(np_array)
# [1 2 4]

# converted_tensor = tf.convert_to_tensor(np_array)
# print(converted_tensor)
# tf.Tensor([1 2 4], shape=(3,), dtype=int32)

# eye_tensor = tf.eye(
#     num_rows=3,
#     num_columns=None,  # automatically equals to 3
#     batch_shape=None,  # specify how many iteration/subset you want
#     # batch leads the list if batch = (2, 4) then output dimension is (2, 4, 3, 3)
#     dtype=tf.dtypes.float32,  # data types
#     name=None
# )
# print(eye_tensor)
'''tf.Tensor(
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]], shape=(3, 3), dtype=float32)'''

# fill_tensor = tf.fill(
#     [2, 3], 9, name=None
# )
# print(fill_tensor)
# it automatically creates tensor object of required dimensional of a certain value
'''
tf.Tensor(
[[9 9 9]
 [9 9 9]], shape=(2, 3), dtype=int32)'''

# ones_tensor = tf.ones([3, 4], tf.int32)
# print(ones_tensor)
# similar to tf.fill but with exact value of 1
'''
tf.Tensor(
[[1 1 1 1]
 [1 1 1 1]
 [1 1 1 1]], shape=(3, 4), dtype=int32)'''

# ones_like_tensor = tf.ones_like(
#     ones_tensor, dtype=None, name=None
# )
# takes the input's dimension and creates a one's tf object
# print(ones_like_tensor)
'''tf.Tensor(
[[1 1 1 1]
 [1 1 1 1]
 [1 1 1 1]], shape=(3, 4), dtype=int32)'''


# rank_tensor = tf.constant([[[1, 2, 0], [3, 5, -1]],
#                            [[10, 2, 0], [1, 0, 2]],
#                            [[5.11, 8.32, 1.2], [2, 7, 0]]])
# print(tf.rank(rank_tensor))
# tf.Tensor(3, shape=(), dtype=int32) rank is 3

# size_tensor = tf.size(
#     rank_tensor,
#     out_type=tf.dtypes.float16,
#     name=None
# )
# print(size_tensor)
# tf.Tensor(18, shape=(), dtype=int32)
# get the number of elements of out_type

# random_tensor = tf.random.normal(
#     [3, 2],
#     mean=10.0,
#     stddev=2.0,
#     dtype=tf.dtypes.float32,
#     seed=None,
#     name=None
# )
# print(random_tensor)
# this function return a dimensional of given shape
# which have random value having average of 10.0 and stddev of 2
# meaning any value is between 8 and 12

'''
tf.Tensor(
[[ 8.114124 11.638804]
 [ 8.868149  8.113518]
 [ 8.33167  14.455563]], shape=(3, 2), dtype=float32)'''

# uniform_tensor = tf.random.uniform(
#     [3, 2],
#     minval=2,
#     maxval=5,
#     dtype=tf.dtypes.float32,
#     seed=None,
#     name=None
# )
# print(uniform_tensor)

# returns a shape with random value ranging from minval and maxval
'''
[0.5763211  0.55317414]
 [0.94574106 0.38361692]
 [0.4886359  0.18659806]], shape=(3, 2), dtype=float32)'''
