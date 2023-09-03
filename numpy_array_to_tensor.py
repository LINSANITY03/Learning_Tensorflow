import numpy as np
import tensorflow as tf
np_array = np.array([1, 2, 4])
print(np_array)
# [1 2 4]

converted_tensor = tf.convert_to_tensor(np_array)
print(converted_tensor)
# tf.Tensor([1 2 4], shape=(3,), dtype=int32)

eye_tensor = tf.eye(
    num_rows=3,
    num_columns=None,  # automatically equals to 3
    batch_shape=None,  # specify how many iteration/subset you want
    # batch leads the list if batch = (2, 4) then output dimension is (2, 4, 3, 3)
    dtype=tf.dtypes.float32,  # data types
    name=None
)
print(eye_tensor)
'''tf.Tensor(
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]], shape=(3, 3), dtype=float32)'''
