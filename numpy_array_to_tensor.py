import numpy as np
import tensorflow as tf
np_array = np.array([1, 2, 4])
print(np_array)
# [1 2 4]

converted_tensor = tf.convert_to_tensor(np_array)
print(converted_tensor)
# tf.Tensor([1 2 4], shape=(3,), dtype=int32)
