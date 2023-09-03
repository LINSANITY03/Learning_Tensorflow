import tensorflow as tf

tensor_zero_d = tf.constant(4)
print(tensor_zero_d)

tensor_one_d = tf.constant([2, 0, -11, 49.012, 5])
casted_tensor_one_d = tf.cast(tensor_one_d, tf.int16)
# tf.Tensor([  2.   0. -11.  49.012   5.], shape=(5,), dtype=float32)
print(tensor_one_d)

# tf.Tensor([  2   0 -11  49   5], shape=(5,), dtype=int16)
print(casted_tensor_one_d)

tensor_two_d = tf.constant([
    [1, 2, 0], [3, 5, -1], [1, 5, 6]
])
print(tensor_two_d)
'''
tf.Tensor(
[[ 1  2  0]
 [ 3  5 -1]
 [ 1  5  6]], shape=(3, 3), dtype=int32)
'''

three_d_input = [[[1, 2, 0], [3, 5, -1]],
                 [[10, 2, 0], [1, 0, 2]],
                 [[5, 8, 0], [2, 7, 0]]]
tensor_three_d = tf.constant(three_d_input)
print(tensor_three_d)

'''
tf.Tensor(
[[[ 1  2  0]
  [ 3  5 -1]]

 [[10  2  0]
  [ 1  0  2]]

 [[ 5  8  0]
  [ 2  7  0]]], shape=(3, 2, 3), dtype=int32)
  '''
four_d_input = [[[[1, 2, 0], [3, 5, -1]],
                 [[10, 2, 0], [1, 0, 2]],
                 [[5, 8, 0], [2, 7, 0]]],
                [[[1, 2, 0], [3, 5, -1]],
                 [[10, 2, 0], [1, 0, 2]],
                 [[5, 8, 0], [2, 7, 0]]]
                ]
tensor_four_d = tf.constant(four_d_input)
print(tensor_four_d)
'''
tf.Tensor(
[[[[ 1  2  0]
   [ 3  5 -1]]

  [[10  2  0]
   [ 1  0  2]]

  [[ 5  8  0]
   [ 2  7  0]]]


 [[[ 1  2  0]
   [ 3  5 -1]]

  [[10  2  0]
   [ 1  0  2]]

  [[ 5  8  0]
   [ 2  7  0]]]], shape=(2, 3, 2, 3), dtype=int32)'''
