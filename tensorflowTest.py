import tensorflow as tf
hello = tf.constant('Hello, Tensorflow!')
sess = tf.Session()
print(sess.run(hello))


a = tf.constant(66)
b = tf.constant(88)
print(sess.run(a + b))