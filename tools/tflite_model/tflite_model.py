import numpy as np
import tensorflow as tf
import tensorflow.contrib.lite as lite

import tensorflow as tf

#img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
#var = tf.get_variable("weights", dtype=tf.float32, shape=(1, 64, 64, 3))
#val = img + var
#out = tf.identity(val, name="out")

#with tf.Session() as sess:
#  sess.run(tf.global_variables_initializer())
#  converter = tf.contrib.lite.TocoConverter.from_session(sess, [img], [out])
#  tflite_model = converter.convert()
#  open("converted_model.tflite", "wb").write(tflite_model)





x_init  = [2]

x       = tf.placeholder(tf.float32,[1], name = 'x')
y       = tf.placeholder(tf.float32,[1], name = 'y')
z       = tf.constant(2.0)
y       = x * z


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    converter       = lite.TocoConverter.from_session(sess, [x], [y])
    tflite_model    = converter.convert()
    open("hello_world.tflite", "wb").write(tflite_model)

    #y_out           = sess.run(y, { x: x_init } )
    #print (y_out)





