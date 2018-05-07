#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf


# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

losses = []
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
learning_rate = 1.0
beta1=0.9
beta2 = 0.999
epsilon=1e-08
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong


#curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
#print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

#losses.append(curr_loss)
#iterations.append(0)

threshold = 10
curr = 0
prev_loss = 0
lowest = 1000
it_found = 0

for i in range(5000):
  sess.run(train, {x: x_train, y: y_train})
  curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
  losses.append(curr_loss)
  print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
  if curr_loss<lowest:
  	lowest = curr_loss
  	it_found=i
  if prev_loss == curr_loss:
  	curr+=1
  if curr == threshold:
  	print('Optimium found at %s'%(i-threshold))
  	break
  prev_loss = curr_loss

print("Lowest: %s on iteration: %s"%(lowest, it_found))
plt.plot(losses)
plt.savefig('adam-%s-%s-%s.png'%(learning_rate,beta1, beta2))
