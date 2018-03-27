from __future__ import division

import argparse

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

import keras
from keras import activations
from keras.datasets import reuters
from keras.layers import Dense, Dropout, Activation
from keras.models import model_from_json
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
from sklearn.metrics import classification_report,confusion_matrix

from autograd import elementwise_grad, value_and_grad
from scipy.optimize import minimize
from collections import defaultdict
from itertools import izip_longest
from functools import partial

import tensorflow as tf

from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons


# Basic config
max_words = 1000
batch_size = 32
epochs = 5
results_history = []
results_score = []
results_acc = []
path_ = []
line = None
point = None
f = None

print 'Using Keras version', keras.__version__

np.random.seed(1)

parser = argparse.ArgumentParser(
	description='Uses the Reuters dataset to compare the results of different optimization functions')
	
parser.add_argument('-w', '--words', action='store_true', default=1000,
	help='max words to use (default: 1000)')
parser.add_argument('-b', '--batch_size', action='store_true', default=32,
	help='size of batches to use (default: 32)')
parser.add_argument('-e', '--epochs', action='store_true', default=5,
	help='amount of epochs to use (default: 5)')
parser.add_argument('-f', '--test_function', action='store_true', default=False,
	help='visualize optimizations using test functions (default: false)')


args = parser.parse_args()

def calculateModel(model, x_train, x_test, y_train, y_test, optimizer, epochs) :

	print 'Using optimizer: ', optimizer

	model.compile(loss='categorical_crossentropy',
	              optimizer=optimizer,
	              metrics=['accuracy'])

	history = model.fit(x_train, y_train,
	                    batch_size=batch_size,
	                    epochs=epochs,
	                    verbose=1,
	                    validation_split=0.1)
	score = model.evaluate(x_test, y_test,
	                       batch_size=batch_size, verbose=1)

	print 'Test score:', score[0]
	print ' Test accuracy:', score[1]

	results_score.append(score[0])
	results_acc.append(score[1])
	results_history.append(history)

	#vis(model, x_train, x_test, y_train, y_test, optimizer)

	#Accuracy plot
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig('results/model_accuracy_' + optimizer +'.png')
	plt.close()
	#Loss plot
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig('results/model_loss_' + optimizer + '.png')
	plt.close()
	#plot_model(model, to_file='results/model_' + optimizer + '.png')

	f = open('results/results.txt', 'a+')
	f.write('Test score {0}: {1}'.format(optimizer, score[0]))
	f.write('Test accuracy {0}: {1} \n'.format(optimizer, score[1]))
	f.close()
	
	return

def vis(model, x_train, x_test, y_train, y_test, optimizer):

	class_idx = 0
	indices = np.where(y_test[:, class_idx] == 1.)[0]

	# pick some random input from here.
	idx = indices[0]
	# Utility to search for layer index by name. 
	# Alternatively we can specify this as -1 since it corresponds to the last layer.
	layer_idx = utils.find_layer_idx(model, 'preds')

	# Swap softmax with linear
	model.layers[layer_idx].activation = activations.linear
	model = utils.apply_modifications(model)

	grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, seed_input=x_test[idx])
	# Plot with 'jet' colormap to visualize as a heatmap.
	
	plt.savefig('results/vis1_' + optimizer + '.png')

	for class_idx in np.arange(10):    
		indices = np.where(y_test[:, class_idx] == 1.)[0]
		idx = indices[0]

		f, ax = plt.subplots(1, 4)
		ax[0].imshow(x_test[idx][..., 0])
		ax[0].savefig('results/vis2_' + optimizer + '.png')

		for i, modifier in enumerate([None, 'guided', 'relu']):
			grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, 
						seed_input=x_test[idx], backprop_modifier=modifier)
			if modifier is None:
				modifier = 'vanilla'
			ax[i+1].set_title(modifier)    
			ax[i+1].imshow(grads, cmap='jet')
			ax[i+1].savefig('results/vis3_' + optimizer + '.png')
	return

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    return line, point

def animate(i):
    line.set_data(path[0,:i], path[1,:i])
    line.set_3d_properties(f(*path[::,:i]))
    point.set_data(path[0,i-1:i], path[1,i-1:i])
    point.set_3d_properties(f(*path[::,i-1:i]))
    return line, point

def make_minimize_cb(path=[]):
    
    def minimize_cb(xk):
        # note that we make a deep copy of xk
        path.append(np.copy(xk))

    return minimize_cb

def visOptimizations(conf):
	bealeFunction(conf)
	return

def bealeFunction(conf): 
	global line, point, path, f

	f  = lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

	xmin, xmax, xstep = -4.5, 4.5, .2
	ymin, ymax, ystep = -4.5, 4.5, .2
	x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
	z = f(x, y)
	minima = np.array([3., .5])
	minima_ = minima.reshape(-1, 1)

	x0 = np.array([3., 4.])
	func = value_and_grad(lambda args: f(*args))

	path_ = [x0]

	res = minimize(func, x0=x0, method='Newton-CG',
               jac=True, tol=1e-20, callback=make_minimize_cb(path_))


	path = np.array(path_).T

	#3D surface plot
	fig = plt.figure(figsize=(8, 5))
	ax = plt.axes(projection='3d', elev=50, azim=-50)

	ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)
	ax.plot(minima_[0], minima_[1], f(minima_[0], minima_[1]), 'r*', markersize=10)

	line, = ax.plot([], [], [], 'b', label='Newton-CG', lw=2)
	point, = ax.plot([], [], [], 'bo')

	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')
	ax.set_zlabel('$z$')

	ax.set_xlim((xmin, xmax)) 
	ax.set_ylim((ymin, ymax))

	anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=path.shape[1], interval=60, 
                               repeat_delay=5, blit=True)
	anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

	return

def comPlot(results_history, results_score, results_acc, conf): 
	for history in results_history:
		plt.plot(history.history['acc'])
	plt.legend(['sgd','adam','rmsprop','adagrad','adamax','adadelta','nadam'], loc='upper left')
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.savefig('results/total_model_accuracy.png')
	plt.close()

	x = [0,1,2,3,4,5,6]

	plt.scatter(x, results_score)
	for i,c in enumerate(conf):
		plt.annotate(c, (x[i], results_score[i]))

	plt.title('model score')
	plt.savefig('results/total_opt_score.png')
	plt.close()

	plt.scatter(x, results_acc)
	for i,c in enumerate(conf):
		plt.annotate(c, (x[i], results_acc[i]))

	plt.title('optimization accuracy')
	plt.savefig('results/total_opt_acc.png')
	plt.close()
	return


def plot_loss_contour(trX, trY, ngrid):
    X = tf.placeholder(tf.float32, [None, nx])
    Y = tf.placeholder(tf.float32, [None, ny])
    W = tf.placeholder(tf.float32, [nx, ny])

    w = tf.Variable(np.random.randn(2, 1))
    y = tf.matmul(X, W)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=Y)
    loss = tf.reduce_mean(loss)

    wx, wy = np.meshgrid(np.linspace(-2, 5, ngrid), np.linspace(-5, 2, ngrid))
    ws = np.stack([wx.flatten(), wy.flatten()], 1)[:, :, np.newaxis]

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=4,
                            inter_op_parallelism_threads=4)
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        zs = []
        for w in tqdm(ws):
            z = sess.run(loss, {X:trX, Y:trY, W:w})
            zs.append(z)
        zs = np.asarray(zs).reshape(ngrid, ngrid)
    plt.contour(wx, wy, zs, 40)
    best_ws = ws[np.argmin(zs.flatten())]
    plt.scatter(best_ws[0], best_ws[1])

def w_trajectory(trX, trY, opt, nbatch, niter):
    nx = trX.shape[1]
    ny = trY.shape[1]
    X = tf.placeholder(tf.float32, [None, nx])
    Y = tf.placeholder(tf.float32, [None, ny])

    wnp = np.asarray([[-0.5], [1.5]], dtype=np.float32)
    w = tf.Variable(wnp)
    y = tf.matmul(X, w)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=Y)
    loss = tf.reduce_mean(loss)
    train = opt.minimize(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        ws = []
        for i in range(niter):
            batch_idxs = np.random.permutation(np.arange(ntrain))[:nbatch]
            iterate_w, _ = sess.run([w, train], {X:trX[batch_idxs], Y:trY[batch_idxs]})
            ws.append(iterate_w)
        ws = np.asarray(ws)
    return ws


def main():

	max_words = args.words
	epochs = args.epochs
	batch_size = args.batch_size
	# Loads reuters dataset 
	# Dataset of 11,228 newswires from Reuters, labeled over 46 topics.
	# Each wire is encoded as a sequence of word indexes.
	# For convenience, words are indexed by overall frequency in the dataset,
	# so that for instance the integer "3" encodes the 3rd most frequent word
	# in the data. This allows for quick filtering operations such as: 
	# "only consider the top 10,000 most common words, but eliminate the top 20
	# most common words".
	# As a convention, "0" does not stand for a specific word, but instead is
	# used to encode any unknown word.
	# https://keras.io/datasets/
	# https://keras.io/preprocessing/sequence/

	(x_train, y_train), (x_test, y_test)  = reuters.load_data(path="reuters.npz",
		num_words=None,skip_top=0,maxlen=None,test_split=0.2,seed=113,start_char=1,
		oov_char=2,index_from=3)

	# Check sizes of dataset
	# Datasets are a sequence of words that are indexed, there for it should be
	# of only one dimension
	print 'Number of train examples', x_train.shape[0]
	print 'Size of train examples', x_train.shape[1:]

	num_classes = np.max(y_train) + 1
	print num_classes, 'classes'

	# Tokenizer: Class for vectorizing texts, or/and turning texts into
	# sequences (=list of word indexes, where the word of rank i in the
	# dataset (starting at 1) has index i).
	# num_words: None or int. Maximum number of words to work with 
	#(if set, tokenization will be restricted to the top num_words most
	# common words in the dataset).

	# We need to vectorize the data to be able to send it to the 
	# Neural Network
	print 'Vectorizing sequence data...'
	tokenizer = Tokenizer(num_words=max_words)

	# Return: numpy array of shape (len(sequences), num_words).
	# Since it is set to binary mode, returns a matrix that specifies
	# if the sequence (x) has a specific word (y) (1 or 0). Could be 
	# set to count (how many times the word is repeated), tfidf 
	# (term frequency inverse document frequency reflects how 
	# important the word is to the sequence) and freq (frecuency 
	# of the word)
	x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
	x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

	print 'x_train shape:', x_train.shape
	print 'x_test shape:', x_test.shape

	print'Convert class vector to binary class matrix (for use with categorical_crossentropy)'
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	print 'y_train shape:', y_train.shape 
	print 'y_test shape:', y_test.shape

	print 'Building model...'
	model = Sequential()
	model.add(Dense(512, input_shape=(max_words,)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax', name='preds'))
	#model.add(Activation('softmax'))

	conf = ['sgd','adam','rmsprop','adagrad','adamax','adadelta','nadam']

	#for o in conf:
		#calculateModel(model, x_train, x_test, y_train, y_test, o, epochs)

	#comPlot(results_history, results_score, results_acc, conf)

	#if args.test_function:
	visOptimizations(conf)


if __name__ == '__main__':
	
	seed = 42
	ntrain = 1000
	nbatch = 16
	np.random.seed(seed)
	trX, trY = make_moons(n_samples=ntrain, shuffle=True, noise=0.3, random_state=seed)
	trX = trX.astype(np.float32)
	trY = trY.astype(np.float32)[:, np.newaxis]
	nx = trX.shape[1]
	ny = trY.shape[1]

	plot_loss_contour(trX, trY, ngrid=101)

	sgd = tf.train.GradientDescentOptimizer(learning_rate=0.5)
	sgd_ws = w_trajectory(trX, trY, sgd, nbatch=16, niter=200)
    
	mom = tf.train.MomentumOptimizer(learning_rate=0.075, momentum=0.94)
	mom_ws = w_trajectory(trX, trY, mom, nbatch=16, niter=200)
    
	plt.plot(sgd_ws[:, 0, 0], sgd_ws[:, 1, 0], label='sgd')
	plt.plot(mom_ws[:, 0, 0], mom_ws[:, 1, 0], label='momentum')
	plt.legend()
	plt.ylim([-5, 2])
	plt.xlim([-2, 5])
	plt.savefig('test')
	main()