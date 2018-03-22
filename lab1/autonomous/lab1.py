from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import keras
from keras.datasets import reuters
from keras.layers import Dense, Dropout, Activation
from keras.models import model_from_json
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
from sklearn.metrics import classification_report,confusion_matrix

# Basic config
max_words = 1000
batch_size = 32
epochs = 5

print 'Using Keras version', keras.__version__

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
	print 'Test accuracy:', score[1]

	#Accuracy plot
	plt.plot(history.history['acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper left')
	plt.savefig('results/model_accuracy_' + optimizer +'.png')
	plt.close()
	#Loss plot
	plt.plot(history.history['loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper left')
	plt.savefig('results/model_loss_' + optimizer + '.png')

	#plot_model(model, to_file='results/model_' + optimizer + '.png')

	f = open('results/results.txt', 'a+')
	f.write('Test score {0}: {1}'.format(optimizer, score[0]))
	f.write('Test accuracy {0}: {1} \n'.format(optimizer, score[1]))
	f.close()
	
	return

def main():

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

	(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
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
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	conf = ['sgd','adam','rmsprop','adagrad','adamax','adadelta','nadam']

	for o in conf:
		calculateModel(model, x_train, x_test, y_train, y_test, o, epochs)


if __name__ == '__main__':
	main()