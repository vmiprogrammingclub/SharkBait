import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling3D, MaxPooling1D, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D
from tensorflow.python.keras.callbacks import TensorBoard ReduceLROnPlateau
import numpy as np
import pickle
import time
# from Ext_Of_Sequential import Sequences
from tensorflow import keras
imdb = keras.datasets.imdb

NAME = "movie-words-cnn-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
reduce_lr = ReduceLROnPlateau

def get_x_y(xName, yName):
	# A dictionary mapping words to an integer index
	word_index = imdb.get_word_index()

	# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=100000)

	# # The first indices are reserved
	word_index = {k:(v+3) for k,v in word_index.items()} 
	word_index["<PAD>"] = 0
	word_index["<START>"] = 1
	word_index["<UNK>"] = 2  # unknown
	word_index["<UNUSED>"] = 3


	X_in = pickle.load(open(xName, "rb"))
	y = pickle.load(open(yName, "rb"))

	temp = []
	max_x = 0
	for comment in range(len(X_in)):
		temp.append([])
		for word in range(len(X_in[comment])):
			inFromDic = word_index.get(X_in[comment][word])
			if (not((inFromDic == None))):
				temp[comment].append(int(float(inFromDic)))
				if int(inFromDic) > max_x:
					max_x = int(inFromDic)

	y = [int(float(y[i])) for i in range(len(y)) if temp[i] != []]
	temp = [x for x in temp if x != []]



	X = np.asarray(temp)
	y = np.asarray(y)
	# print(X)

	X = keras.preprocessing.sequence.pad_sequences(X, value=word_index["<PAD>"], padding='post', maxlen=1024)



	return (X, y, max_x)

def new_model(max_x):
	nodes_per_layer = 256

	# X = X/255.0

	model = Sequential()

	# input shape is the vocabulary count used for the movie reviews (10,000 words)
	vocab_size = max_x+1


	model = keras.Sequential()
	model.add(keras.layers.Embedding(vocab_size, nodes_per_layer))
	model.add(keras.layers.GlobalMaxPooling1D())
	model.add(keras.layers.Dense(nodes_per_layer, activation=tf.nn.relu)) # to draw comparisons from one word to eachother
	model.add(keras.layers.Dense(nodes_per_layer, activation=tf.nn.relu)) # to draw comparisons in sentences
	model.add(keras.layers.Dense(nodes_per_layer, activation=tf.nn.sigmoid)) # geometry of words in sentences
	model.add(keras.layers.Dense(nodes_per_layer, activation=tf.nn.sigmoid))
	model.add(keras.layers.Dense(nodes_per_layer, activation=tf.nn.sigmoid))
	model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid)) # output layer


	model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])


	model.summary()
	return model

def train_model(X, y, model):

	# loss='binary_crossentropy',



	# x_val = train_data[:10000]
	# partial_x_train = X[10000:]

	# y_val = train_labels[:10000]
	# partial_y_train = y[10000:]

	print(model)
	history = model.fit(X, y, epochs=35, shuffle = True, batch_size=64, validation_data=(X, y), verbose=1, callbacks=[tensorboard, reduce_lr])


	# results = model.evaluate(X, y)

	# print(results)

	return model

def save(model):
	 model.save('happy_feet.h5')
# # Save entire model to a HDF5 file
# model.save('my_model.h5')

def load():
	return keras.models.load_model('happy_feet.h5')
# # Recreate the exact same model, including weights and optimizer.
# new_model = keras.models.load_model('my_model.h5')
def test():
	model = load()
	(X, y, max_x) = get_x_y("X_test.pickle", "y_test.pickle")
	results = model.evaluate(X, y, verbose=1, batch_size=32)
	print(results)

# results = model.evaluate(test_data, test_labels)
# print(results)


def train():
	(X, y, max_x) = get_x_y("X_train.pickle", "y_train.pickle")
	model = new_model(max_x)
	fit_model = train_model(X, y, model)
	save(fit_model)



def main():
	# # for training
	# train()
	#
	# # for testing
	test()

main()
