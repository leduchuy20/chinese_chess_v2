from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense

NUM_CLASSES = 15

def cnn_model():
	model = Sequential()
	model.add(Convolution2D(
		input_shape=(56, 56, 3),
		filters=32,
		kernel_size=3,
		strides=1,
		padding='same',
		data_format='channels_last',
		activation='relu'
	))
	model.add(MaxPooling2D(
		pool_size=2,
		strides=2,
		padding='same',
		data_format='channels_last',
	))
	### Conv layer 2
	model.add(Convolution2D(32, 3, strides=1, padding='same', data_format='channels_last', activation='relu'))
	model.add(MaxPooling2D(2, 2, padding='same', data_format='channels_last'))

	### Conv layer 3
	model.add(Convolution2D(64, 3, strides=1, padding='same', data_format='channels_last', activation='relu'))
	model.add(MaxPooling2D(2, 2, padding='same', data_format='channels_last'))
	model.add(Dropout(0.25))

	### FC
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(NUM_CLASSES, activation='softmax', name='output'))

	model.compile(optimizer='Adam',
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])

	return model
