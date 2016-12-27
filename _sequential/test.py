import numpy as np
from chainer import Variable
from chainer import functions as F
from sequential import Sequential
import layers
import functions
import util
import chain

# Linear test
x = np.random.normal(scale=1, size=(2, 28*28)).astype(np.float32)
x = Variable(x)

seq = Sequential(weight_initializer="GlorotNormal", weight_init_std=0.05)
seq.add(layers.Linear(28*28, 500))
seq.add(layers.BatchNormalization(500))
seq.add(layers.Linear(500, 500))
seq.add(functions.Activation("clipped_relu"))
seq.add(layers.Linear(500, 500, use_weightnorm=True))
seq.add(functions.Activation("crelu"))	# crelu outputs 2x 
seq.add(layers.BatchNormalization(1000))
seq.add(layers.Linear(1000, 500))
seq.add(functions.Activation("elu"))
seq.add(layers.Linear(500, 500, use_weightnorm=True))
seq.add(functions.Activation("hard_sigmoid"))
seq.add(layers.BatchNormalization(500))
seq.add(layers.Linear(500, 500))
seq.add(functions.Activation("leaky_relu"))
seq.add(layers.Linear(500, 500, use_weightnorm=True))
seq.add(functions.Activation("relu"))
seq.add(layers.BatchNormalization(500))
seq.add(layers.Linear(500, 500))
seq.add(functions.Activation("sigmoid"))
seq.add(layers.Linear(500, 500, use_weightnorm=True))
seq.add(functions.Activation("softmax"))
seq.add(layers.BatchNormalization(500))
seq.add(layers.Linear(500, 500))
seq.add(functions.Activation("softplus"))
seq.add(layers.Linear(500, 500, use_weightnorm=True))
seq.add(functions.Activation("tanh"))
seq.add(layers.Linear(500, 10))
seq.build()

y = seq(x)
print y.data.shape

# Conv test
x = np.random.normal(scale=1, size=(2, 3, 96, 96)).astype(np.float32)
x = Variable(x)

seq = Sequential(weight_initializer="GlorotNormal", weight_init_std=0.05)
seq.add(layers.Convolution2D(3, 64, ksize=4, stride=2, pad=0))
seq.add(layers.BatchNormalization(64))
seq.add(functions.Activation("relu"))
seq.add(layers.Convolution2D(64, 128, ksize=4, stride=2, pad=0))
seq.add(layers.BatchNormalization(128))
seq.add(functions.Activation("relu"))
seq.add(layers.Convolution2D(128, 256, ksize=4, stride=2, pad=0))
seq.add(layers.BatchNormalization(256))
seq.add(functions.Activation("relu"))
seq.add(layers.Convolution2D(256, 512, ksize=4, stride=2, pad=0))
seq.add(layers.BatchNormalization(512))
seq.add(functions.Activation("relu"))
seq.add(layers.Convolution2D(512, 1024, ksize=4, stride=2, pad=0))
seq.add(layers.BatchNormalization(1024))
seq.add(functions.Activation("relu"))
seq.add(layers.Linear(None, 10, use_weightnorm=True))
seq.add(functions.softmax())
seq.build()

y = seq(x)
print y.data.shape

# Deconv test
x = np.random.normal(scale=1, size=(2, 100)).astype(np.float32)
x = Variable(x)

image_size = 96
# compute projection width
input_size = util.get_in_size_of_deconv_layers(image_size, num_layers=3, ksize=4, stride=2)
# compute required paddings
paddings = util.get_paddings_of_deconv_layers(image_size, num_layers=3, ksize=4, stride=2)

seq = Sequential(weight_initializer="GlorotNormal", weight_init_std=0.05)
seq.add(layers.Linear(100, 64 * input_size ** 2))
seq.add(layers.BatchNormalization(64 * input_size ** 2))
seq.add(functions.Activation("relu"))
seq.add(functions.reshape((-1, 64, input_size, input_size)))
seq.add(layers.Deconvolution2D(64, 32, ksize=4, stride=2, pad=paddings.pop(0), use_weightnorm=True))
seq.add(layers.BatchNormalization(32))
seq.add(functions.Activation("relu"))
seq.add(layers.Deconvolution2D(32, 16, ksize=4, stride=2, pad=paddings.pop(0), use_weightnorm=True))
seq.add(layers.BatchNormalization(16))
seq.add(functions.Activation("relu"))
seq.add(layers.Deconvolution2D(16, 3, ksize=4, stride=2, pad=paddings.pop(0), use_weightnorm=True))
seq.build()

y = seq(x)
print y.data.shape

# train test
x = np.random.normal(scale=1, size=(128, 28*28)).astype(np.float32)
x = Variable(x)

seq = Sequential(weight_initializer="GlorotNormal", weight_init_std=0.05)
seq.add(layers.Linear(28*28, 500, use_weightnorm=True))
seq.add(layers.BatchNormalization(500))
seq.add(functions.Activation("relu"))
seq.add(layers.Linear(None, 500, use_weightnorm=True))
seq.add(layers.BatchNormalization(500))
seq.add(functions.Activation("relu"))
seq.add(layers.Linear(500, 28*28, use_weightnorm=True))
seq.build()

chain = chain.Chain()
chain.add_sequence(seq)
chain.setup_optimizers("adam", 0.001, momentum=0.9, weight_decay=0.000001, gradient_clipping=10)

for i in xrange(100):
	y = chain(x)
	loss = F.mean_squared_error(x, y)
	chain.backprop(loss)
	print float(loss.data)

chain.save("model")