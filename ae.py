'''
TODO : add lrate decay
'''
from inputParser import get_parser_AE
import cPickle as pickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data

class CostType:
	MeanSquared = "MeanSquaredCost"
	CrossEntropy = "CrossEntropy"
	CategoricalCrossEntropy = "CategoricalCrossEntropy"

ActivationType = {'sigmoid' : T.nnet.sigmoid, 'softplus' : T.nnet.softplus, 'tanh' : T.tanh}

class autoencoder(object):
	"""
	Autoencoder class
	"""
	def __init__(self, numpy_rng, theano_rng=None, input=None,
				 n_visible=784, n_hidden=1000,
				 W=None, bhid=None, bvis=None, cost_type = CostType.MeanSquared, n_batchsize = 20, tied_weights = False, activation = T.nnet.sigmoid):
		"""
		:type numpy_rng: numpy.random.RandomState
		:param numpy_rng: number random generator used to generate weights

		:type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
		:param theano_rng: Theano random generator; if None is given one is
					 generated based on a seed drawn from `rng`

		:type input: theano.tensor.TensorType
		:param input: a symbolic description of the input or None

		:type n_visible: int
		:param n_visible: number of visible units

		:type n_hidden: int
		:param n_hidden:  number of hidden units

		:type W: theano.tensor.TensorType
		:param W: Theano variable pointing to a set of weights that should be
				  shared belong the ae

		:type bhid: theano.tensor.TensorType
		:param bhid: Theano variable pointing to a set of biases values (for
					 hidden units)

		:type bvis: theano.tensor.TensorType
		:param bvis: Theano variable pointing to a set of biases values (for
					 visible units)

		:type tied_weights: bool
		:param tied_weights: ties input-hidden, hidden-reconstruction weights

		:type activation: activation function
		:param activation: activation function such as T.nnet.sigmoid, T.tanh

		"""
		self.activation = activation
		self.n_visible = n_visible
		self.n_hidden = n_hidden
		self.n_batchsize = n_batchsize
		if cost_type == CostType.MeanSquared:
			self.cost_type = CostType.MeanSquared
		elif cost_type == CostType.CrossEntropy:
			self.cost_type = CostType.CrossEntropy
		elif cost_type == CostType.CategoricalCrossEntropy:
			self.cost_type = CostType.CategoricalCrossEntropy

		# create a Theano random generator that gives symbolic random values
		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		# note : W' was written as `W_prime` and b' as `b_prime`
		if not W:
			initial_W = numpy.asarray(numpy_rng.uniform(
					  low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
					  high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
					  size=(n_visible, n_hidden)), dtype=theano.config.floatX)
			W = theano.shared(value=initial_W, name='W', borrow=True)

		if not bvis:
			bvis = theano.shared(value=numpy.zeros(n_visible,
										 dtype=theano.config.floatX),
								 borrow=True)

		if not bhid:
			bhid = theano.shared(value=numpy.zeros(n_hidden,
												   dtype=theano.config.floatX),
								 name='b',
								 borrow=True)

		self.W = W
		# b corresponds to the bias of the hidden
		self.b = bhid
		# b_prime corresponds to the bias of the visible
		self.b_prime = bvis

		if tied_weights :
			# tied weights, therefore W_prime is W transpose
			self.W_prime = self.W.T
			self.params = [self.W, self.b, self.b_prime]
		else:
			initial_W = numpy.asarray(numpy_rng.uniform(
					  low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
					  high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
					  size=(n_hidden, n_visible)), dtype=theano.config.floatX)
			self.W_prime = theano.shared(value=initial_W, name='W_prime', borrow=True)
			self.params = [self.W, self.W_prime, self.b, self.b_prime]

		self.theano_rng = theano_rng
		if input == None:
			self.x = T.dmatrix(name='input')
		else:
			self.x = input

	def get_corrupted_input(self, input, corruption_level):
		"""
		corrupt the input as described in the tutorial
		"""
		if corruption_level == 0:
			return input
		else:
			return	self.theano_rng.binomial(size=input.shape, n=1,p = 1 - corruption_level,dtype=theano.config.floatX) * input

	def get_hidden_values(self, input):
		""" Computes the values of the hidden layer """
		return self.activation( T.dot(input, self.W) + self.b)

	def get_reconstructed_input(self, hidden):
		"""Computes the reconstructed input given the values of the
		hidden layer
		"""
		return self.activation( T.dot(hidden, self.W_prime) + self.b_prime)

	def get_rec_cost(self, x_rec):
		"""
		Returns the reconstruction cost.
		"""
		if self.cost_type == CostType.MeanSquared:
			return T.mean(((self.x - x_rec)**2).sum(axis=1))
		elif self.cost_type == CostType.CrossEntropy:
			return T.mean((T.nnet.binary_crossentropy(x_rec, self.x)).mean(axis=1))
		elif self.cost_type == CostType.CategoricalCrossEntropy:
			return T.nnet.categorical_crossentropy(x_rec, self.x).sum()
		else:
			raise NotImplementedError()

	def get_cost_updates(self, corruption_level, learning_rate):
		""" This function computes the cost and the updates for one trainng
		step of the dA """
		tilde_x = self.get_corrupted_input(self.x, corruption_level)
		y = self.get_hidden_values(tilde_x)
		z = self.get_reconstructed_input(y)

		cost = self.get_rec_cost(z)
		return cost

	def kl_divergence(self, p, p_hat):
		"""
		returns kl_divergence for activation sparsity
		"""
		return p * T.log(p) - p * T.log(p_hat) + (1-p) * T.log(1 - p) - (1-p) * T.log(1 - p_hat)

	def sparsity_penalty(self, h, sparsity_level=0.05, sparse_reg=1e-3):
		"""
		calculates sparsity penalty
		"""
		sparsity_level = T.extra_ops.repeat(sparsity_level, self.n_hidden)
		avg_act = h.mean(axis=0)
		kl_div = self.kl_divergence(sparsity_level, avg_act)
		sparsity_penalty = sparse_reg * kl_div.sum()

		return sparsity_penalty

	def get_sa_sgd_updates(self, learning_rate, sparsity_level, sparse_reg):
		"""
		calculates cost for sparse ae
		"""
		hid = self.get_hidden_values(self.x)
		x_rec = self.get_reconstructed_input(hid)

		cost = self.sparsity_penalty(hid, sparsity_level, sparse_reg)
		return cost

	def get_jacobian(self, hidden, W):
		"""Computes the jacobian of the hidden layer with respect to
		the input, reshapes are necessary for broadcasting the
		element-wise product on the right axis
		"""
		return T.reshape(hidden * (1 - hidden),
						 (self.n_batchsize, 1, self.n_hidden)) * T.reshape(
							 W, (1, self.n_visible, self.n_hidden))

	def get_ca_sgd_updates(self,learning_rate, contraction_level):
		""" This function computes the cost and the updates for one trainng
		step of the cA """
		y = self.get_hidden_values(self.x)
		z = self.get_reconstructed_input(y)
		J = self.get_jacobian(y, self.W)

		self.L_jacob = T.sum(J ** 2) / self.n_batchsize

		cost = contraction_level * T.mean(self.L_jacob)
		return cost

	def get_updates(self,cost,learning_rate):
		"""
		calculate update parameters
		"""

		gparams = T.grad(cost, self.params)
		updates = []
		for param, gparam in zip(self.params, gparams):
			updates.append((param, param - learning_rate * gparam))
		return (cost, updates)

def testAE(learning_rate=0.1,
		   n_epochs=50,
		   dataset='mnist.pkl.gz',
		   batch_size=20,
		   visible_size=784,
		   hidden_size=1000,
		   contraction_level=0.1,
		   sparsity_penalty=0.001,
		   sparsity_level=0.05,
		   corruption_rate=0.3,
		   sparse_flag = False,
		   contraction_flag = False,
		   fname='ae',
		   activation = T.nnet.sigmoid,
		   cost_type = CostType.MeanSquared):

	"""
	This demo is tested on MNIST

	:type p: TODO
	:param p: TODO

	"""
	datasets = load_data(dataset)
	train_set_x, train_set_y = datasets[0]

	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

	index = T.lscalar()	   # index to a [mini]batch
	x = T.matrix('x')      # the data is presented as rasterized images

	####################################
	# BUILDING THE MODEL 
	####################################

	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))

	ae = autoencoder(numpy_rng=rng, theano_rng=theano_rng, input=x,
					 n_visible= visible_size, n_hidden=hidden_size, n_batchsize = batch_size, activation = activation, cost_type = cost_type)

	cost = ae.get_cost_updates(corruption_level=corruption_rate,learning_rate=learning_rate)
	if contraction_flag:
		cost += ae.get_ca_sgd_updates(learning_rate, contraction_level)
	else:
		contraction_level = 0
	if sparse_flag:
		cost += ae.get_sa_sgd_updates(learning_rate, sparsity_level, sparsity_penalty)
	else:
		sparsity_level = 0
	(cost, updates) = ae.get_updates(cost, learning_rate)

	train_ae = theano.function([index], cost, updates=updates,
		 givens={x: train_set_x[index * batch_size:
								  (index + 1) * batch_size]})
	start_time = time.clock()

	############
	# TRAINING #
	############

	# go through training epochs
	for epoch in xrange(n_epochs):
		# go through trainng set
		c = []
		for batch_index in xrange(n_train_batches):
			c.append(train_ae(batch_index))
		print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

	end_time = time.clock()
	training_time = (end_time - start_time)
	print >> sys.stderr, ('code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % (training_time / 60.))
	if fname == 'ae':
		fname = fname + '-hid' + str(hidden_size) + '-lr' + str(learning_rate) + '-epoch' + str(n_epochs) + "-denoise" + str(corruption_rate) + "-contraction" + str(contraction_level) + '-sparsity' + str(sparsity_level) + '-data'+ dataset.split('.')[0] + '.pkl'
	output = open(fname, 'wb')
	pickle.dump(ae, output)
	output.close()

if __name__ == '__main__':
	parser = get_parser_AE()
	p = parser.parse_args()

	activation = ActivationType[p.activation]

	testAE(learning_rate = p.learning_rate, n_epochs = p.n_epochs, dataset = p.benchmark, batch_size = p.batch_size, visible_size = p.visible_size, hidden_size = p.hidden_size, contraction_flag = p.contraction_flag, contraction_level = p.contraction_level, sparse_flag = p.sparse_flag, sparsity_penalty = p.sparsity_penalty, sparsity_level = p.sparsity_level, corruption_rate = p.corruption_rate, fname = p.fname, activation = activation, cost_type = p.cost_type)
