#!/usr/bin/python

import numpy as np
import tensorflow as tf

product = lambda l: reduce(lambda x, y: x * y, l, 1)

class ChessNet:
	INPUT_FEATURE_COUNT = 13
	FILTERS = 128
	CONV_SIZE = 5
	NONLINEARITY = [tf.nn.relu]
	BLOCK_COUNT = 8
	OUTPUT_CONV_FILTERS = 16
	FC_SIZES = [OUTPUT_CONV_FILTERS * 64, 128]

	def __init__(self):
		# Construct input/output placeholders.
		self.input_ph = tf.placeholder(
			tf.float32,
			shape=[None, 8, 8, self.INPUT_FEATURE_COUNT],
			name="input_placeholder")
		self.desired_output_ph = tf.placeholder(
			tf.float32,
			shape=[None, 2, 8, 8],
			name="desired_output_placeholder")
		self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name="learning_rate")
		self.is_training_ph = tf.placeholder(tf.bool, shape=[], name="is_training")

		# Count up parameters while we build, just for curiosity.
		self.total_parameters = 0

		# Begin constructing the data flow.
		self.parameters = []
		self.flow = self.input_ph
		# Stack an initial convolution.
		self.stack_convolution(self.CONV_SIZE, self.INPUT_FEATURE_COUNT, self.FILTERS)
		self.stack_nonlinearity()
		# Stack some number of residual blocks.
		for _ in xrange(self.BLOCK_COUNT):
			self.stack_block()
		# Stack a final 1x1 convolution transitioning to fully-connected features.
		self.stack_convolution(1, self.FILTERS, self.OUTPUT_CONV_FILTERS)
		self.stack_nonlinearity()
		# Switch over to fully connected processing by flattening.
		self.flow = tf.reshape(self.flow, [-1, self.FC_SIZES[0]])

		# Switch over to fully connected processing.
		for old_size, new_size in zip(self.FC_SIZES, self.FC_SIZES[1:]):
			W = self.new_weight_variable([old_size, new_size])
			b = self.new_bias_variable([new_size])
			self.flow = tf.matmul(self.flow, W) + b
			self.stack_nonlinearity()

		# Reshape the final FC output into two 8x8 layers, one for piece "pick up" and one for "put down".
		self.final_output = tf.reshape(self.flow, [-1, 2, 64])
		reshaped_desired_output = tf.reshape(self.desired_output_ph, [-1, 2, 64])

		# Construct the training components.
		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
			labels=reshaped_desired_output,
			logits=self.final_output,
		))
		regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
		reg_variables = tf.trainable_variables()
		self.regularization_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
		self.loss = self.cross_entropy + self.regularization_term

		# Associate batch normalization with training.
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.train_step = tf.train.MomentumOptimizer(
				learning_rate=self.learning_rate_ph, momentum=0.9).minimize(self.loss)

	def new_weight_variable(self, shape):
		self.total_parameters += product(shape)
		stddev = (2.0 / product(shape[:-1]))**0.5
		var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
		self.parameters.append(var)
		return var

	def new_bias_variable(self, shape):
		self.total_parameters += product(shape)
		var = tf.Variable(tf.constant(0.1, shape=shape))
		self.parameters.append(var)
		return var

	def stack_convolution(self, kernel_size, old_size, new_size, batch_normalization=True):
		weights = self.new_weight_variable([kernel_size, kernel_size, old_size, new_size])
		self.flow = tf.nn.conv2d(self.flow, weights, strides=[1, 1, 1, 1], padding="SAME")
		if batch_normalization:
			self.flow = tf.layers.batch_normalization(
				self.flow,
				center=False,
				scale=False,
				training=self.is_training_ph)
		else:
			bias = self.new_bias_variable([new_size])
			self.flow = self.flow + bias # TODO: Is += equivalent?

	def stack_nonlinearity(self):
		self.flow = self.NONLINEARITY[0](self.flow)

	def stack_block(self):
		initial_value = self.flow
		# Stack the first convolution.
		self.stack_convolution(self.CONV_SIZE, self.FILTERS, self.FILTERS)
		self.stack_nonlinearity()
		# Stack the second convolution.
		self.stack_convolution(self.CONV_SIZE, self.FILTERS, self.FILTERS)
		# Add the skip connection.
		self.flow = self.flow + initial_value
		# Stack on the deferred non-linearity.
		self.stack_nonlinearity()

	def train(self, samples, learning_rate):
		self.run_on_samples(self.train_step.run, samples, learning_rate=learning_rate, is_training=True)

	def get_loss(self, samples):
		return self.run_on_samples(self.cross_entropy.eval, samples)

	def get_accuracy(self, samples):
		results = self.run_on_samples(self.final_output.eval, samples)
		results = np.argmax(results, axis=-1)
		assert results.shape == (len(samples["features"]), 2)
		correct = 0
		for move, result in zip(samples["moves"], results):
			lhs = np.argmax(move.reshape((2, 64)), axis=-1)
			assert lhs.shape == result.shape == (2,)
			correct += np.all(lhs == result)
		return correct / float(len(samples["features"]))

	def run_on_samples(self, f, samples, learning_rate=0.01, is_training=False):
		input_tensor, output_tensor = samples["features"], samples["moves"]
		return f(feed_dict={
			self.input_ph:          input_tensor,
			self.desired_output_ph: output_tensor,
			self.learning_rate_ph:  learning_rate,
			self.is_training_ph:    is_training,
		})

# XXX: This is horrifically ugly.
# TODO: Once I have a second change it to not do this horrible graph scraping that breaks if you have other things going on.
def get_batch_norm_vars():
	return [
		i for i in tf.global_variables()
		if "batch_normalization" in i.name and ("moving_mean:0" in i.name or "moving_variance:0" in i.name)
	]

def save_model(net, path):
	x_conv_weights = [sess.run(var) for var in net.parameters]
	x_bn_params = [sess.run(i) for i in get_batch_norm_vars()]
	np.save(path, [x_conv_weights, x_bn_params])
	print "\x1b[35mSaved model to:\x1b[0m", path

# XXX: Still horrifically fragile wrt batch norm variables due to the above horrible graph scraping stuff.
def load_model(net, path):
	x_conv_weights, x_bn_params = np.load(path)
	assert len(net.parameters) == len(x_conv_weights), "Parameter count mismatch!"
	for var, value in zip(net.parameters, x_conv_weights):
		sess.run(var.assign(value))
	bn_vars = get_batch_norm_vars()
	assert len(bn_vars) == len(x_bn_params), "Bad batch normalization parameter count!"
	for var, value in zip(bn_vars, x_bn_params):
		sess.run(var.assign(value))

