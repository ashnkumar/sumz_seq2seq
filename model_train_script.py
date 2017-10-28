import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors


def check_tensorflow_has_gpu():
	from distutils.version import LooseVersion
	import warnings
	# Check TensorFlow Version
	assert LooseVersion(tf.__version__) >= LooseVersion('1.1'), 'Please use TensorFlow version 1.1 or newer'
	print('TensorFlow Version: {}'.format(tf.__version__))
	# Check for a GPU
	if not tf.test.gpu_device_name():
			warnings.warn('No GPU found. Please use a GPU to train your neural network.')
	else:
			print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def model_input_placeholders():
		inputs = tf.placeholder(tf.int32, [None,None], name='input')
		targets = tf.placeholder(tf.int32, [None,None])
		learning_rate = tf.placeholder(tf.float32)
		keep_probability = tf.placeholder(tf.float32, name='keep_probability')
		target_seq_len = tf.placeholder(tf.int32, (None,), name='target_seq_len')
		max_target_seq_len = tf.reduce_max(target_seq_len, name='max_target_seq_len')
		source_seq_len = tf.placeholder(tf.int32, (None,), name='source_seq_len')

		return inputs, targets, learning_rate, keep_probability, target_seq_len, max_target_seq_len, source_seq_len

def embedded_encoder_input(input_data, word_embedding_matrix):
		return tf.nn.embedding_lookup(word_embedding_matrix, input_data)

def encoding_layer(encoder_inputs, rnn_size, source_seq_len, num_layers, keep_prob):

		for layer in range(num_layers):
				with tf.variable_scope('encoder_{}'.format(layer)):
						single_rnn_cell_forward = tf.contrib.rnn.LSTMCell(num_units = rnn_size,
																											initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=2))
						single_rnn_cell_forward = tf.contrib.rnn.DropoutWrapper(cell = single_rnn_cell_forward,
																																		input_keep_prob = keep_prob)
						single_rnn_cell_backward = tf.contrib.rnn.LSTMCell(num_units = rnn_size,
																															 initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=2))
						single_rnn_cell_backward = tf.contrib.rnn.DropoutWrapper(cell = single_rnn_cell_backward,
																																		 input_keep_prob = keep_prob)
						enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(single_rnn_cell_forward,
																																		single_rnn_cell_backward,
																																		encoder_inputs,
																																		source_seq_len,
																																		dtype = tf.float32)
		enc_output = tf.concat(enc_output, 2) # Concatenate both outputs together
		return enc_output, enc_state

def process_decoder_input(target_data, vocab_to_int, batch_size):

		# Remove the last word (integer) from each target sequence
		ending = tf.strided_slice(target_data, [0,0], [batch_size,-1], [1,1])

		# Add the <GO> token to each target sequence
		decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

		return decoder_input

def embedded_decoder_input(input_data, word_embedding_matrix):
		return tf.nn.embedding_lookup(word_embedding_matrix, input_data)

def make_decoder_cell(rnn_size,
											num_layers,
											encoder_output,
											source_seq_len,
											keep_prob,
											batch_size,
											encoder_state):

		for layer in range(num_layers):
				with tf.variable_scope('decoder_{}'.format(layer)):
						single_cell = tf.contrib.rnn.LSTMCell(rnn_size,
																									initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
						dec_cell = tf.contrib.rnn.DropoutWrapper(single_cell, input_keep_prob=keep_prob)

		attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
																															 encoder_output,
																															 source_seq_len,
																															 normalize=False,
																															 name='BahdanauAttention')

		dec_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(dec_cell,
																													attention_mechanism,
																													rnn_size)

		initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(encoder_state[0],
																																		_zero_state_tensors(rnn_size,
																																												batch_size,
																																												tf.float32))
		return dec_cell, initial_state

def decoding_layer(input_data,
									 word_embedding_matrix,
									 num_layers,
									 rnn_size,
									 keep_prob,
									 encoder_output,
									 source_seq_len,
									 encoder_state,
									 batch_size,
									 vocab_size,
									 target_seq_len,
									 max_target_seq_len,
									 vocab_to_int):

		decoder_embedded_input = embedded_decoder_input(input_data, word_embedding_matrix)
		decoder_cell, initial_state = make_decoder_cell(rnn_size,
																										num_layers,
																										encoder_output,
																										source_seq_len,
																										keep_prob,
																										batch_size,
																										encoder_state)
		output_layer = Dense(vocab_size,
												kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))

		# Training
		with tf.variable_scope("decode"):
				training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embedded_input,
																														sequence_length = target_seq_len,
																														time_major=False)
				training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
																													 training_helper,
																													 initial_state,
																													 output_layer)
				training_logits, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
																															 output_time_major=False,
																															 impute_finished=True,
																															 maximum_iterations=max_target_seq_len)

		with tf.variable_scope("decode", reuse=True): # Reuse same params for inference

				start_tokens = tf.tile(tf.constant([vocab_to_int['<GO>']], dtype=tf.int32),
															 [batch_size],
															 name='start_tokens')
				inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(word_embedding_matrix,
																																		start_tokens,
																																		vocab_to_int['<EOS>'])
				inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
																														inference_helper,
																														initial_state,
																														output_layer)
				inference_logits, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
																															output_time_major=False,
																															impute_finished=True,
																															maximum_iterations=max_target_seq_len)

		return training_logits, inference_logits

def full_seq2seq(input_data,
								 word_embedding_matrix,
								 rnn_size,
								 source_seq_len,
								 num_layers,
								 keep_prob,
								 target_data,
								 vocab_to_int,
								 batch_size,
								 vocab_size,
								 target_seq_len,
								 max_target_seq_len


								 ):

		# Encoding layer
		encoder_inputs = embedded_encoder_input(input_data, word_embedding_matrix)
		encoder_output, encoder_state = encoding_layer(encoder_inputs,
																									 rnn_size,
																									 source_seq_len,
																									 num_layers,
																									 keep_prob)

		# Decoding layer
		processed_decoder_input = process_decoder_input(target_data,
																										vocab_to_int,
																										batch_size)
		training_logits, inference_logits = decoding_layer(processed_decoder_input,
																											 word_embedding_matrix,
																											 num_layers,
																											 rnn_size,
																											 keep_prob,
																											 encoder_output,
																											 source_seq_len,
																											 encoder_state,
																											 batch_size,
																											 vocab_size,
																											 target_seq_len,
																											 max_target_seq_len,
																											 vocab_to_int)
		return training_logits, inference_logits

def pad_batch(batch_to_pad):
		max_size = max([len(item) for item in batch_to_pad])
		padded_batch = [item + [vocab_to_int['<PAD>']] * (max_size - len(item)) for item in batch_to_pad]
		return padded_batch

def get_batches(summaries, reviews, batch_size):
		for batch_i in range(0, len(reviews)//batch_size):
				start_i = batch_i * batch_size
				summaries_batch = summaries[start_i:start_i + batch_size]
				reviews_batch = reviews[start_i:start_i + batch_size]
				pad_summaries_batch = pad_batch(summaries_batch)
				pad_reviews_batch = pad_batch(reviews_batch)
				pad_summaries_lengths = []
				for summary in pad_summaries_batch:
						pad_summaries_lengths.append(len(summary))
				pad_reviews_lengths = []
				for review in pad_reviews_batch:
						pad_reviews_lengths.append(len(review))

				yield pad_summaries_batch, pad_reviews_batch, pad_summaries_lengths, pad_reviews_lengths

# Hyperparameters
epochs = 100
rnn_size = 256
batch_size = 64
num_layers = 2
lr = 0.005
keep_prob = 0.75

def build_and_train_model(word_embedding_matrix,
								rnn_size,
								num_layers,
								keep_probability,
								vocab_to_int,
								batch_size,
								sorted_summaries,
								sorted_reviews):


		# GRAPH BUILDING
		train_graph = tf.Graph()
		with train_graph.as_default():

				# Model inputs
				inputs, targets, learning_rate, keep_probability, target_seq_len, max_target_seq_len, source_seq_len = model_input_placeholders()

				# Create final logits tensors
				training_logits, inference_logits = full_seq2seq(tf.reverse(inputs, [-1]),
																												 word_embedding_matrix,
																												 rnn_size,
																												 source_seq_len,
																												 num_layers,
																												 keep_probability,
																												 targets,
																												 vocab_to_int,
																												 batch_size,
																												 len(vocab_to_int)+1,
																												 target_seq_len,
																												 max_target_seq_len)

				training_logits = tf.identity(training_logits.rnn_output, 'logits')
				inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

				masks = tf.sequence_mask(target_seq_len, max_target_seq_len, dtype=tf.float32, name='masks')

				# Set up optimizer
				with tf.name_scope("optimization"):

						cost = tf.contrib.seq2seq.sequence_loss(training_logits,
																										targets,
																										masks)

						optimizer = tf.train.AdamOptimizer(learning_rate)

						gradients = optimizer.compute_gradients(cost)
						capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
						train_operation = optimizer.apply_gradients(capped_gradients)

		print("Finished building the graph!")

		start = 200000
		end = start + 50000
		sorted_summaries_short = sorted_summaries[start:end]
		sorted_reviews_short = sorted_reviews[start:end]
		print("The shortest review length:", len(sorted_reviews_short[0]))
		print("The longest review length:", len(sorted_reviews_short[-1]))

#     learning_rate_decay = 0.95
#     min_learning_rate = 0.0005
		display_step = 1 # Check training loss after every 20 batches
		stop = 5 # Stop training if average loss doesn't decrease in this mean update_checks
		per_epoch = 3 # Make 3 update checks per epoch
		update_check = (len(sorted_reviews_short)//batch_size//per_epoch)-1

		update_loss = 0
		batch_loss = 0
		summary_update_loss = [] # Record the update losses for saving improvements in the model

		checkpoint = "./model_checkpoints2/best_model.ckpt"
		with tf.Session(graph=train_graph) as sess:
				sess.run(tf.global_variables_initializer())

				for epoch_i in range(1, epochs+1):
						update_loss = 0
						batch_loss = 0
						for batch_i, (summaries_batch, reviews_batch, summaries_lengths, reviews_lengths) in enumerate(
										get_batches(sorted_summaries_short, sorted_reviews_short, batch_size)):
								start_time = time.time()
								_, loss = sess.run(
										[train_operation, cost],
										{inputs: reviews_batch,
										 targets: summaries_batch,
										 learning_rate: lr,
										 target_seq_len: summaries_lengths,
										 source_seq_len: reviews_lengths,
										 keep_probability: keep_prob})

								batch_loss += loss
								update_loss += loss
								end_time = time.time()
								batch_time = end_time - start_time

								if batch_i % display_step == 0 and batch_i > 0:
										print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
													.format(epoch_i,
																	epochs,
																	batch_i,
																	len(sorted_reviews_short) // batch_size,
																	batch_loss / display_step,
																	batch_time*display_step))
										batch_loss = 0

								if batch_i % update_check == 0 and batch_i > 0:
										print("Average loss for this update:", round(update_loss/update_check,3))
										summary_update_loss.append(update_loss)

										# If the update loss is at a new minimum, save the model
										if update_loss <= min(summary_update_loss):
												print('New Record!')
												stop_early = 0
												saver = tf.train.Saver()
												saver.save(sess, checkpoint)

										else:
												print("No Improvement.")
												stop_early += 1
												if stop_early == stop:
														break
										update_loss = 0


						# Reduce learning rate, but not below its minimum value
#             learning_rate *= learning_rate_decay
#             if learning_rate < min_learning_rate:
#                 learning_rate = min_learning_rate
def load_pickled_data():
		word_dicts_path = './checkpointed_data/word_dicts.p'
		model_input_data_path = './checkpointed_data/model_input_data.p'
		vocab_to_int, int_to_vocab, word_embedding_matrix = pickle.load(open(word_dicts_path, mode='rb'))
		sorted_summaries, sorted_reviews = pickle.load(open(model_input_data_path, mode='rb'))
		return vocab_to_int, int_to_vocab, word_embedding_matrix, sorted_summaries, sorted_reviews

vocab_to_int, int_to_vocab, word_embedding_matrix, sorted_summaries, sorted_reviews = load_pickled_data()
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

build_and_train_model(word_embedding_matrix,
											rnn_size,
											num_layers,
											keep_prob,
											vocab_to_int,
											batch_size,
											sorted_summaries,
											sorted_reviews)
