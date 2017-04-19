# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

pickle_file = '../notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

# Limit training data to simulate overfitting
# Result: big increase in training accuracy, big drop in test accuracy (83.4% from 92.9%)
# Introducing dropout at 0.20 results in improved test accuracy: 84.4%
# train_dataset = train_dataset[:384]
# train_labels  = train_labels[:384]
# print('Overfit training set: ', train_dataset.shape, train_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# Stocastic Gradient Descent
batch_size     = 128
hidden_units_1 = 1024 
hidden_units_2 = 1024 

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_units_1]))
  biases_1  = tf.Variable(tf.zeros([hidden_units_1]))

  weights_2 = tf.Variable(tf.truncated_normal([hidden_units_1, num_labels]))
  biases_2  = tf.Variable(tf.zeros([num_labels]))

  # weights_3 = tf.Variable(tf.truncated_normal([hidden_units_2, num_labels]))
  # biases_3  = tf.Variable(tf.zeros([num_labels]))

  # Training computation
  # Plain logistic regression: 89.2% on test with l2 weight = 0.002
  # Neural net with single relu layer: 92.9% on test with l2 weight = 0.002 
  train_logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
  valid_logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
  test_logits_1  = tf.matmul(tf_test_dataset,  weights_1) + biases_1

  # First nonlinear layer
  # train_logits_2 = tf.nn.dropout(tf.nn.relu(train_logits_1), 0.20)
  train_logits_1r = tf.nn.relu(train_logits_1)
  valid_logits_1r = tf.nn.relu(valid_logits_1)
  test_logits_1r  = tf.nn.relu(test_logits_1)

  train_logits = tf.matmul(train_logits_1r, weights_2) + biases_2
  valid_logits = tf.matmul(valid_logits_1r, weights_2) + biases_2
  test_logits  = tf.matmul(test_logits_1r,  weights_2) + biases_2

  # Second nonlinear layer
  # train_logits_2r = tf.nn.relu(train_logits_2)
  # valid_logits_2r = tf.nn.relu(valid_logits_2)
  # test_logits_2r  = tf.nn.relu(test_logits_2)

  # train_logits = tf.matmul(train_logits_2r, weights_3) + biases_3
  # valid_logits = tf.matmul(valid_logits_2r, weights_3) + biases_3
  # test_logits  = tf.matmul(test_logits_2r, weights_3) + biases_3

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=train_logits))
  # loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=train_logits))
  #          + 0.002 * (tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(train_logits)
  valid_prediction = tf.nn.softmax(valid_logits)
  test_prediction  = tf.nn.softmax(test_logits)


# Minibatch training
num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

