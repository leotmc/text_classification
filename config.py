import tensorflow as tf

tf.flags.DEFINE_string('train_inputs_path', default='data/train_inputs.txt', help='path of training inputs')
tf.flags.DEFINE_string('train_labels_path', default='data/train_labels.txt', help='path of training labels')
tf.flags.DEFINE_string('vocab_path', default='data/vocab.pickle', help='path of vocab')
tf.flags.DEFINE_integer('embedding_size', default=80, help='size of embedding vector')
tf.flags.DEFINE_integer('hidden_size', default=100, help='number of units of hidden layer')
tf.flags.DEFINE_integer('num_classes', default=2, help='number of classes')

FLAGS = tf.flags.FLAGS

