import tensorflow as tf
import codecs
import os
import functools
import logging
import sys

PAD = 0
OOV = 1


# Logging
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def build_vocab(vocab_path, corpus_path):
    ''' Build the vocabulary of the corpus
    Args:
        vocab_path: The path of the vocabulary
        corpus_path: The path of the corpus
    Returns:
        vocab: The vocabulary of the corpus
    '''
    if not os.path.exists(vocab_path):
        vocab = set()
        for line in open(corpus_path, 'r'):
            vocab.update(line)
        vocab = list(vocab)
        with codecs.open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab))
        return vocab
    with codecs.open(vocab_path, 'r', encoding='utf-8') as f:
        return f.read().split('\n')


def features_generator(fname):
    for line in open(fname, 'r'):
        characters = list(line)
        yield (characters, len(characters))


def labels_generator(fname):
    for line in open(fname, 'r'):
        if line.strip() == '0':
            yield [1, 0]
        else:
            yield [0, 1]


def make_dataset(sentences):
    def predict_input_generator(sentences):
        yield list(sentences), len(list(sentences))
    return tf.data.Dataset.from_generator(functools.partial(predict_input_generator, sentences),
                                          output_shapes=([None], ()),
                                          output_types=(tf.string, tf.int32)).batch(1)


def input_fn(features_fname, labels_fname, params=None, shuffle_and_repeat=False):
    params = params if params is not None else dict()
    features_dataset = tf.data.Dataset.from_generator(
        functools.partial(features_generator, features_fname),
        output_shapes=([None], ()),
        output_types=(tf.string, tf.int32)
    )
    labels_dataset = tf.data.Dataset.from_generator(
        functools.partial(labels_generator, labels_fname),
        output_shapes=([None]),
        output_types=tf.int32
    )
    features_dataset = features_dataset.padded_batch(
        batch_size=params.get('batch_size', 1),
        padded_shapes=([None], ())
    )

    labels_dataset = labels_dataset.batch(params.get('batch_size', 1))

    dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer_size']).repeat(params['num_epochs'])

    return dataset


def model_fn(features, labels, mode, params):
    if isinstance(features, dict):
            features = features['words'], features['num_words']
    if mode == tf.estimator.ModeKeys.PREDICT:
        words = features
    else:
        words, num_words = features
    vocab_words = tf.contrib.lookup.index_table_from_file(params['vocab_path'], default_value=OOV, delimiter=' ')
    word_ids = vocab_words.lookup(words)
    embedding_matrix = tf.Variable(tf.random_normal(shape=[params['vocab_size'], params['embedding_size']], stddev=0.01), name='embedding_matrix')
    embedded = tf.nn.embedding_lookup(embedding_matrix, word_ids)
    embedded = tf.transpose(embedded, perm=[1, 0, 2])
    lstm_cell = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_size'])    #Also, notice that the inputs to any FusedRNNCell instance should be time-major, this can be done by just transposing the tensor before calling the cell.
    _, state = lstm_cell(embedded, dtype=tf.float32)
    final_state = state[1]
    logits = tf.layers.dense(final_state, params['num_classes'])

    # Compute predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes_ids': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Compute evaluation metrics

    accuracy = tf.metrics.accuracy(
        labels=tf.argmax(labels, axis=1),
        predictions=tf.argmax(logits, axis=1),
    )
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(loss, axis=0)
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
            'accuracy': accuracy,
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

    # Create training op
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss, global_step=tf.train.get_or_create_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer)


params = {
    'num_classes': 2,
    'num_epochs': 2,
    'batch_size': 64,
    'hidden_size': 50,
    'embedding_size': 50,
    'buffer_size': 1000,
    'learning_rate': 1.0e-4,
    'vocab_path': 'data/vocab.txt',
    'corpus_path': 'data/train_inputs.txt',
    'train_inputs_path': 'data/train_inputs.txt',
    'train_labels_path': 'data/train_labels.txt',
    'dev_inputs_path': 'data/dev_inputs.txt',
    'dev_labels_path': 'data/dev_labels.txt',
    'test_inputs_path': 'data/test_inputs.txt',
    'test_labels_path': 'data/test_labels.txt',
    'model_dir': 'model',
}

vocab = build_vocab(params['vocab_path'], params['corpus_path'])
vocab = dict(zip(vocab, xrange(2, len(vocab) + 2)))     # index 0 for padding value, index 1 for oov word
params['vocab_size'] = len(vocab)


train_input_inpf = functools.partial(input_fn, params['train_inputs_path'], params['train_labels_path'], params=params, shuffle_and_repeat=True)
test_input_inpf = functools.partial(input_fn, params['test_inputs_path'], params['test_labels_path'], params=params)
pred_input_inpf = functools.partial(make_dataset, 'ABSDFGHJK.')

cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
estimator = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=params['model_dir'],
                                   params=params,
                                   config=cfg)

estimator.train(train_input_inpf)
estimator.evaluate(test_input_inpf)
predictions = estimator.predict(pred_input_inpf)
for prediction in predictions:
    print prediction
