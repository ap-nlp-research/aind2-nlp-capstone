import os
from tqdm import tqdm
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell


def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')

def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    x_tk = keras.preprocessing.text.Tokenizer()
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    if length is None:
        length = max([len(sentence) for sentence in x])

    return keras.preprocessing.sequence.pad_sequences(x, maxlen=length, padding='post')


def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    return preprocess_x, preprocess_y, x_tk, y_tk


def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])



def embed_input(x: tf.Tensor, n_words: int, embedding_size: int) -> tf.Tensor:
    # Create embedding
    emdedded_input = tf.nn.embedding_lookup(params=tf.get_variable(name="embedding", shape=(n_words, embedding_size)),
                                            ids=x)
    return emdedded_input


def create_encoder(emdedded_input: tf.Tensor, num_units: int = 64) -> tf.Tensor:
    """

    :param x: tf.placeholder or data input
    :return:
    """
    with tf.variable_scope("encoder"):
        cell = GRUCell(num_units=num_units)
        _, encoder_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=emdedded_input, dtype=tf.float32)

    return encoder_final_state


def create_decoder(encoder_hs: tf.Tensor, sequence_length: int) -> tf.Tensor:
    batch_size = tf.shape(encoder_hs)[0]
    encoder_units = encoder_hs.get_shape().as_list()[-1]
    dtype = encoder_hs.dtype
    # create a decoder cell

    def teacher_forcing_loop(time, cell_output = None, cell_state = None, loop_state = None):
        emit_output = cell_output  # == None for time == 0
        elements_finished = (time >= sequence_length)
        # time == 0 initialize the sequence with encoder hidden state
        # otherwise, force the cell output as RNNCell input
        if cell_output is None:  # time == 0
            next_cell_state = encoder_hs
            next_input = tf.zeros([batch_size, encoder_units], dtype=dtype)
        else:
            next_cell_state = cell_state
            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(
                    finished,
                    lambda: tf.zeros([batch_size, encoder_units], dtype=dtype),
                    lambda: cell_output)
        next_loop_state = None
        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)

    with tf.variable_scope("decoder"):
        cell = GRUCell(num_units=encoder_units)

        # unroll the sequence reusing cell output as the next input
        outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(cell, loop_fn=teacher_forcing_loop)

        # outputs provided in the form of tensor array that should be converted back into a tensor
        decoder_output = outputs_ta.stack()

    return decoder_output


def create_encoder_decoder_model(inputs: tf.Tensor,
                                 source_size: int,
                                 target_size: int,
                                 target_length: int,
                                 embedding_size: int = 16) -> tf.Tensor:
    embedded_input = embed_input(inputs, n_words=source_size, embedding_size=embedding_size)
    encoder_hs = create_encoder(embedded_input)
    decoder_output = create_decoder(encoder_hs, sequence_length=target_length)
    logits = tf.layers.Dense(units=target_size)(decoder_output)

    return logits


def train(train_ops: list, metrics: list, inputs: np.ndarray, targets: np.ndarray, epochs: int, batch_size: int):
    n_samples = len(inputs)
    n_batches = (n_samples + batch_size - 1) // batch_size

    metric_names = [m.op.name for m in metrics]

    with tf.get_default_graph().as_default() as graph:

        input_tensors = tf.get_collection('inputs')[0]
        target_tensors = tf.get_collection('targets')[0]

        # Get a TensorFlow session managed by the supervisor.
        with tf.Session(graph=graph) as sess:
            # Initialize all global variables
            _ = sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            for epoc in range(epochs):

                pbar = tqdm(range(n_batches), desc="Epoch {}".format(epoc))

                for it in pbar:
                    start = it * batch_size
                    end = min(n_samples, start + batch_size)

                    x, y = inputs[start:end], targets[start:end]

                    _, metrics_output = sess.run([train_ops, metrics], feed_dict={input_tensors: x, target_tensors: y})

                    pbar.set_postfix(dict([(m, v) for m, v in zip(metric_names, metrics_output)]), refresh=True)


if __name__ == '__main__':

    # Load English data
    english_sentences = load_data('data/small_vocab_en')
    # Load French data
    french_sentences = load_data('data/small_vocab_fr')

    print('Dataset Loaded')

    preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = preprocess(
            english_sentences, french_sentences
    )

    max_english_sequence_length = preproc_english_sentences.shape[1]
    max_french_sequence_length = preproc_french_sentences.shape[1]
    english_vocab_size = len(english_tokenizer.word_index)
    french_vocab_size = len(french_tokenizer.word_index)

    inputs = tf.placeholder(tf.int32, [None, max_english_sequence_length], name='inputs')
    tf.add_to_collection(name="inputs", value=inputs)
    targets = tf.placeholder(tf.int32, [None, max_french_sequence_length], name='targets')
    tf.add_to_collection(name="targets", value=targets)

    logits = create_encoder_decoder_model(inputs=inputs,
                                          source_size=english_vocab_size+1,
                                          target_size=french_vocab_size+1,
                                          target_length=max_french_sequence_length)

    # build a loss function
    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=targets, logits=logits), name='acc_loss')
    # build accuracy metric
    predictions = keras.backend.argmax(logits)


    accuracy, update_count_op = tf.metrics.accuracy(labels=targets, predictions=predictions)

    variables = tf.trainable_variables()
    gradients = tf.gradients(loss, variables)

    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

    # Optimization
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
    update_step = optimizer.apply_gradients(zip(clipped_gradients, variables))

    train(train_ops=[update_step, update_count_op, global_step],
          metrics=[loss, accuracy],
          inputs=preproc_english_sentences,
          targets=preproc_french_sentences,
          epochs=10,
          batch_size=32)

    print("Finished training")

