import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import re
from tqdm import tqdm

# Function to Fetch the Glove vectors and load them into memory
def fetchglove():

    glove_wordmap = {}
# download Glove vectors from directory
    with open("glove.6B.50d.txt", encoding='utf-8') as file:
        for line in file:
            word, vector = tuple(line.split(" ",1))
            glove_wordmap[word] = np.fromstring(vector, sep=' ')

# Return the dictionary containing the vector representations of words
    return glove_wordmap

# Function to Fetch vector representation for words in a sentence
def sentence2sequence(sentence, glove_wordmap):
    tokens = sentence.lower().split(" ")
    rows = []
    words = []
    for token in tokens:
        i = len(token)
        token = re.sub('[.,()]', '', token)
        while len(token) > 0 and i > 0:
            word = token[:i]
            if word in glove_wordmap:
                rows.append(glove_wordmap[word])
                words.append(word)
                token = token[i:]
                i = len(token)
            else:
                i = i - 1
    return rows, words

def score_setup(row):
    convert_dict = {
        'entailment': 0,
        'neutral': 1,
        'contradiction': 2
    }
    score = np.zeros((3,))
    try:
        score[convert_dict[row['gold_label']]] += 1
    except:
        score[1] += 1
    return score


def fit_to_size(matrix, shape):
    res = np.zeros(shape)
    slices = tuple([slice(0,min(dim,shape[e])) for e, dim in enumerate(matrix.shape)])
    res[slices] = matrix[slices]
    return res

def lstm_cell():
    return tf.nn.rnn_cell.LSTMCell(lstm_size)


train = input("Would you like to train the data set? (Y/N)")
glove_wordmap = {}
sentences = []
file = open("test.jsonl","r")
text = file.read()
arr = []
arr.append(text)
d = "}"
lrr = [e+d for e in arr[0].split(d) if e]
data = []
del lrr[-1]

glove_wordmap = fetchglove()

hyp_sentences = []
evi_sentences = []
scores = []
labels = []

# Define constants
max_hypothesis_length, max_evidence_length = 15, 15
batch_size, vector_size, hidden_size = 128, 50, 50
lstm_size = hidden_size
weight_decay = 0.0001
learning_rate = 1
input_p, output_p = 0.5, 0.5
training_iterations_count = 100000
display_step = 10
num_of_layers = 2

lstm = tf.nn.rnn_cell.LSTMCell(lstm_size) #tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_of_layers)])
lstm_drop = tf.contrib.rnn.DropoutWrapper(lstm, input_p, output_p)

for elem in lrr:
    temp = json.loads(elem)
    hyp_sentences.append(np.vstack(sentence2sequence(temp["sentence1"],glove_wordmap)[0]))
    evi_sentences.append(np.vstack(sentence2sequence(temp["sentence2"],glove_wordmap)[0]))
    scores.append(score_setup(temp))
    labels.append(temp["gold_label"])

# fit the sentences to match 15 word matrix
hyp_sentences = np.stack([fit_to_size(x,(max_hypothesis_length,vector_size)) for x in hyp_sentences])
evi_sentences = np.stack([fit_to_size(x,(max_evidence_length,vector_size)) for x in evi_sentences])
correct_scores = np.array(scores)

# Create placeholders to put the hypothesis and evidence sentences and the output
hyp = tf.placeholder(tf.float32, [batch_size,max_hypothesis_length, vector_size], 'hypothesis')
evi = tf.placeholder(tf.float32, [batch_size, max_evidence_length, vector_size], 'evidence')
y = tf.placeholder(tf.float32, [batch_size,3], 'label')

lstm_back = tf.nn.rnn_cell.LSTMCell(lstm_size)
lstm_drop_back = tf.contrib.rnn.DropoutWrapper(lstm_back, input_p, output_p)

fc_initializer1 = tf.random_normal_initializer(stddev=0.1)

fc_weight1 = tf.get_variable('fc_weight1', [2*hidden_size, 100], initializer= fc_initializer1)

fc_initializer2 = tf.random_normal_initializer(stddev=0.1)

fc_weight2 = tf.get_variable('fc_weight2', [100, 3],
                            initializer = fc_initializer2)

fc_bias1 = tf.get_variable('bias1', [100])

fc_bias2 = tf.get_variable('bias2', [3])

tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                     tf.nn.l2_loss(fc_weight1))

x = tf.concat([hyp,evi], 1)

x = tf.transpose(x, [1,0,2])

x = tf.reshape(x,[-1,vector_size])

x = tf.split(x, max_evidence_length + max_hypothesis_length)

rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm, lstm_back,
                                                            x, dtype=tf.float32)

classification_scores = tf.matmul(rnn_outputs[-1], fc_weight1) + fc_bias1

classification_scores2 = tf.matmul(classification_scores, fc_weight2) + fc_bias2

# Accuracy Calculations
with tf.variable_scope('Accuracy'):
    predicts = tf.cast(tf.argmax(classification_scores2, 1), 'int32')
    #predicts = tf.cast(tf.nn.softmax(tf.reshape(classification_scores2,[-1,batch_size])), 'int32')
    y_label = tf.cast(tf.argmax(y, 1), 'int32')
    corrects = tf.equal(predicts, y_label)
    num_corrects = tf.reduce_sum(tf.cast(corrects,tf.float32))
    accuracy = tf.reduce_mean(tf.cast(corrects,tf.float32))

# Loss Calculation
with tf.variable_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = classification_scores2, labels= y)
    loss = tf.reduce_mean(cross_entropy)
    total_loss = loss + weight_decay * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
opt_op = optimizer.minimize(total_loss)

# Now start the training for the first layer
init = tf.global_variables_initializer()
# Function to save the current trained model
saver = tf.train.Saver()
sess = tf.Session()
# Launch the TensorFlow session
if train == 'Y':
    sess.run(init)

    training_iterations = range(0, training_iterations_count, batch_size)

    training_iterations = tqdm(training_iterations)

    for i in training_iterations:

        # select indices for a random data subset
        batch = np.random.randint(hyp_sentences.shape[0], size = batch_size)

        # initialize the placeholders based on the random values
        hyps, evis, ys = (hyp_sentences[batch,:], evi_sentences[batch,:], correct_scores[batch])
        # Run the optimization
        sess.run([opt_op], feed_dict={hyp: hyps, evi: evis, y: ys})

        if (i/batch_size) % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={hyp: hyps, evi: evis, y: ys})
            # Calculate batch loss
            tmp_loss = sess.run(loss, feed_dict={hyp: hyps, evi: evis, y: ys})
            # Display results
            print("Iter " + str(i/batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(tmp_loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

    # Save the Final Model
    saver.save(sess, 'C:\\Users\\pylak\\Documents\\Fall_2018\\NLP\\Final_model')
    print('Model saved to Root Directory of the python file')
    sess.close()

else:
    # Test the accuracy on a sentence
    # Save the Model and use it for testing
    saver.restore(sess, 'C:\\Users\\pylak\\Documents\\Fall_2018\\NLP\\Final_model')
    test_evidence = ["so i have to find a way to supplement that"]
    test_hypothesis = ["I need a way to add something extra."]

    sentence2 = [fit_to_size(np.vstack(sentence2sequence(hypothesis,glove_wordmap)[0]), (max_hypothesis_length,vector_size))
                  for hypothesis in test_hypothesis]
    sentence1 = [fit_to_size(np.vstack(sentence2sequence(evidence,glove_wordmap)[0]), (max_hypothesis_length,vector_size))
                  for evidence in test_evidence]
    prediction = sess.run(classification_scores2, feed_dict={hyp: (sentence1 * batch_size),
                                                            evi: (sentence2 * batch_size),
                                                            y: [[0,0,0]]*batch_size})
    print(["Positive", "Neutral", "Negative"][np.argmax(prediction[0])]+
          " entailment")
    print(prediction[0])
    sess.close()
