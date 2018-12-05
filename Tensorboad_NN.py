import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
import numpy as np
import re
from nltk.corpus import words, stopwords
import nltk


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """


class nn(object):

    def __init__(self, data_path, data_size, N_class, skip_step=10, batch_size=5000, learning_rate=0.001):
        self.data_path = data_path
        self.data_size = data_size
        self.N_class = N_class
        self.batch_size = batch_size
        self.skip_step = skip_step
        self.lr = learning_rate
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
        self.Layer_1_neurons = 500
        self.Layer_2_neurons = 500
        self.Layer_3_neurons = 500

    @staticmethod
    def _normalize_document(self, sentence):
        wlp = nltk.WordPunctTokenizer()

        stopword_list = stopwords.words('english')
        stopword_list.remove('not')
        stopword_list.remove('no')

        word_list = words.words()

        lemmatizer = nltk.WordNetLemmatizer()

        # lower case and remove special characters\whitespaces
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence, re.I | re.A)
        sentence = sentence.strip().lower()

        # tokenizing
        sentence = wlp.tokenize(sentence)

        # lemmatizing
        sentence = [lemmatizer.lemmatize(x) for x in sentence]

        #  removing stop words
        sentence = [x for x in sentence if x.lower() not in stopword_list and x.lower() in word_list]

        return ' '.join(sentence)

    def _get_data(self):
        """	step 1: import data from the given path	"""
        with tf.name_scope('data'):

            try:
                self.data = pd.read_csv(self.data_path, encoding='utf_8',
                                        names=['rating', 'id', 'date', 'query', 'user', 'tweet'])
                self.data = self.data[['rating', 'tweet']].iloc[
                            800000 - int(self.data_size / 2):800000 + int(self.data_size / 2)]
                self.data['tweet'] = self.data['tweet'].apply(self._normalize_document)
                self.data['rating'] = self.data['rating'].map({0: [0, 1], 4: [1, 0]})
                print('Processing ', len(self.data), 'user reviews')
            except:
                print('Could not load Data')

    def _split_data(self):
        """step 2: spliting data into train and test split		"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data['tweet'], self.data['rating'],
                                                                                random_state=0, test_size=0.1)

    def _word2vec(self):
        """ step 3: converting data into term frequency matrix """
        self.vect = TfidfVectorizer(ngram_range=(1, 2))
        self.X_train = self.vect.fit_transform(self.X_train)
        self.X_test = self.vect.transform(self.X_test)

        print('Dictionary size is ', len(self.vect.idf_))

    def _create_place_holder(self):
        """	step 5: initializing placeholder for graph	"""
        self.X = tf.placeholder(tf.float32, [None, self.X_train.shape[1]])
        self.Y = tf.placeholder(tf.float32)

    def _structure_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """ step 6: defining structure of layer	"""
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = tf.Variable(tf.random_normal([input_dim, output_dim]), dtype=tf.float32)
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.random_normal([output_dim]), dtype=tf.float32)
            with tf.name_scope('activations'):
                activations = act(tf.add(tf.matmul(input_tensor, weights), biases))
                tf.summary.histogram('activations', activations)
            return activations

    def _structure_model(self):
        """ step 7: defining structure of model		"""
        self.Hidden_1 = self._structure_layer(self.X, self.X_train.shape[1], self.Layer_1_neurons, 'layer_1')
        self.Hidden_2 = self._structure_layer(self.Hidden_1, self.Layer_1_neurons, self.Layer_2_neurons, 'layer_2')
        self.Hidden_3 = self._structure_layer(self.Hidden_2, self.Layer_2_neurons, self.Layer_3_neurons, 'layer_3')
        self.output = self._structure_layer(self.Hidden_3, self.Layer_3_neurons, self.N_class, 'output_layer',
                                            act=tf.identity)

    def _create_loss(self):
        """ Step 8: define the loss function """
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.Y))

    def _create_optimizer(self):
        """ Step 9: define optimizer """
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

    def _accuracy(self):
        """ Step 10: finding accuracy """
        with tf.name_scope('accuracy'):
            self.correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.output, 1))
            with tf.name_scope('accuracy'):
                self.accuracy = (tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))) * 100

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accurate', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            # because we have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._get_data()
        self._split_data()
        self._word2vec()
        self._create_place_holder()
        self._structure_model()
        self._create_loss()
        self._create_optimizer()
        self._accuracy()
        self._create_summaries()

    def train(self, num_train_steps):
        saver = tf.train.Saver()  # defaults to saving all variables

        # make dir for checkpoints
        try:
            os.mkdir('checkpoints')
        except OSError:
            pass

        initial_step = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # initializer summary writer and summary path
            writer = tf.summary.FileWriter('logs' + str(self.lr), sess.graph)
            initial_step = self.global_step.eval()

            # bath processing
            for index in range(initial_step, initial_step + num_train_steps):
                training_loss = 0.0
                j = 0
                while j < self.X_train.shape[0]:
                    start = j
                    end = j + self.batch_size
                    self.batch_x = self.X_train[start:end].toarray()
                    self.batch_y = np.array([x for x in self.y_train[start:end]])
                    try:
                        # main training loop
                        loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op],
                                                          feed_dict={self.X: self.batch_x, self.Y: self.batch_y})
                        training_loss += loss_batch
                        j += self.batch_size

                        # write summary after one epoc
                        if end == self.X_train.shape[0]:
                            writer.add_summary(summary, global_step=index)

                    except tf.errors.OutOfRangeError:
                        print('Out of range')

                # Calculate accuracy on test data
                accuracy = sess.run([self.accuracy], feed_dict={self.X: self.X_test.toarray(),
                                                                self.Y: np.array([x for x in self.y_test])})
                if (index + 1) % self.skip_step == 0:
                    print('Step {}   loss: {:5.1f}    accuracy: {:5.1f}'.format(index, training_loss,
                                                                                accuracy[0]))
                    saver.save(sess, 'checkpoints/nn', index)

            writer.close()

    def predict(self, sentacnce):
        # use trained model for prediction
        sentacnce = self.vect.transform(pd.Series(data=sentacnce)).toarray()
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

            # restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            pridiction = tf.argmax(sess.run([self.output], feed_dict={self.X: sentacnce})[0])
            print(pridiction)
            if pridiction == 1:
                print("Entered Sentance is Negative")
            else:
                print('Entered Sentance is Positive')


def main():
    data_path = 'data/training.csv'
    model = nn(data_path, data_size=10000, batch_size=5000, N_class=2, skip_step=1)
    model.build_graph()
    model.train(10)
    model.predict("you are ugly person")
    model.predict("I am a good man")


if __name__ == '__main__':
    main()
