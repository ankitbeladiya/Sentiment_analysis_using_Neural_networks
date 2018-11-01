import numpy as np
import tensorflow as tf
from twitter_data_processing import create_feature_sets_and_labels

X_train, X_test, y_train, y_test = create_feature_sets_and_labels('data/training.csv',featureSize=10000)


l1_node = 550
l2_node = 550
l3_node = 550

n_class = 2
bathch_size = 1000

x=tf.placeholder('float',[None, X_train.shape[1]]) # 28*28 image size = 784
y=tf.placeholder('float')

def nn_model(data):
    # variables weights and bias
    h1_layer = {'weight':tf.Variable(tf.random_normal([X_train.shape[1],l1_node])),'bias':tf.Variable(tf.random_normal([l1_node]))}
    h2_layer = {'weight': tf.Variable(tf.random_normal([l1_node, l2_node])), 'bias': tf.Variable(tf.random_normal([l2_node]))}
    h3_layer = {'weight': tf.Variable(tf.random_normal([l2_node, l3_node])), 'bias': tf.Variable(tf.random_normal([l3_node]))}
    output_layer = {'weight':tf.Variable(tf.random_normal([l3_node,n_class])),'bias':tf.Variable(tf.random_normal([n_class]))}

    #model
    l1=tf.add(tf.matmul(data,h1_layer['weight']),h1_layer['bias'])
    l1=tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, h2_layer['weight']), h2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, h3_layer['weight']), h3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    return output

def train_nn(x):
    prediction = nn_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimize = tf.train.AdamOptimizer().minimize(cost)

    cycle = 1

    with tf.Session() as sesson:
        sesson.run(tf.global_variables_initializer())

        for i in range(cycle):
            cycle_loss = 0

            j = 0
            while j < X_train.shape[0]:
                start = j
                end = j + bathch_size

                bathch_x = np.array(X_train[start:end].toarray())
                bathch_y = np.array([x for x in y_train[start:end]])
                _, c = sesson.run([optimize,cost],feed_dict={x:bathch_x,y:bathch_y})
                cycle_loss += c

                j += bathch_size

            print('Cycle no ', i+1, 'of total cycle ', cycle, 'loss :', cycle_loss)


        check = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy=tf.reduce_mean(tf.cast(check, 'float'))
        print('accuracy:', accuracy.eval({x:X_test.toarray(), y:[x for x in y_test]}))

train_nn(x)


