import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
 
def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_uniform_initializer(minval=-1, maxval=1)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.matmul(input, W) + b

def my_network(input):
    with tf.variable_scope("layer_1"):
        output_1 = layer(input, [7, 10], [10])
    with tf.variable_scope("layer_2"):
        output_2 = layer(output_1, [10, 20], [20])
    with tf.variable_scope("layer_3"):
        output_3 = layer(output_2, [20, 10], [10])
    with tf.variable_scope("layer_4"):
        output_4 = layer(output_3, [10, 2], [2])
    return output_4

def eval_data(X_eval, y_eval):
    loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: X_eval, y: y_eval})
    return loss, acc

train = pd.read_csv('train.csv')
y = train.pop('Survived')

# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
categorical_vars = ['Pclass', 'Sex', 'Embarked']
continues_vars = ['Age', 'SibSp', 'Parch', 'Fare']
train['Embarked'] = train['Embarked'].fillna(' ')
X = train[categorical_vars + continues_vars].fillna(0)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = X, y
train_limit = int(X_train.shape[0]*0.9)

# Process categorical variables into ids.
X_train = X_train.copy()
#X_test = X_test.copy()
categorical_var_encoders = {}
for var in categorical_vars:
  le = LabelEncoder().fit(X_train[var])
  X_train[var + '_ids'] = le.transform(X_train[var])
  #X_test[var + '_ids'] = le.transform(X_test[var])
  X_train.pop(var)
  #X_test.pop(var)
  categorical_var_encoders[var] = le

# Process output into classifier
lb = preprocessing.LabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)
y_train = np.hstack((y_train, 1 - y_train))

#y_test = lb.transform(y_test)
#y_test = np.hstack((y_test, 1 - y_test))

x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name="X")
y = tf.placeholder(tf.float32, shape=[None, 2], name="Y")
learning_rate = tf.placeholder(tf.float32, shape=[], name="rate")

output = my_network(x)

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
loss_op = tf.reduce_mean(entropy)
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(loss_op)

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

EPOCHS = 100
rate = 0.01

if __name__ == '__main__':
    # Train model
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)

        loss = sess.run(train_op, feed_dict={x: X_train[:train_limit], y: y_train[:train_limit], learning_rate: rate})

        val_loss, val_acc = eval_data(X_train[train_limit:], y_train[train_limit:])

        print("EPOCH {} ...".format(i+1))
        print("Validation loss = {:.3f}".format(val_loss))
        print("Validation accuracy = {:.3f}".format(val_acc))

        print()

    # Evaluate on the test data
    #test_loss, test_acc = eval_data(X_test, y_test)
    #print("Test loss = {:.3f}".format(test_loss))
    #print("Test accuracy = {:.3f}".format(test_acc))

    saver.save(sess, './data/titanic_model',global_step=1000)