import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

categorical_vars = ['Pclass', 'Sex', 'Embarked']
continues_vars = ['Age', 'SibSp', 'Parch', 'Fare']

test = pd.read_csv('test.csv')
test['Embarked'] = test['Embarked'].fillna(' ')
X = test[categorical_vars + continues_vars].fillna(0)
# Process categorical variables into ids.
X_test = X.copy()
categorical_var_encoders = {}
for var in categorical_vars:
  le = LabelEncoder().fit(X_test[var])
  X_test[var + '_ids'] = le.transform(X_test[var])
  X_test.pop(var)
  categorical_var_encoders[var] = le

sess=tf.Session()
saver = tf.train.import_meta_graph('./data/titanic_model-1000.meta')
saver.restore(sess, tf.train.latest_checkpoint('./data/'))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
feed_dict = {X:X_test}

predict = graph.get_operation_by_name("predict").outputs[0]

result = sess.run([predict], feed_dict=feed_dict)
result = np.array(result).reshape(-1).astype(int)
passID = np.array(test["PassengerId"]).astype(int)

#result = np.dstack((passID, result))
np.savetxt("test_res.csv", np.c_[passID, result], fmt='%i', delimiter=",")
