"""Import data from the two JSON files"""

import numpy as np
import tensorflow as tf
import json
import math

DATA_FILE_UP = 'data/pushup_up.json'
DATA_FILE_DOWN = 'data/pushup_down.json'
NUM_FEATURES = 39
EXPORT_DIR = 'pushup_classifier_model'

with open(DATA_FILE_UP) as json_data_up:
    data_up_instance_major = json.load(json_data_up)
with open(DATA_FILE_DOWN) as json_data_down:
    data_down_instance_major = json.load(json_data_down)

data_len = len(data_up_instance_major) + len(data_down_instance_major)
train_len = math.floor(0.8 * data_len) # 80% train, 20% eval
eval_len = data_len - train_len

def feature_name(i):
    body_part_names = [ 'neck',
                        'right-shoulder',
                        'right-elbow',
                        'right-hand',
                        'left-shoulder',
                        'left-elbow',
                        'left-hand',
                        'right-hip',
                        'right-knee',
                        'right-foot',
                        'left-hip',
                        'left-knee',
                        'left-foot'
                      ]
    indicators = ['x', 'y', 'c']
    return body_part_names[i // 3] + '-' + indicators[i % 3]


# Add instances to their labels, shuffle together, then split into training and eval sets
data_instance_major = data_up_instance_major + data_down_instance_major
labels = [0]*len(data_up_instance_major) + [1]*len(data_down_instance_major)
combined = []
for i in range(data_len):
    combined.append((data_instance_major[i], labels[i]))
np.random.shuffle(combined)
train_set = combined[:train_len]
eval_set = combined[train_len:]

# Split tuple, construct dict of {feature names : [arrays of features], ...}, construct list of labels
# for key, val in train_set.items():
#     print(key)
#     print(len(val))
features_train_arr, labels_train_arr = zip(*train_set)
features_train_arr_zip = list(zip(*features_train_arr))
features_train = {}
labels_train = np.array(labels_train_arr)
for i in range(NUM_FEATURES):
    features_train[feature_name(i)] = np.array(features_train_arr_zip[i])

features_eval_arr, labels_eval_arr = zip(*eval_set)
features_eval_arr_zip = list(zip(*features_eval_arr))
features_eval = {}
labels_eval = np.array(labels_eval_arr)
for i in range(NUM_FEATURES):
    features_eval[feature_name(i)] = np.array(features_eval_arr_zip[i])

# Construct classifier
feature_cols = [tf.feature_column.numeric_column(feature_name(i)) for i in range(NUM_FEATURES)]
classifier = tf.estimator.LinearClassifier(feature_columns=feature_cols)

# Training
# for key, val in features_train.items():
#     print(key)
#     print(len(val))
    
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    features_train, labels_train, num_epochs=None, shuffle=True)

classifier.train(input_fn=train_input_fn, steps=1000)

# Testing
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    features_train, labels_train, num_epochs=1, shuffle=False)

test_metrics = classifier.evaluate(input_fn=test_input_fn)
print("train metrics: %r"% test_metrics)

# Evaluation / Prediction
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    features_eval, labels_eval, num_epochs=1, shuffle=False)

eval_metrics = classifier.evaluate(input_fn=eval_input_fn)
print("eval metrics: %r"% eval_metrics)

# builder = tf.saved_model.builder.SavedModelBuilder(EXPORT_DIR)

# with tf.Session(graph=tf.Graph()) as sess:
#     builder.add_meta_graph_and_variables(sess, ["tag"])
# builder.save()

feature_spec = {'neck-x': tf.FixedLenFeature([0], tf.float32),
                'neck-y': tf.FixedLenFeature([0], tf.float32),
                'neck-c': tf.FixedLenFeature([0], tf.float32),
                'right-shoulder-x': tf.FixedLenFeature([0], tf.float32),
                'right-shoulder-y': tf.FixedLenFeature([0], tf.float32),
                'right-shoulder-c': tf.FixedLenFeature([0], tf.float32),
                'right-elbow-x': tf.FixedLenFeature([0], tf.float32),
                'right-elbow-y': tf.FixedLenFeature([0], tf.float32),
                'right-elbow-c': tf.FixedLenFeature([0], tf.float32),
                'right-hand-x': tf.FixedLenFeature([0], tf.float32),
                'right-hand-y': tf.FixedLenFeature([0], tf.float32),
                'right-hand-c': tf.FixedLenFeature([0], tf.float32),
                'left-shoulder-x': tf.FixedLenFeature([0], tf.float32),
                'left-shoulder-y': tf.FixedLenFeature([0], tf.float32),
                'left-shoulder-c': tf.FixedLenFeature([0], tf.float32),
                'left-elbow-x': tf.FixedLenFeature([0], tf.float32),
                'left-elbow-y': tf.FixedLenFeature([0], tf.float32),
                'left-elbow-c': tf.FixedLenFeature([0], tf.float32),
                'left-hand-x': tf.FixedLenFeature([0], tf.float32),
                'left-hand-y': tf.FixedLenFeature([0], tf.float32),
                'left-hand-c': tf.FixedLenFeature([0], tf.float32),
                'right-hip-x': tf.FixedLenFeature([0], tf.float32),
                'right-hip-y': tf.FixedLenFeature([0], tf.float32),
                'right-hip-c': tf.FixedLenFeature([0], tf.float32),
                'right-knee-x': tf.FixedLenFeature([0], tf.float32),
                'right-knee-y': tf.FixedLenFeature([0], tf.float32),
                'right-knee-c': tf.FixedLenFeature([0], tf.float32),
                'right-foot-x': tf.FixedLenFeature([0], tf.float32),
                'right-foot-y': tf.FixedLenFeature([0], tf.float32),
                'right-foot-c': tf.FixedLenFeature([0], tf.float32),
                'left-hip-x': tf.FixedLenFeature([0], tf.float32),
                'left-hip-y': tf.FixedLenFeature([0], tf.float32),
                'left-hip-c': tf.FixedLenFeature([0], tf.float32),
                'left-knee-x': tf.FixedLenFeature([0], tf.float32),
                'left-knee-y': tf.FixedLenFeature([0], tf.float32),
                'left-knee-c': tf.FixedLenFeature([0], tf.float32),
                'left-foot-x': tf.FixedLenFeature([0], tf.float32),
                'left-foot-y': tf.FixedLenFeature([0], tf.float32),
                'left-foot-c': tf.FixedLenFeature([0], tf.float32)}

# sirf = tf.build_parsing_serving_input_receiver_fn(feature_spec)

def serving_input_receiver_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[128],
                                         name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

classifier.export_savedmodel(EXPORT_DIR, serving_input_receiver_fn)
