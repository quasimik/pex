from flask import Flask, jsonify, render_template, request
app = Flask(__name__)

import tensorflow as tf
import sys
import numpy as np
import json
import math
import inspect

EXPORT_DIR = 'pushup_classifier_model'
DATA_FILE_UP = 'data/pushup_up.json'
DATA_FILE_DOWN = 'data/pushup_down.json'
NUM_FEATURES = 39



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
    
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    features_train, labels_train, num_epochs=None, shuffle=True)

classifier.train(input_fn=train_input_fn, steps=10)

# Testing
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    features_train, labels_train, num_epochs=1, shuffle=False)

test_metrics = classifier.evaluate(input_fn=test_input_fn)
print("train metrics: %r"% test_metrics)

# Evaluation
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    features_eval, labels_eval, num_epochs=1, shuffle=False)

eval_metrics = classifier.evaluate(input_fn=eval_input_fn)
print("eval metrics: %r"% eval_metrics)



@app.route('/pushup', methods = ['POST'])
def pushup():

    data = request.get_json()
    features_pred = {}
    for i in range(NUM_FEATURES):
        features_pred[feature_name(i)] = np.array([data[i]])

    # Prediction
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        features_pred, num_epochs=1, shuffle=False)

    predictions = classifier.predict(input_fn=pred_input_fn)

    for i, p in enumerate(predictions):
        val = p['class_ids'][0]

    return str(val)


@app.route('/lmao')
def lmao():
    return request.args.get('lmao')

# @app.route('/show_pushup_status')
# def show_pushup_status():


@app.route('/api_predict_pushup_status', methods = ['POST'])
def api_predict_pushup_status():
    return classifier


@app.route('/')
def index():
    # return render_template('index.html')
    return render_template('index.html')
