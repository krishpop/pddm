# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def feedforward_network(inputStates, inputSize, outputSize, num_fc_layers,
                        depth_fc_layers, tf_datatype, scope):

    with tf.variable_scope(str(scope)):

        #concat K entries together [bs x K x sa] --> [bs x ksa]
        inputState = tf.layers.flatten(inputStates)

        #vars
        intermediate_size = depth_fc_layers
        reuse = False
        initializer = tf.glorot_normal_initializer(
            seed=None, dtype=tf_datatype)
        fc = tf.layers.dense

        # make hidden layers
        for i in range(num_fc_layers):
            if i==0:
                fc_i = fc(
                    inputState,
                    units=intermediate_size,
                    activation=None,
                    kernel_initializer=initializer,
                    bias_initializer=initializer,
                    reuse=reuse,
                    trainable=True)
            else:
                fc_i = fc(
                    h_i,
                    units=intermediate_size,
                    activation=None,
                    kernel_initializer=initializer,
                    bias_initializer=initializer,
                    reuse=reuse,
                    trainable=True)
            h_i = tf.nn.relu(fc_i)

        # make output layer
        z = fc(
            h_i,
            units=outputSize,
            activation=None,
            kernel_initializer=initializer,
            bias_initializer=initializer,
            reuse=reuse,
            trainable=True)

    return z
