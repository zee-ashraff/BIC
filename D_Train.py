import D_Dataset
import tensorflow as tf
import os

# Adding Seed so that random initialization is consistent
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

tf.compat.v1.set_random_seed(2)
batch_size = 32

# Prepare input data
parent_directory=os.path.dirname(os.path.realpath(__file__))
train_path = str(parent_directory) + '/training_set'
classes = os.listdir(train_path)
num_classes = len(classes)

#image size is set to 128x128 pixels and 3 channels
image_size = 128
num_channels = 3

# 20% of the data will automatically be used for validation
validation_size = 0.2

#training and validation sets are loaded to memory for training
data = D_Dataset.read_train_sets(train_path, image_size, classes, validation_size=validation_size)

session = tf.compat.v1.Session()
x = tf.compat.v1.placeholder(tf.float32, shape=[None, image_size, image_size, num_channels], name='x')

## labels
y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

##Network graph params
conv1_conv2_filter_size = 3
conv1_conv2_num_filters = 32
conv3_filter_size = 3
conv3_num_filters = 64

fully_connected_layer_size = 128

#initialize weights and biases before training
def create_weights(shape):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


#define the network
def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
    #define weights to be trained with create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## create biases with create_biases function
    biases = create_biases(num_filters)

    #create convolutional layer
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases

    #max-pooling.
    layer = tf.nn.max_pool2d(input=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #activation function ReLU is used
    layer = tf.nn.relu(layer)
    return layer


def create_flatten_layer(layer):
    #get shape
    layer_shape = layer.get_shape()
    ## Number of features = image_height * image_width* num_channels
    num_features = layer_shape[1:4].num_elements()
    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])
    return layer


def create_fully_connected_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

#convolutional layer 1 definition
conv_layer_1 = create_convolutional_layer(input=x,
                                         num_input_channels=num_channels,
                                         conv_filter_size=conv1_conv2_filter_size,
                                         num_filters=conv1_conv2_num_filters)
#apply dropout
dropout_1 = tf.nn.dropout(
    conv_layer_1,
    rate=0.25,
    noise_shape=None,
    seed=None,
    name=None
)

#convolutional layer 2 definition
conv_layer_2 = create_convolutional_layer(input=dropout_1,
                                         num_input_channels=conv1_conv2_num_filters,
                                         conv_filter_size=conv1_conv2_filter_size,
                                         num_filters=conv1_conv2_num_filters)
#apply dropout
dropout_2 = tf.nn.dropout(
    conv_layer_2,
    rate=0.25,
    noise_shape=None,
    seed=None,
    name=None
)

#convolutional layer 3 definition
conv_layer_3 = create_convolutional_layer(input=dropout_2,
                                         num_input_channels=conv1_conv2_num_filters,
                                         conv_filter_size=conv3_filter_size,
                                         num_filters=conv3_num_filters)

#flatten
layer_flat = create_flatten_layer(conv_layer_3)

#fully connected layers definition
fully_connected_layer_1 = create_fully_connected_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fully_connected_layer_size,
                            use_relu=True)

fully_connected_layer_2 = create_fully_connected_layer(input=fully_connected_layer_1,
                            num_inputs=fully_connected_layer_size,
                            num_outputs=num_classes,
                            use_relu=False)

#apply  softmax to outpout of FC2
y_pred = tf.nn.softmax(fully_connected_layer_2, name='y_pred')

y_pred_cls = tf.argmax(y_pred, axis=1)
session.run(tf.compat.v1.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fully_connected_layer_2,
                                                        labels=y_true)
#define the cost functio and optimizer and get accuracy metrics
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.compat.v1.global_variables_initializer())

#show statistics during training
def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


total_iterations = 0

saver = tf.compat.v1.train.Saver()

#defition of train function 
def train(num_iteration):
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iteration):
        #images passed in depending on batch size
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}
        #run the session
        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, 'BIC')

    total_iterations += num_iteration

train(num_iteration=9000)