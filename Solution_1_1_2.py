import tensorflow as tf
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

adam_optimizer = tf.keras.optimizers.Adam()


# softmax function implementation
def softmax(logits):
    return tf.exp(logits) / tf.reduce_sum(tf.exp(logits))


# cross entropy implementation
def cross_entropy(Y, predictedY):
    loss = -tf.reduce_sum(Y * tf.math.log(predictedY), axis=1)
    return loss


def forward_pass(variables, x):
    # forward propagation step
    layer_1_Z = tf.matmul(x, variables["layer_1_weights"]) + variables["layer_1_bias"]  # layer 1 Z calculation
    layer_1_activation = tf.keras.activations.relu(layer_1_Z)  # Activation of layer 1

    layer_2_Z = tf.matmul(layer_1_activation, variables["layer_2_weights"]) + variables[
        "layer_2_bias"]  # layer 2 Z calculation
    layer_2_activation = tf.keras.activations.relu(layer_2_Z)

    layer_3_Z = tf.matmul(layer_2_activation, variables["layer_3_weights"]) + variables[
        "layer_3_bias"]  # layer 3 z calculation
    predictedY = softmax(layer_3_Z)
    return predictedY


# loss function implementation
def loss_func(y, predicted_Y):
    # cross_entropy = tf.keras.losses.BinaryCrossentropy()
    loss = cross_entropy(y, predicted_Y)
    return loss


# calculate accuracy
def calculate_accuracy(y, predicted_y):
    # round the predictions by logistical unit to either 1 or 0
    predictions = tf.round(predicted_y)

    # tf.equal will return a Boolean array : True if prediction correct, False otherwise
    # tf.cast converts the resulting Boolean array to a numerical array
    # if True (correct prediction),0 if False( incorrect prediction)

    predictions_correct = tf.cast(tf.equal(predictions, y), dtype=tf.float32)
    # Finally, we just determine the mean value of predictions_correct
    accuracy = tf.reduce_mean(predictions_correct)
    return accuracy


def gradient_loop(variables, X_train, Y_train, X_val, Y_val, X_test, Y_test, num_Iterations):
    # Iterate our training loop
    train_accuracy = []
    testAccuracy = []
    val_accuracy = []
    train_loss = []
    val_loss = []
    for i in range(num_Iterations):
        # Create an instance of GradientTape to monitor the forward pass
        # and calcualte the gradients for each of the variables m and c
        with tf.GradientTape() as tape:
            predictedY = forward_pass(variables, x=X_train)
            currentLoss = loss_func(Y_train, predictedY)
            gradients = tape.gradient(currentLoss, variables.values())
            accuracy = calculate_accuracy(Y_train, predictedY)
            # print("Iteration", i, ":Loss=", currentLoss.numpy()[-1], "Acc:", accuracy.numpy() * 100)
            train_accuracy.append(accuracy.numpy() * 100)
            train_loss.append(currentLoss.numpy()[-1])
            adam_optimizer.apply_gradients(
                zip(gradients, variables.values()))
            predictedY_val = forward_pass(variables, x=X_val)
            currentLoss = loss_func(Y_val, predictedY_val)
            Val_accuracy = calculate_accuracy(Y_val, predictedY_val)
            # print("Iteration", i, ":Loss=", currentLoss.numpy()[-1], "Acc:", accuracy.numpy() * 100)
            val_loss.append(currentLoss.numpy()[-1])
            val_accuracy.append(Val_accuracy.numpy() * 100)
            predictedY = forward_pass(variables, X_test)
            test_accuracy = calculate_accuracy(Y_test, predictedY)
            # print("TestAccuracy:", test_accuracy.numpy() * 100)
            testAccuracy.append(test_accuracy.numpy() * 100)
    return train_accuracy, val_accuracy, testAccuracy, train_loss, val_loss


def model(X_train, Y_train, X_val, Y_val, X_test, Y_test, hidden_nodes_1, hidden_nodes_2, num_Iterations):
    variables = {}
    # set the number of coefficients equal to the number of features
    input_nodes = X_train.shape[1]  # no. of cols of X
    output_nodes = 10  # no. of output nodes

    # initialise weights going from input_nodes into hidden_nodes
    layer_1_weights = tf.Variable(tf.random.normal(shape=(input_nodes, hidden_nodes_1), mean=0.0, stddev=0.05))

    # initialise weights going from hidden_nodes into next hidden nodes
    layer_2_weights = tf.Variable(tf.random.normal(shape=(hidden_nodes_1, hidden_nodes_2), mean=0.0, stddev=0.05))

    # initialise weights going from hidden_nodes into output nodes
    layer_3_weights = tf.Variable(tf.random.normal(shape=(hidden_nodes_2, output_nodes), mean=0.0, stddev=0.05))

    # initialise bias weights going into hidden_nodes
    layer_1_bias = tf.Variable(tf.random.normal(shape=(1, hidden_nodes_1)))

    # initialise bias weights going into hidden_nodes
    layer_2_bias = tf.Variable(tf.random.normal(shape=(1, hidden_nodes_2)))

    # initialise bias weights going into output_node
    layer_3_bias = tf.Variable(tf.random.normal(shape=(1, output_nodes)))

    variables["layer_1_weights"] = layer_1_weights
    variables["layer_1_bias"] = layer_1_bias
    variables["layer_2_weights"] = layer_2_weights
    variables["layer_2_bias"] = layer_2_bias
    variables["layer_3_weights"] = layer_3_weights
    variables["layer_3_bias"] = layer_3_bias

    # call gradient decent, and get intercept(bias) and coefficients
    train_accuracy, val_accuracy, testAccuracy, train_loss, val_loss = gradient_loop(variables, X_train, Y_train, X_val,
                                                                                     Y_val, X_test, Y_test,
                                                                                     num_Iterations)
    return train_accuracy, val_accuracy, testAccuracy, train_loss, val_loss


def main():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # load the training and test data
    (tr_x, tr_y), (te_x, te_y) = fashion_mnist.load_data()

    tr_x, X_val, tr_y, y_val = train_test_split(tr_x, tr_y, test_size=0.25, random_state=1)
    # reshape the feature data
    tr_x = tr_x.reshape(tr_x.shape[0], 784)
    X_val = X_val.reshape(X_val.shape[0], 784)
    te_x = te_x.reshape(te_x.shape[0], 784)

    # normalise feature data
    tr_x = tr_x / 255.0
    tr_x = tf.convert_to_tensor(tr_x, dtype=tf.float32)

    X_val = X_val / 255.0
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)

    te_x = te_x / 255.0
    te_x = tf.convert_to_tensor(te_x, dtype=tf.float32)
    print("Shape of training features ", tr_x.shape)
    print("Shape of validation features ", X_val.shape)
    print("Shape of test features ", te_x.shape)

    # one hot encode the training labels and get the transpose
    tr_y = np_utils.to_categorical(tr_y, 10)
    tr_y = tf.convert_to_tensor(tr_y, dtype=tf.float32)
    print("Shape of training labels ", tr_y.shape)

    # one hot encode the training labels and get the transpose
    y_val = np_utils.to_categorical(y_val, 10)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    print("Shape of validation labels ", y_val.shape)

    # one hot encode the test labels and get the transpose
    te_y = np_utils.to_categorical(te_y, 10)
    te_y = tf.convert_to_tensor(te_y, dtype=tf.float32)
    print("Shape of testing labels ", te_y.shape)

    train_accuracy, val_accuracy, testAccuracy, train_loss, val_loss = model(X_train=tr_x, Y_train=tr_y, X_val=X_val,
                                                                             Y_val=y_val, X_test=te_x,
                                                                             Y_test=te_y, hidden_nodes_1=300,
                                                                             hidden_nodes_2=100,
                                                                             num_Iterations=4)

    #  "Model Accuracy"
    plt.plot(train_accuracy)
    plt.plot(val_accuracy)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Iteration')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # "Loss"
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model Loss')
    plt.ylabel('loss')
    plt.xlabel('Iteration')
    plt.legend(['val_loss', 'train_loss'], loc='upper left')
    plt.show()

    # "Test Accuracy"
    plt.plot(testAccuracy)
    plt.title('model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.legend(['test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
