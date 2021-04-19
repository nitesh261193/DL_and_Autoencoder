import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

adam_optimizer = tf.keras.optimizers.Adam()


# sigmoid function implementation
def sigmoid(z):
    return tf.math.sigmoid(z)


def forward_pass(variables, x):
    # forward propagation step
    layer_1_Z = tf.matmul(x, variables["layer_1_weights"]) + variables["layer_1_bias"]  # layer 1 Z calculation
    layer_1_activation = tf.keras.activations.relu(layer_1_Z)  # Activation of layer 1

    layer_2_Z = tf.matmul(layer_1_activation, variables["layer_2_weights"]) + variables[
        "layer_2_bias"]  # layer 2 Z calculation
    layer_2_activation = tf.keras.activations.relu(layer_2_Z)

    layer_3_Z = tf.matmul(layer_2_activation, variables["layer_3_weights"]) + variables[
        "layer_3_bias"]  # layer 3 Z calculation
    layer_3_activation = tf.keras.activations.relu(layer_3_Z)

    layer_4_Z = tf.matmul(layer_3_activation, variables["layer_4_weights"]) + variables[
        "layer_4_bias"]  # layer 4 Z calculation
    layer_4_activation = tf.keras.activations.relu(layer_4_Z)

    layer_5_Z = tf.matmul(layer_4_activation, variables["layer_5_weights"]) + variables[
        "layer_5_bias"]  # layer 5 Z calculation
    layer_5_activation = tf.keras.activations.relu(layer_5_Z)

    layer_6_Z = tf.matmul(layer_5_activation, variables["layer_6_weights"]) + variables[
        "layer_6_bias"]  # layer 6 z calculation
    predictedY = sigmoid(layer_6_Z)
    return predictedY


# mean absolute error method implementation
def mean_absolute_error(y, predicted_y):
    return tf.reduce_mean(tf.abs(predicted_y - y), axis=-1)


# Calculate accuracy
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


def gradient_loop(variables, X_train, Y_train, X_test, Y_test, num_Iterations):
    # Iterate our training loop
    train_accuracy = []
    testAccuracy = []
    train_loss = []
    for i in range(num_Iterations):
        final_prediction = tf.random.normal(shape=X_test.shape, mean=0.0, stddev=.05)
        # Create an instance of GradientTape to monitor the forward pass
        # and calcualte the gradients for each of the variables m and c
        with tf.GradientTape() as tape:
            predictedY = forward_pass(variables, x=X_train)
            currentLoss = mean_absolute_error(Y_train, predictedY)
            gradients = tape.gradient(currentLoss, variables.values())
            accuracy = calculate_accuracy(Y_train, predictedY)
            # print("Iteration", i, ":Loss=", currentLoss.numpy()[-1], "Acc:", accuracy.numpy() * 100)
            train_accuracy.append(accuracy.numpy() * 100)
            train_loss.append(currentLoss.numpy()[-1])
            adam_optimizer.apply_gradients(
                zip(gradients, variables.values()))
            predictedY = forward_pass(variables, X_test)
            test_accuracy = calculate_accuracy(Y_test, predictedY)
            # print("TestAccuracy:", test_accuracy.numpy() * 100)
            final_prediction = predictedY
            testAccuracy.append(test_accuracy.numpy() * 100)
    return train_accuracy, testAccuracy, train_loss, final_prediction


def model(X_train, Y_train, X_test, Y_test, num_Iterations):
    variables = {}
    # set the number of coefficients equal to the number of features
    input_nodes = X_train.shape[1]  # no. of cols of X

    # initialise weights going from input_nodes into hidden_nodes
    layer_1_weights = tf.Variable(tf.random.normal(shape=(input_nodes, 128), mean=0.0, stddev=0.05))

    # initialise weights going from hidden_nodes into next hidden nodes
    layer_2_weights = tf.Variable(tf.random.normal(shape=(128, 64), mean=0.0, stddev=0.05))
    layer_3_weights = tf.Variable(tf.random.normal(shape=(64, 32), mean=0.0, stddev=0.05))
    layer_4_weights = tf.Variable(tf.random.normal(shape=(32, 64), mean=0.0, stddev=0.05))
    layer_5_weights = tf.Variable(tf.random.normal(shape=(64, 128), mean=0.0, stddev=0.05))
    layer_6_weights = tf.Variable(tf.random.normal(shape=(128, 784), mean=0.0, stddev=0.05))

    # initialise bias weights going into hidden_nodes
    layer_1_bias = tf.Variable(tf.random.normal(shape=(1, 128)))
    layer_2_bias = tf.Variable(tf.random.normal(shape=(1, 64)))
    layer_3_bias = tf.Variable(tf.random.normal(shape=(1, 32)))
    layer_4_bias = tf.Variable(tf.random.normal(shape=(1, 64)))
    layer_5_bias = tf.Variable(tf.random.normal(shape=(1, 128)))
    # initialise bias weights going into output_node
    layer_6_bias = tf.Variable(tf.random.normal(shape=(1, 784)))

    variables["layer_1_weights"] = layer_1_weights
    variables["layer_1_bias"] = layer_1_bias
    variables["layer_2_weights"] = layer_2_weights
    variables["layer_2_bias"] = layer_2_bias
    variables["layer_3_weights"] = layer_3_weights
    variables["layer_3_bias"] = layer_3_bias
    variables["layer_4_weights"] = layer_4_weights
    variables["layer_4_bias"] = layer_4_bias
    variables["layer_5_weights"] = layer_5_weights
    variables["layer_5_bias"] = layer_5_bias
    variables["layer_6_weights"] = layer_6_weights
    variables["layer_6_bias"] = layer_6_bias

    # call gradient decent, and get intercept(bias) and coefficients
    train_accuracy, testAccuracy, train_loss, final_prediction = gradient_loop(variables, X_train, Y_train, X_test,
                                                                               Y_test,
                                                                               num_Iterations)
    return train_accuracy, testAccuracy, train_loss, final_prediction


def main():
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    # Normalize train and test data
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Reshape so that each instance is a linear array of 784 normalized pixel values
    x_train = x_train.reshape((len(x_train), 784))
    x_test = x_test.reshape((len(x_test), 784))
    print(x_train.shape, x_test.shape)

    # Add random noise to the image
    noise_factor = 0.2
    x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
    x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

    # Clip the resulting values so that they don't fall outside the upper and lower normalized value of 0 and 1
    x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
    x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

    n = 10  # Number of images to display

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test_noisy[i].numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    train_accuracy, testAccuracy, train_loss, final_prediction = model(X_train=x_train_noisy, Y_train=x_train,
                                                                       X_test=x_test_noisy,
                                                                       Y_test=x_test, num_Iterations=1000)

    #  "Accuracy"
    plt.plot(train_accuracy)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Iteration')
    plt.legend(['train'], loc='upper left')
    plt.show()

    # "Loss"
    plt.plot(train_loss)
    plt.title('model Loss')
    plt.ylabel('loss')
    plt.xlabel('Iteration')
    plt.legend(['train_loss'], loc='upper left')
    plt.show()

    # "Accuracy"
    plt.plot(testAccuracy)
    plt.title('model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.legend(['test'], loc='upper left')
    plt.show()

    n = 10  # Number of images to display

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # noisy input
        ax = plt.subplot(3, n, i + 1)
        plt.title("Noisy input ")
        plt.imshow(x_test_noisy[i].numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Reconstructed output images
        ax = plt.subplot(3, n, i + 1 + n)
        plt.title("Reconstructed output")
        plt.imshow(final_prediction[i].numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Original without noise
        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.title("Original Target")
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    main()
