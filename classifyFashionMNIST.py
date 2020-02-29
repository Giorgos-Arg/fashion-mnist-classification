try:
    import h5py
except ImportError:
    h5py = None
from keras.datasets import fashion_mnist
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(cm, classes):
    cmap = plt.cm.get_cmap('Blues')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


# Plots performance over the training epochs.
def show_results(nn_model_train):
    accuracy = nn_model_train.history['accuracy']
    val_accuracy = nn_model_train.history['val_accuracy']
    loss = nn_model_train.history['loss']
    val_loss = nn_model_train.history['val_loss']
    epochs = range(len(accuracy))
    nb_epochs = len(epochs)
    plt.subplot(1, 2, 1)
    plt.axis((0, nb_epochs, 0, 1.2))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.axis((0, nb_epochs, 0, 1.2))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def predict_and_visualize_results(model, input_X, output_Y, classes):
    predicted_classes = model.predict(input_X)
    # Computes for every image in the test dataset a probability distribution over the 10 categories.
    # Choose the prediction with the highest probability.
    predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
    correctIndex = np.where(predicted_classes == output_Y)[0]
    incorrectIndex = np.where(predicted_classes != output_Y)[0]
    print("Found %d correct labels" % len(correctIndex))
    print("Found %d incorrect labels" % len(incorrectIndex))
    cm = confusion_matrix(test_Y, predicted_classes)
    plot_confusion_matrix(cm, classes)


# Get Training, Validation & Test Set.
(trainAndVal_X, trainAndVal_Y), (test_X, test_Y) = fashion_mnist.load_data()
classes = np.unique(trainAndVal_Y)
nClasses = len(classes)
imgPixelDim = 28  # Image pixel dimension.
# Reshape the images to match the Neural Network input format. number of images: -1 for automatically assigned.
trainAndVal_X = trainAndVal_X.reshape(-1, imgPixelDim, imgPixelDim, 1)
# Pixels: imgPixelDim x imgPixelDim. number of channels: 1 for grey scale, 3 for RGB.
test_X = test_X.reshape(-1, imgPixelDim, imgPixelDim, 1)
# Convert data type from int8 to float32
trainAndVal_X = trainAndVal_X.astype('float32')
test_X = test_X.astype('float32')
# Normalize the data: rescale the pixel values in range 0 - 1 inclusive for training purposes.
trainAndVal_X = trainAndVal_X / 255.
test_X = test_X / 255.
# Change the labels from categorical to one-hot encoding.
# Example: image label 7 becomes [0 0 0 0 0 0 0 1 0]. The output neurons of the Neural Network will be trained to match
# the one_hot encoded array.
trainAndVal_Y_one_hot = to_categorical(trainAndVal_Y)
test_Y_one_hot = to_categorical(test_Y)
# Display the change for category label using one-hot encoding
print('Original label:', trainAndVal_Y[0])
print('After conversion to one-hot encoded:', trainAndVal_Y_one_hot[0])
# Split the trainAndVal data into training dataset and validation dataset.
# The model is trained over the training dataset.
# The validation dataset is used to monitor when the model starts overfitting on the training dataset.
train_X, valid_X, train_label, valid_label = train_test_split(trainAndVal_X, trainAndVal_Y_one_hot, test_size=0.2,
                                                              random_state=13)

# ------------------------------ Hyper Parameters -----------------------------------
# How many images with their corresponding categories to use per Neural Network weights update step.
batch_size = 128
epochs = 12  # How many times to loop over the entire training dataset. Example: for a batch_size=64 and training
# dataset size of 48000 then each epoch will consist of 48000/64=750 updates of the network weights.
learning_rate = 0.001
model = Sequential()
# ------------------------------------ Architecture ---------------------------------
model.add(Conv2D(128, kernel_size=(4, 4), activation='relu', input_shape=(28, 28, 1), padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, kernel_size=(4, 4), activation='relu', input_shape=(28, 28, 1), padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(nClasses, activation='softmax'))
model.summary()
# ------------------------------ Optimizer -----------------------------------
opt = keras.optimizers.RMSprop(lr=learning_rate)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
# Classification accuracy is the number of correct predictions made divided by the total number of predictions made,
# multiplied by 100 to turn it into a percentage.
# ----------------------------------------------------------------------------

# Test the performance of the untrained model over the test dataset.
predicted_classes = model.predict(test_X)
# A probability distribution over the 10 categories.
# Choose the prediction with the highest probability.
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
correctIndex = np.where(predicted_classes == test_Y)[0]
incorrectIndex = np.where(predicted_classes != test_Y)[0]
print("Found %d correct labels using the untrained model" % len(correctIndex))
print("Found %d incorrect labels using the untrained model" % len(incorrectIndex))
# Train the Neural Network
start_time = time.time()
fashion_train_dropout = model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1,
                                  validation_data=(valid_X, valid_label))
trainTime = (time.time() - start_time)
print('\nTraining time: {:.2f}'.format(trainTime), 'sec')
# Test the performance of the trained model over the test dataset
test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=0)
# This is the categorical_crossentropy
print('Test loss: {:.2f}'.format(test_eval[0]))
# The accuracy evaluated by the model during training and testing
print('Test accuracy: {:.2f}'.format(test_eval[1]))
show_results(fashion_train_dropout)
predict_and_visualize_results(model, test_X, test_Y, classes)
