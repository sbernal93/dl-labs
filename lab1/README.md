# Lab 1
ANN and CNN.
[Theory](https://upc-mai-dl.github.io/mlp-convnets-theory/)

## Example 1: Basic NN

This guided laboratory introduces a basic example on how to use Keras for training a simple NN. The following example can be find whole in the MAI-DL directory of the cluster, file code_lab1.1.py. Here we split it in parts to review it.

We will work with the MNIST dataset of hand-written digits.

```python
from __future__ import division
import keras
print 'Using Keras version', keras.__version__
from keras.datasets import mnist
    
#Load the MNIST dataset, already provided by Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()
    
#Check sizes of dataset
print 'Number of train examples', x_train.shape[0]
print 'Size of train examples', x_train.shape[1:]
```

We need to adapt the data to the shape of the input layer. Since its a fully connected layer (i.e., Dense), we need to flatten the 28x28 2D image into a 1D vector of 784 values. We also normalize the input, and make sure its type is correct.

```python
    #Adapt the data as an input of a fully-connected (flatten to 1D)
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
        
    #Normalize data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255
```

The labels of the data have to be transformed to the syntax required by the classifier (i.e., softmax)

```python
    #Adapt the labels to the one-hot vector syntax required by the softmax
    from keras.utils import np_utils
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
```

We can now define the architecture of our neural network. The input layer must be coherent with the shape of the data. We add two fully connected layers, and the final softmax classifier.

```python
    #Define the NN architecture
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    #Two hidden layers
    nn = Sequential()
    nn.add(Dense(128,activation='relu',input_shape=(784,)))
    nn.add(Dense(64,activation='relu'))
    nn.add(Dense(64,activation='relu'))
    nn.add(Dense(10, activation='softmax'))
```

To visualize the architecture we just defined we can use the following keras call

```python
    #Model visualization
    #We can plot the model by using the ```plot_model``` function. We need to install *pydot, graphviz and pydot-ng*.
    from keras.util import plot_model
    plot_model(nn, to_file='nn.png', show_shapes=true)
```
    
Notice that this lines are commented on the job placed at Minotauro. That is because Minotauro have not installed the required libraries. If you want to use it you can install Keras in you personal computer, design the model as shown in above and plot the model. Since if you use keras in you personal computer its version will be greater the above lines are written according the new syntaxis.

After defining the model, we need to compile it before running. At this point we define which optimizer we wish to use (e.g., SGD), how is the loss computed (e.g., categorical cross-entropy) and which metric we use to evaluate the model (e.g., accuracy)

```python
nn.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
```
With the model compiled, we can now launch the training procedure. At this point we may define the batch size, and the number of epochs to run.

```python
    history = nn.fit(x_train,y_train,batch_size=128,nb_epoch=20)
```

We can now evaluate the trained model on the test set. Test accuracy provides the real performance of our model.

```python
    score = nn.evaluate(x_test, y_test, verbose=0)
```

And plot the results and information of the training procedure

```python
    ##Store Plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    #Accuracy plot
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('model_accuracy.pdf')
    plt.close()
    #Loss plot
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('model_loss.pdf')
```
  
For more information, we can visualize the confusion matrix. This is helpful to know how is our model behaving, and how can it be improved.

```python
    #Confusion Matrix
    from sklearn.metrics import classification_report,confusion_matrix
    import numpy as np
    #Compute probabilities
    Y_pred = nn.predict(x_test)
    #Assign most probable label
    y_pred = np.argmax(Y_pred, axis=1)
    #Plot statistics
    print 'Analysis of results'
    target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
    print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))
```

Finally, we can store the model. This can be either done in two files (json for the architecture, hdf5 for the weights), or in a single hdf5 file.

```python
    #Saving model and weights
    from keras.models import model_from_json
    nn_json = nn.to_json()
    with open('nn.json', 'w') as json_file:
        json_file.write(nn_json)
    weights_file = "weights-MNIST_"+str(score[1])+".hdf5"
    nn.save_weights(weights_file,overwrite=True)
    
    #Loading model and weights
    json_file = open('nn.json','r')
    nn_json = json_file.read()
    json_file.close()
    nn = model_from_json(nn_json)
    nn.load_weights(weights_file)
```

Example 2: CNN
Convolutional neural networks are designed to process 2D data, by exploiting 2D spatial patterns. Let us now design a CNN to process the same MNIST dataset, and see if we can outperform the basic neural net of the previous example. Most of the code is the same, so in here we will only review those parts that differ significantly. The whole code is in the MAI-DL directory of the cluster, file code_lab1.2.py.

Before getting into convolutions, let’s start with the requirement of a validation set. While training any neural network, and particularly, deep neural networks, lots of experiments are required to adjust parameters and find the best combination. Parameters such as network architecture, learning rate, optiming algorithm, batch size, etc. By using only a train/test split when evaluating these parameters, we will eventually overfit our model to solve the test set. See an explanation of this in [this Udacity video](https://www.youtube.com/watch?v=--E5qo_XnXo).

Simply put, if we use the train data for taking design decisions, the actual performance of our model cannot any longer be measured reliably on the test set. Which is very bad. To avoid overfitting the test data while also using a set of data for tunning hyperparameters, the most common approach is to use a validation set. We may use the validation set for taking certain design decisions, and when we are done we can test the quality of our model on the test set. The important thing to remember in this case is to NEVER use the results of the test set into account when designing and training your model. The test set should only be used at the end of the development process, just before deployment.

In Keras defining a validation set is very easy, as we just need to specify which part of the training set will be used for validation on the fit function. Additionally, we will use measurements on the validation set to evaluate performance.

```python
    #Start training using a 15% for validation
    history = nn.fit(x_train,y_train,batch_size=128,nb_epoch=20, validation_split=0.15)
    #Plot accuracy on the validation set
    plt.plot(history.history['val_acc'])
    #Plot loss on the validation set
    plt.plot(history.history['val_loss'])
```

Now that we know how to validate our models, let’s define a CNN. A simple network will contain 2 pairs of convolution and max pooling layers and a final fully connected layer. To input the data to the first fully connected layer, we need to flatten the output from a 3D volume to a 1D array using the Flatten layer.

```python
    nn = Sequential()
    nn.add(Conv2D(32, 3, 3, activation='relu', input_shape=input_shape))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Conv2D(64, 3, 3, activation='relu'))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Flatten())
    nn.add(Dense(128, activation='relu'))
    nn.add(Dense(10, activation='softmax'))
```

This simple model can solve the MNIST classification problem with an accuracy over 98%. For a more complicated problem, try the CIFAR10 dataset (see the dataset webpage to analyze the type of problem at hand.

```python
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

Remember to adapt the code to the size of this dataset (MNIST is 28x28x1, while CIFAR10 is 32x32x3), or the input of the network will be erroneous. The problem is also a 10 class classification, so the output of the network will be the same. This is clearly a much harder problem, as the same model only achieves roughly a 49% of test accuracy.


