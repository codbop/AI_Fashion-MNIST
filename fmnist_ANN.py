# Evaluating the ANN
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Sequential # initialize neural network library
from tensorflow.keras.layers import Dense # build our layers library
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
import mnist_reader
from matplotlib import pyplot as plt
import seaborn as sns
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.utils.vis_utils import plot_model

def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
    classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'softmax'))
    #opt = RMSprop()
    #opt = SGD(lr = 0.025)
    opt = Adam()
    classifier.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    return classifier

X_train, y_train = mnist_reader.load_mnist('datasets', kind='train')
X_test, y_test = mnist_reader.load_mnist('datasets', kind='t10k')
X_train, X_test = X_train / 255, X_test / 255
y_train = to_categorical(y_train)

#classifier = KerasClassifier(build_fn = build_classifier, epochs = 10, batch_size = 200)
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 2)
#mean = accuracies.mean()
#variance = accuracies.std()

#print('\n' + "Accuracy mean: "+ str(mean))
#print("Accuracy variance: "+ str(variance) + '\n')

classifier = build_classifier()
history = classifier.fit(X_train, y_train, epochs = 2, batch_size = 50, validation_split = 0.33)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
fig.suptitle('Loss and Accuracy Graphs')

# "Plot Accuracy"
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
plt.show()

# "Plot Loss"
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')
plt.show()

y_pred = classifier.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)
target_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
                'Bag', 'Ankle boot']

print('')
print('Confusion Matrix')
cm = confusion_matrix(y_test, y_pred)
print(cm) 

# "Plot Confusion Matrix"
fig, ax = plt.subplots(1, 1, figsize = (20, 10))
sns.heatmap(cm, annot = True, ax = ax, cmap='Greens', fmt = 'g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title(('Confusion Matrix'))
ax.xaxis.set_ticklabels(target_names)
ax.yaxis.set_ticklabels(target_names)

print('\n', classification_report(y_test, y_pred, target_names = target_names))