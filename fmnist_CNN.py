import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mnist_reader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras import regularizers
from keras.constraints import unit_norm

def build_classifier():
    
    classifier = Sequential()
    
    classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', 
                          kernel_constraint = unit_norm(), kernel_initializer = 'he_normal',
                          padding = 'same', input_shape = (28,28,1)))
    classifier.add(MaxPool2D(pool_size = (2, 2), padding = 'same'))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(0.25))
    
    classifier.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', 
                          padding = 'same'))
    classifier.add(MaxPool2D(pool_size = (2, 2), padding = 'same'))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(0.25))
    
    classifier.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', 
                          padding = 'same'))
    classifier.add(MaxPool2D(pool_size = (2, 2), padding = 'same'))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(0.25))
    
    classifier.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', 
                          padding = 'same'))
    classifier.add(MaxPool2D(pool_size = (2, 2), padding = 'same'))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(0.4))
    
    classifier.add(Flatten())
    classifier.add(Dense(256, activation = 'relu', kernel_constraint = unit_norm()))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(0.2))
    classifier.add(Dense(10, activation = 'softmax'))
                   
    #opt = Adam()
    #opt = SGD()
    opt = RMSprop()
    
    classifier.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    #plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    return classifier
    
X_train, y_train = mnist_reader.load_mnist('datasets', kind='train')
X_test, y_test = mnist_reader.load_mnist('datasets', kind='t10k')

X_train, X_test = X_train / 255, X_test / 255

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

datagen = ImageDataGenerator()

classifier = build_classifier()

datagen.fit(X_train)

history = classifier.fit_generator(datagen.flow(X_train, y_train), epochs = 2, 
                                   validation_data = (X_val, y_val))

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







