#!/usr/bin/env python
# coding: utf-8

# In[31]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Import Seaborn


# In[2]:


(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()
X_train.shape


# In[3]:


X_test.shape


# In[4]:


y_train.shape


# In[5]:


y_train[:5]


# In[6]:


y_train = y_train.reshape(-1,)
y_train[:5]


# In[7]:


y_test = y_test.reshape(-1,)


# In[8]:


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[9]:


def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[10]:


plot_sample(X_train, y_train, 0)


# In[11]:


plot_sample(X_train, y_train, 1)


# In[12]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# In[13]:


ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')    
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)


# In[14]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# In[15]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[16]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[17]:


cnn.fit(X_train, y_train, epochs=10)


# In[18]:


cnn.evaluate(X_test,y_test)


# In[19]:


y_pred = cnn.predict(X_test)
y_pred[:5]


# In[20]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[21]:


y_test[:5]


# In[22]:


plot_sample(X_test, y_test,3)


# In[23]:


classes[y_classes[3]]


# In[24]:


classes[y_classes[3]]


# In[32]:


# Predict on the test set
y_pred = cnn.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

# Print classification report
print("Classification Report: \n", classification_report(y_test, y_pred_classes))

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[44]:


import tensorflow as tf
import matplotlib.pyplot as plt

# Define the CNN model (example model)
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Example training data (using MNIST dataset)
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# Train the model and save history
history = cnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Plotting Learning Curves
def plot_learning_curves(history):
    plt.figure(figsize=(12, 5))

    # Plot Training & Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Training & Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the function to plot the learning curves
plot_learning_curves(history)


# In[ ]:




