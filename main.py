import tensorflow as TF
import matplotlib.pyplot as plt
import numpy as np


mnist = TF.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = TF.keras.utils.normalize(x_train, axis=1)
x_test = TF.keras.utils.normalize(x_test, axis=1)


model = TF.keras.models.Sequential()
model.add(TF.keras.layers.Flatten())
model.add(TF.keras.layers.Dense(128, activation=TF.nn.relu))
model.add(TF.keras.layers.Dense(128, activation=TF.nn.relu))
model.add(TF.keras.layers.Dense(10, activation=TF.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

model.fit(x_train, y_train, epochs=1)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_acc)
print(val_loss)

model.save('MNIST_Calssfification')


new_model = TF.keras.models.load_model('MNIST_Calssfification')
Predictions = new_model.predict([x_test])


plt.imshow(x_test[0])
plt.show

print('My best Guess is:')
print(np.argmax(Predictions[0]))



