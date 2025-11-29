import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 🔹 1. Load CIFAR-10 dataset (60,000 color images of 10 classes)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 🔹 2. Normalize and reshape
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = y_train.flatten()
y_test = y_test.flatten()

# 🔹 3. Class names (just for display)
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# 🔹 4. Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')  # 10 classes
])

# 🔹 5. Compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_split=0.1)

# 🔹 6. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# 🔹 7. Predict and visualize
predictions = model.predict(X_test)

for i in range(5):
    plt.imshow(X_test[i])
    plt.title(f"Predicted: {class_names[np.argmax(predictions[i])]} | Actual: {class_names[y_test[i]]}")
    plt.axis('off')
    plt.show()
