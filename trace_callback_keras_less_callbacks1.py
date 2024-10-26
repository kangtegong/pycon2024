import sys
import time
import numpy as np
import keras
from keras import layers

monitor_call_count = 0
callback_cnt = 0

def monitor_call(code, instruction_offset, callable, arg0):
    global callback_cnt
    callback_cnt += 1

    global monitor_call_count
    monitor_call_count += 1

tool_id = sys.monitoring.DEBUGGER_ID

sys.monitoring.use_tool_id(tool_id, "Example Monitoring Tool")
sys.monitoring.register_callback(tool_id, sys.monitoring.events.CALL, monitor_call)
sys.monitoring.set_events(tool_id, sys.monitoring.events.CALL | sys.monitoring.events.LINE | sys.monitoring.events.PY_RETURN)

start_time = time.time()

"""
## Prepare the data
"""

num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
"""

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

"""
## Train the model
"""

batch_size = 128
epochs = 1

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

print("--- sys.monitoring: %s seconds ---" % (time.time() - start_time))
print(f"CALL events: {monitor_call_count}")
print(f"callback counts: {callback_cnt}")
