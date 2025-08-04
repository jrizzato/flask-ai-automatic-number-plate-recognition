from tensorflow.keras.callbacks import TensorBoard
from joblib import dump, load

# load the preprocessed data
x_train, x_test, y_train, y_test = load('./data/preprocessed_data.joblib')
# load model
model = load('./data/model.joblib') 

# Initialize TensorBoard
tfb = TensorBoard('object_detection')
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=10, callbacks=[tfb])

dump(model, './model/object_detection_model.joblib')
print("Modelo guardado en './model/object_detection_model.joblib'")
# model.save('./model/object_detection_model.h5')