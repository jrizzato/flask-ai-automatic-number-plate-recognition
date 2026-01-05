from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from joblib import dump

# Microsoft Visual C++ 2015–2022 Redistributable (x64) para que funcione tensorflow.keras (?)
# https://learn.microsoft.com/en-us/answers/questions/5637122/looking-for-microsoft-visual-c-2015-2022-redistrib

# inception_resnet = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# inception_resnet.trainable = False  # Freeze the base model
inception_v3 = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
inception_v3.trainable = False  # Freeze the base model

# Capas personalizadas mejoradas
headmodel = inception_v3.output
headmodel = Flatten()(headmodel)
headmodel = Dense(512, activation='relu')(headmodel)
headmodel = Dense(256, activation='relu')(headmodel)
headmodel = Dense(128, activation='relu')(headmodel)
headmodel = Dense(4, activation='sigmoid')(headmodel)  # 4 coordenadas

model = Model(inputs=inception_v3.input, outputs=headmodel)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
model.summary()

dump(model, './data/model.joblib')