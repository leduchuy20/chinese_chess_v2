from keras.models import load_model

model = load_model('./new_model_v2.h5')
model.save('./new_model_v2.keras')