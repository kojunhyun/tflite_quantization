
import os, sys
import tensorflow as tf
import pickle


saved_path = 'vggface2_vggnet7_model.h5'
model = tf.keras.models.load_model(saved_path)
model.summary()

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.allow_custom_ops = True
#print(converter.allow_custom_ops)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

# When quantization-aware, setting input image type and output label type 
#converter.inference_input_type = tf.uint8
#converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
#converter = tf.lite.TFLiteConverter.from_keras_model(model, allow_custom_ops=True)
#tflite_model = converter.convert()


with open('vggface2_vggnet7_model_q.tflite', 'wb') as f:
    pickle.dump(tflite_model, f)