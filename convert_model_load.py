
import os, sys
import tensorflow as tf
import pickle
import time
import numpy as np
import vggface2_custom

saved_path = 'vggface2_vggnet7_model_q.tflite'
result_top_k = 1 # top 1

with open(saved_path, 'rb') as f:
    tflite_model = pickle.load(f)
    

# Load TFLite model and allocate tensors.
#interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
#interpreter.allocate_tensors()
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print('input_details : \n', input_details)
print('output_details : \n', output_details)

print('input_details dtype : ', input_details[0]['dtype'])
print('output_details dtype : ', output_details[0]['dtype'])



### dataset load
#(train_x, train_y) = vggface2_custom.vgg_load(train=True)
(valid_x, valid_y) = vggface2_custom.vgg_load(train=False)
#train_x = train_x.astype('float32')
valid_x = valid_x.astype('float32')
#print(train_x.shape)
print(valid_x.shape)


# check weight integer
interpreter.set_tensor(input_details[0]['index'], valid_x[:1])
all_layers_details = interpreter.get_tensor_details()
for layer in all_layers_details:

    if 'conv2d' in layer['name'] or 'dense' in layer['name']:
        print('*'*50)
        print(layer['name'])
        print(interpreter.get_tensor(layer['index']))
        print(interpreter.get_tensor(layer['index']).dtype, interpreter.get_tensor(layer['index']).shape)



total_cal_time = 0
pred_count = 0
for i in range(len(valid_x)):
    interpreter.set_tensor(input_details[0]['index'], valid_x[i:i+1])
    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()
    total_cal_time += stop_time - start_time

    output_data = interpreter.get_tensor(output_details[0]['index'])
    #print('output : ', output_data)
    results = np.squeeze(output_data)
    #print('results : ', results)

    top_k = results.argsort()[-result_top_k:][::-1]
    #print(top_k)

    if top_k == valid_y[i]:
        pred_count += 1

    if i % 100 == 0:
        print(i, ' / ', pred_count)


print('time : ', total_cal_time)
print('avg time : ', total_cal_time/len(valid_x))
print('acc : ', pred_count/len(valid_x))

