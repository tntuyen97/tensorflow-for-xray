import os, sys
import glob
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# change this as you see fit
image_path = ('C:/Users/TUYEN TRAN/tensorflow-for-xray/tf_files/xray-photos')

extension = ['*.jpeg', '*.jpg','*.png']
files=[]

for e in extension:
directory = os.path.join(image_path, e)
fileList = glob.glob(directory)
for f in fileList:
    files.append(f)

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
           in tf.gfile.GFile("C:/Users/TUYEN TRAN/tensorflow-for-xray/tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("C:/Users/TUYEN TRAN/tensorflow-for-xray/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    # Read in the image_data
    for file in files:
        image_data = tf.gfile.FastGFile(file, 'rb').read()

        predictions = sess.run(softmax_tensor, \
                       {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        print("Image Name: " + file)
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))