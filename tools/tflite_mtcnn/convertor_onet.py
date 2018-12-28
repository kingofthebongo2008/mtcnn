import tensorflow as tf
import tensorflow.contrib.lite as lite

onet_input_node_names   = ['Placeholder_2']
onet_output_node_names  = ['softmax_2/softmax', 'onet/conv6-2/onet/conv6-2', 'onet/conv6-3/onet/conv6-3']

output_graph            ='onet.pb'
shape_onet              = [1, 48, 48, 3]

input_node_names        = onet_input_node_names
output_node_names       = onet_output_node_names

input_shapes            = {"Placeholder_2" : shape_onet}


config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:

    saver = tf.train.import_meta_graph('./all_in_one/mtcnn-3000000.meta', clear_devices=True)
    saver.restore(sess, './all_in_one/mtcnn-3000000')

    subgraph = tf.graph_util.extract_sub_graph(tf.get_default_graph().as_graph_def(), onet_input_node_names + onet_output_node_names)

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants( sess,  subgraph, output_node_names) 

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("%d ops in the final graph." % len(output_graph_def.node))

    shape       = shape_onet
        
    input_shape = input_shapes
    file_name   = 'onet.tflite'

    converter = lite.TocoConverter.from_frozen_graph( output_graph, input_node_names, output_node_names, input_shape )
    tflite_model = converter.convert()
    open(file_name, "wb").write(tflite_model)




