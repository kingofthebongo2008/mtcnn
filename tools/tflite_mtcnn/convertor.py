import tensorflow as tf
import tensorflow.contrib.lite as lite

pnet_input_node_names   = ['Placeholder']
pnet_output_node_names  = ['softmax/Reshape_1', 'pnet/conv4-2/BiasAdd']

rnet_input_node_names   = ['Placeholder_1']
rnet_output_node_names  = ['softmax_1/softmax', 'rnet/conv5-2/rnet/conv5-2']

onet_input_node_names   = ['Placeholder_2']
onet_output_node_names  = ['softmax_2/softmax', 'onet/conv6-2/onet/conv6-2', 'onet/conv6-3/onet/conv6-3']

output_graph            ='mtcnn.pb'

shape_pnet              = [1, 1600, 2560, 3]
shape_rnet              = [1, 24, 24, 3]
shape_onet              = [1, 48, 48, 3]

input_node_names        = pnet_input_node_names  + rnet_input_node_names  + onet_input_node_names
output_node_names       = pnet_output_node_names + rnet_output_node_names + onet_output_node_names

input_shapes            = {"Placeholder" : shape_pnet, "Placeholder_1" : shape_rnet, "Placeholder_2" : shape_onet }


config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:

    saver = tf.train.import_meta_graph('./all_in_one/mtcnn-3000000.meta', clear_devices=True)
    saver.restore(sess, './all_in_one/mtcnn-3000000')

    #tf.train.write_graph(sess.graph_def, './tmp/mtcnn', 'model.pbtxt')
    
    #export_dir = './tmp/all_in_one'
    #builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    #builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING], strip_default_attrs=True)
    #builder.add_meta_graph([tf.saved_model.tag_constants.SERVING], strip_default_attrs=True)
    #builder.save()

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants( sess,  tf.get_default_graph().as_graph_def(), output_node_names) 

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

    converter = lite.TocoConverter.from_frozen_graph( output_graph, input_node_names, output_node_names, input_shapes )
    tflite_model = converter.convert()
    open("mtcnn.tflite", "wb").write(tflite_model)




