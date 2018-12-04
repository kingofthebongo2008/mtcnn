import tensorflow as tf
import tensorflow.contrib.lite as lite

pnet_input_node_names   = ['Placeholder']
pnet_output_node_names  = ['softmax/Reshape_1', 'pnet/conv4-2/BiasAdd']

output_graph            ='pnet.pb'
shape_pnet              = [1, 1600, 2560, 3]

960
1536

672
1076

471
753

330
527

231
369

162
259

113
181

80
127

56
89

39
62

28
44

19
31

14
22

shape_pnets             = [
                            { 1, 1600, 2560, 3 },
                            { 1, 1600, 2560, 3 },
                            { 1, 1600, 2560, 3 },
                            { 1, 1600, 2560, 3 },

                            { 1, 1600, 2560, 3 },
                            { 1, 1600, 2560, 3 },
                            { 1, 1600, 2560, 3 },
                            { 1, 1600, 2560, 3 },

                            { 1, 1600, 2560, 3 },
                            { 1, 1600, 2560, 3 },
                            { 1, 1600, 2560, 3 },
                            { 1, 1600, 2560, 3 },
                            
                            { 1, 1600, 2560, 3 }

                          ]

input_node_names        = pnet_input_node_names
output_node_names       = pnet_output_node_names

input_shapes            = {"Placeholder" : shape_pnet}


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

    subgraph = tf.graph_util.extract_sub_graph(tf.get_default_graph().as_graph_def(), pnet_input_node_names + pnet_output_node_names)
    #tf.reset_default_graph()
    #tf.import_graph_def(subgraph)

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants( sess,  subgraph, output_node_names) 

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("%d ops in the final graph." % len(output_graph_def.node))

    converter = lite.TocoConverter.from_frozen_graph( output_graph, input_node_names, output_node_names, input_shapes )
    tflite_model = converter.convert()
    open("pnet.tflite", "wb").write(tflite_model)




