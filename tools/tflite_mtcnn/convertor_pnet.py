import tensorflow as tf
import tensorflow.contrib.lite as lite

pnet_input_node_names   = ['Placeholder']
pnet_output_node_names  = ['softmax/Reshape_1', 'pnet/conv4-2/BiasAdd']

output_graph            ='pnet.pb'
shape_pnet              = [1, 1600, 2560, 3]


#960
#1536

#672
#1076

#471
#753

#330
#527

#231
#369

#162
#259

#113
#181

#80
#127

#56
#89

#39
#62

#28
#44

#19
#31

#14
#22


shape_pnets             = [
                            [1, 960, 1536, 3],
                            [1, 672, 1076, 3] ,
                            [1, 471, 753, 3],
                            [1, 330, 527, 3],

                            [1, 231, 369, 3],
                            [1, 162, 259, 3],
                            [1, 113, 181, 3],
                            [1, 80, 127, 3 ],

                            [1, 56, 89, 3 ],
                            [1, 39, 62, 3 ],
                            [1, 28, 44, 3 ],
                            [1, 19, 31, 3 ],
                            
                            [ 1, 14, 22, 3 ]

                          ]

input_node_names        = pnet_input_node_names
output_node_names       = pnet_output_node_names

input_shapes            = {"Placeholder" : shape_pnet}


config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:

    saver = tf.train.import_meta_graph('./all_in_one/mtcnn-3000000.meta', clear_devices=True)
    saver.restore(sess, './all_in_one/mtcnn-3000000')

    subgraph = tf.graph_util.extract_sub_graph(tf.get_default_graph().as_graph_def(), pnet_input_node_names + pnet_output_node_names)

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants( sess,  subgraph, output_node_names) 

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("%d ops in the final graph." % len(output_graph_def.node))


    for index in range(0,13):

        shape       = shape_pnets[index]
        
        height      = shape[1]
        width       = shape[2]

        input_shape = {"Placeholder" : shape }
        file_name   = 'pnet_{:03}_{:03}.tflite'.format(height, width) 

        converter = lite.TocoConverter.from_frozen_graph( output_graph, input_node_names, output_node_names, input_shape )
        tflite_model = converter.convert()
        open(file_name, "wb").write(tflite_model)




