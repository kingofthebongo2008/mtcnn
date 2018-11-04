import tensorflow as tf
import tensorflow.contrib.lite as lite
#from   mtcnn import MTCNN

#detector = MTCNN()

#x = detector.get_input()
#y = detector.get_output()


#with tf.Graph().as_default():
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:

	saver = tf.train.import_meta_graph('./all_in_one/mtcnn-3000000.meta')
	saver.restore(sess, './all_in_one/mtcnn-3000000')

	tf.train.write_graph(sess.graph_def, './tmp/mtcnn', 'model.pbtxt')
	
	export_dir = './tmp/all_in_one'
	builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
	builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING], strip_default_attrs=True)
	builder.add_meta_graph([tf.saved_model.tag_constants.SERVING], strip_default_attrs=True)
	builder.save()

