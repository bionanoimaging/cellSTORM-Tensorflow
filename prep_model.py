import tensorflow as tf

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 
model_dir = './testdump/'
output_node_names = 'outputs'

#freeze_graph(model_dir, output_node_names)



# We retrieve our checkpoint fullpath
checkpoint = tf.train.get_checkpoint_state(model_dir)
input_checkpoint = checkpoint.model_checkpoint_path

# We precise the file fullname of our freezed graph
absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
output_graph = absolute_model_dir + "/frozen_model.pb"

# We clear devices to allow TensorFlow to control on which device it will load operations
clear_devices = True

# We start a session using a temporary fresh Graph
with tf.Session(graph=tf.Graph()) as sess:
    # We import the meta graph in the current default Graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We restore the weights
    saver.restore(sess, input_checkpoint)

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, # The session is used to retrieve the weights
        tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
        output_node_names.split(",") # The output node names are used to select the usefull nodes
    ) 

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))



##### Optimize graph	
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

inputGraph = tf.GraphDef()
with tf.gfile.Open("frozen.pb", "rb") as f:
    data2read = f.read()
    inputGraph.ParseFromString(data2read)

outputGraph = optimize_for_inference_lib.optimize_for_inference(
                inputGraph,
                ["inputTensor"], # an array of the input node(s)
                ["output/output"], # an array of output nodes
                tf.int32.as_datatype_enum)

        # Save the optimized graph

f = tf.gfile.FastGFile("outputOptimizedGraph.pb", "w")
f.write(outputGraph.SerializeToString())    
