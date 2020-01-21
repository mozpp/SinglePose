import tensorflow as tf
from tensorflow.python.framework import graph_util

def load_pb(pb):
  with tf.gfile.GFile(pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')
    return graph

# MobileNetV2
#g2 = load_pb('mv2_cpm/pb/model458000_224_7stage_tf1.8.pb')
g2 = load_pb('mv3_cpm_tiny/pb/model_mv3_large_38000.pb')
with g2.as_default():
  flops = tf.profiler.profile(g2, options = tf.profiler.ProfileOptionBuilder.float_operation())
  print('FLOP after freezing', flops.total_float_ops)