"""
This is a modified version of tensorflow.nn.seq2seq's attention_decoder and 
embedding_attention_decoder which allow dynamic sequence lengths and implement
the specific calculations dictated in the im2markup paper. It will probably be
rewritten as tensorflow gains dynamic variants of its seq2seq models, such as 
the recently added dynamic_rnn_decoder.
"""
import numpy as np
import tensorflow as tf

from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell._linear  # pylint: disable=protected-access

def attention_decoder(initial_state,
                      attention_states,
                      cell,
                      vocab_size,
                      time_steps,
                      batch_size,
                      output_size=None,
                      loop_function=None,
                      dtype=None,
                      scope=None):
  if attention_states.get_shape()[2].value is None:
    raise ValueError("Shape[2] of attention_states must be known: %s"
                     % attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  with variable_scope.variable_scope(
      scope or "attention_decoder", dtype=dtype) as scope:
    dtype = scope.dtype

    attn_length = attention_states.get_shape()[1].value
    if attn_length is None:
      attn_length = shape(attention_states)[1]
    attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = array_ops.reshape(
        attention_states, [-1, attn_length, 1, attn_size])
    attention_vec_size = attn_size  # Size of query vectors for attention.
    k = variable_scope.get_variable("AttnW",
                                    [1, 1, attn_size, attention_vec_size])
    hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
    v = variable_scope.get_variable("AttnV", [attention_vec_size])

    state = initial_state

    def attention(query):
      """Put attention masks on hidden using hidden_features and query."""
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(1, query_list)
      with variable_scope.variable_scope("Attention_0"):
        y = linear(query, attention_vec_size, True)
        y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
        # Attention mask is a softmax of v^T * tanh(...).
        s = math_ops.reduce_sum(
            v * math_ops.tanh(hidden_features + y), [2, 3])
        a = nn_ops.softmax(s)
        # Now calculate the attention-weighted vector d.
        d = math_ops.reduce_sum(
            array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
            [1, 2])
        ds = array_ops.reshape(d, [-1, attn_size])
      return ds

    prev = array_ops.zeros([batch_size,output_size])
    batch_attn_size = array_ops.pack([batch_size, attn_size])
    attn = array_ops.zeros(batch_attn_size, dtype=dtype)
    attn.set_shape([None, attn_size])

    def cond(time_step, prev_o_t, prev_softmax_input, state_c, state_h, outputs):
      return time_step < time_steps

    def body(time_step, prev_o_t, prev_softmax_input, state_c, state_h, outputs):
      state = tf.nn.rnn_cell.LSTMStateTuple(state_c,state_h)
      with variable_scope.variable_scope("loop_function", reuse=True):
        inp = loop_function(prev_softmax_input, time_step)

      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      x = tf.concat(1,[inp,prev_o_t])
      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      attn = attention(state)

      with variable_scope.variable_scope("AttnOutputProjection"):
        output = math_ops.tanh(linear([cell_output, attn], output_size, False))
        with variable_scope.variable_scope("FinalSoftmax"):
          softmax_input = linear(output,vocab_size,False)

      new_outputs = tf.concat(1, [outputs,tf.expand_dims(softmax_input,1)])
      return (time_step + tf.constant(1, dtype=tf.int32),\
              output, softmax_input, state.c, state.h, new_outputs)

    time_step = tf.constant(0, dtype=tf.int32)
    shape_invariants = [time_step.get_shape(),\
                        prev.get_shape(),\
                        tf.TensorShape([batch_size, vocab_size]),\
                        tf.TensorShape([batch_size,512]),\
                        tf.TensorShape([batch_size,512]),\
                        tf.TensorShape([batch_size, None, vocab_size])]

# START keyword is 0
    init_word = np.zeros([batch_size, vocab_size])

    loop_vars = [time_step,\
                 prev,\
                 tf.constant(init_word, dtype=tf.float32),\
                 initial_state.c,initial_state.h,\
                 tf.zeros([batch_size,1,vocab_size])] # we just need to feed an empty matrix
                                                      # to start off the while loop since you can
                                                      # only concat matrices that agree on all but
                                                      # one dimension. Below, we remove that initial
                                                      # filler index

    outputs = tf.while_loop(cond, body, loop_vars, shape_invariants)

  return outputs[-1][:,1:], tf.nn.rnn_cell.LSTMStateTuple(outputs[-3],outputs[-2])

def embedding_attention_decoder(initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                time_steps,
                                batch_size,
                                embedding_size,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None):
  if output_size is None:
    output_size = cell.output_size
  if output_projection is not None:
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with variable_scope.variable_scope(
      scope or "embedding_attention_decoder", dtype=dtype) as scope:

    embedding = variable_scope.get_variable("embedding",
                                            [num_symbols, embedding_size])
    loop_function = tf.nn.seq2seq._extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None
    return attention_decoder(
        initial_state,
        attention_states,
        cell,
        num_symbols,
        time_steps,
        batch_size,
        output_size=output_size,
        loop_function=loop_function)

