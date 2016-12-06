import Image, random, time, os, decoder
import numpy as np
import tensorflow as tf

def load_data():
  vocab = open('data/latex_vocab.txt').read().split('\n')
  vocab_to_idx = dict([ (vocab[i],i) for i in range(len(vocab))])
  formulas = open('data/formulas.norm.lst').read().split('\n')

  # 0: START
  # 1: END
  # 2: UNKNOWN
  # 3: PADDING

  def formula_to_indices(formula):
    formula = formula.split(' ')
    res = [0]
    for token in formula:
      if token in vocab_to_idx:
        res.append( vocab_to_idx[token] + 4 )
      else:
        res.append( 2 )
    res.append(1)
    return res

  formulas = map( formula_to_indices, formulas)

  train = open('data/train_filter.lst').read().split('\n')[:-1]
  val = open('data/validate_filter.lst').read().split('\n')[:-1]
  test = open('data/test_filter.lst').read().split('\n')[:-1]

  def import_images(datum):
    datum = datum.split(' ')
    img = np.array(Image.open('data/images_processed/'+datum[0]).convert('L'))
    return (img, formulas[ int(datum[1]) ])

  train = map(import_images, train)
  val = map(import_images, val)
  test = map(import_images, test)
  return train, val, test

def batchify(data, batch_size):
# group by image size
  res = {}
  for datum in data:
    if datum[0].shape not in res:
      res[datum[0].shape] = [datum]
    else:
      res[datum[0].shape].append(datum)
  batches = []
  for size in res:
    group = sorted(res[size], key= lambda x: len(x[1]))
    for i in range(0, len(group), batch_size):
      images = map(lambda x: np.expand_dims(np.expand_dims(x[0],0),3), group[i:i+batch_size])
      batch_images = np.concatenate(images, 0)
      seq_len = max([ len(x[1]) for x in group[i:i+batch_size]])
      def preprocess(x):
        arr = np.array(x[1])
        pad = np.pad( arr, (0, seq_len - arr.shape[0]), 'constant', constant_values = 3)
        return np.expand_dims( pad, 0)
      labels = map( preprocess, group[i:i+batch_size])
      batch_labels = np.concatenate(labels, 0)
#[((50, 120), 6000), ((40, 160), 6400), ((40, 200), 8000), ((40, 240), 9600), ((50, 200), 10000), ((40, 280), 11200), ((50, 240), 12000), ((40, 320), 12800), ((50, 280), 14000), ((40, 360), 14400), ((50, 320), 16000), ((50, 360), 18000), ((50, 400), 20000), ((60, 360), 21600), ((100, 360), 36000), ((100, 500), 50000), ((160, 400), 64000)]
      too_big = [(160,400),(100,500),(100,360),(60,360),  (100,800), (200,500), (800,800), (100,600)]
      if batch_labels.shape[0] == batch_size and not (batch_images.shape[1],batch_images.shape[2]) in too_big:
        batches.append( (batch_images, batch_labels) )
#skip the last incomplete batch for now
  return batches

def weight_variable(name,shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.get_variable(name + "_weights", initializer= initial)

def bias_variable(name, shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.get_variable(name + "_bias", initializer= initial)

def conv2d(x,W):
  return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

def init_cnn(inp):
  W_conv1 = weight_variable("conv1", [3,3,1,512])
  b_conv1 = bias_variable("conv1", [512])
  h_conv1 = tf.nn.relu(conv2d(inp,W_conv1) + b_conv1)
  h_bn1   = tf.contrib.layers.batch_norm(h_conv1)

  W_conv2 = weight_variable("conv2", [3,3,512,512])
  b_conv2 = bias_variable("conv2", [512])
  h_pad2  = tf.pad(h_bn1, [[0,0],[1,1],[1,1],[0,0]], "CONSTANT")
  h_conv2 = tf.nn.relu(conv2d(h_pad2, W_conv2) + b_conv2)
  h_bn2   = tf.contrib.layers.batch_norm(h_conv2)
  h_pool2 = tf.nn.max_pool(h_bn2, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')

  W_conv3 = weight_variable("conv3", [3,3,512,256])
  b_conv3 = bias_variable("conv3", [256])
  h_pad3  = tf.pad(h_pool2, [[0,0],[1,1],[1,1],[0,0]], "CONSTANT")
  h_conv3 = tf.nn.relu(conv2d(h_pad3, W_conv3) + b_conv3)

  h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1,2,1,1], strides=[1,2,1,1], padding='SAME')

  W_conv4 = weight_variable("conv4", [3,3,256,256])
  b_conv4 = bias_variable("conv4", [256])
  h_pad4  = tf.pad(h_pool3, [[0,0],[1,1],[1,1],[0,0]], "CONSTANT")
  h_conv4 = tf.nn.relu(conv2d(h_pad4, W_conv4) + b_conv4)
  h_bn4   = tf.contrib.layers.batch_norm(h_conv4)

  W_conv5 = weight_variable("conv5", [3,3,256,128])
  b_conv5 = bias_variable("conv5", [128])
  h_pad5  = tf.pad(h_bn4, [[0,0],[1,1],[1,1],[0,0]], "CONSTANT")
  h_conv5 = tf.nn.relu(conv2d(h_pad5, W_conv5) + b_conv5)
  h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  W_conv6 = weight_variable("conv6", [3,3,128,64])
  b_conv6 = bias_variable("conv6", [64])
  h_pad6  = tf.pad(h_pool5, [[0,0],[1,1],[1,1],[0,0]], "CONSTANT")
  h_conv6 = tf.nn.relu(conv2d(h_pad6, W_conv6) + b_conv6)
  h_pad6  = tf.pad(h_conv6, [[0,0],[2,2],[2,2],[0,0]], "CONSTANT")
  h_pool6 = tf.nn.max_pool(h_pad6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  return h_pool6

def build_model(inp, batch_size, num_rows, num_columns, dec_seq_len):
  #constants
  enc_lstm_dim = 256
  feat_size = 64
  dec_lstm_dim = 512
  vocab_size = 503
  embedding_size = 80

  cnn = init_cnn(inp)

  #function for map to apply the rnn to each row
  def fn(inp):
    enc_init_shape = [batch_size, enc_lstm_dim]
    with tf.variable_scope('encoder_rnn'):
      with tf.variable_scope('forward'):
        lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(enc_lstm_dim)
        init_fw = tf.nn.rnn_cell.LSTMStateTuple(\
                  tf.get_variable("enc_fw_c", enc_init_shape, initializer=tf.contrib.layers.initializers.xavier_initializer()),\
                  tf.get_variable("enc_fw_h", enc_init_shape, initializer=tf.contrib.layers.initializers.xavier_initializer())
                  )
      with tf.variable_scope('backward'):
        lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(enc_lstm_dim)
        init_bw = tf.nn.rnn_cell.LSTMStateTuple(\
                  tf.get_variable("enc_bw_c", enc_init_shape, initializer=tf.contrib.layers.initializers.xavier_initializer()),\
                  tf.get_variable("enc_bw_h", enc_init_shape, initializer=tf.contrib.layers.initializers.xavier_initializer())
                  )
      output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, \
                                                  lstm_cell_bw, \
                                                  inp, \
                                                  sequence_length = tf.fill([batch_size], tf.shape(inp)[1]), \
                                                  initial_state_fw = init_fw, \
                                                  initial_state_bw = init_bw \
                                                  )
    return tf.concat(2,output)

  fun = tf.make_template('fun', fn)
  #shape is (batch size, rows, columns, features)
  #swap axes so rows are first. map splits tensor on first axis, so fn will be applied to tensors of shape (batch_size,time_steps,feat_size)
  rows_first = tf.transpose(cnn,[1,0,2,3])
  res = tf.map_fn(fun, rows_first, dtype=tf.float32)
  encoder_output = tf.transpose(res,[1,0,2,3])

  dec_lstm_cell = tf.nn.rnn_cell.LSTMCell(dec_lstm_dim)
  dec_init_shape = [batch_size, dec_lstm_dim]
  dec_init_state = tf.nn.rnn_cell.LSTMStateTuple( tf.truncated_normal(dec_init_shape),\
                                                  tf.truncated_normal(dec_init_shape) )

  init_words = np.zeros([batch_size,1,vocab_size])

  decoder_output = decoder.embedding_attention_decoder(dec_init_state,\
                                               tf.reshape(encoder_output,\
                                                          [batch_size, -1,\
                                                          2*enc_lstm_dim]),\
                                               dec_lstm_cell,\
                                               vocab_size,\
                                               dec_seq_len,
                                               batch_size,
                                               embedding_size,\
                                               feed_previous=True)

  return (encoder_output, decoder_output)

def main():
  batch_size = 20
  epochs = 100
  lr = 0.1
  min_lr = 0.001
  learning_rate = tf.placeholder(tf.float32)
  inp = tf.placeholder(tf.float32)
  num_rows = tf.placeholder(tf.int32)
  num_columns = tf.placeholder(tf.int32)
  num_words = tf.placeholder(tf.int32)
  true_labels = tf.placeholder(tf.int32)
  start_time = time.time()

  print "Building Model"
  _, (output,state) = build_model(inp, batch_size, num_rows, num_columns, num_words)
  output = output[:,1:]
  cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output,true_labels))
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.to_int32(tf.argmax( output, 2)), true_labels)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  print "Loading Data"
  train, val, test = load_data()
  train = batchify(train, batch_size)
  #train = sorted(train,key= lambda x: x[1].shape[1])
  random.shuffle(train)
  val = batchify(val, batch_size)
  test = batchify(test, batch_size)

  last_val_acc = 0
  reduce_lr = 0
  with tf.Session() as sess:
    try:
      sess.run(tf.global_variables_initializer())
      print "Training"
      for i in range(epochs):
        if reduce_lr == 5:
          lr = max(min_lr, lr-0.005)
          reduce_lr = 0
        print "Epoch %d learning rate %.4f"%(i,lr)
        epoch_start_time = time.time()
        batch_50_start = epoch_start_time
        for j in range(len(train)):
          images, labels = train[j]
          if j<5 or j%50==0:
            train_accuracy = accuracy.eval(feed_dict={inp: images,\
                                            true_labels:labels,\
                                            num_rows: images.shape[1],\
                                            num_columns: images.shape[2],\
                                            num_words:labels.shape[1]})
            new_time = time.time()
            print("step %d/%d, training accuracy %g, took %f mins"%\
                  (j, len(train), train_accuracy, (new_time - batch_50_start)/60))
            batch_50_start = new_time
          train_step.run(feed_dict={learning_rate: lr,\
                                    inp: images,\
                                    true_labels: labels,\
                                    num_rows: images.shape[1],\
                                    num_columns: images.shape[2],\
                                    num_words: labels.shape[1]})
        print "Time for epoch:%f mins"%((time.time()-epoch_start_time)/60)
        print "Running on Validation Set"
        accs = []
        for j in range(len(val)):
          images, labels = val[j]
          val_accuracy = accuracy.eval(feed_dict={inp: images,\
                                        true_labels: labels,\
                                        num_rows: images.shape[1],\
                                        num_columns: images.shape[2],\
                                        num_words: labels.shape[1]})
          accs.append( val_accuracy )
        val_acc = sess.run(tf.reduce_mean(accs))
        if (val_acc - last_val_acc) >= 1:
          reduce_lr = 0
        else:
          reduce_lr = reduce_lr + 1
        last_val_acc = val_acc
        print("val accuracy %g"%val_acc)
    finally:
      print 'Saving model'
      saver = tf.train.Saver()
      id = 'saved_models/model-'+time.strftime("%d-%m-%Y--%H-%M")
      os.mkdir(id)
      save_path = saver.save(sess, id+'/model' )
      print 'Running on Test Set'
      accs = []
      for j in range(len(test)):
        images, labels = test[j]
        test_accuracy = accuracy.eval(feed_dict={inp: images,\
                                                  true_labels: labels,\
                                                  num_rows: images.shape[1],\
                                                  num_columns: images.shape[2],\
                                                  num_words: labels.shape[1]})
        accs.append( test_accuracy )
      test_acc = sess.run(tf.reduce_mean(accs))
      print("test accuracy %g"%test_acc)

if __name__ == "__main__":
  main()
