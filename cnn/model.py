import os
import json
import numpy as np
import tensorflow as tf

class Model(object):
  def __init__(self, config, cate_list):
    self.config = config

    # Summary Writer
    self.train_writer = tf.summary.FileWriter(config['model_dir'] + '/train')
    self.eval_writer = tf.summary.FileWriter(config['model_dir'] + '/eval')

    # Building network
    self.init_placeholders()
    self.build_model(cate_list)
    self.init_optimizer()


  def init_placeholders(self):
    # [B] user id
    self.u = tf.placeholder(tf.int32, [None,])

    # [B] item id
    self.i = tf.placeholder(tf.int32, [None,])

    # [B] item label
    self.y = tf.placeholder(tf.float32, [None,])

    # [B, T] user's history item id
    self.hist_i = tf.placeholder(tf.int32, [None, None])

    # [B, T] user's history item purchase time
    self.hist_t = tf.placeholder(tf.int32, [None, None])

    # [B] valid length of `hist_i`
    self.sl = tf.placeholder(tf.int32, [None,])

    # learning rate
    self.lr = tf.placeholder(tf.float64, [])

    # whether it's training or not
    self.is_training = tf.placeholder(tf.bool, [])


  def build_model(self, cate_list):
    item_emb_w = tf.get_variable(
        "item_emb_w",
        [self.config['item_count'], self.config['itemid_embedding_size']])
    item_b = tf.get_variable(
        "item_b",
        [self.config['item_count'],],
        initializer=tf.constant_initializer(0.0))
    cate_emb_w = tf.get_variable(
        "cate_emb_w",
        [self.config['cate_count'], self.config['cateid_embedding_size']])
    cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

    i_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.i),
        tf.nn.embedding_lookup(cate_emb_w, tf.gather(cate_list, self.i)),
        ], 1)
    i_b = tf.gather(item_b, self.i)

    h_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.hist_i),
        tf.nn.embedding_lookup(cate_emb_w, tf.gather(cate_list, self.hist_i)),
        ], 2)

    if self.config['concat_time_emb'] is True:
      t_emb = tf.one_hot(self.hist_t, 12, dtype=tf.float32)
      h_emb = tf.concat([h_emb, t_emb], -1)
      h_emb = tf.layers.dense(h_emb, self.config['hidden_units'])
    else:
      t_emb = tf.layers.dense(tf.expand_dims(self.hist_t, -1),
                              self.config['hidden_units'],
                              activation=tf.nn.tanh)
      h_emb += t_emb


    filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # num_filters = [32, 32, 32, 16, 16]
    num_filters = [32] * len(filter_sizes)
    u_emb = cnn_net(
        h_emb,
        self.sl,
        filter_sizes,
        num_filters,
        self.config['hidden_units'],
        self.config['dropout'],
        self.is_training)
    u_emb = tf.layers.dense(u_emb, self.config['hidden_units'])

    self.logits = i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1)

    # ============== Eval ===============
    self.eval_logits = self.logits

    # Step variable
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = \
        tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = \
        tf.assign(self.global_epoch_step, self.global_epoch_step+1)

    # Loss
    l2_norm = tf.add_n([
        tf.nn.l2_loss(u_emb),
        tf.nn.l2_loss(i_emb),
        ])

    self.loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.y)
        ) + self.config['regulation_rate'] * l2_norm

    self.train_summary = tf.summary.merge([
        tf.summary.histogram('embedding/1_item_emb', item_emb_w),
        tf.summary.histogram('embedding/2_cate_emb', cate_emb_w),
        tf.summary.histogram('embedding/3_time_raw', self.hist_t),
        tf.summary.histogram('embedding/3_time_dense', t_emb),
        tf.summary.histogram('embedding/4_final', h_emb),
        tf.summary.histogram('attention_output', u_emb),
        tf.summary.scalar('L2_norm Loss', l2_norm),
        tf.summary.scalar('Training Loss', self.loss),
        ])


  def init_optimizer(self):
    # Gradients and SGD update operation for training the model
    trainable_params = tf.trainable_variables()
    if self.config['optimizer'] == 'adadelta':
      self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
    elif self.config['optimizer'] == 'adam':
      self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
    elif self.config['optimizer'] == 'rmsprop':
      self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
    else:
      self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

    # Compute gradients of loss w.r.t. all trainable variables
    gradients = tf.gradients(self.loss, trainable_params)

    # Clip gradients by a given maximum_gradient_norm
    clip_gradients, _ = tf.clip_by_global_norm(
        gradients, self.config['max_gradient_norm'])

    # Update the model
    self.train_op = self.opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=self.global_step)



  def train(self, sess, uij, l, add_summary=False):
    input_feed = {
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.hist_i: uij[3],
        self.hist_t: uij[4],
        self.sl: uij[5],
        self.lr: l,
        self.is_training: True,
        }

    output_feed = [self.loss, self.train_op]

    if add_summary:
      output_feed.append(self.train_summary)

    outputs = sess.run(output_feed, input_feed)

    if add_summary:
      self.train_writer.add_summary(
          outputs[2], global_step=self.global_step.eval())

    return outputs[0]

  def eval(self, sess, uij):
    res1 = sess.run(self.eval_logits, feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.hist_i: uij[3],
        self.hist_t: uij[4],
        self.sl: uij[5],
        self.is_training: False,
        })
    res2 = sess.run(self.eval_logits, feed_dict={
        self.u: uij[0],
        self.i: uij[2],
        self.hist_i: uij[3],
        self.hist_t: uij[4],
        self.sl: uij[5],
        self.is_training: False,
        })
    return np.mean(res1 - res2 > 0)

  def test(self, sess, uij, item_count):
    pos_logit = sess.run(self.eval_logits, feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.hist_i: uij[3],
        self.hist_t: uij[4],
        self.sl: uij[5],
        self.is_training: False,
        })
    neg_list = list(set(range(item_count)) - set(uij[3][0]) - set(uij[1]))
    u_auc = 0
    neg_logit = sess.run(self.eval_logits, feed_dict={
        self.u: [uij[0][0]] * len(neg_list),
        self.i: neg_list,
        self.hist_i: [uij[3][0]] * len(neg_list),
        self.hist_t: [uij[4][0]] * len(neg_list),
        self.sl: [uij[5][0]] * len(neg_list),
        self.is_training: False,
        })
    u_auc = np.sum(pos_logit >= neg_logit) / len(neg_list)
    return u_auc


  def save(self, sess):
    checkpoint_path = os.path.join(self.config['model_dir'], 'atrank')
    saver = tf.train.Saver()
    save_path = saver.save(sess,
                           save_path=checkpoint_path,
                           global_step=self.global_step.eval())
    jf_name = '%s-%d.json' % (checkpoint_path, self.global_step.eval())
    with open(jf_name, 'w') as jf:
      json.dump(self.config, jf, indent=2)
    print('model saved at %s' % save_path, flush=True)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)
    print('model restored from %s' % path, flush=True)


def cnn_net(h_emb,
            sl,
            filter_sizes,
            num_filters,
            embedding_size,
            dropout_rate,
            is_training):
  # h_emb: [B, T, H]
  mask = tf.sequence_mask(sl, tf.shape(h_emb)[1], dtype=tf.float32) # [B, T]
  mask = tf.expand_dims(mask, -1) # [B, T, 1]
  mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]]) # [B, T, H]
  h_emb *= mask # [B, T, H]

  sequence_length = 500
  h_emb = tf.pad(
      h_emb, [[0, 0], [0, sequence_length-tf.shape(h_emb)[1]], [0, 0]])

  h_emb = tf.expand_dims(h_emb, -1)

  pooled_outputs = []
  for filter_size, num_filter in zip(filter_sizes, num_filters):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
      # Convolution Layer
      filter_shape = [filter_size, embedding_size, 1, num_filter]
      W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
      b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
      conv = tf.nn.conv2d(h_emb,
                          W,
                          strides=[1, 1, 1, 1],
                          padding="VALID",
                          name="conv")
      # Apply nonlinearity
      h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
      # Maxpooling over the outputs
      pooled = tf.nn.max_pool(h,
                              ksize=[1, sequence_length - filter_size + 1, 1, 1], # pylint: disable=line-too-long
                              strides=[1, 1, 1, 1],
                              padding='VALID',
                              name="pool")
      pooled_outputs.append(pooled)

  num_filters_total = sum(num_filters)
  h_pool = tf.concat(pooled_outputs, 3)
  h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

  h_pool_dropout = tf.layers.dropout(h_pool_flat,
                                     dropout_rate,
                                     training=is_training)
  return h_pool_dropout
