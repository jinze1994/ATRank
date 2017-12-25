import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell

class Model(object):

    def __init__(self, config, map_list):

        self.config = config

        # Summary Writer
        self.train_writer = tf.summary.FileWriter(config['model_dir'] + '/train')
        self.eval_writer = tf.summary.FileWriter(config['model_dir'] + '/eval')

        # Building network
        self.init_placeholders()
        self.build_model(map_list)
        self.init_optimizer()


    def init_placeholders(self):
        # [B] user id
        self.u = tf.placeholder(tf.int32, [None,])

        # [B] item id, query id, coupon id
        self.i = tf.placeholder(tf.int32, [None,])
        self.q = tf.placeholder(tf.int32, [None,])
        self.c = tf.placeholder(tf.int32, [None,])

        # [B] item label(score): yes(1) or no(0)
        self.y = tf.placeholder(tf.float32, [None,])


        # [B, T1] user's history item id
        self.hist_i_id = tf.placeholder(tf.int32, [None, None])

        # [B, T1] user's history action id: value range [0, 3]
        self.hist_i_act = tf.placeholder(tf.int32, [None, None])

        # [B, T1] user's history item purchase time
        self.hist_i_time = tf.placeholder(tf.int32, [None, None])

        # [B]    valid length of `hist_i`
        self.hist_i_sl = tf.placeholder(tf.int32, [None,])


        # [B, T2] user's history query id
        self.hist_q_id = tf.placeholder(tf.int32, [None, None])

        # [B, T2] user's history query time
        self.hist_q_time = tf.placeholder(tf.int32, [None, None])

        # [B]    valid length of `hist_q`
        self.hist_q_sl = tf.placeholder(tf.int32, [None,])


        # [B, T3] user's history coupon id
        self.hist_c_id = tf.placeholder(tf.int32, [None, None])

        # [B, T3] user's history coupon time
        self.hist_c_time = tf.placeholder(tf.int32, [None, None])

        # [B]    valid length of `hist_c`
        self.hist_c_sl = tf.placeholder(tf.int32, [None,])


        # learning rate
        self.lr = tf.placeholder(tf.float64, [])

        # whether it's training or not
        self.is_training = tf.placeholder(tf.bool, [])

        # multi-behavior type
        self.behavior_type = tf.placeholder(tf.string, [])


    def build_model(self, map_list):

        item_feat_shop_list = tf.convert_to_tensor(map_list[0], dtype=tf.int64)
        item_feat_cate_list = tf.convert_to_tensor(map_list[1], dtype=tf.int64)
        item_feat_brand_list = tf.convert_to_tensor(map_list[2], dtype=tf.int64)
        coupon_feat_shop_list = tf.convert_to_tensor(map_list[3], dtype=tf.int64)
        coupon_feat_cate_list = tf.convert_to_tensor(map_list[4], dtype=tf.int64)

        user_emb_w = tf.get_variable("user_emb_w", [self.config['user_count'], 64])
        item_emb_w = tf.get_variable("item_emb_w", [self.config['item_count'], 64])
        shop_emb_w = tf.get_variable('shop_emb_w', [self.config['shop_count'], 64])
        cate_emb_w = tf.get_variable("cate_emb_w", [self.config['cate_count'], 64])
        brand_emb_w = tf.get_variable("brand_emb_w", [self.config['brand_count'], 64])
        action_emb_w = tf.get_variable('action_emb_w', [self.config['action_count'], 64])
        query_emb_w = tf.get_variable('query_emb_w', [self.config['query_count'], 64])
        coupon_emb_w = tf.get_variable('coupon_emb_w', [self.config['coupon_count'], 64])

        i_layer = Dense(self.config['hidden_units'], name='i_layer')
        q_layer = Dense(self.config['hidden_units'], name='q_layer')
        c_layer = Dense(self.config['hidden_units'], name='c_layer')


        u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)

        i_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.i),
            tf.nn.embedding_lookup(shop_emb_w, tf.gather(item_feat_shop_list, self.i)),
            tf.nn.embedding_lookup(cate_emb_w, tf.gather(item_feat_cate_list, self.i)),
            tf.nn.embedding_lookup(brand_emb_w, tf.gather(item_feat_brand_list, self.i)),
            ], 1)
        i_emb = tf.pad(i_emb, [[0, 0], [0, 64+12]])
        i_emb = i_layer(i_emb)
        item_b = tf.get_variable("item_b", [self.config['item_count'], ],
                initializer=tf.constant_initializer(0.0))
        i_b = tf.gather(item_b, self.i)

        q_emb = tf.nn.embedding_lookup(query_emb_w, self.q)
        q_emb = tf.pad(q_emb, [[0, 0], [0, 12]])
        q_emb = q_layer(q_emb)
        item_q = tf.get_variable("item_q", [self.config['query_count'], ],
                initializer=tf.constant_initializer(0.0))
        q_b = tf.gather(item_q, self.q)

        c_emb = tf.concat([
            tf.nn.embedding_lookup(coupon_emb_w, self.c),
            tf.nn.embedding_lookup(shop_emb_w, tf.gather(coupon_feat_shop_list, self.c)),
            tf.nn.embedding_lookup(cate_emb_w, tf.gather(coupon_feat_cate_list, self.c)),
            ], 1)
        c_emb = tf.pad(c_emb, [[0, 0], [0, 12]])
        c_emb = c_layer(c_emb)
        item_c = tf.get_variable("item_c", [self.config['coupon_count'], ],
                initializer=tf.constant_initializer(0.0))
        c_b = tf.gather(item_c, self.c)


        hist_i_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.hist_i_id),
            tf.nn.embedding_lookup(shop_emb_w, tf.gather(item_feat_shop_list, self.hist_i_id)),
            tf.nn.embedding_lookup(cate_emb_w, tf.gather(item_feat_cate_list, self.hist_i_id)),
            tf.nn.embedding_lookup(brand_emb_w, tf.gather(item_feat_brand_list, self.hist_i_id)),
            tf.nn.embedding_lookup(action_emb_w, self.hist_i_act),
            tf.one_hot(self.hist_i_time, 12, dtype=tf.float32),
            ], 2)
        hist_i_emb = i_layer(hist_i_emb)
        hist_i_mask = tf.sequence_mask(self.hist_i_sl, tf.shape(hist_i_emb)[1])

        hist_q_emb = tf.concat([
            tf.nn.embedding_lookup(query_emb_w, self.hist_q_id),
            tf.one_hot(self.hist_q_time, 12, dtype=tf.float32),
            ], 2)
        hist_q_emb = q_layer(hist_q_emb)
        hist_q_mask = tf.sequence_mask(self.hist_q_sl, tf.shape(hist_q_emb)[1])

        hist_c_emb = tf.concat([
            tf.nn.embedding_lookup(coupon_emb_w, self.hist_c_id),
            tf.nn.embedding_lookup(shop_emb_w, tf.gather(coupon_feat_shop_list, self.hist_c_id)),
            tf.nn.embedding_lookup(cate_emb_w, tf.gather(coupon_feat_cate_list, self.hist_c_id)),
            tf.one_hot(self.hist_c_time, 12, dtype=tf.float32),
            ], 2)
        hist_c_emb = c_layer(hist_c_emb)
        hist_c_mask = tf.sequence_mask(self.hist_c_sl, tf.shape(hist_c_emb)[1])

        hist_emb = tf.concat([hist_i_emb, hist_q_emb, hist_c_emb], 1)
        hist_mask = tf.concat([hist_i_mask, hist_q_mask, hist_c_mask], 1)

        num_blocks = self.config['num_blocks']
        num_heads = self.config['num_heads']
        dropout_rate = self.config['dropout']
        num_units = hist_emb.get_shape().as_list()[-1]

        filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
        num_filters = [32] * len(filter_sizes)

        i_emb = tf.reshape(tf.case([
            (tf.equal(self.behavior_type, 'item_batch'),   lambda: i_emb),
            (tf.equal(self.behavior_type, 'query_batch'),  lambda: q_emb),
            (tf.equal(self.behavior_type, 'coupon_batch'), lambda: c_emb),
            ], default=lambda: tf.zeros([1])), [-1, self.config['hidden_units']])
        i_b = tf.case([
            (tf.equal(self.behavior_type, 'item_batch'),   lambda: i_b),
            (tf.equal(self.behavior_type, 'query_batch'),  lambda: q_b),
            (tf.equal(self.behavior_type, 'coupon_batch'), lambda: c_b),
            ], default=lambda: tf.zeros([1]))

        # ============== Model ==============
        if self.config['net_type'].startswith('att-a2'):
            u_emb, self.att_vec, self.stt_vec = attention_net(hist_emb, hist_mask, i_emb, num_units, num_heads, num_blocks, dropout_rate, self.is_training, False)

        elif self.config['net_type'].startswith('att-i2'):
            u_emb, self.att_vec, self.stt_vec = attention_net(hist_i_emb, hist_i_mask, i_emb, num_units, num_heads, num_blocks, dropout_rate, self.is_training, False)
        elif self.config['net_type'].startswith('att-q2'):
            u_emb, self.att_vec, self.stt_vec = attention_net(hist_q_emb, hist_q_mask, i_emb, num_units, num_heads, num_blocks, dropout_rate, self.is_training, False)
        elif self.config['net_type'].startswith('att-c2'):
            u_emb, self.att_vec, self.stt_vec = attention_net(hist_c_emb, hist_c_mask, i_emb, num_units, num_heads, num_blocks, dropout_rate, self.is_training, False)

        elif self.config['net_type'].startswith('rnn-i2'):
            u_emb = birnn_net(hist_i_emb, self.hist_i_sl, num_units)
        elif self.config['net_type'].startswith('rnn-q2'):
            u_emb = birnn_net(hist_q_emb, self.hist_q_sl, num_units)
        elif self.config['net_type'].startswith('rnn-c2'):
            u_emb = birnn_net(hist_c_emb, self.hist_c_sl, num_units)

        elif self.config['net_type'].startswith('rna-i2'):
            u_emb = birnn_att_net(hist_i_emb, self.hist_i_sl, i_emb, num_units)
        elif self.config['net_type'].startswith('rna-q2'):
            u_emb = birnn_att_net(hist_q_emb, self.hist_q_sl, i_emb, num_units)
        elif self.config['net_type'].startswith('rna-c2'):
            u_emb = birnn_att_net(hist_c_emb, self.hist_c_sl, i_emb, num_units)

        elif self.config['net_type'].startswith('cnn-i2'):
            u_emb = cnn_net(hist_i_emb, self.hist_i_sl, filter_sizes, num_filters, num_units, dropout_rate, self.is_training)
        elif self.config['net_type'].startswith('cnn-q2'):
            u_emb = cnn_net(hist_q_emb, self.hist_q_sl, filter_sizes, num_filters, num_units, dropout_rate, self.is_training)
        elif self.config['net_type'].startswith('cnn-c2'):
            u_emb = cnn_net(hist_c_emb, self.hist_c_sl, filter_sizes, num_filters, num_units, dropout_rate, self.is_training)

        else:
            print('net_type error')
            exit(1)

        # MF predict: u_i > u_j
        self.logits = i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1)

        # ============== Eval ===============
        # self.eval_logits = i_b + tf.reduce_sum(tf.multiply(u_emb_test_i, i_emb), 1)
        self.eval_logits = self.logits
        # self.eval_auc = tf.reduce_mean(tf.to_float(tf.equal(self.eval_logits > 0, self.y > 0.5)))
    
        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
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
            # tf.summary.histogram('embedding/1_item_emb', item_emb_w),
            # tf.summary.histogram('embedding/2_cate_emb', cate_emb_w),
            # tf.summary.histogram('embedding/3_time_raw', self.hist_t),
            # tf.summary.histogram('embedding/3_time_dense', t_emb),
            # tf.summary.histogram('embedding/4_final', h_emb),
            # tf.summary.histogram('attention_output', u_emb),
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
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.config['max_gradient_norm'])

        # Update the model
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)



    def train(self, sess, uij, l, add_summary=False, behavior_type=None):
        if behavior_type == None:
            if self.config['net_type'].endswith('i'):
                behavior_type = 'item_batch'
            elif self.config['net_type'].endswith('q'):
                behavior_type = 'query_batch'
            elif self.config['net_type'].endswith('c'):
                behavior_type = 'coupon_batch'
            else:
                print('net_type error')
                exit(1)
        assert(behavior_type != None)

        input_feed = {
            self.u: uij[0],
            self.i: uij[1] if behavior_type == 'item_batch' else [0] * len(uij[1]),
            self.q: uij[1] if behavior_type == 'query_batch' else [0] * len(uij[1]),
            self.c: uij[1] if behavior_type == 'coupon_batch' else [0] * len(uij[1]),
            self.y: uij[2],
            self.hist_i_id:   uij[3],
            self.hist_i_act:  uij[4],
            self.hist_i_time: uij[5],
            self.hist_i_sl:   uij[6],
            self.hist_q_id:   uij[7],
            self.hist_q_time: uij[8],
            self.hist_q_sl:   uij[9],
            self.hist_c_id:   uij[10],
            self.hist_c_time: uij[11],
            self.hist_c_sl:   uij[12],
            self.is_training: True,
            self.lr: l,
            self.behavior_type: behavior_type,
            }

        output_feed = [self.loss, self.train_op]

        if add_summary:
            output_feed.append(self.train_summary)

        outputs = sess.run(output_feed, input_feed)

        if add_summary:
            self.train_writer.add_summary(outputs[2],
                    global_step=self.global_step.eval())

        return outputs[0]

    def eval(self, sess, uij, behavior_type=None):
        if behavior_type == None:
            if self.config['net_type'].endswith('i'):
                behavior_type = 'item_batch'
            elif self.config['net_type'].endswith('q'):
                behavior_type = 'query_batch'
            elif self.config['net_type'].endswith('c'):
                behavior_type = 'coupon_batch'
            else:
                print('net_type error')
                exit(1)
        assert(behavior_type != None)

        return sess.run(self.eval_logits, feed_dict={
            self.u: uij[0],
            self.i: uij[1] if behavior_type == 'item_batch' else [0] * len(uij[1]),
            self.q: uij[1] if behavior_type == 'query_batch' else [0] * len(uij[1]),
            self.c: uij[1] if behavior_type == 'coupon_batch' else [0] * len(uij[1]),
            self.y: uij[2],
            self.hist_i_id:   uij[3],
            self.hist_i_act:  uij[4],
            self.hist_i_time: uij[5],
            self.hist_i_sl:   uij[6],
            self.hist_q_id:   uij[7],
            self.hist_q_time: uij[8],
            self.hist_q_sl:   uij[9],
            self.hist_c_id:   uij[10],
            self.hist_c_time: uij[11],
            self.hist_c_sl:   uij[12],
            self.is_training: False,
            self.behavior_type: behavior_type,
            })

    def test(self, sess, uij, behavior_type=None):
        assert(len(uij[0]) == 1)
        if behavior_type == None:
            if self.config['net_type'].endswith('i'):
                behavior_type = 'item_batch'
            elif self.config['net_type'].endswith('q'):
                behavior_type = 'query_batch'
            elif self.config['net_type'].endswith('c'):
                behavior_type = 'coupon_batch'
            else:
                print('net_type error')
                exit(1)
        assert(behavior_type != None)

        logit, att_vec, stt_vec = sess.run([self.eval_logits, self.att_vec, self.stt_vec], feed_dict={
            self.u: uij[0],
            self.i: uij[1] if behavior_type == 'item_batch' else [0] * len(uij[1]),
            self.q: uij[1] if behavior_type == 'query_batch' else [0] * len(uij[1]),
            self.c: uij[1] if behavior_type == 'coupon_batch' else [0] * len(uij[1]),
            self.y: uij[2],
            self.hist_i_id:   uij[3],
            self.hist_i_act:  uij[4],
            self.hist_i_time: uij[5],
            self.hist_i_sl:   uij[6],
            self.hist_q_id:   uij[7],
            self.hist_q_time: uij[8],
            self.hist_q_sl:   uij[9],
            self.hist_c_id:   uij[10],
            self.hist_c_time: uij[11],
            self.hist_c_sl:   uij[12],
            self.is_training: False,
            self.behavior_type: behavior_type,
            })
        return logit, att_vec, stt_vec


       
    def save(self, sess):
        checkpoint_path = os.path.join(self.config['model_dir'], 'atrank')
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path=checkpoint_path, global_step=self.global_step.eval())
        json.dump(self.config,
                open('%s-%d.json' % (checkpoint_path, self.global_step.eval()), 'w'),
                indent=2)
        print('model saved at %s' % save_path, flush=True)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path, flush=True)


def attention_net(enc, enc_mask, dec, num_units, num_heads, num_blocks, dropout_rate, is_training, reuse):
    with tf.variable_scope("all", reuse=reuse):
        with tf.variable_scope("user_hist_group"):
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    enc, stt_vec = multihead_attention(queries=enc,
                            queries_mask=enc_mask,
                            keys=enc,
                            keys_mask=enc_mask,
                            num_units=num_units,
                            num_heads=num_heads,
                            dropout_rate=dropout_rate,
                            is_training=is_training,
                            scope="self_attention"
                            )

                    ### Feed Forward
                    enc = feedforward(enc,
                            num_units=[num_units // 4, num_units],
                            scope="feed_forward", reuse=reuse)

        dec = tf.expand_dims(dec, 1)
        with tf.variable_scope("item_feature_group"):
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( vanilla attention)
                    dec, att_vec = multihead_attention(queries=dec,
                            queries_mask=tf.ones_like(dec[:, :, 0], dtype=tf.int32),
                            keys=enc,
                            keys_mask=enc_mask,
                            num_units=num_units,
                            num_heads=num_heads,
                            dropout_rate=dropout_rate,
                            is_training=is_training,
                            scope="vanilla_attention")

                    ## Feed Forward
                    dec = feedforward(dec,
                            num_units=[num_units // 4, num_units],
                            scope="feed_forward", reuse=reuse)

        dec = tf.reshape(dec, [-1, num_units])
        return dec, att_vec, stt_vec


def multihead_attention(queries,
                        queries_mask,
                        keys,
                        keys_mask,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      queries_length: A 1d tensor with shape of [N].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      keys_length:  A 1d tensor with shape of [N].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections, C = # dim or column, T_x = # vectors or actions
        # same size mapping op!
        # in order to reorg the features into different groups
        # but what is the difference?
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        # query-key score matrix
        # each big score matrix is then split into h score matrix with same size
        # w.r.t. different part of the feature
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = keys_mask   # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)

        # Causality = Future blinding: No use, removed

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.to_float(queries_mask)   # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (h*N, T_q, T_k)

        # Attention vector
        att_vec = outputs

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)

    return outputs, att_vec

def feedforward(inputs,
                num_units=[2048, 512],
                scope="feedforward",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def birnn_net(h_emb, sl, hidden_units):
    cell_fw = build_cell(hidden_units)
    cell_bw = build_cell(hidden_units)
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, h_emb, sl, dtype=tf.float32)
    hist = tf.concat([
        extract_axis_1(rnn_output[0], sl-1),
        tf.reshape(rnn_output[1][:, 0, :], [-1, hidden_units]),
        ], axis=1)
    hist = tf.layers.dense(hist, hidden_units)
    return hist

def birnn_att_net(h_emb, sl, i_emb, hidden_units):
    cell_fw = build_cell(hidden_units)
    cell_bw = build_cell(hidden_units)
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, h_emb, sl, dtype=tf.float32)
    rnn_output = tf.concat(rnn_output, 2)

    hist = vanilla_attention(i_emb, rnn_output, sl)
    hist = tf.reshape(hist, [-1, hidden_units * 2])
    hist = tf.layers.dense(hist, hidden_units)
    return hist

'''
    queries:     [B, H]
    keys:        [B, T, H]
    keys_length: [B]
'''
def vanilla_attention(queries, keys, keys_length):
    queries = tf.tile(queries, [1, 2])
    queries = tf.expand_dims(queries, 1) # [B, 1, H]
    # Multiplication
    outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1])) # [B, 1, T]

    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
    key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, 1, T]

    # Weighted sum
    outputs = tf.matmul(outputs, keys)  # [B, 1, H]

    return outputs


def extract_axis_1(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res

def build_single_cell(hidden_units):
    cell_type = LSTMCell
    # cell_type = GRUCell
    cell = cell_type(hidden_units)
    return cell

def build_cell(hidden_units, depth=1):
    cell_list = [build_single_cell(hidden_units) for i in range(depth)]
    return MultiRNNCell(cell_list)
    user_count, item_count, cate_count = pickle.load(f)


def cnn_net(h_emb, sl, filter_sizes, num_filters, embedding_size, dropout_rate, is_training):
    # h_emb: [B, T, H]
    mask = tf.sequence_mask(sl, tf.shape(h_emb)[1], dtype=tf.float32) # [B, T]
    mask = tf.expand_dims(mask, -1) # [B, T, 1]
    mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]]) # [B, T, H]
    h_emb *= mask # [B, T, H]

    sequence_length = 30
    h_emb = tf.pad(h_emb, [[0, 0], [0, sequence_length-tf.shape(h_emb)[1]], [0, 0]])

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
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],  
                    padding='VALID',  
                    name="pool")  
            pooled_outputs.append(pooled)

    num_filters_total = sum(num_filters)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    h_pool_dropout = tf.layers.dropout(h_pool_flat, dropout_rate, training=is_training)
    return h_pool_dropout
