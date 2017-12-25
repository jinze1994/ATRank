import tensorflow as tf

class Model(object):

  def __init__(self, user_count, item_count, cate_count, cate_list):
    self.u = tf.placeholder(tf.int32, [None,])
    self.i = tf.placeholder(tf.int32, [None,])
    self.j = tf.placeholder(tf.int32, [None,])
    self.lr = tf.placeholder(tf.float64, [])

    user_emb_w = tf.get_variable("user_emb_w", [user_count, 128])
    item_emb_w = tf.get_variable("item_emb_w", [item_count, 64])
    item_b = tf.get_variable("item_b", [item_count])
    cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, 64])
    cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

    u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)

    ic = tf.gather(cate_list, self.i)
    i_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.i),
        tf.nn.embedding_lookup(cate_emb_w, ic),
        ], 1)
    i_b = tf.gather(item_b, self.i)

    jc = tf.gather(cate_list, self.j)
    j_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.j),
        tf.nn.embedding_lookup(cate_emb_w, jc),
        ], 1)
    j_b = tf.gather(item_b, self.j)

    # MF predict: u_i > u_j
    x = i_b - j_b + tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1)
    self.logits = tf.sigmoid(x)

    # AUC for one user:
    # reasonable iff all (u,i,j) pairs are from the same user
    # average AUC = mean( auc for each user in test set)
    self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))

    # logits for all item:
    all_emb = tf.concat([
        item_emb_w,
        tf.nn.embedding_lookup(cate_emb_w, cate_list)
        ], axis=1)
    self.logits_all = tf.sigmoid(
        item_b + tf.matmul(u_emb, all_emb, transpose_b=True))

    l2_norm = tf.add_n([
        tf.nn.l2_loss(u_emb),
        tf.nn.l2_loss(i_emb),
        tf.nn.l2_loss(j_emb),
        ])

    reg_rate = 5e-5
    self.bprloss = reg_rate * l2_norm - tf.reduce_mean(tf.log(self.logits))

    opt = tf.train.GradientDescentOptimizer
    self.train_op = opt(self.lr).minimize(self.bprloss)

  def train(self, sess, uij, l):
    loss, _ = sess.run([self.bprloss, self.train_op], feed_dict={
        self.u: uij[:, 0],
        self.i: uij[:, 1],
        self.j: uij[:, 2],
        self.lr: l,
        })
    return loss

  def eval(self, sess, test_set):
    return sess.run(self.mf_auc, feed_dict={
        self.u: test_set[:, 0],
        self.i: test_set[:, 1],
        self.j: test_set[:, 2],
        })

  def test(self, sess, uid):
    return sess.run(self.logits_all, feed_dict={
        self.u: uid,
        })

  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)
