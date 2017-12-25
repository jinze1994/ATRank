import os
import pickle
import numpy as np
import tensorflow as tf

from input import DataInput
from model import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(1234)
tf.set_random_seed(1234)

train_batch_size = 32

with open('dataset.pkl', 'rb') as f:
  train_set = pickle.load(f)
  test_set = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count = pickle.load(f)


gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(
    config=tf.ConfigProto(gpu_options=gpu_options)
    ) as sess:

  model = Model(user_count, item_count, cate_count, cate_list)
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  best_auc = 0.0
  lr = 1.0
  for epoch in range(50):

    if epoch % 100 == 0 and epoch != 0:
      lr *= 0.5

    epoch_size = train_set.shape[0] // train_batch_size
    loss_sum = 0.0
    for _, uij in DataInput(train_set, train_batch_size):
      loss = model.train(sess, uij, lr)
      loss_sum += loss

    epoch += 1
    print('epoch: %d\ttrain_loss: %.2f\tlr: %.2f' %
          (epoch, loss_sum / epoch_size, lr), end='\t')

    test_auc = model.eval(sess, test_set)
    print('test_auc: %.4f' % test_auc, flush=True)

    if best_auc < test_auc:
      best_auc = test_auc
      model.save(sess, 'save_path/ckpt')

  print('best test_auc:', best_auc)
