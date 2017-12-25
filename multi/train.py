import os
import time
import json
import math
import pickle
import random
import numpy as np
import tensorflow as tf
from sklearn import metrics
from collections import OrderedDict

from input import DataInput
from model import Model

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

# Network parameters
tf.app.flags.DEFINE_integer('hidden_units', 256, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_blocks', 1, 'Number of blocks in each attention')
tf.app.flags.DEFINE_integer('num_heads', 4, 'Number of heads in each attention')
tf.app.flags.DEFINE_float('dropout', 0.1, 'Dropout probability(0.0: no dropout)')
tf.app.flags.DEFINE_float('regulation_rate', 0.00001, 'L2 regulation rate')

tf.app.flags.DEFINE_integer('itemid_embedding_size', 64, 'Item id embedding size')
tf.app.flags.DEFINE_integer('cateid_embedding_size', 64, 'Cate id embedding size')

tf.app.flags.DEFINE_integer('concat_time_emb', True, 'Concat time-embedding instead of Add')

tf.app.flags.DEFINE_string('net_type', 'att-i2i', 'a: attention  i: rnn-item  q: rnn-query  c: rnn-coupon')

# Training parameters
tf.app.flags.DEFINE_boolean('from_scratch', True, 'Romove model_dir, and train from scratch, default: False')
tf.app.flags.DEFINE_string('model_dir', 'save_path', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('optimizer', 'sgd', 'Optimizer for training: (adadelta, adam, rmsprop,sgd*)')
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'Learning rate')
tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm')

tf.app.flags.DEFINE_integer('train_batch_size', 32, 'Training Batch size')
tf.app.flags.DEFINE_integer('test_batch_size', 512, 'Testing Batch size')
tf.app.flags.DEFINE_integer('max_epochs', 10, 'Maximum # of training epochs')

tf.app.flags.DEFINE_integer('display_freq', 100, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('eval_freq', 100, 'Display training status every this iteration')

# Runtime parameters
tf.app.flags.DEFINE_string('cuda_visible_devices', '1', 'Choice which GPU to use')
tf.app.flags.DEFINE_float('per_process_gpu_memory_fraction', 0.0, 'Gpu memory use fraction, 0.0 for allow_growth=True')

FLAGS = tf.app.flags.FLAGS

def create_model(sess, config, cate_list):

    print(json.dumps(config, indent=4), flush=True)
    model = Model(config, cate_list)

    print('All global variables:')
    for v in tf.global_variables():
        if v not in tf.trainable_variables():
            print('\t', v)
        else:
            print('\t', v, 'trainable')

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..', flush=True)
        model.restore(sess, ckpt.model_checkpoint_path)
    else:
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        print('Created new model parameters..', flush=True)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

    return model

def eval(sess, test_set, model, behavior_type=None):

    predict_prob_y, test_y = [], []
    for iter, uij in DataInput(test_set, FLAGS.test_batch_size):
        test_y.extend(uij[2])
        predict_prob_y.extend(model.eval(sess, uij, behavior_type).tolist())
    assert(len(test_y) == len(predict_prob_y))
    test_auc = metrics.roc_auc_score(test_y, predict_prob_y)

    # model.eval_writer.add_summary(
    #         summary=tf.Summary(value=[tf.Summary.Value(tag='Eval AUC', simple_value=test_auc)]),
    #         global_step=model.global_step.eval())

    return test_auc


def train():
    start_time = time.time()

    if FLAGS.from_scratch:
        if tf.gfile.Exists(FLAGS.model_dir):
            tf.gfile.DeleteRecursively(FLAGS.model_dir)
        tf.gfile.MakeDirs(FLAGS.model_dir)

    # Loading data
    print('Loading data..', flush=True)
    with open('dataset.pkl', 'rb') as f:
        ai_train_set = pickle.load(f)
        ai_test_set = pickle.load(f)
        item_feat_shop_list = pickle.load(f)
        item_feat_cate_list = pickle.load(f)
        item_feat_brand_list = pickle.load(f)
        coupon_feat_shop_list = pickle.load(f)
        coupon_feat_cate_list = pickle.load(f)
        coupon_feat_type_list = pickle.load(f)
        user_count, item_count, shop_count, cate_count, brand_count, action_count, query_count, coupon_count = pickle.load(f)
        aq_train_set = pickle.load(f)
        aq_test_set = pickle.load(f)
        ac_train_set = pickle.load(f)
        ac_test_set = pickle.load(f)

    # Config GPU options
    if FLAGS.per_process_gpu_memory_fraction == 0.0:
        gpu_options = tf.GPUOptions(allow_growth=True)
    elif FLAGS.per_process_gpu_memory_fraction == 1.0:
        gpu_options = tf.GPUOptions()
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_visible_devices

    # Build Config
    config = OrderedDict(sorted(FLAGS.__flags.items()))
    config['user_count'] = user_count
    config['item_count'] = item_count
    config['shop_count'] = shop_count
    config['cate_count'] = cate_count
    config['brand_count'] = brand_count
    config['action_count'] = action_count
    config['query_count'] = query_count
    config['coupon_count'] = coupon_count

    if config['net_type'].endswith('i'):
        train_set = ai_train_set
        test_set = ai_test_set
    elif config['net_type'].endswith('q'):
        train_set = aq_train_set
        test_set = aq_test_set
    elif config['net_type'].endswith('c'):
        train_set = ac_train_set
        test_set = ac_test_set
    else:
        print('net_type error')
        exit(1)

    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options)) as sess:

        # Create a new model or reload existing checkpoint
        model = create_model(sess, config,
                (item_feat_shop_list, item_feat_cate_list, item_feat_brand_list, coupon_feat_shop_list, coupon_feat_cate_list, coupon_feat_type_list))
        print('Init finish.\tCost time: %.2fs' % (time.time()-start_time), flush=True)

        # Eval init AUC
        best_auc = eval(sess, test_set, model)
        print('Init AUC: %.4f' % best_auc)

        # Start training
        lr = FLAGS.learning_rate
        epoch_size = round(len(train_set) / FLAGS.train_batch_size)
        print('Training..\tmax_epochs: %d\tepoch_size: %d' %
                (FLAGS.max_epochs, epoch_size), flush=True)

        start_time, avg_loss = time.time(), 0.0
        for _ in range(FLAGS.max_epochs):

            random.shuffle(train_set)

            for _, uij in DataInput(train_set, FLAGS.train_batch_size):

                add_summary = True if model.global_step.eval() % FLAGS.display_freq == 0 else False
                step_loss = model.train(sess, uij, lr, add_summary)
                avg_loss += step_loss

                if model.global_step.eval() % FLAGS.eval_freq == 0:
                    test_auc = eval(sess, test_set, model)
                    print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_AUC: %.4f' % 
                            (model.global_epoch_step.eval(), model.global_step.eval(), avg_loss / FLAGS.eval_freq, test_auc),
                            flush=True)
                    avg_loss = 0.0

                    if test_auc > 0.60 and test_auc > best_auc:
                        best_auc = test_auc
                    #     model.save(sess)

            # if model.global_epoch_step.eval() == 2:
            #     lr = 0.1

            print('Epoch %d DONE\tCost time: %.2f' % (model.global_epoch_step.eval(), time.time()-start_time), flush=True)
            model.global_epoch_step_op.eval()
            
        print('best test_auc:', best_auc)
        print('Finished', flush=True)
    

def train_all():
    start_time = time.time()

    if FLAGS.from_scratch:
        if tf.gfile.Exists(FLAGS.model_dir):
            tf.gfile.DeleteRecursively(FLAGS.model_dir)
        tf.gfile.MakeDirs(FLAGS.model_dir)

    # Loading data
    print('Loading data..', flush=True)
    with open('dataset.pkl', 'rb') as f:
        ai_train_set = pickle.load(f)
        ai_test_set = pickle.load(f)
        item_feat_shop_list = pickle.load(f)
        item_feat_cate_list = pickle.load(f)
        item_feat_brand_list = pickle.load(f)
        coupon_feat_shop_list = pickle.load(f)
        coupon_feat_cate_list = pickle.load(f)
        coupon_feat_type_list = pickle.load(f)
        user_count, item_count, shop_count, cate_count, brand_count, action_count, query_count, coupon_count = pickle.load(f)
        aq_train_set = pickle.load(f)
        aq_test_set = pickle.load(f)
        ac_train_set = pickle.load(f)
        ac_test_set = pickle.load(f)

    # Config GPU options
    if FLAGS.per_process_gpu_memory_fraction == 0.0:
        gpu_options = tf.GPUOptions(allow_growth=True)
    elif FLAGS.per_process_gpu_memory_fraction == 1.0:
        gpu_options = tf.GPUOptions()
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_visible_devices

    # Build Config
    config = OrderedDict(sorted(FLAGS.__flags.items()))
    config['user_count'] = user_count
    config['item_count'] = item_count
    config['shop_count'] = shop_count
    config['cate_count'] = cate_count
    config['brand_count'] = brand_count
    config['action_count'] = action_count
    config['query_count'] = query_count
    config['coupon_count'] = coupon_count
    assert(config['net_type'] == 'att-a2a')

    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options)) as sess:

        # Create a new model or reload existing checkpoint
        model = create_model(sess, config,
                (item_feat_shop_list, item_feat_cate_list, item_feat_brand_list, coupon_feat_shop_list, coupon_feat_cate_list, coupon_feat_type_list))
        print('Init finish.\tCost time: %.2fs' % (time.time()-start_time), flush=True)

        def eval_all():
            return eval(sess, ai_test_set, model, 'item_batch'),\
                   eval(sess, aq_test_set, model, 'query_batch'),\
                   eval(sess, ac_test_set, model, 'coupon_batch')

        # Eval init AUC
        print('Init AUC:\tai %.4f\taq %.4f\tac %.4f' % eval_all())

        # Start training
        lr = FLAGS.learning_rate
        ai_epoch_size = math.ceil(len(ai_train_set) / FLAGS.train_batch_size)
        aq_epoch_size = math.ceil(len(aq_train_set) / FLAGS.train_batch_size)
        ac_epoch_size = math.ceil(len(ac_train_set) / FLAGS.train_batch_size)
        sum_epoch_size = ai_epoch_size + aq_epoch_size + ac_epoch_size
        print('Training..\tmax_epochs: %d\tepoch_size: %d (%d %d %d)' %
                (FLAGS.max_epochs, sum_epoch_size, ai_epoch_size, aq_epoch_size, ac_epoch_size), flush=True)

        start_time, avg_loss, best_auc = time.time(), 0.0, 0.0
        for _ in range(FLAGS.max_epochs):

            random.shuffle(ai_train_set)
            random.shuffle(aq_train_set)
            random.shuffle(ac_train_set)

            ai_input = DataInput(ai_train_set, FLAGS.train_batch_size)
            aq_input = DataInput(aq_train_set, FLAGS.train_batch_size)
            ac_input = DataInput(ac_train_set, FLAGS.train_batch_size)
            inputs = (ai_input, aq_input, ac_input)
            behavior_type = ('item_batch', 'query_batch', 'coupon_batch')

            epoch_t = [0]*ai_epoch_size + [1]*aq_epoch_size + [2]*ac_epoch_size
            random.shuffle(epoch_t)

            for t in epoch_t:
                _, uij = next(inputs[t])

                # Apply different learning rate according to the behavior type,
                # in case of varying degrees of overfitting.
                if behavior_type[t][0] == 'i':
                    llr = 0.1
                elif behavior_type[t][0] == 'q':
                    llr = 0.04
                else:
                    llr = 0.08
                step_loss = model.train(sess, uij, llr, False, behavior_type[t])
                avg_loss += step_loss

                if model.global_step.eval() % FLAGS.eval_freq == 0:
                    test_auc = eval_all()
                    print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_AUC: %.4f %.4f %.4f %.4f' % 
                            (model.global_epoch_step.eval(), model.global_step.eval(), avg_loss / FLAGS.eval_freq, \
                                    test_auc[0], test_auc[1], test_auc[2], sum(test_auc)),
                            flush=True)
                    avg_loss = 0.0

                    if sum(test_auc) > 2.1 and sum(test_auc) > best_auc:
                        best_auc = sum(test_auc)
                        model.save(sess)

                # if model.global_step.eval() == 50000:
                #     lr = 0.1

            print('Epoch %d DONE\tCost time: %.2f' % (model.global_epoch_step.eval(), time.time()-start_time), flush=True)
            model.global_epoch_step_op.eval()
            
        model.save(sess)
        print('best test_auc:', best_auc)
        print('Finished', flush=True)


def main(_):
    if FLAGS.net_type == 'att-a2a':
        train_all()
    else:
        train()

if __name__ == '__main__':
    tf.app.run()
