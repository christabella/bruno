import argparse
import importlib
import json
import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import logger
import utils

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config_name',
                    type=str,
                    required=True,
                    help='Configuration name')
parser.add_argument('--nr_gpu',
                    type=int,
                    default=1,
                    help='How many GPUs to distribute the training across?')
parser.add_argument('--resume',
                    type=int,
                    default=0,
                    help='Resume training from a checkpoint?')
parser.add_argument('--debug',
                    type=int,
                    default=0,
                    help='Enable tensorflow debugger (TFDBG)?')
parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-3,
                    help='Learning rate.')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
assert args.nr_gpu == len(''.join(
    filter(str.isdigit, os.environ["CUDA_VISIBLE_DEVICES"])))
# -----------------------------------------------------------------------------
np.random.seed(seed=42)
# Just in case, since there's only one default tensorflow graph for all of the tensors accessed during runtime
tf.reset_default_graph()
tf.set_random_seed(0)

# config
configs_dir = __file__.split('/')[-2]
config = importlib.import_module('%s.%s' % (configs_dir, args.config_name))
if not args.resume:
    experiment_id = '%s-%s' % (args.config_name.split('.')[-1],
                               time.strftime("%Y_%m_%d", time.localtime()))
    utils.autodir('metadata')
    save_dir = 'metadata/' + experiment_id
    utils.autodir(save_dir)
else:
    save_dir = utils.find_model_metadata('metadata/', args.config_name)
    experiment_id = os.path.dirname(save_dir).split('/')[-1]
    with open(save_dir + '/meta.pkl', 'rb') as f:
        resumed_metadata = pickle.load(f)
        last_lr = resumed_metadata['lr']
        last_iteration = resumed_metadata['iteration']
        print('Last iteration', last_iteration)
        print('Last learning rate', last_lr)

# logs
utils.autodir('logs')
sys.stdout = logger.Logger('logs/%s.log' % experiment_id)
sys.stderr = sys.stdout

print('exp_id', experiment_id)
if args.resume:
    print('Resuming training')

# create the model
model = tf.make_template('model', config.build_model)

# run once for data dependent initialization of parameters
x_init = tf.placeholder(tf.float32,
                        shape=(config.batch_size, ) + config.obs_shape)
y_init = tf.placeholder(tf.float32,
                        shape=(config.batch_size, ) + config.label_shape)
init_pass = model(x_init, y_init, init=True)[0]

all_params = tf.trainable_variables()
n_parameters = 0
for variable in all_params:
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    n_parameters += variable_parameters
print('Number of parameters', n_parameters)

# get loss gradients over multiple GPUs
xs = []
ys = []
grads = []
train_losses = []

# evaluation in case we want to validate
x_in_eval = tf.placeholder(tf.float32,
                           shape=(config.batch_size, ) + config.obs_shape)
y_in_eval = tf.placeholder(tf.float32,
                           shape=(config.batch_size, ) + config.label_shape)
log_probs = model(x_in_eval, y_in_eval)[0]
eval_loss = config.eval_loss(log_probs) if hasattr(
    config, 'eval_loss') else config.loss(log_probs)

for i in range(args.nr_gpu):
    xs.append(
        tf.placeholder(tf.float32,
                       shape=(config.batch_size / args.nr_gpu, ) +
                       config.obs_shape))
    ys.append(
        tf.placeholder(tf.float32,
                       shape=(config.batch_size / args.nr_gpu, ) +
                       config.label_shape))
    with tf.device('/gpu:%d' % i):
        # train
        with tf.variable_scope('gpu_%d' % i):
            with tf.variable_scope('train'):
                log_probs = model(xs[i], ys[i])[0]
                # log probs is Tensor("gpu_0/train/model/stack:0", shape=(4, 15), dtype=float32, device=/device:GPU:0)
                train_loss = config.loss(log_probs)
                train_losses.append(train_loss)
                # Op to calculate every variable gradient
                grad = tf.gradients(train_losses[i], all_params)
                grads.append(grad)

# add gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
tf_gp_grad_scale = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    # Losses and grads of first gpu is the average of all of them? ?_?
    num_variables = len(grads[0])
    for i in range(1, args.nr_gpu):  # Skip if only 1 GPU
        train_losses[0] += train_losses[i]
        for j in range(num_variables):
            grads[0][j] += grads[i][j]
    # average over gpus
    train_losses[0] /= args.nr_gpu
    for j in range(num_variables):
        grads[0][j] /= args.nr_gpu

    # scale gradients of gp_params
    gp_params = ['prior_nu', 'prior_mean', 'prior_var', 'prior_corr']
    for j in range(num_variables):
        if any(name in all_params[j].name for name in gp_params):
            grads[0][j] *= tf_gp_grad_scale

    # training op; grads[0] is the average over all GPUs
    grads_and_vars = list(zip(grads[0], all_params))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops_gpu0 = []
    for u in update_ops:
        if u.name.startswith('gpu_0/train'):
            update_ops_gpu0.append(u)

    with tf.control_dependencies(update_ops_gpu0):
        if hasattr(config, 'optimizer') and config.optimizer == 'rmsprop':
            print('using rmsprop')
            train_step = tf.train.RMSPropOptimizer(
                learning_rate=tf_lr).apply_gradients(
                    grads_and_vars=grads_and_vars,
                    global_step=None,
                    name='rmsprop')
        else:
            print('using adam')
            train_step = tf.train.AdamOptimizer(
                learning_rate=tf_lr).apply_gradients(
                    grads_and_vars=grads_and_vars,
                    global_step=None,
                    name='adam')

train_loss = train_losses[0]
# Create a summary to monitor cost tensor
train_loss_summary = tf.summary.scalar('train_loss', train_loss)
# Maybe plot GP correlations.
essential_ops = [train_loss_summary]
corr = config.gp_layer.corr
mean = tf.reduce_mean(corr)
essential_ops.append(tf.summary.scalar('gp_corr/mean', mean))
stddev = tf.sqrt(tf.reduce_mean(tf.square(corr - mean)))
essential_ops.append(tf.summary.scalar('gp_corr/stddev', stddev))
essential_ops.append(tf.summary.scalar('gp_corr/max', tf.reduce_max(corr)))
essential_ops.append(tf.summary.scalar('gp_corr/min', tf.reduce_min(corr)))
essential_ops.append(tf.summary.histogram('gp_correlations', corr))

detailed_ops = []
# Create summaries to visualize weights
for var in tf.trainable_variables():
    if "gaussian" in var.name:
        essential_ops.append(
            tf.summary.histogram(var.name.replace(':', '_'), var))
    else:
        detailed_ops.append(
            tf.summary.histogram(var.name.replace(':', '_'), var))
# Summarize all gradients
for grad, var in grads_and_vars:
    if "gaussian" in var.name:
        essential_ops.append(
            tf.summary.histogram('gradients/' + var.name.replace(':', '_'),
                                 grad))
    else:
        detailed_ops.append(
            tf.summary.histogram('gradients/' + var.name.replace(':', '_'),
                                 grad))
# Merge all essential summaries into a single op
merged_essential_ops = tf.summary.merge(essential_ops)
# Merge all extra detailed summaries into a single op
merged_detailed_ops = tf.summary.merge(detailed_ops)

# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

print('\n Start training')
train_data_iter = config.train_data_iter
lr = args.learning_rate
gp_grad_scale = config.scale_gp_grad
batch_idxs = range(0, config.max_iter)
print_every = 100
train_iter_losses = []
if args.resume:
    losses_eval_train = resumed_metadata['losses_eval_train']
    losses_avg_train = resumed_metadata['losses_avg_train']
else:
    losses_eval_train, losses_avg_train = [], []

start_time = time.time()
with tf.Session() as sess:
    # Write the session graph and TF event files to summary directory, which
    # can be anywhere in the current working directory (Guild run directory).
    writer = tf.summary.FileWriter(f'summaries', sess.graph)
    detailed_writer = tf.summary.FileWriter(f'../detailed_summaries',
                                            sess.graph)
    if args.resume:
        ckpt_file = save_dir + 'params.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    prev_time = time.clock()
    for iteration, (x_batch, y_batch) in zip(batch_idxs,
                                             train_data_iter.generate()):
        assert not np.any(np.isnan(y_batch))
        assert not np.any(np.isnan(x_batch))
        if hasattr(config, 'learning_rate_schedule'
                   ) and iteration in config.learning_rate_schedule:
            lr = np.float32(config.learning_rate_schedule[iteration])
        elif hasattr(config, 'lr_decay'):
            lr *= config.lr_decay

        if hasattr(
                config,
                'gp_grad_schedule') and iteration in config.gp_grad_schedule:
            gp_grad_scale = np.float32(config.gp_grad_schedule[iteration])
            print('setting gp grad scale to %.7f' %
                  config.gp_grad_schedule[iteration])

        if args.resume and iteration < last_iteration:
            if iteration % (print_every * 10) == 0:
                print(iteration, 'skipping training')
            continue

        # init
        if iteration == 0:
            print('initializing the model...')
            sess.run(initializer)
            if args.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                sess.add_tensor_filter("has_inf_or_nan",
                                       tf_debug.has_inf_or_nan)
            init_loss = sess.run(init_pass, {x_init: x_batch, y_init: y_batch})
            print(f'Initial loss: {init_loss}')
            if np.isnan(init_loss).any():
                print('Loss is NaN')
                import pdb
                pdb.set_trace()
            sess.graph.finalize()
        else:
            xfs = np.split(x_batch, args.nr_gpu)
            yfs = np.split(y_batch, args.nr_gpu)
            feed_dict = {tf_lr: lr, tf_gp_grad_scale: gp_grad_scale}
            feed_dict.update({xs[i]: xfs[i] for i in range(args.nr_gpu)})
            feed_dict.update({ys[i]: yfs[i] for i in range(args.nr_gpu)})
            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, l = sess.run([train_step, train_loss], feed_dict)
            train_iter_losses.append(l)
            if np.isnan(l):
                print('Loss is NaN')
                import pdb
                pdb.set_trace()
                sys.exit(0)

            if (iteration + 1) % print_every == 0:
                summary_essential, summary_detailed = sess.run(
                    [merged_essential_ops, merged_detailed_ops], feed_dict)
                # Write logs at every 100th iteration
                writer.add_summary(summary_essential, iteration + 1)
                detailed_writer.add_summary(summary_detailed, iteration + 1)
                avg_train_loss = np.mean(train_iter_losses)
                losses_avg_train.append(avg_train_loss)
                train_iter_losses = []
                print('%d/%d train_loss=%6.8f bits/value=%.3f' %
                      (iteration + 1, config.max_iter, avg_train_loss,
                       avg_train_loss / config.ndim / np.log(2.)))
                corr = config.gp_layer.corr.eval().flatten()

            if (iteration + 1) % config.save_every == 0:
                current_time = time.time()
                eta_time = (config.max_iter - iteration
                            ) / config.save_every * (current_time - prev_time)
                prev_time = current_time
                print('ETA: ', time.strftime("%H:%M:%S",
                                             time.gmtime(eta_time)))
                print('Saving model (iteration %s):' % iteration,
                      experiment_id)
                print('current learning rate:', lr)
                saver.save(sess, save_dir + '/params.ckpt')

                with open(save_dir + '/meta.pkl', 'wb') as f:
                    pickle.dump(
                        {
                            'lr': lr,
                            'iteration': iteration + 1,
                            'losses_avg_train': losses_avg_train,
                            'losses_eval_train': losses_eval_train
                        }, f)

                corr = config.gp_layer.corr.eval().flatten()
                print('0.01', np.sum(corr > 0.01))
                print('0.1', np.sum(corr > 0.1))
                print('0.2', np.sum(corr > 0.2))
                print('0.3', np.sum(corr > 0.3))
                print('0.5', np.sum(corr > 0.5))
                print('0.7', np.sum(corr > 0.7))
                print('corr min-max:', np.min(corr), np.max(corr))
                var = config.gp_layer.var.eval().flatten()
                print('var min-max:', np.min(var), np.max(var))

                if hasattr(config.gp_layer, 'nu'):
                    nu = config.gp_layer.nu.eval().flatten()
                    print('nu median-min-max:', np.median(nu), np.min(nu),
                          np.max(nu))

print('Total time: ',
      time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
