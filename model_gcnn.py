from __future__ import division
import tensorflow as tf
import numpy as np
import scipy
from utils import show_all_variables
from attn import attn

tfversion_ = tf.VERSION.split(".")
global tfversion
if int(tfversion_[0]) < 1:
    raise EnvironmentError("TF version should be above 1.0!!")

if int(tfversion_[1]) < 1:
    print("Working in TF version 1.0....")
    tfversion = "old"
else:
    print("Working in TF version 1.%d...." % int(tfversion_[1]))
    tfversion = "new"

EPSILON = 1e-3
nonlinearity = tf.nn.elu
hid_units = [8] # numbers of hidden units per each attention head in each layer

def kl_divergence(p, p_hat):
    p_hat = tf.clip_by_value(p_hat, clip_value_min=0, clip_value_max=1)
    return p * tf.log(p / p_hat + EPSILON) + (1 - p) * tf.log((1 - p) / (1 - p_hat) + EPSILON)


class Model(object):
    def __init__(self, config, output_node, E=None):
        self.server_name = config.server_name

        self.p = config.pool_size

        M_0 = output_node

        self.E = E
        self.regularizers = []

        self.regularization = config.regularization
        self.dropout = config.dropout
        self.model_type = config.model_type
        self.batch_size = config.batch_size
        self.num_node = M_0
        self.output_node = output_node
        self.residual = config.residual
        self.congc = config.congc

        self.temporal_dim = config.temporal_dim
        self.temporal_attn = config.temporal_attn
        self.n_heads_spt= config.n_heads_spt
        self.n_heads_tmp = config.n_heads_tmp
        self.layers = config.layers
        self.is_pooling = config.is_pooling
        self.is_removing_T = config.is_removing_T
        self.is_sparse = config.is_sparse
        self.beta = config.beta
        self.p_o = config.p_o
        self.min_relu = config.min_relu
        self.is_removing_attn = config.is_removing_attn

        self.scl_evl_E = config.scl_evl_E
        self.scl_evl_T = config.scl_evl_T

        # periodic paramters
        self.period_dim = config.period_dim
        self.n_heads_prd = config.n_heads_prd
        self.no_scl = config.no_scl
        self.mask_weight_no_scl = config.mask_weight_no_scl
        self.spt_scl = config.spt_scl

        # hypergraph flag
        self.hypergraph_flag = config.hypergraph_flag
        self.bias_hyper = config.bias_hyper
        self.n_hid = config.n_hid
        self.dropout_hyper = config.dropout_hyper
        self.hyper_weight_stddev = config.hyper_weight_stddev
        self.hyper_bias_stddev= config.hyper_bias_stddev
        self.hgnn_residual = config.hgnn_residual
        self.distance_shunxu = config.distance_shunxu
        self.hyper_w_transform_flag = config.hyper_w_transform_flag

        # vae flag
        self.vae_flag = config.vae_flag
        self.stddev1 = config.stddev1
        self.stddev2 = config.stddev2
        self.loss_vae = config.loss_vae
        self.onlyB = config.onlyB
        self.vae_residual = config.vae_residual

        if self.scl_evl_E == 1 or self.scl_evl_T == 1:
            self.H = np.random.randint(0, 2, size=(self.num_node, 1)).astype(np.float32)
        else:
            import csv
            with open(
                    'osm_json/' + self.server_name + '/' + str(self.distance_shunxu) + 'id-nodes-shunxu.csv') as f2:
                xianp = csv.reader(f2)
                xianpoi = []
                for xi in xianp:
                    xianpoi.append(xi)
            for index, item in enumerate(xianpoi):
                xianpoi[index] = list(map(int, item))

            self.H = np.zeros((self.num_node, len(xianpoi)))
            for i in range(len(xianpoi)):
                for n_j in xianpoi[i]:
                    if n_j - 1 > self.num_node - 1: continue
                    self.H[n_j - 1][i] = 1


        if config.model_type == 'hist':
            self.feat_in = len(config.hist_range) - 1
            self.feat_out = len(config.hist_range) - 1
        else:
            self.feat_in = config.feat_in
            self.feat_out = config.feat_out

        self.classif_loss = config.classif_loss
        print("Config Learning Rate: ", config.learning_rate)
        self.start_lr = config.learning_rate
        self.decay_step = config.decay_step
        self.decay_rate = config.decay_rate
        self.max_grad_norm = None
        if config.max_grad_norm > 0:
            self.max_grad_norm = config.max_grad_norm
        self.optimizer = config.optimizer
        
        self._build_placeholders()
        self._build_model()
        self._build_steps()
        self._build_optim()
        
        show_all_variables()

    def _build_placeholders(self):
        self.cnn_input = tf.placeholder(tf.float32,
                                        [self.batch_size, self.num_node, self.feat_in, self.temporal_dim, self.period_dim],
                                        name="cnn_input")

        self.output_label = tf.placeholder(tf.float32,
                                         [self.batch_size, self.output_node, self.feat_out],
                                         name="final_output")
        self.ph_labels_weight = tf.placeholder(
            tf.float32, (self.batch_size, self.output_node), 'labels_weight')

        self.weight_mt = tf.placeholder(
            tf.float32, (self.batch_size, self.output_node), 'weight_mt')
        self.t_weight_mt = tf.placeholder(tf.float32, (self.batch_size, self.temporal_dim*self.output_node))
        self.p_weight_mt = tf.placeholder(tf.float32, (self.batch_size, self.period_dim*self.output_node))

        self.attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        self.bias_in = tf.placeholder(dtype=tf.float32, shape=(self.num_node, self.num_node))
        if  'xian'in self.server_name and self.scl_evl_T == 0 and self.scl_evl_E == 0:
            self.context_matrix_in = tf.placeholder(dtype=tf.float32, shape=(self.num_node, 20))
        if self.server_name == 'chengdu' and self.scl_evl_T == 0 and self.scl_evl_E == 0:
            self.context_matrix_in = tf.placeholder(dtype=tf.float32, shape=(self.num_node, 22))
        if self.scl_evl_T == 1 or self.scl_evl_E == 1:
            self.context_matrix_in = tf.placeholder(dtype=tf.float32, shape=(self.num_node, 5))
        if self.server_name == 'HK' and self.scl_evl_T == 0 and self.scl_evl_E == 0:
            self.context_matrix_in = tf.placeholder(dtype=tf.float32, shape=(self.num_node, 28))
        self.bias_in_temp = tf.placeholder(dtype=tf.float32, shape=(self.num_node*self.temporal_dim, self.num_node*self.temporal_dim))
        self.bias_in_prd = tf.placeholder(dtype=tf.float32, shape=(self.num_node*self.period_dim, self.num_node*self.period_dim))

        self.ids_neighbor = tf.placeholder(tf.int32, (self.num_node, None))
        self.weight_mt_ids = tf.placeholder(tf.float32, (self.batch_size, self.num_node, None))
        self.bias_in_ids = tf.placeholder(dtype=tf.float32, shape=(self.num_node, None))


        # Place holder for embedding layer if any
        if self.E is not None:
            self.ph_embeds = []
            for i, E_i in enumerate(self.E):
                self.ph_embeds.append(tf.placeholder(
                    tf.int32, (self.batch_size, 1), name='embed_{}'.format(i)))
        else:
            self.ph_embeds = None

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.model_step = tf.Variable(0, name='model_step', trainable=False, dtype=tf.int32)
        print("Starting lr: ", self.start_lr)
        self.learning_rate = tf.train.exponential_decay(
            self.start_lr, self.model_step, self.decay_step,
            self.decay_rate, staircase=False)
        # self.learning_rate = self.start_lr

    def _build_model(self, reuse=None, bn=False):
        sparse_loss = 0
        with tf.variable_scope("gconv_model", reuse=reuse) as sc:
            with tf.name_scope('feature_context'):
                context_f_ac_2 = self.context_matrix_in

            if self.hypergraph_flag == 1:
                with tf.variable_scope('hypergraph_conv'):
                    print('start hypergraph_conv')
                    x_logits = tf.transpose(self.cnn_input, perm=(0, 1, 3, 4, 2))
                    N, M, T, P, B = x_logits.get_shape()
                    N, M, T, P, B = int(N), int(M), int(T), int(P), int(B)
                    x_logits = tf.reshape(x_logits, [N, M, T * P, B])

                    if self.hyper_w_transform_flag ==1:
                        x_logits1 = tf.reshape(x_logits, [N,M,-1])
                        x_logits2 = tf.layers.conv1d(x_logits1, T*P*B, 1, use_bias=False)
                        x = tf.reshape(x_logits2, [N, M, T*P, B])
                    else:
                        x = x_logits

                    n_edge = self.H.shape[1]
                    # the weight of the hyperedge
                    W = np.ones(n_edge)
                    # the degree of the node
                    DV = np.sum(self.H * W, axis=1)

                    # only remain nodes whose DV >= 1:
                    DV_indexs = np.where(DV>=1)[0]
                    DV_larger_1 = DV[DV_indexs]

                    H = self.H[DV_indexs,:]

                    # the degree of the hyperedge
                    DE = np.sum(H, axis=0)

                    invDE = np.mat(np.diag(np.power(DE, -1)))
                    DV2 = np.mat(np.diag(np.power(DV_larger_1, -0.5)))
                    W = np.mat(np.diag(W))
                    H = np.mat(H)
                    HT = H.T

                    G = DV2 * H * W * invDE * HT * DV2

                    x = tf.stack([x[:,int(i),:,:] for i in DV_indexs], axis=1)

                    # transfer to x in R^{nxC1}
                    N, M, K_s, TPB = x.get_shape()
                    N, M, K_s, TPB = int(N), int(M), int(K_s), int(TPB)

                    x = tf.reshape(x, [N, M, -1])

                    in_d = int(x.get_shape()[-1])
                    x = tf.nn.relu(self.HGNN_conv(in_d, self.n_hid, x, G, self.hyper_weight_stddev,self.hyper_bias_stddev))
                    x = tf.layers.dropout(x, rate=self.dropout_hyper, training=self.is_training)
                    x_logits_DV_larger_1 = self.HGNN_conv(self.n_hid, in_d, x, G, self.hyper_weight_stddev,self.hyper_bias_stddev)
                    x_logits_DV_larger_1 = tf.reshape(x_logits_DV_larger_1, [N, M, K_s, TPB])

                    if self.hgnn_residual:
                        x_logits = tf.stack([x_logits[:, int(i), :, :] if i not in DV_indexs
                                             else x_logits_DV_larger_1[:,np.where(DV_indexs == i)[0][0],:, :] + x_logits[:, int(i), :,:]
                                             for i in range(x_logits.get_shape()[1])], axis=1)
                    else:
                        x_logits = tf.stack([x_logits[:, int(i), :, :] if i not in DV_indexs
                                             else x_logits_DV_larger_1[:,np.where(DV_indexs == i)[0][0],:, :]
                                             for i in range(x_logits.get_shape()[1])], axis=1)
            # graph neighbor node conv
            with tf.variable_scope('spatial_attention_conv'):
                print('start spatial attn')
                N, M, TP, B = x_logits.get_shape()
                N, M, TP, B = int(N), int(M), int(TP), int(B)
                seq = tf.reshape(x_logits, [N, M, TP * B])

                if self.congc == 1:
                    x_logits = attn.inference(seq, self.attn_drop, self.ffd_drop,weight_mt=self.weight_mt,
                                     bias_mat=self.bias_in,context_f_ac_2=context_f_ac_2,
                                     hid_units=hid_units, n_heads=self.n_heads_spt,
                                     residual=self.residual, activation=nonlinearity)
                if self.congc == 0:
                    x_logits = attn.inference_gc(seq, self.attn_drop, self.ffd_drop, weight_mt=self.weight_mt,
                                             bias_mat=self.bias_in, context_f_ac_2=context_f_ac_2,
                                             hid_units=hid_units, n_heads=self.n_heads_spt,
                                             residual=self.residual, activation=nonlinearity)

            if self.is_pooling == 1:
                with tf.name_scope('pooling_1'):
                    if int(int(x_logits.get_shape()[2])/2) > 1:
                        x_logits = self.mpool1_new(x_logits, int(int(x_logits.get_shape()[2])/2))
                    elif int(int(x_logits.get_shape()[2]) / 2) == 1:
                        x_logits = self.mpool1_new(x_logits, 2)

            with tf.variable_scope('temporal_attention_conv'):
                print('start temporal attn')
                N, M, K_s, TPB = x_logits.get_shape()
                N, M, K_s, TPB = int(N), int(M), int(K_s), int(TPB)
                x_temp_input = tf.reshape(x_logits, [N, M, K_s, T, P, -1])
                x_temp_input = tf.transpose(x_temp_input, perm=(0, 3, 1, 2, 4, 5))
                x_temp_input = tf.reshape(x_temp_input, [N, T * M, -1])

                x_temp_output = attn.inference_no_scl(x_temp_input, self.attn_drop, self.ffd_drop,weight_mt=self.t_weight_mt,
                                              bias_mat=self.bias_in_temp,
                                              hid_units=hid_units, n_heads=self.n_heads_tmp,
                                              residual=self.residual, activation=nonlinearity)

                if self.is_pooling == 1:
                    with tf.name_scope('pooling_2'):
                        if int(int(x_temp_output.get_shape()[2])/2) > 1:
                            x_temp_output = self.mpool1_new(x_temp_output, int(int(x_temp_output.get_shape()[2])/2))
                        elif int(int(x_temp_output.get_shape()[2])/2) == 1:
                            x_temp_output = self.mpool1_new(x_temp_output, 2)

            with tf.variable_scope('periodic_attention_conv'):
                # first layer period attn
                print('start periodic attn')
                N, TM, K_t_prime, K_sPB = x_temp_output.get_shape()
                N, TM, K_t_prime, K_sPB = int(N), int(TM), int(K_t_prime), int(K_sPB)
                x_periodic_intput = tf.reshape(x_temp_output, [N, T, M,K_t_prime, K_s, P, -1])
                x_periodic_intput = tf.transpose(x_periodic_intput, perm=(0, 5,2,1, 3, 4, 6))
                x_periodic_intput = tf.reshape(x_periodic_intput, [N, P * M, -1])

                x = attn.inference_no_scl(x_periodic_intput, self.attn_drop, self.ffd_drop,weight_mt=self.p_weight_mt,
                                  bias_mat=self.bias_in_prd,
                                  hid_units=hid_units, n_heads=self.n_heads_prd,
                                  residual=self.residual, activation=nonlinearity)

                if self.is_pooling == 1:
                    with tf.name_scope('pooling_1tmp_attn'):
                        if int(int(x.get_shape()[2])/2) > 1:
                            x = self.mpool1_new(x, int(int(x.get_shape()[2])/2))
                        elif int(int(x.get_shape()[2]) / 2) == 1:
                            x = self.mpool1_new(x, 2)

                x = tf.layers.dropout(
                    x, rate=self.dropout, training=self.is_training)
                N, PM, K_p_1, TK_t_primeK_sB = x.get_shape()
                N, PM, K_p_1, TK_t_primeK_sB = int(N), int(P), int(K_p_1), int(TK_t_primeK_sB)

                x_output = tf.reshape(x, [N, P, M,K_p_1, T, K_t_prime, K_s, -1])
                x_output = tf.transpose(x_output, perm=(0, 2, 1,  3, 4, 5,6, 7))
                x_output = tf.reshape(x_output,[N,M,-1,B])


                with tf.name_scope('bias_relu_2tmp_attn'):
                    x_relu = self.b1relu_new(x_output)

                if self.is_pooling == 1:
                    with tf.name_scope('pooling_2tmp_attn'):
                        x_p = self.mpool1(x_relu, self.p[0])
                    x = tf.layers.dropout(
                        x_p, rate=self.dropout, training=self.is_training)

                _, M_1, T_1, B_1 = x.get_shape()
                M_1, T_1, B_1 = int(M_1), int(T_1), int(B_1)
                x = tf.reshape(x, [N, M_1, -1, B])

            N, M, F, B = x.get_shape()
            print("The number of output node is ", M)

            if self.vae_flag == 1:
                x_new = x
                if self.onlyB == 1:
                    mu_w = tf.get_variable("mu_w", [B, B], tf.float32,
                                           tf.random_normal_initializer(stddev=self.stddev1))
                    mu_b = tf.get_variable("mu_b", [B], tf.float32,
                                           initializer=tf.constant_initializer(0.0))
                    sigma_w = tf.get_variable("sigma_w", [B, B], tf.float32,
                                              tf.random_normal_initializer(stddev=self.stddev1))
                    sigma_b = tf.get_variable("sigma_b", [B], tf.float32,
                                              initializer=tf.constant_initializer(0.0))

                    mu = tf.matmul(x_new, mu_w) + mu_b
                    log_sigma_sq = tf.matmul(x_new, sigma_w) + sigma_b
                    eps = tf.random_normal(shape=tf.shape(log_sigma_sq), mean=0, stddev=self.stddev2,
                                           dtype=tf.float32)

                    x_new = tf.cond(tf.equal(self.is_training, tf.constant(False)), lambda: mu, lambda: mu + tf.sqrt(tf.exp(log_sigma_sq)) * eps)

                    if self.vae_residual:
                        x = x_new + x
                    else:
                        x = x_new

                    if self.loss_vae == 0:
                        loss_tmp = -0.5 * tf.reduce_sum(
                            tf.reduce_sum(1 + log_sigma_sq - tf.square(mu) - tf.exp(log_sigma_sq), -1), -1)
                        self.vae_loss = tf.reduce_mean(loss_tmp)
                    if self.loss_vae == 1:
                        loss_tmp = -0.5 * tf.reduce_sum(
                            1 + log_sigma_sq - tf.square(mu) - tf.exp(log_sigma_sq), -1)
                        self.vae_loss = tf.reduce_mean(loss_tmp)
                    if self.loss_vae == 2:
                        self.vae_loss = -0.5 * tf.reduce_sum(
                            1 + log_sigma_sq - tf.square(mu) - tf.exp(log_sigma_sq))

            list_tensor = []
            for k in range(B):
                hist_i = x[:, :, :, k]
                hist_i = tf.reshape(hist_i, (int(N), int(M * F)))  # N x M
                # concatenate with embedding layer
                embedding_x = []
                embedding_x.append(hist_i)
                hist_i = tf.concat(embedding_x, 1)
                hist_i = tf.expand_dims(hist_i, -1)
                list_tensor.append(hist_i)
            x = tf.concat(list_tensor, axis=-1)

            # fullly connected layer with normalization afterwards.
            with tf.variable_scope('logits'):
                x = self.fc(x, self.output_node, relu=False)
                if self.model_type == 'hist':
                    self.predictions = tf.nn.softmax(x, dim=-1)
                else:
                    self.predictions = tf.nn.sigmoid(x)

                print("Info prediction is ", self.predictions)

            self.model_vars = tf.contrib.framework.get_variables(
                sc, collection=tf.GraphKeys.TRAINABLE_VARIABLES)

            if self.is_sparse == 1:
                p_hat = tf.reduce_mean(x_relu, axis=1)
                p_o = self.p_o
                kl = kl_divergence(p_o, p_hat)
                sparse_loss = self.beta * tf.reduce_sum(kl)

        self._build_loss(sparse_loss)

    def _build_loss(self,sparse_loss):
        if self.classif_loss == "kl":
            loss_batchmean = self.weighted_kl_tf(
                self.output_label, self.predictions, self.ph_labels_weight) + sparse_loss

        with tf.name_scope("losses"):
            if self.vae_flag == 1:
                self.kl_loss = loss_batchmean + self.vae_loss
            else:
                self.kl_loss = loss_batchmean

        if len(self.regularizers) > 0:
           with tf.name_scope('regularization'):
               regularization = self.regularization * tf.add_n(self.regularizers)
        else:
            regularization = 0

        self.loss = self.kl_loss + regularization

        self.model_summary = tf.summary.merge(
            [tf.summary.scalar("model_loss/loss",
                               self.kl_loss),
             tf.summary.scalar("model_loss/regularization",
                               regularization),
             tf.summary.scalar("model_loss/loss_reg",
                               self.loss)])
    def _build_steps(self):
        def run(sess, feed_dict, fetch,
                summary_op, summary_writer, output_op=None, output_img=None):
            if summary_writer is not None:
                fetch['summary'] = summary_op
            if output_op is not None:
                fetch['output'] = output_op
            # run_opts = tf.RunOptions(report_tensor_allocation_upon_oom = True)
            result = sess.run(fetch, feed_dict=feed_dict)
            if "summary" in result.keys() and "step" in result.keys():
                summary_writer.add_summary(result['summary'], result['step'])
                summary_writer.flush()
            return result
        
        def train(sess, feed_dict, summary_writer=None,
                  with_output=False):
            fetch = {'loss': self.kl_loss,
                     'optim': self.model_optim, #?
                     'step': self.model_step, #?
                     'lr': self.learning_rate
            }
            return run(sess, feed_dict, fetch,
                       self.model_summary, summary_writer,
                       output_op=self.output_label if with_output else None,)
        
        def test(sess, feed_dict, summary_writer=None,
                 with_output=False):
            fetch = {'loss': self.kl_loss,
                     'pred': self.predictions,
                    'step': self.model_step}
            return run(sess, feed_dict, fetch,
                       self.model_summary, summary_writer,
                       output_op=self.output_label if with_output else None,)
        self.train = train
        self.test = test
        
    def _build_optim(self):
        def minimize(loss, step, var_list, learning_rate, optimizer):
            if optimizer == "sgd":
                optim = tf.train.GradientDescentOptimizer(learning_rate)
            elif optimizer == "adam":
                optim = tf.train.AdamOptimizer(learning_rate)
            elif optimizer == "rmsprop":
                optim = tf.train.RMSPropOptimizer(learning_rate)
            else:
                raise Exception("[!] Unkown optimizer: {}".format(
                    optimizer))
            ## Gradient clipping ##    
            if self.max_grad_norm is not None:
                grads_and_vars = optim.compute_gradients(
                    loss, var_list=var_list)
                new_grads_and_vars = []
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None and var in var_list:
                        grad = tf.clip_by_norm(grad, self.max_grad_norm)
                        grad = tf.check_numerics(
                            grad, "Numerical error in gradient for {}".format(
                                var.name))
                        new_grads_and_vars.append((grad, var))
                return optim.apply_gradients(new_grads_and_vars, global_step=step)
            else:
                grads_and_vars = optim.compute_gradients(
                    loss, var_list=var_list)
                return optim.apply_gradients(grads_and_vars,
                                             global_step=step)
        
        # optim #
        self.model_optim = minimize(
            self.loss,
            self.model_step,
            self.model_vars,
            self.learning_rate,
            self.optimizer)

    def _weight_variable(self, shape, index=0, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable(
            'weights_{}'.format(index), shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)

        return var

    def _weight_variable_hyper(self, shape, index='0', regularization=True, hyper_weight_stddev=0.1):
        initial = tf.truncated_normal_initializer(0, hyper_weight_stddev)
        var = tf.get_variable(
            'weights_hypergraph_{}'.format(index), shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)

        return var

    def _bias_variable(self, shape, index=0, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias_{}'.format(
            index), shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)

        return var

    def _bias_variable_hyper(self, shape, index='0', regularization=True, hyper_bias_stddev=0.1):
        initial = tf.constant_initializer(hyper_bias_stddev)
        var = tf.get_variable('bias_hypergraph_{}'.format(
            index), shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)

        return var

    def _bias_variable_new(self, shape, index=0, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias_new_{}'.format(
            index), shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)

        return var

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min, B = x.get_shape()
        list_tensor = []
        for i in range(B):
            hist_i = x[:, :, i]

            W = self._weight_variable(
                [int(Min), Mout], index=i, regularization=True)
            b = self._bias_variable([Mout], index=i, regularization=True)
            hist_i = tf.matmul(hist_i, W) + b
            hist_i = tf.nn.relu(hist_i) if relu else hist_i
            hist_i = tf.expand_dims(hist_i, axis=-1)
            list_tensor.append(hist_i)
        fc_result = tf.concat(list_tensor, axis=-1)

        return fc_result

    def b1relu_new(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F, B = x.get_shape()
        N, M, F, B = int(N), int(M), int(F), int(B)
        b = self._bias_variable_new([1, 1, F, B], regularization=True)

        # return tf.nn.tanh(x + b)
        # return tf.nn.leaky_relu(x + b)
        return tf.nn.tanh(x + b)

    def mpool1_new(self, x, p):
        """
        Max pooling of size p.
        The size of the input x is [batch, len_feature, nb_kernels, nb_bins].

        x: [batch, height, width, channels]
        """

        if p > 1:
            x = tf.nn.max_pool(x, ksize=[1, 1, p, 1], strides=[
                1, 1, p, 1], padding='SAME')
            # tf.maximum
            return x  # N x M/p x F
        else:
            return x

    def mpool1(self, x, p):
        """
        Max pooling of size p.
        The size of the input x is [batch, len_feature, nb_kernels, nb_bins].

        x: [batch, height, width, channels]
        """

        if p > 1:
            x = tf.nn.max_pool(x, ksize=[1, p, 1, 1], strides=[
                1, p, 1, 1], padding='SAME')
            # tf.maximum
            return x  # N x M/p x F
        else:
            return x

    def log10(self, x):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    def weighted_kl_tf(self, y_true, y_pred, weight, epsilon=EPSILON):

        N, M, B = y_pred.get_shape()
        N, M, B = int(N), int(M), int(B)
        w_N, w_M = weight.get_shape()
        w_N, w_M = int(w_N), int(w_M)
        assert w_N == N, w_M == M

        log_pred = self.log10(y_pred + epsilon)
        log_true = self.log10(y_true + epsilon)
        log_sub = tf.subtract(log_pred, log_true)
        mul_op = tf.multiply(y_pred, log_sub)
        sum_hist = tf.reduce_sum(mul_op, 2)
        if weight is not None:
            sum_hist = tf.multiply(weight, sum_hist)
            #        avg_kl_div = tf.reduce_mean(sum_hist)
        weight_avg_kl_div = tf.reduce_sum(sum_hist)
        avg_kl_div = weight_avg_kl_div / tf.reduce_sum(weight)

        return avg_kl_div

    def weighted_kl(self, y_true, y_pred, weight, epsilon=EPSILON):

        N, M, B = np.shape(y_pred)
        N, M, B = int(N), int(M), int(B)
        w_N, w_M = np.shape(weight)
        w_N, w_M = int(w_N), int(w_M)
        assert w_N == N, w_M == M

        log_pred = np.log10(y_pred + epsilon)
        log_true = np.log10(y_true + epsilon)
        log_sub = np.subtract(log_pred, log_true)
        mul_op = np.multiply(y_pred, log_sub)
        sum_hist = np.sum(mul_op, 2)
        if weight is not None:
            sum_hist = np.multiply(weight, sum_hist)
            #        avg_kl_div = tf.reduce_mean(sum_hist)
        weight_avg_kl_div = np.sum(sum_hist)
        # avg_kl_div = weight_avg_kl_div / np.sum(weight)

        # return avg_kl_div
        return weight_avg_kl_div

    def HGNN_conv(self, in_ch, n_hid, x, G, hyper_weight_stddev,hyper_bias_stddev):
        theta = self._weight_variable_hyper([in_ch, n_hid], index=str(in_ch)+str(n_hid), regularization=True, hyper_weight_stddev=hyper_weight_stddev)
        x = tf.matmul(x, theta)
        if self.bias_hyper:
            b = self._bias_variable_hyper([n_hid], index=str(in_ch)+str(n_hid), regularization=True,hyper_bias_stddev=hyper_bias_stddev)
            x = x + b
        x = tf.matmul(tf.cast(tf.convert_to_tensor(G), tf.float32), x)
        return x