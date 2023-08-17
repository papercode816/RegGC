import numpy as np
import tensorflow as tf
from tqdm import trange
from model_gcnn import Model
from utils import BatchLoader
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from itertools import zip_longest
import time
"""
Trainer: 

1. Initializes model
2. Train
3. Test
"""
EPSILON = 1e-6

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def adj_to_bias(adj, sizes, nhood=1):
    mt = np.eye(adj.shape[1])
    for _ in range(nhood):
        mt = np.matmul(mt, (adj + np.eye(adj.shape[1])))
    for i in range(sizes[0]):
        for j in range(sizes[0]):
            if mt[i][j] > 0.0:
                mt[i][j] = 1.0
    return -1e9 * (1.0 - mt)

def feature_engineering_context(context_data):
    context_data = np.asarray(context_data)
    print(context_data)
    gen_ohe = OneHotEncoder()
    cat_encoded = gen_ohe.fit_transform(context_data[:,1:4]).toarray()
    continus_d = np.stack([context_data[:,0],context_data[:,4]], axis=1).astype(np.float)
    est = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='uniform')
    est.fit(continus_d)
    cont_encoded = est.transform(continus_d)
    encoded = np.concatenate([cat_encoded,cont_encoded], axis=1)
    print('obtained context feature by one-hot and binning')
    return encoded

class Trainer(object):
    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.model_dir = config.model_dir
        self.gpu_memory_fraction = config.gpu_memory_fraction
        self.checkpoint_secs = config.checkpoint_secs
        self.log_step = config.log_step
        self.num_epoch = config.num_epochs
        self.stop_win_size = config.stop_win_size
        self.stop_early = config.stop_early

        ## import data Loader ##ir
        batch_size = config.batch_size
        server_name = config.server_name
        mode = config.mode
        target = config.target
        sample_rate = config.sample_rate
        win_size = config.win_size
        hist_range = config.hist_range
        s_month = config.s_month
        e_month = config.e_month
        e_date = config.e_date
        s_date = config.s_date
        data_rm = config.data_rm
        coarsening_level = config.coarsening_level
        cnn_mode = config.conv
        is_coarsen = config.is_coarsen
        least_threshold = config.least_threshold
        trip_table = config.trip_table
        temporal_dim = config.temporal_dim
        p_dim = config.period_dim
        n_heads_tmp = config.n_heads_tmp
        n_heads_prd = config.n_heads_prd

        self.log_loss = open("loss_"+str(self.config.learning_rate)+"_"+str(self.config.decay_step)+"_"+str(self.config.decay_rate)+ ".csv", "a")

        self.data_loader = BatchLoader(server_name, mode, target, sample_rate, win_size,
                                       hist_range, s_month, s_date, e_month, e_date,
                                       data_rm, least_threshold,trip_table,temporal_dim,p_dim, n_heads_tmp,n_heads_prd,config.ds_ind,config,batch_size, coarsening_level, cnn_mode,
                                       is_coarsen)

        actual_node = self.data_loader.adj.shape[0]


        tf.reset_default_graph()
        ## define model ##
        self.model = Model(config, actual_node)

        ## model saver / summary writer ##
        self.saver = tf.train.Saver()
        self.model_saver = tf.train.Saver(self.model.model_vars)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                 is_chief=True,
                                 saver=self.saver,
                                 summary_op=None,
                                 summary_writer=self.summary_writer,
                                 save_summaries_secs=300,
                                 save_model_secs=self.checkpoint_secs,
                                 global_step=self.model.model_step)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=self.gpu_memory_fraction,
            allow_growth=True)  # seems to be not working
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=True,
                                     gpu_options=gpu_options)
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        adj = self.data_loader.adj.todense()
        context_data = self.data_loader.context_data
        self.context_matrix = feature_engineering_context(context_data)
        # adj = self.data_loader.adj.todense()*self.data_loader.adj.todense()
        self.biases = adj_to_bias(np.asarray(adj), [np.shape(adj)[0]], nhood=config.hood)
        self.biases_temp = -1e9 * (1.0 - np.tile(np.eye(actual_node), (temporal_dim, temporal_dim)))
        self.biases_prd = -1e9 * (1.0 - np.tile(np.eye(actual_node), (p_dim, p_dim)))

        adj_neighb = [np.nonzero(adj_tmp)[1] for adj_tmp in adj ]
        self.ids_neighbor = np.array(list(zip_longest(*adj_neighb, fillvalue=0))).T

    def train(self, val_best_score=10, save=False, index=1, best_model=None):
        print("[*] Checking if previous run exists in {}"
              "".format(self.model_dir))
        latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)

        print("[*] No previous result")
        self.b_pretrain_loaded = False

        print("[*] Training starts...")
        self.model_summary_writer = None

        val_loss = 0
        lr = 0
        tmp_best_loss = float('+inf')
        validation_loss_window = np.zeros(self.stop_win_size)
        validation_loss_window[:] = float('+inf')
        ##Training
        train_running_time = 0.0
        c_train = 0
        ep_n = 0
        for n_epoch in trange(self.num_epoch, desc="Training[epoch]"):
            self.data_loader.reset_batch_pointer(0)
            loss_epoch = []
            for k in trange(self.data_loader.sizes[0], desc="[per_batch]"):
                c_train += 1
                # Fetch training data
                batch_x, batch_y, weight_y,\
                count_y, batch_vel_list = self.data_loader.next_batch(0)

                if self.model.n_heads_prd>0 and self.model.n_heads_tmp>0:
                    weight_mt = (np.sum(batch_x[:, :, :, 1, 1], axis=2) > 0) * 1
                    t_weight_mt = (np.sum(np.reshape(np.transpose(batch_x[:, :, :, :, 1], axes=[0,3,1,2]), [np.shape(batch_x)[0],-1,np.shape(batch_x)[2]]), axis=2) > 0 ) * 1
                    p_weight_mt = (np.sum(np.reshape(np.transpose(batch_x[:, :, :, 1, :], axes=[0,3,1,2]), [np.shape(batch_x)[0],-1,np.shape(batch_x)[2]]), axis=2) > 0 ) * 1
                if  self.model.n_heads_prd==0 and self.model.n_heads_tmp>0:
                    weight_mt = (np.sum(batch_x[:, :, :, 1], axis=2) > 0) * 1
                    t_weight_mt = (np.sum(np.reshape(np.transpose(batch_x[:, :, :, :], axes=[0, 3, 1, 2]),
                                                     [np.shape(batch_x)[0], -1, np.shape(batch_x)[2]]), axis=2) > 0) * 1
                if self.model.n_heads_prd>0 and self.model.n_heads_tmp==0:
                    weight_mt = (np.sum(batch_x[:, :, :, 1], axis=2) > 0) * 1
                    # t_weight_mt = (np.sum(np.reshape(np.transpose(batch_x[:, :, :, :, 1], axes=[0,3,1,2]), [np.shape(batch_x)[0],-1,np.shape(batch_x)[2]]), axis=2) > 0 ) * 1
                    p_weight_mt = (np.sum(np.reshape(np.transpose(batch_x[:, :, :, :], axes=[0,3,1,2]), [np.shape(batch_x)[0],-1,np.shape(batch_x)[2]]), axis=2) > 0 ) * 1

                weight_mt_tmp = np.tile(np.expand_dims(weight_mt, axis=1), [1, np.shape(batch_x)[1], 1])
                eye_m = [(np.eye(weight_mt_tmp.shape[1])).tolist() for i in range(weight_mt_tmp.shape[0])]
                tmp = np.clip((weight_mt_tmp + eye_m), 0, 1)
                weight_mt_ids = np.asarray([[tmp[i,j,self.ids_neighbor[j]] for j in range(tmp.shape[1])] for i in range(tmp.shape[0])])

                # bias_tmp = np.tile(self.biases, [np.shape(batch_x)[0], 1, 1])
                bias_in_ids = np.asarray([self.biases[i,self.ids_neighbor[i]] for i in range(self.biases.shape[0])])
                if self.model.n_heads_prd > 0 and self.model.n_heads_tmp > 0:
                    feed_dict = {
                        self.model.cnn_input: batch_x,
                        self.model.output_label: batch_y,
                        self.model.ph_labels_weight: weight_y,
                        self.model.is_training: True,
                        self.model.attn_drop: 0.0,
                        self.model.ffd_drop: 0.0,
                        self.model.bias_in: self.biases,
                        self.model.weight_mt: weight_mt,
                        self.model.context_matrix_in: self.context_matrix,
                        self.model.t_weight_mt: t_weight_mt,
                        self.model.p_weight_mt: p_weight_mt,
                        self.model.bias_in_temp:self.biases_temp,
                        self.model.bias_in_prd:self.biases_prd,
                        self.model.ids_neighbor: self.ids_neighbor,
                        self.model.weight_mt_ids: weight_mt_ids,
                        self.model.bias_in_ids: bias_in_ids
                    }
                t0 = time.time()
                res = self.model.train(self.sess, feed_dict, self.model_summary_writer,
                                       with_output=True)
                t1 = time.time()
                train_running_time += (t1 - t0)
                loss_epoch.append(res['loss'])
                lr = res['lr']
                print('lr:%f' %lr)
                print('global step:%d' %res['step'])
                print('loss:%f' %res['loss'])
                self.log_loss.write(str(res['loss']) + ",\n")
                self.log_loss.flush()
                self.model_summary_writer = self._get_summary_writer(res)
            val_loss,_ = self.validate()
            train_loss = np.mean(loss_epoch)

            validation_loss_window[n_epoch % self.stop_win_size] = val_loss

            if self.stop_early:
                # hxl changes
                if np.abs(validation_loss_window.mean() - val_loss) < 1e-4:
                # if np.abs(validation_loss_window.mean() - val_loss) < 1e-5:
                    print('Validation loss did not decrease. Stopping early.')
                    break

            if n_epoch % 10 == 0:
                if save:
                    self.saver.save(self.sess, self.model_dir)
                if val_loss < val_best_score:
                    val_best_score = val_loss
                    best_model = self.model_dir
                if val_loss < tmp_best_loss:
                    tmp_best_loss = val_loss
                print("Searching {}...".format(index))
                print("Epoch {}: ".format(n_epoch))
                print("LR: ", lr)
                print("  Train Loss: ", train_loss)
                print("  Validate Loss: ", val_loss)
                print("  Current Best Loss: ", val_best_score)
                print("  Current Model Dir: ", best_model)
        train_vel = np.asarray(self.data_loader.all_batches[0][4])
        HA = [np.hstack(np.concatenate(train_vel[:,:, i])) for i in range(np.shape(train_vel)[2])]
        ave_train_time = train_running_time / c_train / self.model.batch_size

        val_loss, mklr = self.validate()

        # # # only for finetune,should comment after finetune
        # self.sess.close()
        # tf.reset_default_graph()

        return tmp_best_loss,HA,val_loss,ave_train_time,mklr

    def validate(self):
        gt_y = []
        pred_y = []
        w_y = []
        counts_y = []
        vel_list_y = []
        loss = []
        for n_sample in trange(self.data_loader.sizes[1], desc="Validating"):
            batch_x, batch_y, weight_y, count_y, \
            vel_list = self.data_loader.next_batch(1)

            if self.model.n_heads_prd > 0 and self.model.n_heads_tmp > 0:
                weight_mt = (np.sum(batch_x[:, :, :, 1, 1], axis=2) > 0) * 1
                t_weight_mt = (np.sum(np.reshape(np.transpose(batch_x[:, :, :, :, 1], axes=[0, 3, 1, 2]),
                                                 [np.shape(batch_x)[0], -1, np.shape(batch_x)[2]]), axis=2) > 0) * 1
                p_weight_mt = (np.sum(np.reshape(np.transpose(batch_x[:, :, :, 1, :], axes=[0, 3, 1, 2]),
                                                 [np.shape(batch_x)[0], -1, np.shape(batch_x)[2]]), axis=2) > 0) * 1

            weight_mt_tmp = np.tile(np.expand_dims(weight_mt, axis=1), [1, np.shape(batch_x)[1], 1])
            eye_m = [(np.eye(weight_mt_tmp.shape[1])).tolist() for i in range(weight_mt_tmp.shape[0])]
            tmp = np.clip((weight_mt_tmp + eye_m), 0, 1)
            weight_mt_ids = np.asarray(
                [[tmp[i, j, self.ids_neighbor[j]] for j in range(tmp.shape[1])] for i in range(tmp.shape[0])])

            # bias_tmp = np.tile(self.biases, [np.shape(batch_x)[0], 1, 1])
            bias_in_ids = np.asarray([self.biases[i, self.ids_neighbor[i]] for i in range(self.biases.shape[0])])

            if self.model.n_heads_prd > 0 and self.model.n_heads_tmp > 0:
                feed_dict = {
                    self.model.cnn_input: batch_x,
                    self.model.output_label: batch_y,
                    self.model.ph_labels_weight: weight_y,
                    self.model.is_training: False,
                    self.model.attn_drop: 0.0,
                    self.model.ffd_drop: 0.0,
                    self.model.bias_in: self.biases,
                    self.model.weight_mt: weight_mt,
                    self.model.context_matrix_in: self.context_matrix,
                    self.model.t_weight_mt: t_weight_mt,
                    self.model.p_weight_mt: p_weight_mt,
                    self.model.bias_in_temp: self.biases_temp,
                    self.model.bias_in_prd: self.biases_prd,
                    self.model.ids_neighbor: self.ids_neighbor,
                    self.model.weight_mt_ids: weight_mt_ids,
                    self.model.bias_in_ids: bias_in_ids
                }
            res = self.model.test(self.sess, feed_dict, self.summary_writer,
                                  with_output=True)
            loss.append(res['loss'])
            gt_y.append(batch_y)
            w_y.append(weight_y)
            counts_y.append(count_y)
            vel_list_y.append(vel_list)
            pred_y.append(res['pred'])

        final_gt = np.concatenate(gt_y, axis=0)
        final_pred = np.concatenate(pred_y, axis=0)
        final_weight = np.concatenate(w_y, axis=0)
        final_count = np.concatenate(counts_y, axis=0)
        final_vel_list = np.concatenate(vel_list_y, axis=0)

        result_dict = {'ground_truth': final_gt,
                       'prediction': final_pred,
                       'weight': final_weight,
                       'count': final_count,
                       'vel_list': final_vel_list}
        train_vel = np.asarray(self.data_loader.all_batches[0][4])
        HA = [np.hstack(np.concatenate(train_vel[:, :, i])) for i in range(np.shape(train_vel)[2])]
        if self.config.model_type == 'hist':
            mklr = self.cal_mklr(result_dict, HA)
            print('mklr of validation dataset: %f' % mklr)

        return np.nanmean(loss),mklr

    def test(self, HA):
        loss = []
        gt_y = []
        pred_y = []
        w_y = []
        counts_y = []
        test_running_time = 0.0
        c_test = 0
        ep_n = 0
        vel_list_y = []
        for n_sample in trange(self.data_loader.sizes[2], desc="Testing"):
            c_test += 1
            batch_x, batch_y, weight_y, \
            count_y, vel_list = self.data_loader.next_batch(2)

            if self.model.n_heads_prd > 0 and self.model.n_heads_tmp > 0:
                weight_mt = (np.sum(batch_x[:, :, :, 1, 1], axis=2) > 0) * 1
                t_weight_mt = (np.sum(np.reshape(np.transpose(batch_x[:, :, :, :, 1], axes=[0, 3, 1, 2]),
                                                 [np.shape(batch_x)[0], -1, np.shape(batch_x)[2]]), axis=2) > 0) * 1
                p_weight_mt = (np.sum(np.reshape(np.transpose(batch_x[:, :, :, 1, :], axes=[0, 3, 1, 2]),
                                                 [np.shape(batch_x)[0], -1, np.shape(batch_x)[2]]), axis=2) > 0) * 1

            weight_mt_tmp = np.tile(np.expand_dims(weight_mt, axis=1), [1, np.shape(batch_x)[1], 1])
            eye_m = [(np.eye(weight_mt_tmp.shape[1])).tolist() for i in range(weight_mt_tmp.shape[0])]
            tmp = np.clip((weight_mt_tmp + eye_m), 0, 1)
            weight_mt_ids = np.asarray(
                [[tmp[i, j, self.ids_neighbor[j]] for j in range(tmp.shape[1])] for i in range(tmp.shape[0])])

            # bias_tmp = np.tile(self.biases, [np.shape(batch_x)[0], 1, 1])
            bias_in_ids = np.asarray([self.biases[i, self.ids_neighbor[i]] for i in range(self.biases.shape[0])])

            if self.model.n_heads_prd > 0 and self.model.n_heads_tmp > 0:
                feed_dict = {
                    self.model.cnn_input: batch_x,
                    self.model.output_label: batch_y,
                    self.model.ph_labels_weight: weight_y,
                    self.model.is_training: False,
                    self.model.attn_drop: 0.0,
                    self.model.ffd_drop: 0.0,
                    self.model.bias_in: self.biases,
                    self.model.weight_mt: weight_mt,
                    self.model.context_matrix_in: self.context_matrix,
                    self.model.t_weight_mt: t_weight_mt,
                    self.model.p_weight_mt: p_weight_mt,
                    self.model.bias_in_temp: self.biases_temp,
                    self.model.bias_in_prd: self.biases_prd,
                    self.model.ids_neighbor: self.ids_neighbor,
                    self.model.weight_mt_ids: weight_mt_ids,
                    self.model.bias_in_ids: bias_in_ids
                }
            t0 = time.time()
            res = self.model.test(self.sess, feed_dict, self.summary_writer,
                                  with_output=True)
            t1 = time.time()
            test_running_time += (t1 - t0)
            loss.append(res['loss'])
            gt_y.append(batch_y)
            w_y.append(weight_y)
            counts_y.append(count_y)
            vel_list_y.append(vel_list)
            pred_y.append(res['pred'])

        final_gt = np.concatenate(gt_y, axis=0)
        final_pred = np.concatenate(pred_y, axis=0)
        final_weight = np.concatenate(w_y, axis=0)
        final_count = np.concatenate(counts_y, axis=0)
        final_vel_list = np.concatenate(vel_list_y, axis=0)

        result_dict = {'ground_truth': final_gt,
                       'prediction': final_pred,
                       'weight': final_weight,
                       'count': final_count,
                       'vel_list': final_vel_list}

        test_loss = np.mean(loss)
        print("Test Loss: ", test_loss)
        ave_test_time = test_running_time / c_test / self.model.batch_size

        if self.config.model_type == 'hist':
            mklr = self.cal_mklr(result_dict, HA)
            print('mklr of test dataset: %f' % mklr)

            flr = self.cal_flr(result_dict, HA)
            print('flr of test dataset: %f' % flr)

            return result_dict,mklr,flr,ave_test_time
        else:
            mape = self.cal_mape(result_dict)
            print('mape of test dataset: %f' % mape)
            return result_dict,mape
            # self.model_summary_writer = self._get_summary_writer(res)

    def _get_summary_writer(self, result):
        if result['step'] % self.log_step == 0:
            return self.summary_writer
        else:
            return None


    def cal_mklr(self, result_dict, HA):
        print('calculating mklr for test dataset')
        final_gt,final_pred,final_weight,final_count,final_vel_list = result_dict['ground_truth'],result_dict['prediction']\
            ,result_dict['weight'],result_dict['count'],result_dict['vel_list']
        HA_temp = np.asarray([np.histogram(HA[i], self.config.hist_range, density=True)[0]* (self.config.hist_range[1] - self.config.hist_range[0]) for i in range(len(HA))])
        nan_index = np.argwhere(np.isnan(HA_temp))
        if len(nan_index) > 0:
            HA_temp[nan_index[0][0],:] = 0
        HA_hist = np.asarray([HA_temp for j in range(np.shape(final_pred)[0])])

        pred_mklr = self.model.weighted_kl(final_gt, final_pred, final_weight)
        ha_mklr = self.model.weighted_kl(final_gt, HA_hist, final_weight)
        mklr = pred_mklr / ha_mklr

        return mklr

    def cal_flr(self, result_dict, HA):
        print('calculating flr for test dataset')
        final_gt,final_pred,final_weight,final_count,final_vel_list = result_dict['ground_truth'],result_dict['prediction']\
            ,result_dict['weight'],result_dict['count'],result_dict['vel_list']
        HA_temp = np.asarray([np.histogram(HA[i], self.config.hist_range, density=True)[0]* (self.config.hist_range[1] - self.config.hist_range[0]) for i in range(len(HA))])
        nan_index = np.argwhere(np.isnan(HA_temp))
        if len(nan_index) > 0:
            HA_temp[nan_index[0][0],:] = 0
        HA_hist = np.asarray([HA_temp for j in range(np.shape(final_pred)[0])])

        # likelihoods of observing gt vel list from HA distribution
        lr_HA = self.lr_Ok_distribution(HA_hist, final_vel_list)
        # likelihoods of observing gt vel list from prediction distribution
        lr_pred = self.lr_Ok_distribution(final_pred, final_vel_list)

        lr_ij_mat = lr_pred / lr_HA > 1

        sum_hist = np.multiply(final_weight, lr_ij_mat)
        weight_sum = np.sum(sum_hist)
        flr = weight_sum / np.sum(final_weight)


        return flr

    def lr_Ok_distribution(self, hist, final_vel_list):

        # ok_mat = [[np.asarray(final_vel_list[i][j])//5 for j in range(len(final_vel_list[0]))] for i in range(len(final_vel_list))]
        ok_mat = [[(np.asarray(final_vel_list[i][j]) - self.config.hist_range[0]) // (self.config.hist_range[1]-self.config.hist_range[0]) for j in range(len(final_vel_list[0]))] for i in range(len(final_vel_list))]
        # log version
        # lr = np.asarray([[np.sum([np.log((hist[i][j][int(ok_mat[i][j][z])] + EPSILON)) for z in range(len(ok_mat[i][j]))]) for j in range(len(ok_mat[0]))] for i in range(len(ok_mat))])
        # citation [29] version
        lr = np.asarray([[np.prod([(hist[i][j][int(ok_mat[i][j][z])] + EPSILON) for z in range(len(ok_mat[i][j]))]) for j in range(len(ok_mat[0]))] for i in range(len(ok_mat))])
        return lr

    def cal_mape(self, result_dict):
        print('calculating mape for test dataset')
        scaler = MinMaxScaler().fit([[self.config.hist_range[0]],
                                          [self.config.hist_range[1]]])

        final_gt,final_pred,final_weight,final_count,final_vel_list = result_dict['ground_truth'],result_dict['prediction']\
            ,result_dict['weight'],result_dict['count'],result_dict['vel_list']
        # real = np.asarray([[np.average(final_vel_list[i][j]) for j in range(len(final_vel_list[0]))] for i in range(len(final_vel_list))])
        real = np.asarray([[1e10 if math.isnan(np.average(final_vel_list[i][j])) else np.average(final_vel_list[i][j]) for j in range(len(final_vel_list[0]))] for i in range(len(final_vel_list))])


        final_pred = final_pred.reshape(final_pred.shape[0],final_pred.shape[1])
        final_pred_scale = scaler.inverse_transform(final_pred)
        avg_mape = np.sum(np.abs((real - final_pred_scale) / real) * final_weight) / np.sum(final_weight) * 100
        return avg_mape