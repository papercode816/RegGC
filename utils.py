import tensorflow as tf
import numpy as np
import math
import pickle
import os
import sys
import json
from datetime import datetime
import tensorflow.contrib.slim as slim
from sklearn.model_selection import train_test_split

from pyemd import emd
import geopy.distance

CONF_DIR = os.path.join('.', 'conf')
EPSILON = 1e-6
topk = 50000
min_nb = 1

def save_config(model_dir, config):
    '''
    save config params in a form of param.json in model directory
    '''

    param_path = os.path.join(model_dir, "params.json")

    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def prepare_dirs(config, test=False):
    config.model_name = "{}_{}_{}_{}_{}_{}_rm{}".format(
        config.mode, config.server_name,
        config.target, config.conv, config.filter,
        config.ds_ind, config.data_rm)

    sub_folder = 'lr_{0:.0e}_reg_{1:.0e}_dp_{2:.0e}_decay_{3:.0e}' \
                 '_{4}_{5}_{6}_{7}_{8}_{9}/'.format(
        config.learning_rate, config.regularization,
        config.dropout, config.decay_rate,
        config.num_kernels[0], config.num_kernels[1],
        config.conv_size[0], config.conv_size[1],
        config.pool_size[0], config.pool_size[1])

    config.model_dir = os.path.join(
        config.log_dir, config.mode,
        config.server_name, config.target,
        config.conv, config.filter,
        '{}_rm{}'.format(config.ds_ind, config.data_rm))

    if config.sub_folder:
        config.model_dir = os.path.join(
            config.model_dir, sub_folder)

    for path in [config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
            print("Model Directory '%s' created" % path)

    if test:
        config.result_dir = os.path.join(
            config.output_dir,
            'test',
            # config.classif_loss,
            config.model_name)
    else:
        config.result_dir = os.path.join(
            config.output_dir,
            config.model_name)

    for path in [config.result_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
            print("Output Directory '%s' created" % path)


def prepare_config_date(config, ind):
    """
    prepare the config date

    :param config:
    :param s_month:
    :param s_date:
    :param e_month:
    :param e_date:
    :return:
    """

    if config.server_name == 'HK':
        start_months = [1, 2, 4, 6, 8]
        start_dates = [1, 28, 30, 30, 31]
        end_months = [2, 4, 6, 8, 10]
        end_dates = [28, 30, 30, 31, 31]
    if config.server_name == 'chengdu':
        start_months = [10, 10, 10, 11, 11]
        start_dates = [1, 13, 25, 6, 18]
        end_months = [10, 10, 11, 11, 11]
        end_dates = [13, 25, 6, 18, 30]
    if 'xian' in config.server_name:
        start_months = [10, 10, 10, 11, 11]
        start_dates = [1, 13, 25, 6, 18]
        end_months = [10, 10, 11, 11, 11]
        end_dates = [13, 25, 6, 18, 30]

    config.s_month = start_months[ind]
    config.s_date = start_dates[ind]
    config.e_month = end_months[ind]
    config.e_date = end_dates[ind]

def pklLoad(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def pklSave(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def save_results(result_dir, obj):
    fname = os.path.join(result_dir, 'result_dict.pickle')
    pklSave(fname, obj)

def cal_kl_div(y_pred, y_true, base=2, eps=1e-6):
    log_op = np.log2(y_pred+eps) - np.log2(y_true+eps)
    mul_op = np.multiply(y_pred, log_op)
    sum_hist = np.sum(mul_op, axis=1)
    multi_factor = np.log2(base)
    sum_hist = sum_hist/multi_factor

    return sum_hist

def weighted_kl_div(y_true, y_pred,
                    epsilon=EPSILON, metric='kl', base=2):
    if metric == 'kl':
        y_true = y_true
        y_pred = y_pred
        sum_hist = cal_kl_div(y_pred, y_true, base, eps=epsilon)

    sum_hist = sum_hist[np.isnan(sum_hist) == False]
    print("The total test number is ", sum_hist.shape[0])
    avg_kl_div = np.nanmean(sum_hist)
    std_kl_div = np.std(sum_hist)

    return avg_kl_div, std_kl_div

def euclidean_dist(y_true, y_pred, p=2):
    abs_minus = np.abs(y_true - y_pred)
    abs_minus_pow = abs_minus ** p
    sum_abs_minus_pow = np.sum(abs_minus_pow, axis=1)
    root_sum_abs_minus_pow = sum_abs_minus_pow ** 1 / p
    p_wasser_dist = np.nanmean(root_sum_abs_minus_pow)

    return p_wasser_dist

def weighted_emd(y_true, y_pred):
    n_bucket = y_true.shape[1]
    nb_result = y_true.shape[0]
    d_matrix = np.zeros((n_bucket, n_bucket))
    for i in range(n_bucket):
        d_matrix[i, i:n_bucket] = np.arange(1, n_bucket+1)[:n_bucket - i]
    d_matrix = np.maximum(d_matrix, d_matrix.T)
    emds = []
    for j in range(nb_result):
        hist_true = y_true[j, :].astype(np.float64)
        hist_pred = y_pred[j, :].astype(np.float64)
        emd_j = emd(hist_pred, hist_true, d_matrix)
        emds.append(emd_j)
    emds = np.array(emds)

    return emds

def sig_test_zvalue(vel_list, pmf_array):
    num_bins = len(pmf_array)
    bucket_size = 40 / num_bins
    mean_bucket = np.arange(0, 40, bucket_size)
    mean_bucket = mean_bucket + bucket_size / 2
    pred_mean = np.sum(np.array(pmf_array) * mean_bucket)
    true_mean = np.mean(vel_list)
    true_std = np.std(vel_list)
    true_num = len(vel_list)
    z_value = np.abs(pred_mean - true_mean) / (true_std / np.sqrt(true_num))
    # print("True Mean: ", true_mean)
    # print("Pred Mean: ", pred_mean)
    return z_value

def eval_sig_test_vel_list(vel_list, y_pred, alpha=0.05):
    accept_num = 0
    for i in range(y_pred.shape[0]):
        pred_pmf = y_pred[i]
        vel_list_i = vel_list[i]
        max_dis = sig_test_zvalue(vel_list_i, pred_pmf)
        # scale = np.sqrt((len(vel_list_i) + 40) / (len(vel_list_i) * 40)) * np.sqrt(-0.5 * np.log(alpha / 2))
        scale = 1.282
        if max_dis < scale:
            accept_num += 1

    accept_rate = accept_num / y_pred.shape[0]

    return accept_rate


def cal_max_wasser_dis(pmf_array, vel_list):
    round_vel = np.round(np.array(vel_list), decimals=0).astype(int)
    round_vel = round_vel[round_vel <= 40]
    sorted_array = np.sort(round_vel)
    true_cdf = np.zeros(41)
    sum_i = 0
    for i in range(41):
        if i in round_vel:
            record_i = np.sum(round_vel == i)
            sum_i += record_i / len(sorted_array)
        else:
            sum_i += 0
        true_cdf[i] = sum_i

    pred_cdf = np.zeros(41)
    unit = 40 / len(pmf_array)
    sum_j = 0
    for i in range(40):
        bucket_i = pmf_array[int(i / unit)]
        sum_j += bucket_i * 1 / unit
        pred_cdf[i] = sum_j
    pred_cdf[-1] = 1.0

    # print("True cdf: ", true_cdf)
    # print("Pred cdf: ", pred_cdf)
    wasser = np.max(np.abs(true_cdf - pred_cdf))

    return wasser

def eval_ks_test_vel_list(vel_list, y_pred, alpha=0.05):
    accept_num = 0
    for i in range(y_pred.shape[0]):
        pred_pmf = y_pred[i]
        vel_list_i = vel_list[i]
        max_dis = cal_max_wasser_dis(pred_pmf, vel_list_i)
        scale = np.sqrt((len(vel_list_i) + 40) / (len(vel_list_i) * 40)) * np.sqrt(-0.5 * np.log(alpha / 2))
        # scale = np.sqrt(2 / 40) * np.sqrt(-0.5 * np.log(alpha / 2))
        # print("max_dis  |   scale")
        # print("{}   |   {}".format(max_dis, scale))
        if max_dis < scale:
            accept_num += 1

    accept_rate = accept_num / y_pred.shape[0]

    return accept_rate


def eval_wasser_vel_list(vel_list, y_pred):
    wasser_dist_sum = 0
    for i in range(y_pred.shape[0]):
        pred_pmf = y_pred[i]
        vel_list_i = vel_list[i]
        max_dis = cal_max_wasser_dis(pred_pmf, vel_list_i)
        wasser_dist_sum += max_dis

    avg_wasser_dist = wasser_dist_sum / y_pred.shape[0]

    return avg_wasser_dist


def real_output(result_dict):
    final_gt = result_dict['ground_truth']
    final_pred = result_dict['prediction']
    final_weight = result_dict['weight']
    final_count = result_dict['count']
    final_vel_list = result_dict['vel_list']

    print("final weight shape", final_weight.shape)
    print("Final vel list shape", final_vel_list.shape)
    selected_pos = final_weight == 1
    y_true = final_gt[selected_pos]
    y_pred = final_pred[selected_pos]
    count_y = final_count[selected_pos]
    vel_list = final_vel_list[selected_pos]

    return y_true, y_pred, count_y, vel_list

def evaluate_array(y_true, y_pred, count_y,
                   vel_list=None, method='KL',
                   alpha=0.05, count_thres=0):
    select_pos = count_y > count_thres
    print("Total number in evaluating is ", np.sum(select_pos))
    y_true = y_true[select_pos]
    y_pred = y_pred[select_pos]
    if vel_list is not None:
        vel_list = vel_list[select_pos]

    if method == 'KL':
        loss_value, loss_std = weighted_kl_div(y_true, y_pred)
    elif method == 'KS-test':
        loss_value = eval_ks_test_vel_list(vel_list, y_pred, alpha)
    elif method == 'emd':
        loss_value = weighted_emd(y_true, y_pred)
    elif method == 'wasser':
        loss_value = eval_wasser_vel_list(vel_list, y_pred)
    elif method == 'euclidean':
        loss_value = euclidean_dist(y_true, y_pred, p=2)
    elif method == 'sig_test':
        loss_value = eval_sig_test_vel_list(vel_list, y_pred)
        # loss_value = eval_sig_test(y_true, y_pred, vel_list)
    else:
        loss_value = None
        print("Please specify a valid metric...")

    return loss_value


def evaluate_result(result_dict, method='KL', alpha=0.05):
    y_true, y_pred, count_y, vel_list = real_output(result_dict)

    return evaluate_array(y_true, y_pred, count_y, vel_list, method, alpha)

def softmax(x, n_axis=-1, exp=True):
    if exp:
        # take the sum along the specified axis
        x = np.exp(x)
    else:
        # in case there's negative value in the output
        # x_min = np.expand_dims(np.min(x, axis=n_axis), n_axis)
        # x = (x - x_min)
        x[x < 0] = 0.

    ax_sum = np.expand_dims(np.sum(x, axis=n_axis), n_axis)

    return x / ax_sum


class BatchLoader(object):
    def __init__(self, sname, mode, target, sample_rate, win_size,
                 hist_range, s_month, s_date, e_month, e_date,
                 data_rm, least_threshold,trip_table,temporal_dim,p_dim,n_heads_tmp,n_heads_prd,ds_ind,config,batch_size=-1, coarsening_level=4,
                 conv_mode='gcnn', is_coarsen=True):

        base_dir = os.path.join('..', 'data', sname)
        if target == 'avg':
            data_dir = os.path.join(base_dir, '{}_{}'.format(sample_rate, win_size), mode, target,
                                    '{}_{}-{}_{}'.format(s_date,
                                                         s_month, e_date, e_month),
                                    'rm{}t{}p{}'.format(data_rm,temporal_dim,p_dim))
        else:
            data_dir = os.path.join(base_dir, '{}_{}'.format(sample_rate, win_size), mode, target,
                                    '{}_{}_{}'.format(
                                        hist_range[0], hist_range[-1] + 1, hist_range[1] - hist_range[0]),
                                    '{}_{}-{}_{}'.format(s_date,
                                                         s_month, e_date, e_month),
                                    'rm{}t{}p{}'.format(data_rm,temporal_dim,p_dim))

        dict_normal_fname = os.path.join(data_dir, 'dict_normal.pickle')
        train_data_dict_fname = os.path.join(
            data_dir, 'train_data_dict.pickle')
        validate_data_dict_fname = os.path.join(
            data_dir, 'validate_data_dict.pickle')
        Adj_fname = os.path.join(base_dir, 'edge_adj.pickle')

        # for debug only
        edges_file = base_dir + '/edges.pickle'
        edges_list = pklLoad(edges_file)

        edge_adj_file = base_dir + '/edge_adj.pickle'
        edge_adj_list = pklLoad(edge_adj_file)

        edge_dict_file = base_dir + '/edge_dict.pickle'
        edge_dict_list = pklLoad(edge_dict_file)

        if not os.path.exists(dict_normal_fname) or \
                not os.path.exists(train_data_dict_fname) or \
                not os.path.exists(validate_data_dict_fname) or \
                not os.path.exists(Adj_fname):
            print("Creating Data...")
            self.data_generator(base_dir, data_dir, sname, mode, target,
                                sample_rate, win_size, hist_range, s_month,
                                s_date, e_month, e_date, data_rm,least_threshold,trip_table,temporal_dim,p_dim)
        if config.scl_evl_E == 0 and config.scl_evl_T == 0:
            print("Loading data...")
            # for context data construction, hxl added, 20210525:
            # hxl added, 20210616, need to add context construction for HK and mtr datasets
            context_file = base_dir + '/context_' + sname + '.pickle'
            if 'mtr' in context_file and os.path.exists(context_file):
                os.remove(context_file)
            if not os.path.exists(context_file):
                edges_file = base_dir + '/edges.pickle'
                edges_list = pklLoad(edges_file)
                if sname == 'chengdu':
                    osm_f = 'Chengdu_osm.json'
                elif 'xian' in sname:
                    osm_f = 'Xian_osm.json'
                elif 'HK' in sname:
                    osm_f = 'HKmap181217.osm'

                if 'mtr' != sname:
                    osm_json_file = base_dir+'/../'+osm_f

                if sname == 'chengdu' or 'xian' in sname:
                    context_dict = {}
                    with open(osm_json_file, 'r') as f:
                        s = json.load(f)
                    nodes_dict = {}
                    for node in s['elements']:
                        if node['type'] == 'node':
                            id = node['id']
                            lat = node['lat']
                            lon = node['lon']

                            nodes_dict[id] = [float(lat), float(lon)]
                    print('loaded nodes!')

                    for way in s['elements']:
                        if way['type'] == 'way':
                            wayid = str(way['id'])
                            st = str(way['nodes'][0]) + '-' + str(way['nodes'][-1])
                            nodes = way['nodes']

                            # default values
                            speed_limit = 120
                            one_way = 'no'
                            highway = 'none'
                            lanes = 1

                            if 'maxspeed' in way['tags']:
                                speed_limit = way['tags']['maxspeed']
                            if 'oneway' in way['tags']:
                                one_way = way['tags']['oneway']
                            if 'highway' in way['tags']:
                                highway = way['tags']['highway']
                            if 'lanes' in way['tags']:
                                lanes = way['tags']['lanes']
                            lat_lon_list = [nodes_dict[n] for n in nodes]

                            len_way = 0.0
                            for i in range(len(lat_lon_list) - 1):
                                coords_1 = lat_lon_list[i]
                                coords_2 = lat_lon_list[i + 1]
                                dist = np.abs(geopy.distance.distance(coords_1, coords_2).km)
                                len_way += dist

                            context_dict[st] = [speed_limit, one_way, highway, lanes, len_way, st]
                    print('loaded way!')

                if sname == 'mtr':
                    s_e_dic = {'WHA-HOM': 0.830, 'HOM-YMT': 1.804, 'YMT-MOK': 0.766, 'MOK-PRE': 0.585, 'PRE-SKM': 0.797,
                               'SKM-KOT': 1.266, 'KOT-LOF': 0.990, 'LOF-WTS': 0.938,
                               'WTS-DIH': 0.809, 'DIH-CHH': 0.949, 'CHH-KOB': 1.443, 'KOB-NTK': 1.143, 'NTK-KWT': 0.867,
                               'KWT-LAT': 0.930, 'LAT-YAT': 1.089, 'YAT-TIK': 2.087}
                    context_dict = {}
                    for k in s_e_dic.keys():
                        context_dict[k] = [140.0, 'yes', 'none', 1, s_e_dic[k], k]
                    print('loaded mtr context!')



                if sname == 'HK':
                    import xml.etree.ElementTree as ET
                    root = ET.parse(osm_json_file).getroot()
                    # for event, elem in ET.iterparse(osm_json_file, events=("start",)):
                    context_dict = {}
                    # with open(osm_json_file, 'r') as f:
                    #     s = json.load(f)
                    nodes_dict = {}
                    for node in root.findall('node'):
                        if node.tag == 'node':
                            id = node.attrib['id']
                            lat = node.attrib['lat']
                            lon = node.attrib['lon']

                            nodes_dict[id] = [float(lat), float(lon)]
                    print('loaded nodes!')

                    for way in root.findall('way'):
                        wayid = str(way.attrib['id'])
                        nodes = way.findall('nd')
                        st = str(nodes[0].attrib['ref']) + '-' + str(nodes[-1].attrib['ref'])
                        nodes = [nodes[i].attrib['ref'] for i in range(len(nodes))]
                        speed_limit = 120
                        one_way = 'no'
                        highway = 'none'
                        lanes = 1

                        for tag in way.findall('tag'):
                            if 'maxspeed' == tag.attrib['k']:
                                speed_limit = tag.attrib['v']
                            if 'oneway' == tag.attrib['k']:
                                one_way = tag.attrib['v']
                            if 'highway' == tag.attrib['k']:
                                highway = tag.attrib['v']
                            if 'lanes' == tag.attrib['k']:
                                lanes = tag.attrib['v']
                        lat_lon_list = [nodes_dict[n] for n in nodes]

                        len_way = 0.0
                        for i in range(len(lat_lon_list) - 1):
                            coords_1 = lat_lon_list[i]
                            coords_2 = lat_lon_list[i + 1]
                            dist = np.abs(geopy.distance.distance(coords_1, coords_2).km)
                            len_way += dist

                        context_dict[st] = [speed_limit, one_way, highway, lanes, len_way, st]
                    print('loaded way!')

                context_data = [context_dict[e] for e in edges_list]
                pklSave(context_file, context_data)
                print("context data constructed!")


            adj = pklLoad(Adj_fname)
            context_data = pklLoad(context_file)
            dict_normal = pklLoad(dict_normal_fname)
            train_data_dict = pklLoad(train_data_dict_fname)
            validate_data_dict = pklLoad(validate_data_dict_fname)

            if target == 'avg':
                # hxl changes
                # self.y_scaler = dict_normal['velocity']
                self.y_scaler = None
            else:
                self.y_scaler = None

            train_data_temp = train_data_dict['velocity_x']
            if n_heads_tmp==n_heads_prd==0:
                train_data = []
                for i in range(np.shape(train_data_temp)[0]):
                    temp_i = []
                    for j in range(np.shape(train_data_temp)[1]):
                        temp_j = []
                        for k in range(np.shape(train_data_temp)[2]):
                            temp_j.append(train_data_temp[i][j][k][1][1])
                        temp_i.append(np.asarray(temp_j))
                    train_data.append(np.asarray(temp_i))
                train_data = np.asarray(train_data)
            if n_heads_tmp>0 and n_heads_prd>0:
                train_data = train_data_temp
            if n_heads_tmp>0 and n_heads_prd==0:
                train_data = []
                for i in range(np.shape(train_data_temp)[0]):
                    temp_i = []
                    for j in range(np.shape(train_data_temp)[1]):
                        temp_j = []
                        for k in range(np.shape(train_data_temp)[2]):
                            temp_k = []
                            for l in range(np.shape(train_data_temp)[3]):
                                temp_k.append(train_data_temp[i][j][k][l][1])
                            temp_j.append(np.asarray(temp_k))
                        temp_i.append(np.asarray(temp_j))
                    train_data.append(np.asarray(temp_i))
                train_data = np.asarray(train_data)

            train_labels = train_data_dict['velocity_y']
            train_label_weight = train_data_dict['weight_y']
            train_counts = train_data_dict['count_y']
            train_vel_lists = train_data_dict['vel_list']
            cat_train = train_data_dict['cat']
            con_train = train_data_dict['con']


            test_data_temp = validate_data_dict['velocity_x']
            if n_heads_tmp==n_heads_prd==0:
                test_data = []
                for i in range(np.shape(test_data_temp)[0]):
                    temp_i = []
                    for j in range(np.shape(test_data_temp)[1]):
                        temp_j = []
                        for k in range(np.shape(test_data_temp)[2]):
                            temp_j.append(test_data_temp[i][j][k][1][1])
                        temp_i.append(np.asarray(temp_j))
                    test_data.append(np.asarray(temp_i))
                test_data = np.asarray(test_data)
            if n_heads_tmp>0 and n_heads_prd>0:
                test_data = test_data_temp
            if n_heads_tmp>0 and n_heads_prd==0:
                test_data = []
                for i in range(np.shape(test_data_temp)[0]):
                    temp_i = []
                    for j in range(np.shape(test_data_temp)[1]):
                        temp_j = []
                        for k in range(np.shape(test_data_temp)[2]):
                            temp_k = []
                            for l in range(np.shape(test_data_temp)[3]):
                                temp_k.append(test_data_temp[i][j][k][l][1])
                            temp_j.append(np.asarray(temp_k))
                        temp_i.append(np.asarray(temp_j))
                    test_data.append(np.asarray(temp_i))
                test_data = np.asarray(test_data)


            test_labels = validate_data_dict['velocity_y']
            test_labels_weight = validate_data_dict['weight_y']
            test_counts = validate_data_dict['count_y']
            test_vel_lists = validate_data_dict['vel_list']
            cat_test = validate_data_dict['cat']
            con_test = validate_data_dict['con']

            if conv_mode == 'gcnn':
                perm_file = os.path.join(base_dir, 'adj_perm.pickle')
                graph_file = os.path.join(base_dir, 'perm_graphs.pickle')
                if os.path.exists(perm_file) and os.path.exists(graph_file):
                    self.perm = pklLoad(perm_file)
                    self.graphs = pklLoad(graph_file)

            val_x, train_x, val_y, train_y, \
            val_y_weight, train_y_weight, \
            val_con, train_con, val_cat, \
            train_cat, val_count, train_count, \
            val_vel_list, train_vel_list = \
                train_test_split(train_data, train_labels, train_label_weight,
                                 con_train, cat_train, train_counts,
                                 train_vel_lists, test_size=0.80)

        print("Reshaping tensors...")
        self.all_batches = []
        self.sizes = []
        # Split train, val, test data into batches
        # self.construct_batches(train_data, train_labels, train_label_weight,
        #                        cat_train, con_train, train_counts, train_vel_lists, batch_size)
        # self.construct_batches(val_x, val_y, val_y_weight,
        #                        val_cat, val_con, val_count, val_vel_list, batch_size)
        # self.construct_batches(test_x, test_y, test_y_weight,
        #                        test_cat, test_con, test_count, test_vel_list, batch_size)


        self.construct_batches(train_x, train_y, train_y_weight,
                               train_cat, train_con, train_count, train_vel_list, batch_size)
        self.construct_batches(val_x, val_y, val_y_weight,
                               val_cat, val_con, val_count, val_vel_list, batch_size)
        self.construct_batches(test_data, test_labels, test_labels_weight,
                               cat_test, con_test, test_counts, test_vel_lists, batch_size)

        self.adj = adj
        self.context_data = context_data
        self.batch_idx = [0, 0, 0]
        print("data load done. Number of batches in train: %d, val: %d, test: %d"
              % (self.sizes[0], self.sizes[1], self.sizes[2]))


    def split_into_batch(self, data_array, batch_size=-1, sample_size=20):

        shape_list = list(data_array.shape)
        if batch_size == -1:
            batch_size = sample_size * \
                         int(math.floor(shape_list[0] / sample_size))

        data_array = data_array[: batch_size *
                                  int(math.floor(shape_list[0] / batch_size)), ...]
        shape_list[0] = batch_size
        reshape_size = [-1] + shape_list
        batches = list(data_array.reshape(reshape_size))

        return batches

    def construct_batches(self, train_data, train_labels, train_label_weight,
                          cat_train, con_train, count_train, vel_list, batch_size):
        # Split train, val, test data into batches
        train_data_batches = self.split_into_batch(train_data, batch_size, 20)
        train_label_batches = self.split_into_batch(
            train_labels, batch_size, 20)
        train_label_weight_batches = self.split_into_batch(
            train_label_weight, batch_size, 20)
        train_count_batches = self.split_into_batch(
            count_train, batch_size, 20)
        vel_list_batches = self.split_into_batch(vel_list, batch_size, 20)
        # cat_train_batches = self.split_into_batch(cat_train, batch_size)
        # con_train_batches = self.split_into_batch(con_train, batch_size)
        self.all_batches.append([train_data_batches, train_label_batches,
                                 train_label_weight_batches, train_count_batches,
                                 vel_list_batches])
        self.sizes.append(len(train_data_batches))

    def next_batch(self, split_idx):
        # cycle around to beginning
        if self.batch_idx[split_idx] >= self.sizes[split_idx]:
            self.batch_idx[split_idx] = 0
        idx = self.batch_idx[split_idx]
        self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
        return self.all_batches[split_idx][0][idx], \
               self.all_batches[split_idx][1][idx], \
               self.all_batches[split_idx][2][idx], \
               self.all_batches[split_idx][3][idx], \
               self.all_batches[split_idx][4][idx]

    def reset_batch_pointer(self, split_idx, batch_idx=None):
        if batch_idx == None:
            batch_idx = 0
        self.batch_idx[split_idx] = batch_idx

    def data_generator(self, base_dir, data_dir, sname, mode, target,
                       sample_rate, win_size, hist_range, s_month,
                       s_date, e_month, e_date, data_rm,least_threshold,trip_table,temporal_dim,period_dim):

        try:
            os.stat(data_dir)
        except:
            os.makedirs(data_dir)

        if sname == 'HK' or sname == 'HK_local':
            year = 2010
            Training_start_date = datetime(year, 1, 1)
        # else:
        #     year = 2014
        #     Training_start_date = datetime(year, 8, 2)
        if sname == 'mtr_local':
            year = 2019
            Training_start_date = datetime(year, 12, 6)
        if sname == 'mtr':
            year = 2019
            Training_start_date = datetime(year, 11, 1)
        if sname == 'chengdu':
            year = 2016
            Training_start_date = datetime(year, 10, 1)
        if  'xian' in sname:
            year = 2016
            Training_start_date = datetime(year, 10, 1)

        Val_start_date = datetime(year, s_month, s_date)
        Val_end_date = datetime(year, e_month, e_date)
        cat_head = []  # ['time_index', 'dayofweek']
        con_head = []
        prep_param = {'data_dir': data_dir,
                      'base_dir': base_dir,
                      'server_name': sname,
                      'conf_dir': CONF_DIR,
                      'random_node': True,
                      'data_rm_ratio': data_rm,
                      'cat_head': cat_head,
                      'con_head': con_head,
                      'sample_rate': sample_rate,
                      'window_size': win_size,
                      'start_date': Training_start_date,
                      'small_threshold': hist_range[0],
                      'big_threshold': hist_range[-1],
                      'min_nb': min_nb,
                      'test_start_date': Val_start_date,
                      'test_end_date': Val_end_date,
                      'trip_table' : trip_table,
                      'temporal_dim' : temporal_dim,
                      'period_dim' : period_dim}
        try:
            if sname == 'HK' or sname == 'HK_local':
            #     dataset = ds.KDD_Data(**prep_param)
            # else:
                prep_param['topk'] = topk
                dataset = ds.GPS_Data(**prep_param)
            elif sname == 'mtr' or sname == 'mtr_local':
                prep_param['topk'] = topk
                dataset = ds.mtr_Data(**prep_param)
            elif sname == 'chengdu':
                prep_param['topk'] = topk
                dataset = ds.chengdu_Data(**prep_param)
            elif 'xian' in sname:
                prep_param['topk'] = topk
                dataset = ds.xian_Data(**prep_param)


            if temporal_dim > 0 and period_dim > 0:

                dict_normal, train_data_dict, validate_data_dict = \
                    dataset.prepare_est_pred_with_date_plus_temporal_periodic_dim(
                        method=target,
                        window=win_size,
                        sample_rate=sample_rate,
                        mode=mode,
                        hist_range=hist_range,
                        # least=True,
                        least=False,
                        # hxl change from 0.5 to 0.96
                        least_threshold=least_threshold,
                        temporal_dim=temporal_dim,
                        period_dim=period_dim)
        except KeyboardInterrupt:
            print("Ctrl-C is pressed, quiting...")
            sys.exit(0)