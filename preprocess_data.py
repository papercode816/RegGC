
import pandas as pd
import pickle as pkl
import os
import numpy as np
import networkx as nx
import scipy
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import copy
import configparser
from sqlalchemy import create_engine


def connect_sql_server(server, dir):
    """Connect to a sql server with configure file

    This function is used to connect to a sql server using sqlalchemy package by
    specifying a configure file and the a configure name in this file.

    :param server (string): the name of the sql server in configure file
    :param dir (string): the directory of configure file

    :return: the connected sql engine.
    """
    db_conf_file = os.path.join(dir, "dbconf.conf")
    db_conf = configparser.ConfigParser()
    db_conf.read(db_conf_file)
    if 'xian' in server:
        server = 'xian'
    connect_str = db_conf.get(server, "conn_str")
    engine = create_engine(connect_str)

    return engine

def vel_list(array_like, edge_len):
    """
    Convert the array like content into list

    :param array_like (numpy.array or related):

    :return: list: List format of the argument.
    """

    if len(array_like) > 0:
        vel_array = array_like.values[array_like.values > 0]
        if len(vel_array) <= 0:
            vel_array = []
        else:
            # vel_array = edge_len / array_like.values
            vel_array = array_like.values
    else:
        vel_array = []

    return list(vel_array)


def my_rolling_apply_list(frame, func, hist_range, window=1):
    """
    Construct histogram in all cells

    :param frame (pd.Dataframe): The source dataframe
    :param func: The function need to be applied
    :param link_len (double): the length of the current link
    :param hist_range (np.array): the histogram buckets
    :param window (int): The size of the window

    :return: pd.Series
    """
    index = frame.index[window - 1:]
    values = [func(frame.iloc[i:i + window], hist_range)
              for i in range(len(frame) - window + 1)]

    return pd.Series(data=values, index=index).reindex(frame.index)


def my_rolling_apply_avg(frame, func, window=1):
    """
    Construct average value in all cells

    :param frame (pd.Dataframe): The source dataframe
    :param window (int): The size of the window
    :param func: The function need to be applied
    :param link_len (double): the length of the current link

    :return: pd.Series
    """
    index = frame.index[window - 1:]
    values = [func(frame.iloc[i:i + window])
              for i in range(len(frame) - window + 1)]

    return pd.Series(data=values, index=index).reindex(frame.index)

def construct_temp_dim(x_train, temporal_dim):
    x_train_temp = np.asarray([x_train[i - temporal_dim + 1:i+1] for i in range(x_train.shape[0])])

    return x_train_temp


def is_connected_temp_dim(df_x_all, temporal_dim):
    t_list = df_x_all['time_index'].values
    conn_temp_dim = [True if i - temporal_dim + 1>=0 and t_list[i] - t_list[i - temporal_dim + 1]==temporal_dim-1 else False for i in range(t_list.shape[0])]
    return np.asarray(conn_temp_dim)

def construct_temp_periodic_dim(df_x_all, x_train, temporal_dim, period_dim,sample_rate):
    x_train_temp = []
    conn_temp_dim = []
    index_list = list(df_x_all.index.values)
    for i in range(len(index_list)):
        i_temp = index_list[i]
        target_t = df_x_all.loc[i_temp]['time']
        temporal_t_list = [target_t-pd.Timedelta(minutes=sample_rate*ti) for ti in reversed(range(temporal_dim))]
        period_list = np.asarray([np.asarray([ti - pd.Timedelta(minutes=7*24*60*pi) for pi in reversed(range(period_dim))]) for ti in temporal_t_list])
        period_t_list = [[df_x_all.loc[df_x_all['time'] == ti - pd.Timedelta(minutes=7*24*60*pi)] for pi in reversed(range(period_dim))] for ti in temporal_t_list]
        period_t_index = [[df_x_all.index[df_x_all['time'] == ti - pd.Timedelta(minutes=7*24*60*pi)].values for pi in reversed(range(period_dim))] for ti in temporal_t_list]
        p_check = [1 if period_t_list[i][j].empty==True else 0 for j in range(len(period_t_list[0]))  for i in range(len(period_t_list))]
        if 1 in p_check:
            conn_temp_dim.append(False)
            x_train_temp.append(None)
        else:
            conn_temp_dim.append(True)
            true_period_t_index = [[index_list.index(period_t_index[i][j][0]) for j in range(len(period_t_index[0]))] for i in range(len(period_t_index))]
            x_train_i = np.asarray([np.asarray([x_train[true_period_t_index[i][j], :, :] for j in range(len(true_period_t_index[0]))]) for i in range(len(true_period_t_index))])
            x_train_temp.append(x_train_i)

    return np.asarray(x_train_temp),np.asarray(conn_temp_dim)

class DataSet(object):
    def __init__(self, data_dir, base_dir, server_name, conf_dir, random_node,
                 cat_head, con_head, start_date, test_start_date, test_end_date,trip_table,
                 data_rm_ratio=0.5,
                 source_ratio=0.5, sample_rate=5, windows_size=1, predict_size=4,
                 small_threshold=0.0, big_threshold=50.0, min_nb=5, unit=1609.34,
                 combo_random=True, num_combos=3, temporal_dim=1, period_dim=1):

        self.data_dir = data_dir
        self.base_dir = base_dir
        self.random_node = random_node
        self.min_nb = min_nb
        self.unit = unit
        self.num_combos = num_combos
        self.combo_random = combo_random
        self.engine = connect_sql_server(server_name, conf_dir)
        # Get a road graph with the max number of nodes
        self.edges_adj, self.graph_edges = self.construct_road_graph()

        self.data_rm_ratio = data_rm_ratio
        self.source_ratio = source_ratio
        self.data_rm = [int(len(self.graph_edges) * data_rm_ratio)]
        self.data_needed = [len(self.graph_edges) - self.data_rm[0]]
        self.nb_source = [int(len(self.graph_edges) * source_ratio)]
        self.cat_head = cat_head
        self.con_head = con_head
        self.start_date = start_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.sample_rate = sample_rate
        self.window_size = windows_size
        self.predict_size = predict_size
        self.small_threshold = small_threshold
        self.big_threshold = big_threshold
        self.scaler = MinMaxScaler().fit([[self.small_threshold],
                                          [self.big_threshold]])
        self.trip_table = trip_table
        self.temporal_dim = temporal_dim
        self.period_dim = period_dim
        self.server_name = server_name

    def construct_road_graph(self):
        """
        Construct a edge-noded graph

        :return: csr matrix, list of nodes
        """

        edge_adj_file, edges_file = self._get_graph_file()

        if os.path.exists(edge_adj_file) and os.path.exists(edges_file):
            print("Reading road graph from existing file...")
            print('{}\n{}'.format(edge_adj_file, edges_file))
            with open(edge_adj_file, 'rb') as f:
                edges_adj = pkl.load(f)
            with open(edges_file, 'rb') as f:
                edges = pkl.load(f)
        else:
            edge_dict_file = os.path.join(self.base_dir, 'edge_dict.pickle')
            if os.path.exists(edge_dict_file):
                with open(edge_dict_file, 'rb') as f:
                    edge_dict = pkl.load(f)
            else:
                print("Constructing road graph...")
                edge_dict = self._get_edge_connection()
                with open(edge_dict_file, 'wb') as f:
                    pkl.dump(edge_dict, f)
                print("Construction Done! ")
            selected_nodes = list(edge_dict.keys())
            print("The number of nodes is ", len(selected_nodes))

            edges_adj, edges = self._convert_edge_graph(
                edge_dict, self.random_node)

            final_selected_nodes = set(selected_nodes).intersection(edges)
            print("Number of selected edge in "
                  "final edges is {0}/{1}".format(
                len(final_selected_nodes), len(edges)))

            with open(edge_adj_file, 'wb') as f:
                pkl.dump(edges_adj, f)
            with open(edges_file, 'wb') as f:
                pkl.dump(edges, f)

        return edges_adj, edges

    def construct_node_graph(self):
        """
        convert a graph from edge connections to node connection

        :param random_node: bool, whether to random shuffle the data
        :param engine: sqlalchemy sql server engine
        :return:
            dict_edge_node, a dictionary contains the mapping of edge node to real node
            link_ids, list of link ids used in this road network
            W, Adjacency matrix of the resulted graph
            L, Laplacian matrix of the resulted graph
            D, Degree matrix of the resulted graph
        """

        path_directed_dict = os.path.join(
            self.data_dir, 'dict_edge_node.pickle')
        if not os.path.exists(path_directed_dict):
            dict_edge_node = self.construct_dict_edge_node()
        else:
            with open(path_directed_dict, 'rb') as f_di_dict:
                dict_edge_node = pkl.load(f_di_dict)

        self.row_ind, self.col_ind = self.obtain_col_row(dict_edge_node)
        # get a directed graph and convert it into undirected
        di_graph = self.construct_DiGraph(dict_edge_node)

        nodes = di_graph.nodes()
        A = nx.adjacency_matrix(di_graph, nodes)
        # convert adjacency to proximity matrix
        A = A.todense()
        # W = self.get_hop_proximity_matrix(di_graph, hop=2)
        W = self.get_proximity_matrix(A)
        W_row_sum = W.sum(axis=1)
        D = np.diag(W_row_sum)
        # W_row_sum = [sum(W.sum(axis=1).tolist(), [])]
        # D = scipy.sparse.diags(W_row_sum, [0])
        L = D - W

        return dict_edge_node, self.graph_edges, W, L, D

    def construct_dict_edge_node(self):
        return self._construct_dict_edge_node()

    def construct_DiGraph(self, dict_edg_node):
        return self._construct_DiGraph(dict_edg_node)

    def get_proximity_matrix(self, A):
        """
        Get a proximity matrix from adjacency matrix

        :param A: 2d dimension of array
        :return: 2d numpy array
        """
        assert A.ndim == 2
        W = np.zeros(A.shape, dtype=np.float32)
        m = np.sum(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                W[i, j] = np.sum(A[i, :]) * np.sum(A[:, j]) / (2 * m)
        return W

    def get_df_vel(self):

        print("Entering into constructing avg...")
        if 'avg' in self.data_dir:
            df_vel_path = os.path.join(self.data_dir, '..', '..', '..', '..')
        else:
            df_vel_path = os.path.join(self.data_dir, '..', '..', '..', '..', '..')
        df_vel_file = os.path.join(df_vel_path, 'df_vel.pickle')
        # Data of origin to all destinations.
        if not os.path.exists(df_vel_file):
            df_vel = self._construct_link_vel_list()
            df_vel = df_vel.drop_duplicates('time')
            with open(df_vel_file, 'wb') as f:
                pkl.dump(df_vel, f)
        else:
            print("Reading generated df_vel.pickle file...")
            with open(df_vel_file, 'rb') as f:
                df_vel = pkl.load(f)

        df_vel = df_vel.set_index('time')

        return df_vel

    def construct_speed_table(self):
        """
        Construct an average travel speed of nodes within certain time range.

        :param nodes: list, seg_ids that we are required.
        :param sample_freq: str, sample rate.
        :param write_db: bool, write the constructed results into database or not/
        :return: Dataframe, table contains the avg travel speed of all nodes.
        """

        df_link_vel = self._construct_link_vel_list()
        # df_link_vel = self._convert_tt_vel(df_link_tt)
        # Get the time_index and day of week

        df_link_vel['time_index'] = (df_link_vel['time'].dt.hour * 60 / self.sample_rate +
                                     df_link_vel['time'].dt.minute / self.sample_rate).astype(int)
        df_link_vel['dayofweek'] = df_link_vel['time'].dt.dayofweek

        return df_link_vel

    def construct_estimation_df(self, method, window, hist_range=None):

        df_vel_list = self.get_df_vel()
        if method == 'avg':
            df_vel_result = df_vel_list.apply(
                my_rolling_apply_avg, axis=0,
                args=(self.get_vel_avg_rolling,
                      window))
        elif method == 'hist':
            assert hist_range is not None
            df_vel_result = df_vel_list.apply(
                my_rolling_apply_list, axis=0,
                args=(self.get_vel_hist_rolling,
                      hist_range, window))
        else:
            raise Exception("[!] Unkown method type: {}".format(method))

        df_vel_result = df_vel_result.reset_index()
        # Get the time_index and day of week
        df_vel_result['time_index'] = (df_vel_result['time'].dt.hour * 60 / self.sample_rate +
                                       df_vel_result['time'].dt.minute / self.sample_rate).astype(int)
        df_vel_result['dayofweek'] = df_vel_result['time'].dt.dayofweek

        return df_vel_result

    def construct_count_df(self, method, window):

        df_vel_list = self.get_df_vel()
        df_count = df_vel_list.apply(
            my_rolling_apply_avg, axis=0, args=(
                self.get_vel_count_rolling, window))

        df_count = df_count.reset_index()

        return df_count

    def construct_origin_df(self, method, window):

        df_vel_list = self.get_df_vel()
        df_count = df_vel_list.apply(
            my_rolling_apply_avg, axis=0, args=(
                self.get_vel_value_rolling, window))

        df_count = df_count.reset_index()

        return df_count

    def get_vel_hist_rolling(self, pdSeries_like, hist_bin):
        data_lists = pdSeries_like.values.flatten()
        data_list = []
        for i, item in enumerate(data_lists):
            if type(item) == list or type(item) == np.ndarray:
                for j, item_j in enumerate(item):
                    data_list.append(item_j)
            else:
                continue
        # data_list = [item for sublist in data_lists for item in sublist]
        # tt_array = pdSeries_like.values.flatten()
        data_array = np.array(data_list)
        data_keep = (data_array < self.big_threshold) & (
            data_array >= self.small_threshold)
        data_array = data_array[data_keep]

        if len(data_array) < self.min_nb:
            # print("length smaller than {}...".format(self.min_nb))
            return np.nan

        hist, bin_edges = np.histogram(data_array, hist_bin, density=True)

        if np.isnan(hist).any():
            print('nan hist returned!')
            print('data', data_array)
            print(hist)
            return np.nan
        hist *= hist_bin[1] - hist_bin[0]

        return hist

    def get_vel_avg_rolling(self, pdSeries_like):

        data_lists = pdSeries_like.values.flatten()
        data_list = []
        for i, item in enumerate(data_lists):
            if type(item) == list or type(item) == np.ndarray:
                for j, item_j in enumerate(item):
                    data_list.append(item_j)
            else:
                continue

        data_array = np.array(data_list)
        data_keep = (data_array < self.big_threshold) & (
            data_array >= self.small_threshold)

        data_array = data_array[data_keep]
        if len(data_array) < self.min_nb:
            # print("length smaller than {}...".format(self.min_nb))
            return np.nan

        mean = np.nanmean(data_array)
        f_mean = self.scaler.transform([[mean]])[0]

        return f_mean

    def get_vel_count_rolling(self, pdSeries_like):

        data_lists = pdSeries_like.values.flatten()
        data_list = []
        for i, item in enumerate(data_lists):
            if type(item) == list or type(item) == np.ndarray:
                for j, item_j in enumerate(item):
                    data_list.append(item_j)
            else:
                continue

        data_array = np.array(data_list)
        data_keep = (data_array < self.big_threshold) & (
            data_array >= self.small_threshold)

        data_array = data_array[data_keep]

        return len(data_array)

    def get_vel_value_rolling(self, pdSeries_like):

        data_lists = pdSeries_like.values.flatten()
        data_list = []
        for i, item in enumerate(data_lists):
            if type(item) == list or type(item) == np.ndarray:
                for j, item_j in enumerate(item):
                    data_list.append(item_j)
            else:
                continue

        data_array = np.array(data_list)
        data_keep = (data_array < self.big_threshold) & (
            data_array >= self.small_threshold)

        data_array = data_array[data_keep]

        return list(data_array)


    def convert_multi_channel_array(self, df_all_array, nb_bins):

        multi_channel_array = []
        all_shape = df_all_array.shape

        for i in range(all_shape[0]):
            channel_i = np.zeros((all_shape[1], nb_bins))
            for j in range(all_shape[1]):
                if pd.isnull([df_all_array[i, j]]).any():
                    continue
                else:
                    for k in range(nb_bins):
                        channel_i[j, k] = df_all_array[i, j][k]
            multi_channel_array.append(channel_i)
        multi_channel_array = np.array(multi_channel_array)

        return multi_channel_array


    def categorical_trans_file(self, data, cat_head):
        # make the training data categorical
        les = {}
        for i in range(data.shape[1]):
            le = LabelEncoder()
            data[:, i] = le.fit_transform(data[:, i])
            les[cat_head[i]] = le

        with open('les.pickle', 'wb') as f:
            pkl.dump(les, f, -1)

        return les, data

    def prepare_est_pred_with_date(self, method, window, mode='prediction',
                                   hist_range=None,
                                   least=True, least_threshold=0.5):

        if method == 'hist':
            assert hist_range is not None
            nb_bins = len(hist_range) - 1
        elif method == 'avg':
            nb_bins = 1
        else:
            raise Exception("[!] Unkown method: {}".format(method))
        assert mode == 'prediction' or mode == 'estimation'

        dict_normal_file = os.path.join(self.data_dir, 'dict_normal.pickle')
        train_data_dict_file = os.path.join(
            self.data_dir, 'train_data_dict.pickle')
        validate_data_dict_file = os.path.join(
            self.data_dir, 'validate_data_dict.pickle')

        if os.path.exists(dict_normal_file) and os.path.exists(train_data_dict_file) \
                and os.path.exists(validate_data_dict_file):
            print("Reading feature from existing file...")
            print('{}\n{}\n{}'.format(dict_normal_file,
                                      train_data_dict_file,
                                      validate_data_dict_file))

            with open(dict_normal_file, 'rb') as f:
                dict_normal = pkl.load(f)
            with open(train_data_dict_file, 'rb') as f:
                train_data_dict = pkl.load(f)
            with open(validate_data_dict_file, 'rb') as f:
                validate_data_dict = pkl.load(f)
        else:
            df_all = self.construct_estimation_df(
                method, window=window, hist_range=hist_range)
            # get the number of records
            df_count = self.construct_count_df(method, window)

            df_vel_list = self.construct_origin_df(method, window)

            row_notnull = pd.notnull(df_all[self.graph_edges].values)

            if least:
                num_needed = int(len(self.graph_edges) * (1 - least_threshold))
            else:
                num_needed = len(self.graph_edges) - self.data_rm[0]

            row_keep = row_notnull.sum(axis=1) >= num_needed
            df_x_all = df_all[row_keep]
            if mode == 'estimation':
                df_y_all = df_x_all.copy()
                df_count = df_count[row_keep]
                df_vel_list = df_vel_list[row_keep]
            else:
                df_y_index = df_x_all.index + window
                df_y_all = df_all.loc[df_y_index]
                df_count = df_count.loc[df_y_index]
                df_vel_list = df_vel_list.loc[df_y_index]

            print("The dimension of the dataframe is {}".format(df_all.shape))
            x_all_array = df_x_all[self.graph_edges].values
            y_all_array = df_y_all[self.graph_edges].values
            y_all_count = df_count[self.graph_edges].values
            vel_list = df_vel_list[self.graph_edges].values

            con_array = df_x_all[self.con_head].values
            cat_array = df_x_all[self.cat_head].values.astype(np.int)
            # position indicator of meaningful data
            row_notnull = row_notnull[row_keep, :]
            validate_row = (df_x_all['time'] > self.test_start_date) & \
                           (df_x_all['time'] < self.test_end_date)
            training_row = validate_row == False

            dict_normal = {}
            # 2. max_min normalization for continuous data (wind speed,
            # temperature)
            for i, con_head_i in enumerate(self.con_head):
                scaler_i = MinMaxScaler()
                con_array[:, i] = scaler_i.fit_transform(con_array[:, i])
                dict_normal[con_head_i] = scaler_i
            # 3. categorical type data
            les, cat_array = self.categorical_trans_file(
                cat_array, self.cat_head)
            dict_normal.update(les)
            con_validate = con_array[validate_row, :]
            cat_validate = cat_array[validate_row, :]
            cat_train = cat_array[training_row, :]
            con_train = con_array[training_row, :]

            df_all_x_array = self.convert_multi_channel_array(
                x_all_array, nb_bins)
            df_all_y_array = self.convert_multi_channel_array(
                y_all_array, nb_bins)

            array_y_weight = np.zeros(row_notnull.shape, dtype=np.int8)
            y_pos = np.sum(df_all_y_array, axis=2) > 0
            array_y_weight[y_pos] = 1

            df_all_x_array = self.construct_x_array(df_all_x_array, row_notnull)
            x_train = df_all_x_array[training_row, :, :]

            x_validate = df_all_x_array[validate_row, :, :]
            y_validate = df_all_y_array[validate_row, :, :]

            if mode == 'estimation':
                y_train = x_train.copy()
                train_y_pos = np.sum(y_train, axis=2) > 0
                train_y_weight = np.zeros(train_y_pos.shape, dtype=np.int8)
                train_y_weight[train_y_pos] = 1

                val_y_pos = np.sum(x_validate, axis=2) > 0
                val_y_weight = array_y_weight[validate_row, :]
                val_y_weight[val_y_pos] = 1

            else:
                y_train = df_all_y_array[training_row, :, :]
                train_y_weight = array_y_weight[training_row, :]
                val_y_weight = array_y_weight[validate_row, :]

            # get the count of records to construct histograms
            y_train_count = y_all_count[training_row, :]
            y_val_count = y_all_count[validate_row, :]

            # get the original records
            train_vel_list = vel_list[training_row, :]
            val_vel_list = vel_list[validate_row, :]

            train_data_dict = {'velocity_x': x_train,
                               'velocity_y': y_train,
                               'weight_y': train_y_weight,
                               'count_y': y_train_count,
                               'vel_list': train_vel_list,
                               'cat': cat_train,
                               'con': con_train}
            validate_data_dict = {'velocity_x': x_validate,
                                  'velocity_y': y_validate,
                                  'weight_y': val_y_weight,
                                  'count_y': y_val_count,
                                  'vel_list': val_vel_list,
                                  'cat': cat_validate,
                                  'con': con_validate}

            # write these three dict into files
            dict_list = ['dict_normal',
                         'train_data_dict', 'validate_data_dict']
            dicts = [dict_normal, train_data_dict, validate_data_dict]
            for i, dict_i in enumerate(dict_list):
                path_dict_i = os.path.join(self.data_dir, dict_i + '.pickle')
                with open(path_dict_i, 'wb') as f:
                    pkl.dump(dicts[i], f)


        if mode == 'estimation':
            self.verify_data_correctness(train_data_dict, validate_data_dict)
        else:
            self.verify_pred_data_correctness(train_data_dict, validate_data_dict)

        return dict_normal, train_data_dict, validate_data_dict



    def prepare_est_pred_with_date_plus_temporal_dim(self, method, window, mode='prediction',
                                                     hist_range=None,
                                                     least=True, least_threshold=0.5, temporal_dim=1):

        if method == 'hist':
            assert hist_range is not None
            nb_bins = len(hist_range) - 1
        elif method == 'avg':
            nb_bins = 1
        else:
            raise Exception("[!] Unkown method: {}".format(method))
        assert mode == 'prediction' or mode == 'estimation'

        dict_normal_file = os.path.join(self.data_dir, 'dict_normal.pickle')
        train_data_dict_file = os.path.join(
            self.data_dir, 'train_data_dict.pickle')
        validate_data_dict_file = os.path.join(
            self.data_dir, 'validate_data_dict.pickle')

        if os.path.exists(dict_normal_file) and os.path.exists(train_data_dict_file) \
                and os.path.exists(validate_data_dict_file):
            print("Reading feature from existing file...")
            print('{}\n{}\n{}'.format(dict_normal_file,
                                      train_data_dict_file,
                                      validate_data_dict_file))

            with open(dict_normal_file, 'rb') as f:
                dict_normal = pkl.load(f)
            with open(train_data_dict_file, 'rb') as f:
                train_data_dict = pkl.load(f)
            with open(validate_data_dict_file, 'rb') as f:
                validate_data_dict = pkl.load(f)
        else:
            df_all = self.construct_estimation_df(
                method, window=window, hist_range=hist_range)
            print('df_all shape:%s' %(str(df_all.shape)))
            # get the number of records
            df_count = self.construct_count_df(method, window)

            df_vel_list = self.construct_origin_df(method, window)

            row_notnull = pd.notnull(df_all[self.graph_edges].values)

            if least:
                num_needed = int(len(self.graph_edges) * (1 - least_threshold))
            else:
                num_needed = len(self.graph_edges) - self.data_rm[0]

            row_keep = row_notnull.sum(axis=1) >= num_needed
            df_x_all = df_all[row_keep]
            print('df_x_all shape:%s' % (str(df_x_all.shape)))

            if mode == 'estimation':
                df_y_all = df_x_all.copy()
                df_count = df_count[row_keep]
                df_vel_list = df_vel_list[row_keep]
            else:
                df_y_index = df_x_all.index + window
                df_y_all = df_all.loc[df_y_index]
                df_count = df_count.loc[df_y_index]
                df_vel_list = df_vel_list.loc[df_y_index]

            print("The dimension of the dataframe is {}".format(df_all.shape))
            x_all_array = df_x_all[self.graph_edges].values
            y_all_array = df_y_all[self.graph_edges].values
            y_all_count = df_count[self.graph_edges].values
            vel_list = df_vel_list[self.graph_edges].values

            con_array = df_x_all[self.con_head].values
            cat_array = df_x_all[self.cat_head].values.astype(np.int)
            # position indicator of meaningful data
            row_notnull = row_notnull[row_keep, :]
            validate_row = (df_x_all['time'] > self.test_start_date) & \
                           (df_x_all['time'] < self.test_end_date)
            training_row = validate_row == False

            dict_normal = {}
            # 2. max_min normalization for continuous data (wind speed,
            # temperature)
            for i, con_head_i in enumerate(self.con_head):
                scaler_i = MinMaxScaler()
                con_array[:, i] = scaler_i.fit_transform(con_array[:, i])
                dict_normal[con_head_i] = scaler_i
            # 3. categorical type data
            les, cat_array = self.categorical_trans_file(
                cat_array, self.cat_head)
            dict_normal.update(les)
            con_validate = con_array[validate_row, :]
            cat_validate = cat_array[validate_row, :]
            cat_train = cat_array[training_row, :]
            con_train = con_array[training_row, :]

            df_all_x_array = self.convert_multi_channel_array(
                x_all_array, nb_bins)
            df_all_y_array = self.convert_multi_channel_array(
                y_all_array, nb_bins)

            array_y_weight = np.zeros(row_notnull.shape, dtype=np.int8)
            y_pos = np.sum(df_all_y_array, axis=2) > 0
            array_y_weight[y_pos] = 1

            df_all_x_array = self.construct_x_array(df_all_x_array, row_notnull)
            x_train = df_all_x_array[training_row, :, :]

            x_validate = df_all_x_array[validate_row, :, :]
            y_validate = df_all_y_array[validate_row, :, :]

            if mode == 'estimation':
                y_train = x_train.copy()
                train_y_pos = np.sum(y_train, axis=2) > 0
                train_y_weight = np.zeros(train_y_pos.shape, dtype=np.int8)
                train_y_weight[train_y_pos] = 1

                val_y_pos = np.sum(x_validate, axis=2) > 0
                val_y_weight = array_y_weight[validate_row, :]
                val_y_weight[val_y_pos] = 1

            else:
                y_train = df_all_y_array[training_row, :, :]
                train_y_weight = array_y_weight[training_row, :]
                val_y_weight = array_y_weight[validate_row, :]

            # get the count of records to construct histograms
            y_train_count = y_all_count[training_row, :]
            y_val_count = y_all_count[validate_row, :]

            # get the original records
            train_vel_list = vel_list[training_row, :]
            val_vel_list = vel_list[validate_row, :]

            # add temporal dim here
            # 1. construct temporal dim tensor for x_train, x_validate
            x_all_temp = construct_temp_dim(df_all_x_array, temporal_dim)
            x_train_temp = x_all_temp[training_row]
            x_validate_temp = x_all_temp[validate_row]

            # 2. calculate boolean vector of connectivity of tem_dim
            conn_temp_dim = is_connected_temp_dim(df_x_all, temporal_dim)

            # 3. filter train_data_dict, validate_data_dict by boolean vector in step 2
            x_train = np.asarray(x_train_temp[conn_temp_dim[training_row.values]].tolist()).transpose((0, 2, 3, 1))
            y_train = y_train[conn_temp_dim[training_row.values]]
            train_y_weight = train_y_weight[conn_temp_dim[training_row.values]]
            y_train_count = y_train_count[conn_temp_dim[training_row.values]]
            train_vel_list = train_vel_list[conn_temp_dim[training_row.values]]
            cat_train = cat_train[conn_temp_dim[training_row.values]]
            con_train = con_train[conn_temp_dim[training_row.values]]

            x_validate = np.asarray(x_validate_temp[conn_temp_dim[validate_row.values]].tolist()).transpose(
                (0, 2, 3, 1))
            y_validate = y_validate[conn_temp_dim[validate_row.values]]
            val_y_weight = val_y_weight[conn_temp_dim[validate_row.values]]
            y_val_count = y_val_count[conn_temp_dim[validate_row.values]]
            val_vel_list = val_vel_list[conn_temp_dim[validate_row.values]]
            cat_validate = cat_validate[conn_temp_dim[validate_row.values]]
            con_validate = con_validate[conn_temp_dim[validate_row.values]]

            train_data_dict = {'velocity_x': x_train,
                               'velocity_y': y_train,
                               'weight_y': train_y_weight,
                               'count_y': y_train_count,
                               'vel_list': train_vel_list,
                               'cat': cat_train,
                               'con': con_train}
            validate_data_dict = {'velocity_x': x_validate,
                                  'velocity_y': y_validate,
                                  'weight_y': val_y_weight,
                                  'count_y': y_val_count,
                                  'vel_list': val_vel_list,
                                  'cat': cat_validate,
                                  'con': con_validate}

            # write these three dict into files
            dict_list = ['dict_normal',
                         'train_data_dict', 'validate_data_dict']
            dicts = [dict_normal, train_data_dict, validate_data_dict]
            for i, dict_i in enumerate(dict_list):
                path_dict_i = os.path.join(self.data_dir, dict_i + '.pickle')
                with open(path_dict_i, 'wb') as f:
                    pkl.dump(dicts[i], f)

        # if mode == 'estimation':
        #     self.verify_data_correctness(train_data_dict, validate_data_dict)
        # else:
        #     self.verify_pred_data_correctness(train_data_dict, validate_data_dict)

        return dict_normal, train_data_dict, validate_data_dict



    def prepare_est_pred_with_date_plus_temporal_periodic_dim(self, method, window, sample_rate, mode='prediction',
                                                     hist_range=None,
                                                     least=True, least_threshold=0.5, temporal_dim=1, period_dim=1):

        if method == 'hist':
            assert hist_range is not None
            nb_bins = len(hist_range) - 1
        elif method == 'avg':
            nb_bins = 1
        else:
            raise Exception("[!] Unkown method: {}".format(method))
        assert mode == 'prediction' or mode == 'estimation'

        dict_normal_file = os.path.join(self.data_dir, 'dict_normal.pickle')
        train_data_dict_file = os.path.join(
            self.data_dir, 'train_data_dict.pickle')
        validate_data_dict_file = os.path.join(
            self.data_dir, 'validate_data_dict.pickle')

        if os.path.exists(dict_normal_file) and os.path.exists(train_data_dict_file) \
                and os.path.exists(validate_data_dict_file):
            print("Reading feature from existing file...")
            print('{}\n{}\n{}'.format(dict_normal_file,
                                      train_data_dict_file,
                                      validate_data_dict_file))

            with open(dict_normal_file, 'rb') as f:
                dict_normal = pkl.load(f)
            with open(train_data_dict_file, 'rb') as f:
                train_data_dict = pkl.load(f)
            with open(validate_data_dict_file, 'rb') as f:
                validate_data_dict = pkl.load(f)
        else:
            df_all = self.construct_estimation_df(
                method, window=window, hist_range=hist_range)
            print('df_all shape:%s' %(str(df_all.shape)))
            # get the number of records
            df_count = self.construct_count_df(method, window)

            df_vel_list = self.construct_origin_df(method, window)

            row_notnull = pd.notnull(df_all[self.graph_edges].values)

            if least:
                num_needed = int(len(self.graph_edges) * (1 - least_threshold))
            else:
                num_needed = len(self.graph_edges) - self.data_rm[0]

            row_keep = row_notnull.sum(axis=1) >= num_needed
            df_x_all = df_all[row_keep]
            print('df_x_all shape:%s' % (str(df_x_all.shape)))

            if mode == 'estimation':
                df_y_all = df_x_all.copy()
                df_count = df_count[row_keep]
                df_vel_list = df_vel_list[row_keep]
            else:
                df_y_index = df_x_all.index + window
                print(len(df_y_index))
                print(df_y_index)
                # df_y_all = df_all.loc[df_y_index]
                # df_count = df_count.loc[df_y_index]
                # df_vel_list = df_vel_list.loc[df_y_index]
                df_y_all = df_all.reindex(df_y_index)
                df_count = df_count.reindex(df_y_index)
                df_vel_list = df_vel_list.reindex(df_y_index)

            print("The dimension of the dataframe is {}".format(df_all.shape))
            x_all_array = df_x_all[self.graph_edges].values
            y_all_array = df_y_all[self.graph_edges].values
            y_all_count = df_count[self.graph_edges].values
            vel_list = df_vel_list[self.graph_edges].values

            con_array = df_x_all[self.con_head].values
            cat_array = df_x_all[self.cat_head].values.astype(np.int)
            # position indicator of meaningful data
            row_notnull = row_notnull[row_keep, :]
            validate_row = (df_x_all['time'] > self.test_start_date) & \
                           (df_x_all['time'] < self.test_end_date)
            training_row = validate_row == False

            dict_normal = {}
            # 2. max_min normalization for continuous data (wind speed,
            # temperature)
            for i, con_head_i in enumerate(self.con_head):
                scaler_i = MinMaxScaler()
                con_array[:, i] = scaler_i.fit_transform(con_array[:, i])
                dict_normal[con_head_i] = scaler_i
            # 3. categorical type data
            les, cat_array = self.categorical_trans_file(
                cat_array, self.cat_head)
            dict_normal.update(les)
            con_validate = con_array[validate_row, :]
            cat_validate = cat_array[validate_row, :]
            cat_train = cat_array[training_row, :]
            con_train = con_array[training_row, :]

            df_all_x_array = self.convert_multi_channel_array(
                x_all_array, nb_bins)
            df_all_y_array = self.convert_multi_channel_array(
                y_all_array, nb_bins)

            array_y_weight = np.zeros(row_notnull.shape, dtype=np.int8)
            y_pos = np.sum(df_all_y_array, axis=2) > 0
            array_y_weight[y_pos] = 1

            df_all_x_array = self.construct_x_array(df_all_x_array, row_notnull)
            x_train = df_all_x_array[training_row, :, :]

            x_validate = df_all_x_array[validate_row, :, :]
            y_validate = df_all_y_array[validate_row, :, :]

            if mode == 'estimation':
                y_train = x_train.copy()
                train_y_pos = np.sum(y_train, axis=2) > 0
                train_y_weight = np.zeros(train_y_pos.shape, dtype=np.int8)
                train_y_weight[train_y_pos] = 1

                val_y_pos = np.sum(x_validate, axis=2) > 0
                val_y_weight = array_y_weight[validate_row, :]
                val_y_weight[val_y_pos] = 1

            else:
                y_train = df_all_y_array[training_row, :, :]
                train_y_weight = array_y_weight[training_row, :]
                val_y_weight = array_y_weight[validate_row, :]

            # get the count of records to construct histograms
            y_train_count = y_all_count[training_row, :]
            y_val_count = y_all_count[validate_row, :]

            # get the original records
            train_vel_list = vel_list[training_row, :]
            val_vel_list = vel_list[validate_row, :]



            # add temporal dim here
            # 1. construct temporal dim tensor for x_train, x_validate
            x_all_temp,conn_temp_dim = construct_temp_periodic_dim(df_x_all, df_all_x_array, temporal_dim, period_dim, sample_rate)
            x_train_temp = x_all_temp[training_row]
            print(len(x_train_temp))
            x_validate_temp = x_all_temp[validate_row]
            print(len(x_validate_temp))

            # 2. calculate boolean vector of connectivity of tem_dim
            # conn_temp_dim = is_connected_temp_periodic_dim(df_x_all, temporal_dim, period_dim)

            # 3. filter train_data_dict, validate_data_dict by boolean vector in step 2
            print(len(training_row.values))
            print(len(conn_temp_dim[training_row.values]))
            print(len(x_train_temp[conn_temp_dim[training_row.values]]))
            x_train = np.asarray(x_train_temp[conn_temp_dim[training_row.values]].tolist()).transpose((0, 3, 4, 1,2))
            y_train = y_train[conn_temp_dim[training_row.values]]
            train_y_weight = train_y_weight[conn_temp_dim[training_row.values]]
            y_train_count = y_train_count[conn_temp_dim[training_row.values]]
            train_vel_list = train_vel_list[conn_temp_dim[training_row.values]]
            cat_train = cat_train[conn_temp_dim[training_row.values]]
            con_train = con_train[conn_temp_dim[training_row.values]]

            print(len(validate_row.values))
            print(len(conn_temp_dim[validate_row.values]))
            print(len(x_validate_temp[conn_temp_dim[validate_row.values]]))
            x_validate = np.asarray(x_validate_temp[conn_temp_dim[validate_row.values]].tolist()).transpose((0, 3, 4, 1,2))
            y_validate = y_validate[conn_temp_dim[validate_row.values]]
            val_y_weight = val_y_weight[conn_temp_dim[validate_row.values]]
            y_val_count = y_val_count[conn_temp_dim[validate_row.values]]
            val_vel_list = val_vel_list[conn_temp_dim[validate_row.values]]
            cat_validate = cat_validate[conn_temp_dim[validate_row.values]]
            con_validate = con_validate[conn_temp_dim[validate_row.values]]




            train_data_dict = {'velocity_x': x_train,
                               'velocity_y': y_train,
                               'weight_y': train_y_weight,
                               'count_y': y_train_count,
                               'vel_list': train_vel_list,
                               'cat': cat_train,
                               'con': con_train}
            validate_data_dict = {'velocity_x': x_validate,
                                  'velocity_y': y_validate,
                                  'weight_y': val_y_weight,
                                  'count_y': y_val_count,
                                  'vel_list': val_vel_list,
                                  'cat': cat_validate,
                                  'con': con_validate}

            # write these three dict into files
            dict_list = ['dict_normal',
                         'train_data_dict', 'validate_data_dict']
            dicts = [dict_normal, train_data_dict, validate_data_dict]
            for i, dict_i in enumerate(dict_list):
                path_dict_i = os.path.join(self.data_dir, dict_i + '.pickle')
                with open(path_dict_i, 'wb') as f:
                    pkl.dump(dicts[i], f)

        # if mode == 'estimation':
        #     self.verify_data_correctness(train_data_dict, validate_data_dict)
        # else:
        #     self.verify_pred_data_correctness(train_data_dict, validate_data_dict)

        return dict_normal, train_data_dict, validate_data_dict



    def construct_x_array(self, df_all_x_array, row_notnull):
        """random select num needed nodes with data"""
        if 'HK' in self.server_name:
            return df_all_x_array
        else:
            for i in range(row_notnull.shape[0]):
                not_selected_bool = np.array([True] * len(self.graph_edges))
                notnull_idx_i = np.where(row_notnull[i, :])[0]
                num_needed = len(self.graph_edges) - self.data_rm[0]
                if len(notnull_idx_i) < num_needed:
                    continue
                rand_choice_idx = np.random.choice(
                    notnull_idx_i, num_needed, replace=False)
                not_selected_bool[rand_choice_idx] = False
                df_all_x_array[i, not_selected_bool] = 0

            return df_all_x_array

    def verify_data_correctness(self, train_data_dict, validate_data_dict):
        """verify the correctness of the estimation data"""
        num_needed = len(self.graph_edges) - self.data_rm[0]
        x_train = train_data_dict['velocity_x']
        y_train = train_data_dict['velocity_y']
        train_y_weight = train_data_dict['weight_y']

        x_val = validate_data_dict['velocity_x']
        y_val = validate_data_dict['velocity_y']
        val_y_weight = validate_data_dict['weight_y']

        x_train_sum2 = np.sum(x_train, axis=2)

        assert np.array_equal(x_train, y_train)
        assert np.array_equal(x_train_sum2 > 0, train_y_weight == 1)

        x_val_sum2 = np.sum(x_val, axis=2)
        y_val_sum2 = np.sum(y_val, axis=2)

        assert np.array_equal(y_val_sum2>0, val_y_weight == 1)

    def verify_pred_data_correctness(self, train_data_dict, validate_data_dict):
        """verify the correctness of the prediction data"""

        num_needed = len(self.graph_edges) - self.data_rm[0]
        x_train = train_data_dict['velocity_x']
        y_train = train_data_dict['velocity_y']
        train_y_weight = train_data_dict['weight_y']

        x_val = validate_data_dict['velocity_x']
        y_val = validate_data_dict['velocity_y']
        val_y_weight = validate_data_dict['weight_y']

        x_train_sum3 = np.sum(x_train, axis=2)
        y_train_sum3 = np.sum(y_train, axis=2)

        assert np.array_equal(y_train_sum3 > 0, train_y_weight == 1)

        x_val_sum3 = np.sum(x_val, axis=2)
        y_val_sum3 = np.sum(y_val, axis=2)

        assert np.array_equal(y_val_sum3>0, val_y_weight==1)

    def _get_graph_file(self):
        return NotImplementedError

    def _get_edge_connection(self):
        return NotImplementedError

    def _convert_edge_graph(self, edge_dict, random_node=False):
        return NotImplementedError

    def _construct_dict_edge_node(self):
        return NotImplementedError

    def _construct_DiGraph(self, dict_edg_node):
        return NotImplementedError

    def _construct_link_vel_list(self):
        return NotImplementedError

    def _construct_seq_data(self, df_vel, list_inters, nb_bins, combo_random=False):
        """
        Construct sequence data with specified intervals

        :param df_vel: Dataframe, source data
        :param list_inters: list, specified intervals
        :param combo_random: bool, default is False, prefer combinations
        :return: np.arrays
        """
        list_x_array = []
        list_y_array = []
        list_y_weight = []
        list_con = []
        list_cat = []
        for inter_i in list_inters:
            cur_inters = list(range(inter_i, inter_i + self.window_size))
            cur_inters.append(int(inter_i + self.window_size + self.predict_size - 1))
            # print("Now it's in {0}/{1}".format(test, len(list_inters)))
            dt_series = pd.Series([pd.Timedelta(minutes=i * self.sample_rate)
                                   for i in cur_inters])
            dt_series += self.start_date
            row_selected = df_vel.loc[dt_series, self.graph_edges].values
            row_selected = self.convert_multi_channel_array(row_selected, nb_bins)

            tmp_con = df_vel.loc[dt_series[0], self.con_head]
            tmp_cat = df_vel.loc[dt_series[0], self.cat_head]

            if combo_random:
                list_x_array, list_y_array, list_y_weight = \
                    self._arange_data_random(row_selected, list_x_array,
                                             list_y_array, list_y_weight)
                list_con.append(tmp_con)
                list_cat.append(tmp_cat)
            else:
                list_x_array, list_y_array, list_y_weight, com_num = \
                    self._arange_data_combos(row_selected, list_x_array,
                                             list_y_array, list_y_weight)
                list_con += [tmp_con] * com_num
                list_cat += [tmp_cat] * com_num

        x_array = np.array(list_x_array)
        y_array = np.array(list_y_array)
        y_weight = np.array(list_y_weight)
        con_array = np.array(list_con)
        cat_array = np.array(list_cat)

        train_data_dict = {'velocity_x': x_array,
                           'velocity_y': y_array,
                           'weight_y': y_weight,
                           'cat': cat_array,
                           'con': con_array}

        return train_data_dict

    def _arange_data_random(self, row_selected, list_x_array,
                            list_y_array, list_y_weight):
        """
        Process the data in a random order

        :param row_selected:
        :param list_x_array:
        :param list_y_array:
        :param list_y_weight:
        :return:
        """

        x_data = self.tailor_data_needed_single_random(row_selected[:self.window_size, :])
        # print("The shape of the x_data after is ", x_data.shape)
        y_weight, y_data = self.tailor_predicted_val_weights(
            row_selected[self.window_size:, :])

        list_x_array.append(x_data)
        list_y_array.append(y_data)
        list_y_weight.append(y_weight)

        return list_x_array, list_y_array, list_y_weight

    def _arange_data_combos(self, row_selected, list_x_array, list_y_array, list_y_weight):
        """
        Process the data in a random order

        :param row_selected:
        :param list_x_array:
        :param list_y_array:
        :param list_y_weight:
        :return:
        """

        list_x_data = self.tailor_data_needed_combination(row_selected[:self.window_size, :])
        # print("The shape of the x_data after is ", x_data.shape)
        y_weight, y_data = self.tailor_predicted_val_weights(
            row_selected[self.window_size:, :])
        list_y_data = [y_data] * len(list_x_data)
        list_y_weight_i = [y_weight] * len(list_x_data)
        list_x_array += list_x_data
        list_y_array += list_y_data
        list_y_weight += list_y_weight_i

        return list_x_array, list_y_array, list_y_weight, len(list_x_data)

    def tailor_data_needed_combination(self, row_array):

        row_notnull = pd.notnull(row_array)
        combinations_list = [row_array]
        for row_i in range(row_notnull.shape[0]):
            notnull_idx_i = np.where(row_notnull[row_i, :])[0]
            assert len(notnull_idx_i) >= self.data_needed[0]
            if (len(notnull_idx_i) - self.data_needed[0] + 1) >= self.num_combos:
                row_i_combos_num = self.num_combos
            else:
                row_i_combos_num = len(notnull_idx_i) - self.data_needed[0] + 1
            prev_num_combinations = len(combinations_list)
            combinations_list = combinations_list * row_i_combos_num

            for i in range(row_i_combos_num):
                combo_i = notnull_idx_i[i:self.data_needed[0] + i]
                for prev_i in range(prev_num_combinations):
                    row_array_i = copy.copy(combinations_list[i*prev_num_combinations + prev_i])
                    not_selected_bool = np.array([True] * row_array.shape[1])
                    not_selected_bool[combo_i] = False
                    row_array_i[row_i, not_selected_bool] = 0

                    combinations_list[i*prev_num_combinations + prev_i] = row_array_i

        return combinations_list

    def tailor_data_needed_single_random(self, row_array):

        row_notnull = pd.notnull(row_array)
        for row_i in range(row_notnull.shape[0]):
            not_selected_bool = np.array([True] * row_array.shape[1])
            notnull_idx_i = np.where(row_notnull[row_i, :])[0]
            rand_choice_idx = np.random.choice(
                notnull_idx_i, self.data_needed[0], replace=False)
            not_selected_bool[rand_choice_idx] = False
            row_array[row_i, not_selected_bool] = 0

        return row_array

    def tailor_predicted_val_weights(self, row_array):

        row_notnull = pd.notnull(row_array)
        y_weight = np.zeros(row_notnull.shape, dtype=np.int)
        y_weight[row_notnull] = 1
        row_array[row_notnull == False] = 0.0

        return y_weight, row_array

    # def get_effective_inters(self, df_all, least=True, least_threshold=0.5):
    #     """
    #     Convert the dataframe of single origin to sequence data format.
    #
    #     :param df_all: pd.dataframe, average speed from single origin to all destinations.
    #     :param least: bool, whether to keep all the data or not.
    #     :param least_threshold: float, the threshold of the minimum percentage needed.
    #     :return: list, "inter"s that meet the requirement of least threshold
    #     """
    #
    #     nb_rows = df_all.shape[0]
    #     # select the destinations that meet the requirement
    #     columns_selected = self.graph_edges.copy()
    #     columns_selected.append('time')
    #     df_all = df_all.loc[:, columns_selected]
    #     df_all.loc[:, df_all.columns != 'time'] = \
    #         df_all.loc[:, df_all.columns != 'time'].replace([np.inf, -np.inf], np.nan)
    #
    #     data_array = df_all.loc[:, df_all.columns != 'time'].values
    #     print("The shape of data array is ", data_array.shape)
    #     row_notnull = pd.notnull(data_array)
    #     row_keep = np.array([True] * nb_rows)
    #     tmp_row_notnull = row_notnull.copy()
    #     df_all['inter'] = ((df_all['time'].dt.date -
    #                         self.start_date.date()).dt.days * (60 / self.sample_rate) * 24 +
    #                        df_all['time'].dt.hour * (60 / self.sample_rate) +
    #                        df_all['time'].dt.minute / self.sample_rate).astype(int)
    #
    #     # if least if true, a maximum least threshold should be met.
    #     if least:
    #         num_needed = len(self.graph_edges) - \
    #                      int(len(self.graph_edges) * least_threshold)
    #     else:
    #         num_needed = len(self.graph_edges) - self.data_rm[0]
    #     # Since df_all contains one column of 'time', we need num_needed "+ 1" here.
    #     curr_row_keep = tmp_row_notnull.sum(axis=1) >= num_needed
    #     print("The maximum number of records is ", np.max(tmp_row_notnull.sum(axis=1)))
    #     row_keep = np.logical_and(row_keep, curr_row_keep)
    #     df_all = df_all[row_keep]
    #     df_all = df_all.sort_values('inter', ascending=True)
    #     start_inters = []
    #     for i in range(df_all.shape[0] - self.window_size - self.predict_size):
    #         start_ind = int(i)
    #         end_ind = int(i+self.window_size)
    #         train_pred_inters = list(range(start_ind, end_ind))
    #         train_pred_inters.append(int(i+self.window_size+self.predict_size - 1))
    #         inters = df_all.iloc[train_pred_inters]['inter'].tolist()
    #         train_start_inter = inters[0]
    #         train_end_inter = inters[-2]
    #         predict_inter = train_end_inter + self.predict_size
    #         if train_end_inter - train_start_inter + 1 == self.window_size \
    #                 and predict_inter == inters[-1]:
    #             start_inters.append(train_start_inter)
    #
    #     return start_inters

    def obtain_col_row(self, dict_edge_node):

        row_indicator = []
        col_indicator = []
        # construct the indicator Ys for training and validating
        for ind, link_i in enumerate(self.graph_edges):
            if self.name == 'kdd':
                node_y = dict_edge_node['{}_i'.format(link_i)]
                node_x = dict_edge_node['{}_o'.format(link_i)]
                row_indicator.append(node_y)
                col_indicator.append(node_x)
            else:
                node_y = dict_edge_node[link_i.split('-')[0]]
                node_x = dict_edge_node[link_i.split('-')[1]]
                row_indicator.append(node_y)
                col_indicator.append(node_x)

        return row_indicator, col_indicator

    def prepare_lsm_data(self, data_value, v_list=False):

        assert  data_value.ndim == 2 or data_value.ndim == 1
        if data_value.ndim == 1:
            data_value = np.expand_dims(data_value, -1)

        nb_nodes = max(max(self.row_ind), max(self.col_ind)) + 1
        ndim_data_value = data_value.shape[-1]

        list_sp_matrix = []
        validate_sum = 0
        for dim_i  in range(ndim_data_value):
            if v_list:
                dict_list = {}
                for node_i in range(nb_nodes):
                    dict_list[node_i] = [[]] * nb_nodes
                pd_list = pd.DataFrame(data=dict_list)
                for ind, link_i in enumerate(self.graph_edges):
                    row_i = self.row_ind[ind]
                    col_i = self.col_ind[ind]
                    pd_list.at[row_i, col_i] = data_value[ind, dim_i]
                list_sp_matrix.append(pd_list.values)
            else:
                values = []
                for ind, link_i in enumerate(self.graph_edges):
                    values.append(data_value[ind, dim_i])
                mean_matrix = scipy.sparse.csr_matrix(
                    (values, (self.row_ind, self.col_ind)),
                    shape=(nb_nodes, nb_nodes), dtype=np.float32)
                validate_sum += np.sum(values)
                list_sp_matrix.append(mean_matrix)

        # print("data value is ", data_value)
        # print("Sum of Data is ", np.sum(data_value))
        # print("Validate sum is ", validate_sum)
        if not v_list:
            assert np.isclose(np.sum(data_value), validate_sum)

        return list_sp_matrix

    def prepare_lsm_learning_data(self, data, test_label, test_weight,
                                  test_count, mean_y, test_vel_list):
        """
        This function is specifically for preparing avg data transform

        :param data: np.array (BZ, N, B)
        :param test_label: np.array (BZ, N, B)
        :param test_weight: np.array (BZ, N)
        :param test_count: np.array (BZ, N)
        :param dict_edge_node: dictionary = {node: ind}
        :return:
        """

        nb_rows = data.shape[0]

        list_G_ts = []
        list_trainY_ts = []
        list_weight_ts = []
        list_count_ts = []
        list_val_weight_ts = []
        list_vel_list_ts = []
        for row_i in range(nb_rows):
            data_i = data[row_i, ...]
            test_label_i = test_label[row_i, ...]
            test_weight_i = test_weight[row_i]
            count_i = test_count[row_i]
            vel_list_i = test_vel_list[row_i]

            no_data_pos = np.sum(data_i, axis=-1) == 0.
            in_data_ones = np.ones(test_weight_i.shape, dtype=np.int8)
            in_data_ones[no_data_pos] = 0
            data_i[no_data_pos, :] = mean_y[no_data_pos, :]

            # May have several inputs as time varyin
            list_data_i = self.prepare_lsm_data(data_i)
            list_val_weight_i = self.prepare_lsm_data(in_data_ones)
            list_test_label_i = self.prepare_lsm_data(test_label_i)
            list_test_weight_i = self.prepare_lsm_data(test_weight_i)
            list_count_i = self.prepare_lsm_data(count_i)
            list_vel_list_i = self.prepare_lsm_data(vel_list_i, v_list=True)

            list_G_ts.append(list_data_i)
            list_trainY_ts.append(list_test_label_i)
            list_weight_ts.append(list_test_weight_i)
            list_count_ts.append(list_count_i)
            list_val_weight_ts.append(list_val_weight_i)
            list_vel_list_ts.append(list_vel_list_i)

        result_dict = {'Gt': list_G_ts,
                       'Vt': list_val_weight_ts,
                       'Yt': list_trainY_ts,
                       'Wt': list_weight_ts,
                       'Ct': list_count_ts,
                       'Velt': list_vel_list_ts}

        return result_dict


class GPS_Data(DataSet):

    def __init__(self, data_dir, base_dir, server_name, conf_dir, random_node,
                 cat_head, con_head, start_date, test_start_date, test_end_date, trip_table,
                 data_rm_ratio=0.5, source_ratio=0.5, topk=1000,
                 sample_rate=15, window_size=20, predict_size=4,
                 small_threshold=0.0, big_threshold=50.0, dist=1.,
                 min_nb=5, unit=1, combo_random=True, num_combos=3,
                 custom_rect=None, temporal_dim=1, period_dim=1):

        self.name = 'gps'
        self.topk = topk
        self.dist = dist
        self.custom_rect = custom_rect
        self.trip_table = trip_table

        super().__init__(data_dir, base_dir, server_name, conf_dir, random_node,
                         cat_head, con_head, start_date, test_start_date, test_end_date, trip_table,
                         data_rm_ratio,
                         source_ratio, sample_rate, window_size, predict_size,
                         small_threshold, big_threshold, min_nb, unit,
                         combo_random, num_combos,temporal_dim, period_dim)


    def _get_graph_file(self):
        """
        Construct graph related file

        :return: directories to store graph related file
        """

        edge_adj_dir = os.path.join(self.base_dir)
        edge_adj_file = os.path.join(edge_adj_dir,
                                     'edge_adj.pickle')
        edges_file = os.path.join(edge_adj_dir,
                                  'edges.pickle')

        try:
            os.stat(edge_adj_dir)
        except:
            os.makedirs(edge_adj_dir)

        return edge_adj_file, edges_file

    def _construct_dict_edge_node(self):

        dict_edge_node = {}
        node_id = 0
        for index, edge in enumerate(self.graph_edges):
            link_id_left = edge.split('-')[0]
            if link_id_left not in dict_edge_node.keys():
                dict_edge_node[link_id_left] = node_id
                node_id += 1
            link_id_right = edge.split('-')[1]
            if link_id_right not in dict_edge_node.keys():
                dict_edge_node[link_id_right] = node_id
                node_id += 1

        return dict_edge_node

    def _construct_DiGraph(self, dict_edge_node):

        di_graph = nx.DiGraph()
        for link_id in self.graph_edges:
            left_node = link_id.split('-')[0]
            right_node = link_id.split('-')[1]
            left_node_id = dict_edge_node[left_node]
            right_node_id = dict_edge_node[right_node]
            di_graph.add_edge(left_node_id, right_node_id)
        return di_graph

    def _get_edge_connection(self):
        """
        Get a edge connection dictionary with the topk frequent edges

        :param topk: int, top k frequent edges
        :return: dictionary
        """

        conn = self.engine.connect()
        sql_all_usable_edge = 'SELECT COUNT(travel_speed), seg_id FROM '+self.trip_table+' ' \
                              'WHERE travel_speed > 0 ' \
                              'GROUP BY seg_id ' \
                              'ORDER BY COUNT(travel_speed) DESC ' \
                              'LIMIT {}'.format(self.topk)

        df_frequent_edge = pd.read_sql(sql_all_usable_edge, conn)
        # df_frequent_edge = self._custom_rect_edges(df_frequent_edge, self.custom_rect)
        edge_graph_dict = self._get_edge_connection_dict(df_frequent_edge)
        conn.close()

        return edge_graph_dict

    # def _custom_rect_edges(self, df, custom_rect):
    #     """
    #     Get seg_ids within a given rectangle
    #
    #     :param df: Dataframe, with certain seg_ids
    #     :param custom_rect: dict, positions
    #     :return: Dataframe
    #     """
    #     nodes = df['seg_id'].tolist()
    #
    #     max_x = custom_rect['max_x']
    #     min_x = custom_rect['min_x']
    #     max_y = custom_rect['max_y']
    #     min_y = custom_rect['min_y']
    #
    #     df_within = self._get_rectangle_segids(max_x, max_y, min_x, min_y)
    #     nodes_within = df_within['seg_id'].tolist()
    #     df_segid_intersect = pd.DataFrame()
    #     df_segid_intersect['seg_id'] = list(set(nodes).intersection(nodes_within))
    #
    #     return df_segid_intersect

    def _get_edge_connection_dict(self, df):
        directed_dict = {}
        df = df.drop_duplicates('seg_id', keep='first')
        df_frequent_edge = df.join(
            df.seg_id.str.split('-', expand=True).rename(
                columns={0: 'start_vertex', 1: 'end_vertex'}))

        for index, row in df_frequent_edge.iterrows():
            seg_id = row['seg_id']
            end_vertex = row['end_vertex']
            connected_segs = df_frequent_edge[
                df_frequent_edge['start_vertex'] == end_vertex]
            if seg_id not in directed_dict.keys():
                directed_dict[seg_id] = connected_segs['seg_id'].tolist()
            else:
                directed_dict[seg_id] += connected_segs['seg_id'].tolist()

            start_vertex = row['start_vertex']
            connected_segs = df_frequent_edge[df_frequent_edge['end_vertex']
                                              == start_vertex]
            directed_dict[seg_id] += connected_segs['seg_id'].tolist()

        return directed_dict

    # def _get_gps_boundary(self, nodes):
    #     """Get outer boundary of those road segment
    #
    #     :param nodes:
    #     :return:
    #     """
    #     conn = self.engine.connect()
    #     opposite_nodes = [
    #         '{0}-{1}'.format(x.split('-')[1], x.split('-')[0]) for x in nodes]
    #     di_nodes = nodes + opposite_nodes
    #     sql = "SELECT * from new_vertices_polyline WHERE seg_id IN {}".format(
    #         tuple(di_nodes))
    #     # df_nodes_lon_lat = conn.execute(text(sql), nodes=nodes)
    #     # sql = 'SELECT * FROM new_vertices_polyline'
    #     df_trips_all = pd.read_sql(sql, conn)
    #     conn.close()
    #
    #     df_nodes_lon_lat = df_trips_all[df_trips_all['seg_id'].isin(di_nodes)]
    #     max_x = df_nodes_lon_lat['xpos'].max()
    #     min_x = df_nodes_lon_lat['xpos'].min()
    #     max_y = df_nodes_lon_lat['ypos'].max()
    #     min_y = df_nodes_lon_lat['ypos'].min()
    #     max_x = (math.ceil(100 * max_x) / 100)
    #     max_y = (math.ceil(100 * max_y) / 100)
    #     min_x = (math.floor(100 * min_x) / 100)
    #     min_y = (math.floor(100 * min_y) / 100)
    #
    #     return max_x, max_y, min_x, min_y

    # def _get_rect_edge_graph(self, max_x, max_y, min_x, min_y):
    #     """
    #     Get the largest connected edge graph within the rectangle
    #
    #     :param max_x: float, max x position
    #     :param max_y: float, max y position
    #     :param min_x: float, min x position
    #     :param min_y: float, min y position
    #     :return: Networkx graph, the largest connected edge graph
    #     """
    #
    #     df_matched = self._get_rectangle_segids(max_x, max_y, min_x, min_y)
    #
    #     df_all = self._construct_undirected_df(df_matched)
    #     eg_dict = self._get_edge_connection_dict(df_all)
    #     di_graph = nx.from_dict_of_lists(eg_dict)
    #     undi_graph = di_graph.to_undirected()
    #     Gcc = sorted(nx.connected_component_subgraphs(undi_graph), key=len, reverse=True)
    #     G0 = Gcc[0]
    #
    #     return G0

    # def _get_rectangle_segids(self, max_x, max_y, min_x, min_y):
    #     """
    #     Get the seg_ids within a given rectangle
    #
    #     :param max_x: float, max x position
    #     :param max_y: float, max y position
    #     :param min_x: float, min x position
    #     :param min_y: float, min y position
    #     :return: Networkx graph, the largest connected edge graph
    #     """
    #
    #     conn = self.engine.connect()
    #     sql_match = 'SELECT seg_id from new_vertices_polyline ' \
    #                 'where xpos >= {0} and xpos <= {1} ' \
    #                 'and ypos >= {2} and ypos <= {3}'.format(
    #         min_x, max_x, min_y, max_y)
    #     df_matched = pd.read_sql(sql_match, conn)
    #     conn.close()
    #
    #     return df_matched

    # def _get_boundary_graph(self, nodes):
    #
    #     max_x, max_y, min_x, min_y = self._get_gps_boundary(nodes)
    #
    #     max_sub_graph = self._get_rect_edge_graph(max_x, max_y, min_x, min_y)
    #
    #     max_nodes = max_sub_graph.nodes()
    #     nodes_intersect = set(nodes).intersection(set(max_nodes))
    #
    #     return max_sub_graph, list(nodes_intersect)

    def _construct_link_vel_list(self):
        """
        Construct an average travel time of nodes with certain sample rate.

        :param sample_freq: int, minutes.
        :param write_db: bool, write the constructed results into database or not/
        :return: Dataframe, table contains the avg travel time of all nodes.
        """
        conn = self.engine.connect()
        sample_str = '{}T'.format(self.sample_rate)

        print("Reading needed records for trips from Database...")
        link_sql = "select start_time, seg_id, travel_speed from "+self.trip_table+"" \
                   " where travel_speed > 0 AND seg_id IN {}".format(
                       tuple(self.graph_edges))
        df_link = pd.read_sql(link_sql, conn, parse_dates=['start_time'])

        df_link['travel_speed'] = df_link['travel_speed'].astype(float)
        df_link['time'] = pd.to_datetime(df_link['start_time'])
        del df_link['start_time']
        print("Try to get solid velocity of links at sample rate {} minutes"
              "".format(self.sample_rate))

        df_link = df_link.set_index('time')
        df_link_gb = df_link.groupby('seg_id')

        # check the equality of graph edges and queried edges
        assert len(df_link_gb.groups) == len(self.graph_edges)

        # load the edge length from db
        graph_edges_reverse = [
            '{}-{}'.format(e.split('-')[1], e.split('-')[0]) for e in self.graph_edges]
        graph_edges_reverse += self.graph_edges
        sql_link_dis = 'select seg_id, len from '+self.trip_table+'' \
                       ' where seg_id IN {}'.format(tuple(graph_edges_reverse))
        df_link_dis = pd.read_sql(sql_link_dis, conn)

        list_dfs = []
        for link_name, df_link_i in df_link_gb:
            # Get the length of the current seg_id
            node_dis = df_link_dis.loc[
                df_link_dis['seg_id'] == link_name, 'len']
            if node_dis.empty:
                change_s_t = '{0}-{1}'.format(link_name.split('-')
                                              [1], link_name.split('-')[0])
                node_dis = df_link_dis.loc[df_link_dis[
                    'seg_id'] == change_s_t, 'len']
            link_len = node_dis.iloc[0] * self.unit

            linki_tt = pd.DataFrame()
            linki_resample = df_link_i.travel_speed.resample(
                sample_str).apply(vel_list, edge_len=link_len)
            linki_tt[link_name] = linki_resample
            list_dfs.append(linki_tt)

        df_link_tb = pd.concat(list_dfs, axis=1, join='outer')

        df_link_tb = df_link_tb.reset_index(level=['time'])

        conn.close()

        return df_link_tb

    def _get_directed_dict_from_list(self, nodes):

        directed_dict = {}
        nodes = np.array(list(nodes))
        start_vertices = np.array([node.split('-')[0] for node in nodes])
        end_vertices = np.array([node.split('-')[1] for node in nodes])

        for index, node in enumerate(nodes):
            seg_id = node
            end_vertex = end_vertices[index]
            connected_ind = np.where(start_vertices == end_vertex)
            if seg_id not in directed_dict.keys():
                directed_dict[seg_id] = nodes[connected_ind]
            else:
                directed_dict[seg_id] += nodes[connected_ind]

        return directed_dict

    def _convert_edge_graph(self, directed_dict, random_node=True):
        # get a directed graph and convert it into undirected, why?
        # 1. In paper "Convolutional Neural Networks on Graphs with Fast
        # Localized Spectral Filtering", it deals with undirected and connected graph G.
        # 2. We just consider the spatial relation of these links, direction
        # has not been utilized yet.

        edge_graph = nx.from_dict_of_lists(directed_dict)
        adj = nx.adjacency_matrix(edge_graph)
        # check the resulted graph is undirected graph
        assert (adj != adj.T).nnz == 0
        undi_graph = edge_graph.to_undirected()

        # get connected subgraphs
        graphs = list(nx.connected_component_subgraphs(undi_graph))
        nb_nodes = [len(graph_i.nodes()) for graph_i in graphs]

        max_nb = max(nb_nodes)
        max_index = nb_nodes.index(max_nb)
        max_sub_graph = graphs[max_index]

        # get the adjacency matrix of the largest sub graph
        nodes = max_sub_graph.nodes()
        adj = nx.adjacency_matrix(max_sub_graph)
        adj.setdiag(1)
        # adj.setdiag(0)
        adj = adj.astype(np.float64)
        if random_node:
            adj = adj.todense()
            rnd_order = np.random.permutation(len(nodes))
            adj = adj[rnd_order, :]
            adj = adj[:, rnd_order]
            nodes = list(np.array(nodes)[rnd_order])
            # convert into a sparse format
            adj = scipy.sparse.csr_matrix(adj)
        adj.eliminate_zeros()
        nodes = [str(node) for node in nodes]

        return adj, nodes

    def _construct_undirected_df(self, df):

        df = df.drop_duplicates('seg_id', keep='first')
        df = df.join(
            df.seg_id.str.split('-', expand=True).rename(
                columns={0: 'start_vertex', 1: 'end_vertex'}))
        df_copy = df.copy()
        del df_copy['seg_id']

        df_copy['seg_id'] = df[['end_vertex', 'start_vertex']]. \
            apply(lambda x: '-'.join(x), axis=1)
        df_merge = pd.concat([df, df_copy])

        print("Before intersect with distinct edges, ", df_merge.shape)
        unique_ids = self._get_distinct_segids()
        df_merge = df_merge[df_merge.seg_id.isin(unique_ids)]
        print("After intersect with distinct edges, ", df_merge.shape)

        del df_merge['end_vertex'], df_merge['start_vertex']

        return df_merge

    def _get_distinct_segids(self):
        """
        Get all the distinct seg_ids.

        :return: list, distinct seg_ids
        """

        print("Entering into querying distinct edges...")
        # sql unique seg_ids
        conn = self.engine.connect()
        sql_unique_segids = "SELECT DISTINCT seg_id FROM "+self.trip_table+""
        df_unique_segids = pd.read_sql(sql_unique_segids, conn)
        conn.close()
        unique_segids = df_unique_segids['seg_id'].tolist()

        return unique_segids

    # def _construct_combined_nodes_added(self, graphs, total_nodes, dist=1.):
    #     """Construct combined graphs with added edges
    #
    #     :param graphs: list, connected graphs with top_k edges
    #     :param total_nodes: list, top_k edges
    #     :param dist: float, distance threshold
    #     :return: list, graphs with the biggest connected nodes
    #     """
    #
    #     outer_graph, nodes_intersects = \
    #         self._get_boundary_graph(total_nodes)
    #
    #     outer_nodes = outer_graph.nodes()
    #     print("The dist threshold is ", dist)
    #     print("The total number of outer nodes is ", len(outer_nodes))
    #     print("The total nodes in outer graph is {0}/{1}".format(
    #         len(nodes_intersects), len(total_nodes)))
    #
    #     max_nodes = -1
    #     num_graphs = len(graphs)
    #     print("number of subgraph is ", num_graphs)
    #     nodes_list = []
    #     for i in range(num_graphs):
    #         nodes_i = graphs[i].nodes()
    #         nodes_i = list(set(nodes_i).intersection(outer_nodes))
    #         if len(nodes_i) < 1:
    #             continue
    #         nodes_set = set()
    #         nodes_set = nodes_set.union(set(nodes_i))
    #         for j in range(num_graphs):
    #             nodes_j = graphs[j].nodes()
    #             nodes_j = list(set(nodes_j).intersection(outer_nodes))
    #             if len(nodes_j) < 1:
    #                 continue
    #             nodes_add = self._get_shortest_path_nodes(
    #                 outer_graph, nodes_i, nodes_j, total_nodes, dist)
    #             nodes_set = nodes_set.union(nodes_add)
    #         if len(nodes_set) > max_nodes:
    #             nodes_list = list(nodes_set)
    #             max_nodes = len(nodes_set)
    #             print("Max nodes is {0}, for graph {1}".format(max_nodes, i))
    #
    #     print("The number of nodes is ", len(nodes_list))
    #     connected_graph = outer_graph.subgraph(nodes_list)
    #
    #     return connected_graph, nodes_list

    def _get_shortest_path_nodes(self, graph, nodes_i,
                                nodes_j, total_nodes, dist=1.):
        """
        Get the nodes in shortest paths

        :param graph: networkx, graph
        :param nodes_i: list, Starting nodes for path
        :param nodes_j: list, Ending nodes for path
        :param total_nodes: list,
        :param dist: float, shortest paths with certain distance
        :return: set, set of added nodes
        """

        # print("The size of the graph is ", len(graph.nodes()))
        # print("nodes_j is ", nodes_j)
        # print("intersection of nodes is ", set(nodes_j).intersection(graph.nodes()))
        # print("number of nodes_j is ", len(nodes_j))
        # print("number of graph nodes is ", len(set(nodes_j).intersection(graph.nodes())))

        # assert len(nodes_i) == len(set(nodes_i).intersection(graph.nodes()))
        # assert len(nodes_j) == len(set(nodes_j).intersection(graph.nodes()))

        node_added = set()
        for node_i in nodes_i:
            if node_i not in graph.nodes():
                continue
            else:
                dict_len_nodes = nx.single_source_shortest_path_length(graph, node_i, dist)
                all_nodes = dict_len_nodes.keys()
                diff_all_nodes = set(all_nodes)
                common_nodes = list(diff_all_nodes.intersection(nodes_j))
                sub_graph = graph.subgraph(diff_all_nodes)
                node_added_i = self._get_optimal_added_nodes(
                    sub_graph, node_i, common_nodes, total_nodes)
                node_added = node_added.union(node_added_i)

        return node_added

    def _get_optimal_added_nodes(self, sub_graph, node_i,
                                 target_nodes, total_nodes):
        if len(target_nodes) < 1:
            return set()
        else:
            node_added = set()
            for node_j in target_nodes:
                paths_i_j = nx.all_shortest_paths(sub_graph, node_i, node_j)
                selected_path = list()
                max_matched = 2
                paths_i_j = [p for p in paths_i_j]
                for path_i_j in paths_i_j:
                    num_matched = len(set(path_i_j).intersection(total_nodes))
                    if num_matched > max_matched:
                        selected_path = path_i_j.copy()
                if max_matched == 2:
                    selected_path = paths_i_j[0]
                node_added = node_added.union(set(selected_path))

            return node_added

    # def _combine_gps_boundary(self, rect_i, rect_j, threshold=3.):
    #     """
    #     combine two boundaries into one
    #
    #     :param rect_i: tuple, first rectangle (max_x, max_y, min_x, min_y)
    #     :param rect_j: tuple, second rectangle (max_x, max_y, min_x, min_y)
    #     :param threshold: float, distance between two gps points.
    #     :return:
    #     """
    #
    #     center_i = ((rect_i[1] + rect_i[3]) / 2, (rect_i[0] + rect_i[2]) / 2)
    #     center_j = ((rect_j[1] + rect_j[3]) / 2, (rect_j[0] + rect_j[2]) / 2)
    #     distance = vincenty(center_i, center_j).km
    #     # print("The distance between two centers is {} km".format(distance))
    #     if distance > threshold:
    #         return None
    #
    #     max_x = max(rect_i[0], rect_j[0])
    #     max_y = max(rect_i[1], rect_j[1])
    #     min_x = min(rect_i[2], rect_j[2])
    #     min_y = min(rect_i[3], rect_j[3])
    #
    #     biggest_graph = self._get_rect_edge_graph(max_x, max_y, min_x, min_y)
    #
    #     return biggest_graph

    # def _convert_tt_vel(self, df_link_tt):
    #     """
    #     Convert the travel time into travel speed.
    #
    #     :param df_link_tt: Dataframe, table contains the avg travel time of all nodes.
    #     :return: Dataframe, table contain the avg travel speed of all nodes.
    #     """
    #     conn = self.engine.connect()
    #     # load the edge length from db
    #     graph_edges_reverse = [
    #         '{}-{}'.format(e.split('-')[1], e.split('-')[0]) for e in self.graph_edges]
    #     graph_edges_reverse += self.graph_edges
    #     sql_link_dis = 'select seg_id, len from trips' \
    #                    ' where seg_id IN {}'.format(tuple(graph_edges_reverse))
    #     df_link_dis = pd.read_sql(sql_link_dis, conn)
    #
    #     link_dis = []
    #     for node in self.graph_edges:
    #         node_dis = df_link_dis.loc[df_link_dis['seg_id'] == node, 'len']
    #         if node_dis.empty:
    #             change_s_t = '{0}-{1}'.format(node.split('-')
    #                                           [1], node.split('-')[0])
    #             node_dis = df_link_dis.loc[df_link_dis[
    #                 'seg_id'] == change_s_t, 'len']
    #         node_dis = node_dis.iloc[0]
    #         link_dis.append(node_dis)
    #     link_dis = np.array(link_dis)
    #     nodes = [str(node) for node in self.graph_edges]
    #     col_name = list(df_link_tt.columns.values)
    #     diff_nodes1 = set(nodes).difference(col_name)
    #     diff_dict = dict.fromkeys(diff_nodes1, np.nan)
    #     df_link_tt = df_link_tt.assign(**diff_dict)
    #
    #     # verify that all graph edges are in the dataframe
    #     col_name = list(df_link_tt.columns.values)
    #     diff_nodes = set(nodes).difference(col_name)
    #     if len(diff_nodes) > 0:
    #         print("The dataframe is not correct... \n"
    #               "the following seg_id is not included!")
    #         print(diff_nodes)
    #         raise ValueError
    #
    #     # this step is necessary to ensure the order is correct
    #     # since the link_dis is in this order.
    #     df_link_tt[self.graph_edges] = df_link_tt[self.graph_edges]
    #     df_link_tt[nodes] = 1.0 / (df_link_tt[nodes].values) * link_dis * 1000
    #     conn.close()
    #     # if the velocity is too small, make it to be nan
    #     df_link_tt[nodes] = df_link_tt[nodes].apply(
    #         apply_small_num, axis=1, args=[nodes, self.small_threshold])
    #
    #     return df_link_tt

    def _construct_link_vel_hist(self, hist_range):
        """
        Construct a table with average travel time of a link

        :param hist_range: The pre-defined histogram bins

        :return:
        """

        sample_str = '{}T'.format(self.sample_rate)
        conn = self.engine.connect()
        link_sql = "select start_time, seg_id, travel_speed from "+self.trip_table+"" \
                   " where seg_id IN {} and travel_speed > 0".format(
                       tuple(self.graph_edges))
        df_link = pd.read_sql(link_sql, conn, parse_dates=['start_time'])
        df_link['travel_speed'] = df_link['travel_speed'].astype(float)
        df_link['time'] = pd.to_datetime(df_link['start_time'])
        del df_link['start_time']
        print("Try to get the links' historical travel times at sample rate {}".format(
            sample_str))

        df_link = df_link.set_index('time')
        df_link_gb = df_link.groupby('seg_id')

        # load the edge length from db
        graph_edges_reverse = [
            '{}-{}'.format(e.split('-')[1], e.split('-')[0]) for e in self.graph_edges]
        graph_edges_reverse += self.graph_edges
        sql_link_dis = 'select seg_id, len from '+self.trip_table+'' \
                       ' where seg_id IN {}'.format(tuple(graph_edges_reverse))
        df_link_dis = pd.read_sql(sql_link_dis, conn)

        def get_vel_hist(array_like, link_len, hist_bin):
            if len(array_like) == 0:
                return np.nan
            tt_array = np.array(array_like)
            # tt_array = link_len / tt_array
            hist, bin_edges = np.histogram(tt_array, hist_bin, density=True)
            if np.isnan(hist).any():
                print('nan hist returned!')
                print('data', tt_array)
                print(hist)
                return np.nan
            # hist *= hist_bin[1] - hist_bin[0]
            return hist

        list_dfs = []
        link_names = []
        for link_name, df_link_i in df_link_gb:
            # Get the length of the current seg_id
            node_dis = df_link_dis.loc[
                df_link_dis['seg_id'] == link_name, 'len']
            if node_dis.empty:
                change_s_t = '{0}-{1}'.format(link_name.split('-')
                                              [1], link_name.split('-')[0])
                node_dis = df_link_dis.loc[df_link_dis[
                    'seg_id'] == change_s_t, 'len']
            link_len = node_dis.iloc[0]

            linki_vel_hist = pd.DataFrame()
            linki_vel_hist[link_name] = df_link_i.travel_speed.resample(
                sample_str).apply(vel_list)
            linki_vel_hist[link_name] = linki_vel_hist[link_name].apply(
                get_vel_hist, args=(link_len, hist_range))
            link_names.append(str(link_name))
            list_dfs.append(linki_vel_hist)
        df_link_tb = pd.concat(list_dfs, axis=1, join='outer')

        nodes = [str(node) for node in self.graph_edges]
        col_name = list(df_link_tb.columns.values)
        diff_nodes1 = set(nodes).difference(col_name)
        diff_dict = dict.fromkeys(diff_nodes1, np.nan)
        df_link_tb = df_link_tb.assign(**diff_dict)
        df_link_tb[self.graph_edges] = df_link_tb[self.graph_edges]

        df_link_tb = df_link_tb.reset_index(level=['time'])
        df_link_tb.rename(columns=lambda x: str(x), inplace=True)

        conn.close()

        return df_link_tb

    def _construct_overlap_link_vel_hist(self, hist_range):
        """
        Construct a table with average travel time of a link

        :param hist_range: The pre-defined histogram bins

        :return:
        """

        sample_str = '{}T'.format(self.sample_rate)
        conn = self.engine.connect()
        link_sql = "select start_time, seg_id, travel_speed from "+self.trip_table+"" \
                   " where seg_id IN {} and travel_speed > 0".format(
                       tuple(self.graph_edges))
        df_link = pd.read_sql(link_sql, conn, parse_dates=['start_time'])
        df_link['travel_speed'] = df_link['travel_speed'].astype(float)
        df_link['time'] = pd.to_datetime(df_link['start_time'])
        del df_link['start_time']
        print("Try to get the links' historical travel times at sample rate {}".format(
            sample_str))

        df_link = df_link.set_index('time')
        df_link_gb = df_link.groupby('seg_id')

        # load the edge length from db
        graph_edges_reverse = [
            '{}-{}'.format(e.split('-')[1], e.split('-')[0]) for e in self.graph_edges]
        graph_edges_reverse += self.graph_edges
        sql_link_dis = 'select seg_id, len from '+self.trip_table+'' \
                       ' where seg_id IN {}'.format(tuple(graph_edges_reverse))
        df_link_dis = pd.read_sql(sql_link_dis, conn)

        list_dfs = []
        link_names = []
        for link_name, df_link_i in df_link_gb:
            # Get the length of the current seg_id
            node_dis = df_link_dis.loc[
                df_link_dis['seg_id'] == link_name, 'len']
            if node_dis.empty:
                change_s_t = '{0}-{1}'.format(link_name.split('-')
                                              [1], link_name.split('-')[0])
                node_dis = df_link_dis.loc[df_link_dis[
                    'seg_id'] == change_s_t, 'len']
            link_len = node_dis.iloc[0]
            linki_vel_hist = pd.DataFrame()
            linki_resample = df_link_i.travel_speed.resample(
                sample_str).apply(vel_list)

            # linki_vel_hist[link_name] = linki_resample.rolling(window=self.window_size).apply(
            #     get_vel_hist_rolling, args=(link_len, hist_range))
            linki_vel_hist[link_name] = my_rolling_apply_list(linki_resample,
                                                              self.get_vel_hist_rolling,
                                                              link_len, hist_range)

            link_names.append(str(link_name))
            list_dfs.append(linki_vel_hist)
        df_link_tb = pd.concat(list_dfs, axis=1, join='outer')

        nodes = [str(node) for node in self.graph_edges]
        col_name = list(df_link_tb.columns.values)
        diff_nodes1 = set(nodes).difference(col_name)
        diff_dict = dict.fromkeys(diff_nodes1, np.nan)
        df_link_tb = df_link_tb.assign(**diff_dict)
        df_link_tb[self.graph_edges] = df_link_tb[self.graph_edges]

        df_link_tb = df_link_tb.reset_index(level=['time'])
        df_link_tb.rename(columns=lambda x: str(x), inplace=True)

        conn.close()

        return df_link_tb

    # def construct_node_graph(self):
    #     """
    #     convert a graph from edge connections to node connection
    #
    #     :param random_node: bool, whether to random shuffle the data
    #     :param engine: sqlalchemy sql server engine
    #     :return:
    #         dict_edge_node, a dictionary contains the mapping of edge node to real node
    #         link_ids, list of link ids used in this road network
    #         W, Adjacency matrix of the resulted graph
    #         L, Laplacian matrix of the resulted graph
    #         D, Degree matrix of the resulted graph
    #     """
    #
    #     path_directed_dict = os.path.join(
    #         self.data_dir, 'dict_edge_node.pickle')
    #     path_links = os.path.join(self.data_dir, 'links.pickle')
    #     if not os.path.exists(path_directed_dict) or not os.path.exists(path_links):
    #         link_ids = self.graph_edges
    #         dict_edge_node = {}
    #         node_id = 0
    #         if self.random_node:
    #             self.graph_edges = np.random.permutation(self.graph_edges)
    #         for index, edge in enumerate(self.graph_edges):
    #
    #             link_id_left = edge.split('-')[0]
    #             if link_id_left not in dict_edge_node.keys():
    #                 dict_edge_node[link_id_left] = node_id
    #                 node_id += 1
    #             link_id_right = edge.split('-')[1]
    #             if link_id_right not in dict_edge_node.keys():
    #                 dict_edge_node[link_id_right] = node_id
    #                 node_id += 1
    #
    #         with open(path_directed_dict, 'wb') as f_di_dict:
    #             pkl.dump(dict_edge_node, f_di_dict)
    #
    #         with open(path_links, 'wb') as f_links:
    #             pkl.dump(link_ids, f_links)
    #     else:
    #         with open(path_directed_dict, 'rb') as f_di_dict:
    #             dict_edge_node = pkl.load(f_di_dict)
    #         with open(path_links, 'rb') as f_links:
    #             link_ids = pkl.load(f_links)
    #
    #     self.row_ind, self.col_ind = self.obtain_col_row(dict_edge_node)
    #     # get a directed graph and convert it into undirected
    #     di_graph = nx.DiGraph()
    #     for link_id in self.graph_edges:
    #         left_node = link_id.split('-')[0]
    #         right_node = link_id.split('-')[1]
    #         left_node_id = dict_edge_node[left_node]
    #         right_node_id = dict_edge_node[right_node]
    #         di_graph.add_edge(left_node_id, right_node_id)
    #
    #     nodes = di_graph.nodes()
    #
    #     A = nx.adjacency_matrix(di_graph, nodes)
    #     # convert adjacency to proximity matrix
    #     A = A.todense()
    #     # W = self.get_hop_proximity_matrix(di_graph, hop=2)
    #     W = self.get_proximity_matrix(A)
    #     W_row_sum = W.sum(axis=1)
    #     D = np.diag(W_row_sum)
    #     # W_row_sum = [sum(W.sum(axis=1).tolist(), [])]
    #     # D = scipy.sparse.diags(W_row_sum, [0])
    #     L = D - W
    #
    #     return dict_edge_node, link_ids, W, L, D


class mtr_Data(DataSet):

    def __init__(self, data_dir, base_dir, server_name, conf_dir, random_node,
                 cat_head, con_head, start_date, test_start_date, test_end_date, trip_table,
                 data_rm_ratio=0.5, source_ratio=0.5, topk=1000,
                 sample_rate=15, window_size=20, predict_size=4,
                 small_threshold=0.0, big_threshold=50.0, dist=1.,
                 min_nb=5, unit=1, combo_random=True, num_combos=3,
                 custom_rect=None, temporal_dim=1, period_dim=1):

        self.name = 'mtr'
        self.topk = topk
        self.dist = dist
        self.custom_rect = custom_rect
        self.trip_table = trip_table

        super().__init__(data_dir, base_dir, server_name, conf_dir, random_node,
                         cat_head, con_head, start_date, test_start_date, test_end_date, trip_table,
                         data_rm_ratio,
                         source_ratio, sample_rate, window_size, predict_size,
                         small_threshold, big_threshold, min_nb, unit,
                         combo_random, num_combos, temporal_dim, period_dim)


    def _get_graph_file(self):
        """
        Construct graph related file

        :return: directories to store graph related file
        """

        edge_adj_dir = os.path.join(self.base_dir)
        edge_adj_file = os.path.join(edge_adj_dir,
                                     'edge_adj.pickle')
        edges_file = os.path.join(edge_adj_dir,
                                  'edges.pickle')

        try:
            os.stat(edge_adj_dir)
        except:
            os.makedirs(edge_adj_dir)

        return edge_adj_file, edges_file

    def _construct_dict_edge_node(self):

        dict_edge_node = {}
        node_id = 0
        for index, edge in enumerate(self.graph_edges):
            link_id_left = edge.split('-')[0]
            if link_id_left not in dict_edge_node.keys():
                dict_edge_node[link_id_left] = node_id
                node_id += 1
            link_id_right = edge.split('-')[1]
            if link_id_right not in dict_edge_node.keys():
                dict_edge_node[link_id_right] = node_id
                node_id += 1

        return dict_edge_node

    def _construct_DiGraph(self, dict_edge_node):

        di_graph = nx.DiGraph()
        for link_id in self.graph_edges:
            left_node = link_id.split('-')[0]
            right_node = link_id.split('-')[1]
            left_node_id = dict_edge_node[left_node]
            right_node_id = dict_edge_node[right_node]
            di_graph.add_edge(left_node_id, right_node_id)
        return di_graph

    def _get_edge_connection(self):
        """
        Get a edge connection dictionary with the topk frequent edges

        :param topk: int, top k frequent edges
        :return: dictionary
        """

        conn = self.engine.connect()
        sql_all_usable_edge = 'SELECT COUNT(s2s_speed), s2s FROM '+self.trip_table+' ' \
                              'WHERE s2s_speed > 0 ' \
                              'GROUP BY s2s ' \
                              'ORDER BY COUNT(s2s_speed) DESC ' \
                              'LIMIT {}'.format(self.topk)

        df_frequent_edge = pd.read_sql(sql_all_usable_edge, conn)
        # df_frequent_edge = self._custom_rect_edges(df_frequent_edge, self.custom_rect)
        edge_graph_dict = self._get_edge_connection_dict(df_frequent_edge)
        conn.close()

        return edge_graph_dict

    def _get_edge_connection_dict(self, df):
        directed_dict = {}
        df = df.drop_duplicates('s2s', keep='first')
        df_frequent_edge = df.join(
            df.s2s.str.split('-', expand=True).rename(
                columns={0: 'start_vertex', 1: 'end_vertex'}))

        for index, row in df_frequent_edge.iterrows():
            s2s = row['s2s']
            end_vertex = row['end_vertex']
            connected_segs = df_frequent_edge[
                df_frequent_edge['start_vertex'] == end_vertex]
            if s2s not in directed_dict.keys():
                directed_dict[s2s] = connected_segs['s2s'].tolist()
            else:
                directed_dict[s2s] += connected_segs['s2s'].tolist()

            start_vertex = row['start_vertex']
            connected_segs = df_frequent_edge[df_frequent_edge['end_vertex']
                                              == start_vertex]
            directed_dict[s2s] += connected_segs['s2s'].tolist()

        return directed_dict

    def _construct_link_vel_list(self):
        """
        Construct an average travel time of nodes with certain sample rate.

        :param sample_freq: int, minutes.
        :param write_db: bool, write the constructed results into database or not/
        :return: Dataframe, table contains the avg travel time of all nodes.
        """
        conn = self.engine.connect()
        sample_str = '{}T'.format(self.sample_rate)

        print("Reading needed records for trips from Database...")
        link_sql = "select act_dep_time, s2s, s2s_speed from "+self.trip_table+"" \
                   " where s2s_speed > 0 AND s2s IN {}".format(
                       tuple(self.graph_edges))
        df_link = pd.read_sql(link_sql, conn, parse_dates=['act_dep_time'])

        df_link['s2s_speed'] = df_link['s2s_speed'].astype(float)
        df_link['time'] = pd.to_datetime(df_link['act_dep_time'])
        del df_link['act_dep_time']
        print("Try to get solid velocity of links at sample rate {} minutes"
              "".format(self.sample_rate))

        df_link = df_link.set_index('time')
        df_link_gb = df_link.groupby('s2s')

        # check the equality of graph edges and queried edges
        assert len(df_link_gb.groups) == len(self.graph_edges)

        # load the edge length from db
        graph_edges_reverse = [
            '{}-{}'.format(e.split('-')[1], e.split('-')[0]) for e in self.graph_edges]
        graph_edges_reverse += self.graph_edges
        sql_link_dis = 'select s2s, s2s_len from '+self.trip_table+'' \
                       ' where s2s IN {}'.format(tuple(graph_edges_reverse))
        df_link_dis = pd.read_sql(sql_link_dis, conn)

        list_dfs = []
        for link_name, df_link_i in df_link_gb:
            # Get the length of the current s2s
            node_dis = df_link_dis.loc[
                df_link_dis['s2s'] == link_name, 's2s_len']
            if node_dis.empty:
                change_s_t = '{0}-{1}'.format(link_name.split('-')
                                              [1], link_name.split('-')[0])
                node_dis = df_link_dis.loc[df_link_dis[
                    's2s'] == change_s_t, 's2s_len']
            link_len = node_dis.iloc[0] * self.unit

            linki_tt = pd.DataFrame()
            linki_resample = df_link_i.s2s_speed.resample(
                sample_str).apply(vel_list, edge_len=link_len)
            linki_tt[link_name] = linki_resample
            list_dfs.append(linki_tt)

        df_link_tb = pd.concat(list_dfs, axis=1, join='outer')

        df_link_tb = df_link_tb.reset_index(level=['time'])

        conn.close()

        return df_link_tb

    def _get_directed_dict_from_list(self, nodes):

        directed_dict = {}
        nodes = np.array(list(nodes))
        start_vertices = np.array([node.split('-')[0] for node in nodes])
        end_vertices = np.array([node.split('-')[1] for node in nodes])

        for index, node in enumerate(nodes):
            s2s = node
            end_vertex = end_vertices[index]
            connected_ind = np.where(start_vertices == end_vertex)
            if s2s not in directed_dict.keys():
                directed_dict[s2s] = nodes[connected_ind]
            else:
                directed_dict[s2s] += nodes[connected_ind]

        return directed_dict

    def _convert_edge_graph(self, directed_dict, random_node=True):
        # get a directed graph and convert it into undirected, why?
        # 1. In paper "Convolutional Neural Networks on Graphs with Fast
        # Localized Spectral Filtering", it deals with undirected and connected graph G.
        # 2. We just consider the spatial relation of these links, direction
        # has not been utilized yet.

        edge_graph = nx.from_dict_of_lists(directed_dict)
        adj = nx.adjacency_matrix(edge_graph)
        # check the resulted graph is undirected graph
        assert (adj != adj.T).nnz == 0
        undi_graph = edge_graph.to_undirected()

        # get connected subgraphs
        graphs = list(nx.connected_component_subgraphs(undi_graph))
        nb_nodes = [len(graph_i.nodes()) for graph_i in graphs]

        max_nb = max(nb_nodes)
        max_index = nb_nodes.index(max_nb)
        max_sub_graph = graphs[max_index]

        # get the adjacency matrix of the largest sub graph
        nodes = max_sub_graph.nodes()
        adj = nx.adjacency_matrix(max_sub_graph)
        adj.setdiag(1)
        # adj.setdiag(0)
        adj = adj.astype(np.float64)
        if random_node:
            adj = adj.todense()
            rnd_order = np.random.permutation(len(nodes))
            adj = adj[rnd_order, :]
            adj = adj[:, rnd_order]
            nodes = list(np.array(nodes)[rnd_order])
            # convert into a sparse format
            adj = scipy.sparse.csr_matrix(adj)
        adj.eliminate_zeros()
        nodes = [str(node) for node in nodes]

        return adj, nodes

    def _construct_undirected_df(self, df):

        df = df.drop_duplicates('s2s', keep='first')
        df = df.join(
            df.s2s.str.split('-', expand=True).rename(
                columns={0: 'start_vertex', 1: 'end_vertex'}))
        df_copy = df.copy()
        del df_copy['s2s']

        df_copy['s2s'] = df[['end_vertex', 'start_vertex']]. \
            apply(lambda x: '-'.join(x), axis=1)
        df_merge = pd.concat([df, df_copy])

        print("Before intersect with distinct edges, ", df_merge.shape)
        unique_ids = self._get_distinct_segids()
        df_merge = df_merge[df_merge.s2s.isin(unique_ids)]
        print("After intersect with distinct edges, ", df_merge.shape)

        del df_merge['end_vertex'], df_merge['start_vertex']

        return df_merge

    def _get_distinct_segids(self):
        """
        Get all the distinct s2ss.

        :return: list, distinct s2ss
        """

        print("Entering into querying distinct edges...")
        # sql unique s2ss
        conn = self.engine.connect()
        sql_unique_segids = "SELECT DISTINCT s2s FROM "+self.trip_table+""
        df_unique_segids = pd.read_sql(sql_unique_segids, conn)
        conn.close()
        unique_segids = df_unique_segids['s2s'].tolist()

        return unique_segids

    def _get_shortest_path_nodes(self, graph, nodes_i,
                                nodes_j, total_nodes, dist=1.):
        """
        Get the nodes in shortest paths

        :param graph: networkx, graph
        :param nodes_i: list, Starting nodes for path
        :param nodes_j: list, Ending nodes for path
        :param total_nodes: list,
        :param dist: float, shortest paths with certain distance
        :return: set, set of added nodes
        """

        # print("The size of the graph is ", len(graph.nodes()))
        # print("nodes_j is ", nodes_j)
        # print("intersection of nodes is ", set(nodes_j).intersection(graph.nodes()))
        # print("number of nodes_j is ", len(nodes_j))
        # print("number of graph nodes is ", len(set(nodes_j).intersection(graph.nodes())))

        # assert len(nodes_i) == len(set(nodes_i).intersection(graph.nodes()))
        # assert len(nodes_j) == len(set(nodes_j).intersection(graph.nodes()))

        node_added = set()
        for node_i in nodes_i:
            if node_i not in graph.nodes():
                continue
            else:
                dict_len_nodes = nx.single_source_shortest_path_length(graph, node_i, dist)
                all_nodes = dict_len_nodes.keys()
                diff_all_nodes = set(all_nodes)
                common_nodes = list(diff_all_nodes.intersection(nodes_j))
                sub_graph = graph.subgraph(diff_all_nodes)
                node_added_i = self._get_optimal_added_nodes(
                    sub_graph, node_i, common_nodes, total_nodes)
                node_added = node_added.union(node_added_i)

        return node_added

    def _get_optimal_added_nodes(self, sub_graph, node_i,
                                 target_nodes, total_nodes):
        if len(target_nodes) < 1:
            return set()
        else:
            node_added = set()
            for node_j in target_nodes:
                paths_i_j = nx.all_shortest_paths(sub_graph, node_i, node_j)
                selected_path = list()
                max_matched = 2
                paths_i_j = [p for p in paths_i_j]
                for path_i_j in paths_i_j:
                    num_matched = len(set(path_i_j).intersection(total_nodes))
                    if num_matched > max_matched:
                        selected_path = path_i_j.copy()
                if max_matched == 2:
                    selected_path = paths_i_j[0]
                node_added = node_added.union(set(selected_path))

            return node_added

    def _construct_link_vel_hist(self, hist_range):
        """
        Construct a table with average travel time of a link

        :param hist_range: The pre-defined histogram bins

        :return:
        """

        sample_str = '{}T'.format(self.sample_rate)
        conn = self.engine.connect()
        link_sql = "select act_dep_time, s2s, s2s_speed from "+self.trip_table+"" \
                   " where s2s IN {} and s2s_speed > 0".format(
                       tuple(self.graph_edges))
        df_link = pd.read_sql(link_sql, conn, parse_dates=['act_dep_time'])
        df_link['s2s_speed'] = df_link['s2s_speed'].astype(float)
        df_link['time'] = pd.to_datetime(df_link['act_dep_time'])
        del df_link['act_dep_time']
        print("Try to get the links' historical travel times at sample rate {}".format(
            sample_str))

        df_link = df_link.set_index('time')
        df_link_gb = df_link.groupby('s2s')

        # load the edge length from db
        graph_edges_reverse = [
            '{}-{}'.format(e.split('-')[1], e.split('-')[0]) for e in self.graph_edges]
        graph_edges_reverse += self.graph_edges
        sql_link_dis = 'select s2s, s2s_len from '+self.trip_table+'' \
                       ' where s2s IN {}'.format(tuple(graph_edges_reverse))
        df_link_dis = pd.read_sql(sql_link_dis, conn)

        def get_vel_hist(array_like, link_len, hist_bin):
            if len(array_like) == 0:
                return np.nan
            tt_array = np.array(array_like)
            # tt_array = link_len / tt_array
            hist, bin_edges = np.histogram(tt_array, hist_bin, density=True)
            if np.isnan(hist).any():
                print('nan hist returned!')
                print('data', tt_array)
                print(hist)
                return np.nan
            # hist *= hist_bin[1] - hist_bin[0]
            return hist

        list_dfs = []
        link_names = []
        for link_name, df_link_i in df_link_gb:
            # Get the length of the current s2s
            node_dis = df_link_dis.loc[
                df_link_dis['s2s'] == link_name, 's2s_len']
            if node_dis.empty:
                change_s_t = '{0}-{1}'.format(link_name.split('-')
                                              [1], link_name.split('-')[0])
                node_dis = df_link_dis.loc[df_link_dis[
                    's2s'] == change_s_t, 's2s_len']
            link_len = node_dis.iloc[0]

            linki_vel_hist = pd.DataFrame()
            linki_vel_hist[link_name] = df_link_i.s2s_speed.resample(
                sample_str).apply(vel_list)
            linki_vel_hist[link_name] = linki_vel_hist[link_name].apply(
                get_vel_hist, args=(link_len, hist_range))
            link_names.append(str(link_name))
            list_dfs.append(linki_vel_hist)
        df_link_tb = pd.concat(list_dfs, axis=1, join='outer')

        nodes = [str(node) for node in self.graph_edges]
        col_name = list(df_link_tb.columns.values)
        diff_nodes1 = set(nodes).difference(col_name)
        diff_dict = dict.fromkeys(diff_nodes1, np.nan)
        df_link_tb = df_link_tb.assign(**diff_dict)
        df_link_tb[self.graph_edges] = df_link_tb[self.graph_edges]

        df_link_tb = df_link_tb.reset_index(level=['time'])
        df_link_tb.rename(columns=lambda x: str(x), inplace=True)

        conn.close()

        return df_link_tb

    def _construct_overlap_link_vel_hist(self, hist_range):
        """
        Construct a table with average travel time of a link

        :param hist_range: The pre-defined histogram bins

        :return:
        """

        sample_str = '{}T'.format(self.sample_rate)
        conn = self.engine.connect()
        link_sql = "select act_dep_time, s2s, s2s_speed from "+self.trip_table+"" \
                   " where s2s IN {} and s2s_speed > 0".format(
                       tuple(self.graph_edges))
        df_link = pd.read_sql(link_sql, conn, parse_dates=['act_dep_time'])
        df_link['s2s_speed'] = df_link['s2s_speed'].astype(float)
        df_link['time'] = pd.to_datetime(df_link['act_dep_time'])
        del df_link['act_dep_time']
        print("Try to get the links' historical travel times at sample rate {}".format(
            sample_str))

        df_link = df_link.set_index('time')
        df_link_gb = df_link.groupby('s2s')

        # load the edge length from db
        graph_edges_reverse = [
            '{}-{}'.format(e.split('-')[1], e.split('-')[0]) for e in self.graph_edges]
        graph_edges_reverse += self.graph_edges
        sql_link_dis = 'select s2s, s2s_len from '+self.trip_table+'' \
                       ' where s2s IN {}'.format(tuple(graph_edges_reverse))
        df_link_dis = pd.read_sql(sql_link_dis, conn)

        list_dfs = []
        link_names = []
        for link_name, df_link_i in df_link_gb:
            # Get the length of the current s2s
            node_dis = df_link_dis.loc[
                df_link_dis['s2s'] == link_name, 's2s_len']
            if node_dis.empty:
                change_s_t = '{0}-{1}'.format(link_name.split('-')
                                              [1], link_name.split('-')[0])
                node_dis = df_link_dis.loc[df_link_dis[
                    's2s'] == change_s_t, 's2s_len']
            link_len = node_dis.iloc[0]
            linki_vel_hist = pd.DataFrame()
            linki_resample = df_link_i.s2s_speed.resample(
                sample_str).apply(vel_list)

            # linki_vel_hist[link_name] = linki_resample.rolling(window=self.window_size).apply(
            #     get_vel_hist_rolling, args=(link_len, hist_range))
            linki_vel_hist[link_name] = my_rolling_apply_list(linki_resample,
                                                              self.get_vel_hist_rolling,
                                                              link_len, hist_range)

            link_names.append(str(link_name))
            list_dfs.append(linki_vel_hist)
        df_link_tb = pd.concat(list_dfs, axis=1, join='outer')

        nodes = [str(node) for node in self.graph_edges]
        col_name = list(df_link_tb.columns.values)
        diff_nodes1 = set(nodes).difference(col_name)
        diff_dict = dict.fromkeys(diff_nodes1, np.nan)
        df_link_tb = df_link_tb.assign(**diff_dict)
        df_link_tb[self.graph_edges] = df_link_tb[self.graph_edges]

        df_link_tb = df_link_tb.reset_index(level=['time'])
        df_link_tb.rename(columns=lambda x: str(x), inplace=True)

        conn.close()

        return df_link_tb



class chengdu_Data(DataSet):

    def __init__(self, data_dir, base_dir, server_name, conf_dir, random_node,
                 cat_head, con_head, start_date, test_start_date, test_end_date, trip_table,
                 data_rm_ratio=0.5, source_ratio=0.5, topk=1000,
                 sample_rate=15, window_size=20, predict_size=4,
                 small_threshold=0.0, big_threshold=50.0, dist=1.,
                 min_nb=5, unit=1, combo_random=True, num_combos=3,
                 custom_rect=None, temporal_dim=1, period_dim=1):

        self.name = 'chengdu'
        self.topk = topk
        self.dist = dist
        self.custom_rect = custom_rect
        self.trip_table = trip_table

        super().__init__(data_dir, base_dir, server_name, conf_dir, random_node,
                         cat_head, con_head, start_date, test_start_date, test_end_date, trip_table,
                         data_rm_ratio,
                         source_ratio, sample_rate, window_size, predict_size,
                         small_threshold, big_threshold, min_nb, unit,
                         combo_random, num_combos, temporal_dim, period_dim)


    def _get_graph_file(self):
        """
        Construct graph related file

        :return: directories to store graph related file
        """

        edge_adj_dir = os.path.join(self.base_dir)
        edge_adj_file = os.path.join(edge_adj_dir,
                                     'edge_adj.pickle')
        edges_file = os.path.join(edge_adj_dir,
                                  'edges.pickle')

        try:
            os.stat(edge_adj_dir)
        except:
            os.makedirs(edge_adj_dir)

        return edge_adj_file, edges_file

    def _construct_dict_edge_node(self):

        dict_edge_node = {}
        node_id = 0
        for index, edge in enumerate(self.graph_edges):
            link_id_left = edge.split('-')[0]
            if link_id_left not in dict_edge_node.keys():
                dict_edge_node[link_id_left] = node_id
                node_id += 1
            link_id_right = edge.split('-')[1]
            if link_id_right not in dict_edge_node.keys():
                dict_edge_node[link_id_right] = node_id
                node_id += 1

        return dict_edge_node

    def _construct_DiGraph(self, dict_edge_node):

        di_graph = nx.DiGraph()
        for link_id in self.graph_edges:
            left_node = link_id.split('-')[0]
            right_node = link_id.split('-')[1]
            left_node_id = dict_edge_node[left_node]
            right_node_id = dict_edge_node[right_node]
            di_graph.add_edge(left_node_id, right_node_id)
        return di_graph

    def _get_edge_connection(self):
        """
        Get a edge connection dictionary with the topk frequent edges

        :param topk: int, top k frequent edges
        :return: dictionary
        """

        conn = self.engine.connect()
        sql_all_usable_edge = 'SELECT COUNT(travel_speed), st_seg_id FROM '+self.trip_table+' ' \
                              'WHERE travel_speed > 0 ' \
                              'GROUP BY st_seg_id ' \
                              'ORDER BY COUNT(travel_speed) DESC ' \
                              'LIMIT {}'.format(self.topk)

        df_frequent_edge = pd.read_sql(sql_all_usable_edge, conn)
        # df_frequent_edge = self._custom_rect_edges(df_frequent_edge, self.custom_rect)
        edge_graph_dict = self._get_edge_connection_dict(df_frequent_edge)
        conn.close()

        return edge_graph_dict

    def _get_edge_connection_dict(self, df):
        directed_dict = {}
        df = df.drop_duplicates('st_seg_id', keep='first')
        df_frequent_edge = df.join(
            df.st_seg_id.str.split('-', expand=True).rename(
                columns={0: 'start_vertex', 1: 'end_vertex'}))

        for index, row in df_frequent_edge.iterrows():
            st_seg_id = row['st_seg_id']
            end_vertex = row['end_vertex']
            connected_segs = df_frequent_edge[
                df_frequent_edge['start_vertex'] == end_vertex]
            if st_seg_id not in directed_dict.keys():
                directed_dict[st_seg_id] = connected_segs['st_seg_id'].tolist()
            else:
                directed_dict[st_seg_id] += connected_segs['st_seg_id'].tolist()

            start_vertex = row['start_vertex']
            connected_segs = df_frequent_edge[df_frequent_edge['end_vertex']
                                              == start_vertex]
            directed_dict[st_seg_id] += connected_segs['st_seg_id'].tolist()

        return directed_dict

    def _construct_link_vel_list(self):
        """
        Construct an average travel time of nodes with certain sample rate.

        :param sample_freq: int, minutes.
        :param write_db: bool, write the constructed results into database or not/
        :return: Dataframe, table contains the avg travel time of all nodes.
        """
        conn = self.engine.connect()
        sample_str = '{}T'.format(self.sample_rate)

        print("Reading needed records for trips from Database...")
        link_sql = "select start_time, st_seg_id, travel_speed from "+self.trip_table+"" \
                   " where travel_speed > 0 AND st_seg_id IN {}".format(
                       tuple(self.graph_edges))
        df_link = pd.read_sql(link_sql, conn, parse_dates=['start_time'])

        df_link['travel_speed'] = df_link['travel_speed'].astype(float)
        df_link['time'] = pd.to_datetime(df_link['start_time'])
        del df_link['start_time']
        print("Try to get solid velocity of links at sample rate {} minutes"
              "".format(self.sample_rate))

        df_link = df_link.set_index('time')
        df_link_gb = df_link.groupby('st_seg_id')

        # check the equality of graph edges and queried edges
        assert len(df_link_gb.groups) == len(self.graph_edges)

        # load the edge length from db
        graph_edges_reverse = [
            '{}-{}'.format(e.split('-')[1], e.split('-')[0]) for e in self.graph_edges]
        graph_edges_reverse += self.graph_edges
        sql_link_dis = 'select st_seg_id, len from '+self.trip_table+'' \
                       ' where st_seg_id IN {}'.format(tuple(graph_edges_reverse))
        df_link_dis = pd.read_sql(sql_link_dis, conn)

        list_dfs = []
        for link_name, df_link_i in df_link_gb:
            # Get the length of the current st_seg_id
            node_dis = df_link_dis.loc[
                df_link_dis['st_seg_id'] == link_name, 'len']
            if node_dis.empty:
                change_s_t = '{0}-{1}'.format(link_name.split('-')
                                              [1], link_name.split('-')[0])
                node_dis = df_link_dis.loc[df_link_dis[
                    'st_seg_id'] == change_s_t, 'len']
            link_len = node_dis.iloc[0] * self.unit

            linki_tt = pd.DataFrame()
            linki_resample = df_link_i.travel_speed.resample(
                sample_str).apply(vel_list, edge_len=link_len)
            linki_tt[link_name] = linki_resample
            list_dfs.append(linki_tt)

        df_link_tb = pd.concat(list_dfs, axis=1, join='outer')

        df_link_tb = df_link_tb.reset_index(level=['time'])

        conn.close()

        return df_link_tb

    def _get_directed_dict_from_list(self, nodes):

        directed_dict = {}
        nodes = np.array(list(nodes))
        start_vertices = np.array([node.split('-')[0] for node in nodes])
        end_vertices = np.array([node.split('-')[1] for node in nodes])

        for index, node in enumerate(nodes):
            st_seg_id = node
            end_vertex = end_vertices[index]
            connected_ind = np.where(start_vertices == end_vertex)
            if st_seg_id not in directed_dict.keys():
                directed_dict[st_seg_id] = nodes[connected_ind]
            else:
                directed_dict[st_seg_id] += nodes[connected_ind]

        return directed_dict

    def _convert_edge_graph(self, directed_dict, random_node=True):
        # get a directed graph and convert it into undirected, why?
        # 1. In paper "Convolutional Neural Networks on Graphs with Fast
        # Localized Spectral Filtering", it deals with undirected and connected graph G.
        # 2. We just consider the spatial relation of these links, direction
        # has not been utilized yet.

        edge_graph = nx.from_dict_of_lists(directed_dict)
        adj = nx.adjacency_matrix(edge_graph)
        # check the resulted graph is undirected graph
        assert (adj != adj.T).nnz == 0
        undi_graph = edge_graph.to_undirected()

        # get connected subgraphs
        graphs = list(nx.connected_component_subgraphs(undi_graph))
        nb_nodes = [len(graph_i.nodes()) for graph_i in graphs]

        max_nb = max(nb_nodes)
        max_index = nb_nodes.index(max_nb)
        max_sub_graph = graphs[max_index]

        # get the adjacency matrix of the largest sub graph
        nodes = max_sub_graph.nodes()
        adj = nx.adjacency_matrix(max_sub_graph)
        adj.setdiag(1)
        # adj.setdiag(0)
        adj = adj.astype(np.float64)
        if random_node:
            adj = adj.todense()
            rnd_order = np.random.permutation(len(nodes))
            adj = adj[rnd_order, :]
            adj = adj[:, rnd_order]
            nodes = list(np.array(nodes)[rnd_order])
            # convert into a sparse format
            adj = scipy.sparse.csr_matrix(adj)
        adj.eliminate_zeros()
        nodes = [str(node) for node in nodes]

        return adj, nodes

    def _construct_undirected_df(self, df):

        df = df.drop_duplicates('st_seg_id', keep='first')
        df = df.join(
            df.st_seg_id.str.split('-', expand=True).rename(
                columns={0: 'start_vertex', 1: 'end_vertex'}))
        df_copy = df.copy()
        del df_copy['st_seg_id']

        df_copy['st_seg_id'] = df[['end_vertex', 'start_vertex']]. \
            apply(lambda x: '-'.join(x), axis=1)
        df_merge = pd.concat([df, df_copy])

        print("Before intersect with distinct edges, ", df_merge.shape)
        unique_ids = self._get_distinct_segids()
        df_merge = df_merge[df_merge.st_seg_id.isin(unique_ids)]
        print("After intersect with distinct edges, ", df_merge.shape)

        del df_merge['end_vertex'], df_merge['start_vertex']

        return df_merge

    def _get_distinct_segids(self):
        """
        Get all the distinct seg_ids.

        :return: list, distinct seg_ids
        """

        print("Entering into querying distinct edges...")
        # sql unique seg_ids
        conn = self.engine.connect()
        sql_unique_segids = "SELECT DISTINCT st_seg_id FROM "+self.trip_table+""
        df_unique_segids = pd.read_sql(sql_unique_segids, conn)
        conn.close()
        unique_segids = df_unique_segids['st_seg_id'].tolist()

        return unique_segids

    def _get_shortest_path_nodes(self, graph, nodes_i,
                                nodes_j, total_nodes, dist=1.):
        """
        Get the nodes in shortest paths

        :param graph: networkx, graph
        :param nodes_i: list, Starting nodes for path
        :param nodes_j: list, Ending nodes for path
        :param total_nodes: list,
        :param dist: float, shortest paths with certain distance
        :return: set, set of added nodes
        """

        # print("The size of the graph is ", len(graph.nodes()))
        # print("nodes_j is ", nodes_j)
        # print("intersection of nodes is ", set(nodes_j).intersection(graph.nodes()))
        # print("number of nodes_j is ", len(nodes_j))
        # print("number of graph nodes is ", len(set(nodes_j).intersection(graph.nodes())))

        # assert len(nodes_i) == len(set(nodes_i).intersection(graph.nodes()))
        # assert len(nodes_j) == len(set(nodes_j).intersection(graph.nodes()))

        node_added = set()
        for node_i in nodes_i:
            if node_i not in graph.nodes():
                continue
            else:
                dict_len_nodes = nx.single_source_shortest_path_length(graph, node_i, dist)
                all_nodes = dict_len_nodes.keys()
                diff_all_nodes = set(all_nodes)
                common_nodes = list(diff_all_nodes.intersection(nodes_j))
                sub_graph = graph.subgraph(diff_all_nodes)
                node_added_i = self._get_optimal_added_nodes(
                    sub_graph, node_i, common_nodes, total_nodes)
                node_added = node_added.union(node_added_i)

        return node_added

    def _get_optimal_added_nodes(self, sub_graph, node_i,
                                 target_nodes, total_nodes):
        if len(target_nodes) < 1:
            return set()
        else:
            node_added = set()
            for node_j in target_nodes:
                paths_i_j = nx.all_shortest_paths(sub_graph, node_i, node_j)
                selected_path = list()
                max_matched = 2
                paths_i_j = [p for p in paths_i_j]
                for path_i_j in paths_i_j:
                    num_matched = len(set(path_i_j).intersection(total_nodes))
                    if num_matched > max_matched:
                        selected_path = path_i_j.copy()
                if max_matched == 2:
                    selected_path = paths_i_j[0]
                node_added = node_added.union(set(selected_path))

            return node_added

    def _construct_link_vel_hist(self, hist_range):
        """
        Construct a table with average travel time of a link

        :param hist_range: The pre-defined histogram bins

        :return:
        """

        sample_str = '{}T'.format(self.sample_rate)
        conn = self.engine.connect()
        link_sql = "select start_time, st_seg_id, travel_speed from "+self.trip_table+"" \
                   " where st_seg_id IN {} and travel_speed > 0".format(
                       tuple(self.graph_edges))
        df_link = pd.read_sql(link_sql, conn, parse_dates=['start_time'])
        df_link['travel_speed'] = df_link['travel_speed'].astype(float)
        df_link['time'] = pd.to_datetime(df_link['start_time'])
        del df_link['start_time']
        print("Try to get the links' historical travel times at sample rate {}".format(
            sample_str))

        df_link = df_link.set_index('time')
        df_link_gb = df_link.groupby('st_seg_id')

        # load the edge length from db
        graph_edges_reverse = [
            '{}-{}'.format(e.split('-')[1], e.split('-')[0]) for e in self.graph_edges]
        graph_edges_reverse += self.graph_edges
        sql_link_dis = 'select st_seg_id, len from '+self.trip_table+'' \
                       ' where st_seg_id IN {}'.format(tuple(graph_edges_reverse))
        df_link_dis = pd.read_sql(sql_link_dis, conn)

        def get_vel_hist(array_like, link_len, hist_bin):
            if len(array_like) == 0:
                return np.nan
            tt_array = np.array(array_like)
            # tt_array = link_len / tt_array
            hist, bin_edges = np.histogram(tt_array, hist_bin, density=True)
            if np.isnan(hist).any():
                print('nan hist returned!')
                print('data', tt_array)
                print(hist)
                return np.nan
            # hist *= hist_bin[1] - hist_bin[0]
            return hist

        list_dfs = []
        link_names = []
        for link_name, df_link_i in df_link_gb:
            # Get the length of the current st_seg_id
            node_dis = df_link_dis.loc[
                df_link_dis['st_seg_id'] == link_name, 'len']
            if node_dis.empty:
                change_s_t = '{0}-{1}'.format(link_name.split('-')
                                              [1], link_name.split('-')[0])
                node_dis = df_link_dis.loc[df_link_dis[
                    'st_seg_id'] == change_s_t, 'len']
            link_len = node_dis.iloc[0]

            linki_vel_hist = pd.DataFrame()
            linki_vel_hist[link_name] = df_link_i.travel_speed.resample(
                sample_str).apply(vel_list)
            linki_vel_hist[link_name] = linki_vel_hist[link_name].apply(
                get_vel_hist, args=(link_len, hist_range))
            link_names.append(str(link_name))
            list_dfs.append(linki_vel_hist)
        df_link_tb = pd.concat(list_dfs, axis=1, join='outer')

        nodes = [str(node) for node in self.graph_edges]
        col_name = list(df_link_tb.columns.values)
        diff_nodes1 = set(nodes).difference(col_name)
        diff_dict = dict.fromkeys(diff_nodes1, np.nan)
        df_link_tb = df_link_tb.assign(**diff_dict)
        df_link_tb[self.graph_edges] = df_link_tb[self.graph_edges]

        df_link_tb = df_link_tb.reset_index(level=['time'])
        df_link_tb.rename(columns=lambda x: str(x), inplace=True)

        conn.close()

        return df_link_tb

    def _construct_overlap_link_vel_hist(self, hist_range):
        """
        Construct a table with average travel time of a link

        :param hist_range: The pre-defined histogram bins

        :return:
        """

        sample_str = '{}T'.format(self.sample_rate)
        conn = self.engine.connect()
        link_sql = "select start_time, st_seg_id, travel_speed from "+self.trip_table+"" \
                   " where st_seg_id IN {} and travel_speed > 0".format(
                       tuple(self.graph_edges))
        df_link = pd.read_sql(link_sql, conn, parse_dates=['start_time'])
        df_link['travel_speed'] = df_link['travel_speed'].astype(float)
        df_link['time'] = pd.to_datetime(df_link['start_time'])
        del df_link['start_time']
        print("Try to get the links' historical travel times at sample rate {}".format(
            sample_str))

        df_link = df_link.set_index('time')
        df_link_gb = df_link.groupby('st_seg_id')

        # load the edge length from db
        graph_edges_reverse = [
            '{}-{}'.format(e.split('-')[1], e.split('-')[0]) for e in self.graph_edges]
        graph_edges_reverse += self.graph_edges
        sql_link_dis = 'select st_seg_id, len from '+self.trip_table+'' \
                       ' where st_seg_id IN {}'.format(tuple(graph_edges_reverse))
        df_link_dis = pd.read_sql(sql_link_dis, conn)

        list_dfs = []
        link_names = []
        for link_name, df_link_i in df_link_gb:
            # Get the length of the current st_seg_id
            node_dis = df_link_dis.loc[
                df_link_dis['st_seg_id'] == link_name, 'len']
            if node_dis.empty:
                change_s_t = '{0}-{1}'.format(link_name.split('-')
                                              [1], link_name.split('-')[0])
                node_dis = df_link_dis.loc[df_link_dis[
                    'st_seg_id'] == change_s_t, 'len']
            link_len = node_dis.iloc[0]
            linki_vel_hist = pd.DataFrame()
            linki_resample = df_link_i.travel_speed.resample(
                sample_str).apply(vel_list)

            # linki_vel_hist[link_name] = linki_resample.rolling(window=self.window_size).apply(
            #     get_vel_hist_rolling, args=(link_len, hist_range))
            linki_vel_hist[link_name] = my_rolling_apply_list(linki_resample,
                                                              self.get_vel_hist_rolling,
                                                              link_len, hist_range)

            link_names.append(str(link_name))
            list_dfs.append(linki_vel_hist)
        df_link_tb = pd.concat(list_dfs, axis=1, join='outer')

        nodes = [str(node) for node in self.graph_edges]
        col_name = list(df_link_tb.columns.values)
        diff_nodes1 = set(nodes).difference(col_name)
        diff_dict = dict.fromkeys(diff_nodes1, np.nan)
        df_link_tb = df_link_tb.assign(**diff_dict)
        df_link_tb[self.graph_edges] = df_link_tb[self.graph_edges]

        df_link_tb = df_link_tb.reset_index(level=['time'])
        df_link_tb.rename(columns=lambda x: str(x), inplace=True)

        conn.close()

        return df_link_tb


class xian_Data(DataSet):

    def __init__(self, data_dir, base_dir, server_name, conf_dir, random_node,
                 cat_head, con_head, start_date, test_start_date, test_end_date, trip_table,
                 data_rm_ratio=0.5, source_ratio=0.5, topk=1000,
                 sample_rate=15, window_size=20, predict_size=4,
                 small_threshold=0.0, big_threshold=50.0, dist=1.,
                 min_nb=5, unit=1, combo_random=True, num_combos=3,
                 custom_rect=None, temporal_dim=1, period_dim=1):

        self.name = 'xian'
        self.topk = topk
        self.dist = dist
        self.custom_rect = custom_rect
        self.trip_table = trip_table

        super().__init__(data_dir, base_dir, server_name, conf_dir, random_node,
                         cat_head, con_head, start_date, test_start_date, test_end_date, trip_table,
                         data_rm_ratio,
                         source_ratio, sample_rate, window_size, predict_size,
                         small_threshold, big_threshold, min_nb, unit,
                         combo_random, num_combos, temporal_dim, period_dim)


    def _get_graph_file(self):
        """
        Construct graph related file

        :return: directories to store graph related file
        """

        edge_adj_dir = os.path.join(self.base_dir)
        edge_adj_file = os.path.join(edge_adj_dir,
                                     'edge_adj.pickle')
        edges_file = os.path.join(edge_adj_dir,
                                  'edges.pickle')

        try:
            os.stat(edge_adj_dir)
        except:
            os.makedirs(edge_adj_dir)

        return edge_adj_file, edges_file

    def _construct_dict_edge_node(self):

        dict_edge_node = {}
        node_id = 0
        for index, edge in enumerate(self.graph_edges):
            link_id_left = edge.split('-')[0]
            if link_id_left not in dict_edge_node.keys():
                dict_edge_node[link_id_left] = node_id
                node_id += 1
            link_id_right = edge.split('-')[1]
            if link_id_right not in dict_edge_node.keys():
                dict_edge_node[link_id_right] = node_id
                node_id += 1

        return dict_edge_node

    def _construct_DiGraph(self, dict_edge_node):

        di_graph = nx.DiGraph()
        for link_id in self.graph_edges:
            left_node = link_id.split('-')[0]
            right_node = link_id.split('-')[1]
            left_node_id = dict_edge_node[left_node]
            right_node_id = dict_edge_node[right_node]
            di_graph.add_edge(left_node_id, right_node_id)
        return di_graph

    def _get_edge_connection(self):
        """
        Get a edge connection dictionary with the topk frequent edges

        :param topk: int, top k frequent edges
        :return: dictionary
        """

        conn = self.engine.connect()
        sql_all_usable_edge = 'SELECT COUNT(travel_speed), st_seg_id FROM '+self.trip_table+' ' \
                              'WHERE travel_speed > 0 ' \
                              'GROUP BY st_seg_id ' \
                              'ORDER BY COUNT(travel_speed) DESC ' \
                              'LIMIT {}'.format(self.topk)

        df_frequent_edge = pd.read_sql(sql_all_usable_edge, conn)
        # df_frequent_edge = self._custom_rect_edges(df_frequent_edge, self.custom_rect)
        edge_graph_dict = self._get_edge_connection_dict(df_frequent_edge)
        conn.close()

        return edge_graph_dict

    def _get_edge_connection_dict(self, df):
        directed_dict = {}
        df = df.drop_duplicates('st_seg_id', keep='first')
        df_frequent_edge = df.join(
            df.st_seg_id.str.split('-', expand=True).rename(
                columns={0: 'start_vertex', 1: 'end_vertex'}))

        for index, row in df_frequent_edge.iterrows():
            st_seg_id = row['st_seg_id']
            end_vertex = row['end_vertex']
            connected_segs = df_frequent_edge[
                df_frequent_edge['start_vertex'] == end_vertex]
            if st_seg_id not in directed_dict.keys():
                directed_dict[st_seg_id] = connected_segs['st_seg_id'].tolist()
            else:
                directed_dict[st_seg_id] += connected_segs['st_seg_id'].tolist()

            start_vertex = row['start_vertex']
            connected_segs = df_frequent_edge[df_frequent_edge['end_vertex']
                                              == start_vertex]
            directed_dict[st_seg_id] += connected_segs['st_seg_id'].tolist()

        return directed_dict

    def _construct_link_vel_list(self):
        """
        Construct an average travel time of nodes with certain sample rate.

        :param sample_freq: int, minutes.
        :param write_db: bool, write the constructed results into database or not/
        :return: Dataframe, table contains the avg travel time of all nodes.
        """
        conn = self.engine.connect()
        sample_str = '{}T'.format(self.sample_rate)

        print("Reading needed records for trips from Database...")
        link_sql = "select start_time, st_seg_id, travel_speed from "+self.trip_table+"" \
                   " where travel_speed > 0 AND st_seg_id IN {}".format(
                       tuple(self.graph_edges))
        df_link = pd.read_sql(link_sql, conn, parse_dates=['start_time'])

        df_link['travel_speed'] = df_link['travel_speed'].astype(float)
        df_link['time'] = pd.to_datetime(df_link['start_time'])
        del df_link['start_time']
        print("Try to get solid velocity of links at sample rate {} minutes"
              "".format(self.sample_rate))

        df_link = df_link.set_index('time')
        df_link_gb = df_link.groupby('st_seg_id')

        # check the equality of graph edges and queried edges
        assert len(df_link_gb.groups) == len(self.graph_edges)

        # load the edge length from db
        graph_edges_reverse = [
            '{}-{}'.format(e.split('-')[1], e.split('-')[0]) for e in self.graph_edges]
        graph_edges_reverse += self.graph_edges
        sql_link_dis = 'select st_seg_id, len from '+self.trip_table+'' \
                       ' where st_seg_id IN {}'.format(tuple(graph_edges_reverse))
        df_link_dis = pd.read_sql(sql_link_dis, conn)

        list_dfs = []
        for link_name, df_link_i in df_link_gb:
            # Get the length of the current st_seg_id
            node_dis = df_link_dis.loc[
                df_link_dis['st_seg_id'] == link_name, 'len']
            if node_dis.empty:
                change_s_t = '{0}-{1}'.format(link_name.split('-')
                                              [1], link_name.split('-')[0])
                node_dis = df_link_dis.loc[df_link_dis[
                    'st_seg_id'] == change_s_t, 'len']
            link_len = node_dis.iloc[0] * self.unit

            linki_tt = pd.DataFrame()
            linki_resample = df_link_i.travel_speed.resample(
                sample_str).apply(vel_list, edge_len=link_len)
            linki_tt[link_name] = linki_resample
            list_dfs.append(linki_tt)

        df_link_tb = pd.concat(list_dfs, axis=1, join='outer')

        df_link_tb = df_link_tb.reset_index(level=['time'])

        conn.close()

        return df_link_tb

    def _get_directed_dict_from_list(self, nodes):

        directed_dict = {}
        nodes = np.array(list(nodes))
        start_vertices = np.array([node.split('-')[0] for node in nodes])
        end_vertices = np.array([node.split('-')[1] for node in nodes])

        for index, node in enumerate(nodes):
            st_seg_id = node
            end_vertex = end_vertices[index]
            connected_ind = np.where(start_vertices == end_vertex)
            if st_seg_id not in directed_dict.keys():
                directed_dict[st_seg_id] = nodes[connected_ind]
            else:
                directed_dict[st_seg_id] += nodes[connected_ind]

        return directed_dict

    def _convert_edge_graph(self, directed_dict, random_node=True):
        # get a directed graph and convert it into undirected, why?
        # 1. In paper "Convolutional Neural Networks on Graphs with Fast
        # Localized Spectral Filtering", it deals with undirected and connected graph G.
        # 2. We just consider the spatial relation of these links, direction
        # has not been utilized yet.

        edge_graph = nx.from_dict_of_lists(directed_dict)
        adj = nx.adjacency_matrix(edge_graph)
        # check the resulted graph is undirected graph
        assert (adj != adj.T).nnz == 0
        undi_graph = edge_graph.to_undirected()

        # get connected subgraphs
        graphs = list(nx.connected_component_subgraphs(undi_graph))
        nb_nodes = [len(graph_i.nodes()) for graph_i in graphs]

        max_nb = max(nb_nodes)
        max_index = nb_nodes.index(max_nb)
        max_sub_graph = graphs[max_index]

        # get the adjacency matrix of the largest sub graph
        nodes = max_sub_graph.nodes()
        adj = nx.adjacency_matrix(max_sub_graph)
        adj.setdiag(1)
        # adj.setdiag(0)
        adj = adj.astype(np.float64)
        if random_node:
            adj = adj.todense()
            rnd_order = np.random.permutation(len(nodes))
            adj = adj[rnd_order, :]
            adj = adj[:, rnd_order]
            nodes = list(np.array(nodes)[rnd_order])
            # convert into a sparse format
            adj = scipy.sparse.csr_matrix(adj)
        adj.eliminate_zeros()
        nodes = [str(node) for node in nodes]

        return adj, nodes

    def _construct_undirected_df(self, df):

        df = df.drop_duplicates('st_seg_id', keep='first')
        df = df.join(
            df.st_seg_id.str.split('-', expand=True).rename(
                columns={0: 'start_vertex', 1: 'end_vertex'}))
        df_copy = df.copy()
        del df_copy['st_seg_id']

        df_copy['st_seg_id'] = df[['end_vertex', 'start_vertex']]. \
            apply(lambda x: '-'.join(x), axis=1)
        df_merge = pd.concat([df, df_copy])

        print("Before intersect with distinct edges, ", df_merge.shape)
        unique_ids = self._get_distinct_segids()
        df_merge = df_merge[df_merge.st_seg_id.isin(unique_ids)]
        print("After intersect with distinct edges, ", df_merge.shape)

        del df_merge['end_vertex'], df_merge['start_vertex']

        return df_merge

    def _get_distinct_segids(self):
        """
        Get all the distinct seg_ids.

        :return: list, distinct seg_ids
        """

        print("Entering into querying distinct edges...")
        # sql unique seg_ids
        conn = self.engine.connect()
        sql_unique_segids = "SELECT DISTINCT st_seg_id FROM "+self.trip_table+""
        df_unique_segids = pd.read_sql(sql_unique_segids, conn)
        conn.close()
        unique_segids = df_unique_segids['st_seg_id'].tolist()

        return unique_segids

    def _get_shortest_path_nodes(self, graph, nodes_i,
                                nodes_j, total_nodes, dist=1.):
        """
        Get the nodes in shortest paths

        :param graph: networkx, graph
        :param nodes_i: list, Starting nodes for path
        :param nodes_j: list, Ending nodes for path
        :param total_nodes: list,
        :param dist: float, shortest paths with certain distance
        :return: set, set of added nodes
        """

        # print("The size of the graph is ", len(graph.nodes()))
        # print("nodes_j is ", nodes_j)
        # print("intersection of nodes is ", set(nodes_j).intersection(graph.nodes()))
        # print("number of nodes_j is ", len(nodes_j))
        # print("number of graph nodes is ", len(set(nodes_j).intersection(graph.nodes())))

        # assert len(nodes_i) == len(set(nodes_i).intersection(graph.nodes()))
        # assert len(nodes_j) == len(set(nodes_j).intersection(graph.nodes()))

        node_added = set()
        for node_i in nodes_i:
            if node_i not in graph.nodes():
                continue
            else:
                dict_len_nodes = nx.single_source_shortest_path_length(graph, node_i, dist)
                all_nodes = dict_len_nodes.keys()
                diff_all_nodes = set(all_nodes)
                common_nodes = list(diff_all_nodes.intersection(nodes_j))
                sub_graph = graph.subgraph(diff_all_nodes)
                node_added_i = self._get_optimal_added_nodes(
                    sub_graph, node_i, common_nodes, total_nodes)
                node_added = node_added.union(node_added_i)

        return node_added

    def _get_optimal_added_nodes(self, sub_graph, node_i,
                                 target_nodes, total_nodes):
        if len(target_nodes) < 1:
            return set()
        else:
            node_added = set()
            for node_j in target_nodes:
                paths_i_j = nx.all_shortest_paths(sub_graph, node_i, node_j)
                selected_path = list()
                max_matched = 2
                paths_i_j = [p for p in paths_i_j]
                for path_i_j in paths_i_j:
                    num_matched = len(set(path_i_j).intersection(total_nodes))
                    if num_matched > max_matched:
                        selected_path = path_i_j.copy()
                if max_matched == 2:
                    selected_path = paths_i_j[0]
                node_added = node_added.union(set(selected_path))

            return node_added

    def _construct_link_vel_hist(self, hist_range):
        """
        Construct a table with average travel time of a link

        :param hist_range: The pre-defined histogram bins

        :return:
        """

        sample_str = '{}T'.format(self.sample_rate)
        conn = self.engine.connect()
        link_sql = "select start_time, st_seg_id, travel_speed from "+self.trip_table+"" \
                   " where st_seg_id IN {} and travel_speed > 0".format(
                       tuple(self.graph_edges))
        df_link = pd.read_sql(link_sql, conn, parse_dates=['start_time'])
        df_link['travel_speed'] = df_link['travel_speed'].astype(float)
        df_link['time'] = pd.to_datetime(df_link['start_time'])
        del df_link['start_time']
        print("Try to get the links' historical travel times at sample rate {}".format(
            sample_str))

        df_link = df_link.set_index('time')
        df_link_gb = df_link.groupby('st_seg_id')

        # load the edge length from db
        graph_edges_reverse = [
            '{}-{}'.format(e.split('-')[1], e.split('-')[0]) for e in self.graph_edges]
        graph_edges_reverse += self.graph_edges
        sql_link_dis = 'select st_seg_id, len from '+self.trip_table+'' \
                       ' where st_seg_id IN {}'.format(tuple(graph_edges_reverse))
        df_link_dis = pd.read_sql(sql_link_dis, conn)

        def get_vel_hist(array_like, link_len, hist_bin):
            if len(array_like) == 0:
                return np.nan
            tt_array = np.array(array_like)
            # tt_array = link_len / tt_array
            hist, bin_edges = np.histogram(tt_array, hist_bin, density=True)
            if np.isnan(hist).any():
                print('nan hist returned!')
                print('data', tt_array)
                print(hist)
                return np.nan
            # hist *= hist_bin[1] - hist_bin[0]
            return hist

        list_dfs = []
        link_names = []
        for link_name, df_link_i in df_link_gb:
            # Get the length of the current st_seg_id
            node_dis = df_link_dis.loc[
                df_link_dis['st_seg_id'] == link_name, 'len']
            if node_dis.empty:
                change_s_t = '{0}-{1}'.format(link_name.split('-')
                                              [1], link_name.split('-')[0])
                node_dis = df_link_dis.loc[df_link_dis[
                    'st_seg_id'] == change_s_t, 'len']
            link_len = node_dis.iloc[0]

            linki_vel_hist = pd.DataFrame()
            linki_vel_hist[link_name] = df_link_i.travel_speed.resample(
                sample_str).apply(vel_list)
            linki_vel_hist[link_name] = linki_vel_hist[link_name].apply(
                get_vel_hist, args=(link_len, hist_range))
            link_names.append(str(link_name))
            list_dfs.append(linki_vel_hist)
        df_link_tb = pd.concat(list_dfs, axis=1, join='outer')

        nodes = [str(node) for node in self.graph_edges]
        col_name = list(df_link_tb.columns.values)
        diff_nodes1 = set(nodes).difference(col_name)
        diff_dict = dict.fromkeys(diff_nodes1, np.nan)
        df_link_tb = df_link_tb.assign(**diff_dict)
        df_link_tb[self.graph_edges] = df_link_tb[self.graph_edges]

        df_link_tb = df_link_tb.reset_index(level=['time'])
        df_link_tb.rename(columns=lambda x: str(x), inplace=True)

        conn.close()

        return df_link_tb

    def _construct_overlap_link_vel_hist(self, hist_range):
        """
        Construct a table with average travel time of a link

        :param hist_range: The pre-defined histogram bins

        :return:
        """

        sample_str = '{}T'.format(self.sample_rate)
        conn = self.engine.connect()
        link_sql = "select start_time, st_seg_id, travel_speed from "+self.trip_table+"" \
                   " where st_seg_id IN {} and travel_speed > 0".format(
                       tuple(self.graph_edges))
        df_link = pd.read_sql(link_sql, conn, parse_dates=['start_time'])
        df_link['travel_speed'] = df_link['travel_speed'].astype(float)
        df_link['time'] = pd.to_datetime(df_link['start_time'])
        del df_link['start_time']
        print("Try to get the links' historical travel times at sample rate {}".format(
            sample_str))

        df_link = df_link.set_index('time')
        df_link_gb = df_link.groupby('st_seg_id')

        # load the edge length from db
        graph_edges_reverse = [
            '{}-{}'.format(e.split('-')[1], e.split('-')[0]) for e in self.graph_edges]
        graph_edges_reverse += self.graph_edges
        sql_link_dis = 'select st_seg_id, len from '+self.trip_table+'' \
                       ' where st_seg_id IN {}'.format(tuple(graph_edges_reverse))
        df_link_dis = pd.read_sql(sql_link_dis, conn)

        list_dfs = []
        link_names = []
        for link_name, df_link_i in df_link_gb:
            # Get the length of the current st_seg_id
            node_dis = df_link_dis.loc[
                df_link_dis['st_seg_id'] == link_name, 'len']
            if node_dis.empty:
                change_s_t = '{0}-{1}'.format(link_name.split('-')
                                              [1], link_name.split('-')[0])
                node_dis = df_link_dis.loc[df_link_dis[
                    'st_seg_id'] == change_s_t, 'len']
            link_len = node_dis.iloc[0]
            linki_vel_hist = pd.DataFrame()
            linki_resample = df_link_i.travel_speed.resample(
                sample_str).apply(vel_list)

            # linki_vel_hist[link_name] = linki_resample.rolling(window=self.window_size).apply(
            #     get_vel_hist_rolling, args=(link_len, hist_range))
            linki_vel_hist[link_name] = my_rolling_apply_list(linki_resample,
                                                              self.get_vel_hist_rolling,
                                                              link_len, hist_range)

            link_names.append(str(link_name))
            list_dfs.append(linki_vel_hist)
        df_link_tb = pd.concat(list_dfs, axis=1, join='outer')

        nodes = [str(node) for node in self.graph_edges]
        col_name = list(df_link_tb.columns.values)
        diff_nodes1 = set(nodes).difference(col_name)
        diff_dict = dict.fromkeys(diff_nodes1, np.nan)
        df_link_tb = df_link_tb.assign(**diff_dict)
        df_link_tb[self.graph_edges] = df_link_tb[self.graph_edges]

        df_link_tb = df_link_tb.reset_index(level=['time'])
        df_link_tb.rename(columns=lambda x: str(x), inplace=True)

        conn.close()

        return df_link_tb