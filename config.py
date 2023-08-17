#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Parse input command to hyper-parameters

import ast
import argparse

parser = argparse.ArgumentParser()
arg_list = []

def str2bool(v):
    return v.lower() in ('true', '1')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_list.append(arg)
    return arg


# Data
data_arg = add_argument_group('Data')

# set for hypergraph conv evaluation
data_arg.add_argument('--hypergraph_flag', type=int, default=1, help='')
data_arg.add_argument('--bias_hyper', type=int, default=1, help='')
data_arg.add_argument('--n_hid', type=int, default=128, help='')
data_arg.add_argument('--dropout_hyper', type=float, default=0.0, help='')
data_arg.add_argument('--hyper_weight_stddev', type=float, default=0.1, help='')
data_arg.add_argument('--hyper_bias_stddev', type=float, default=0.1, help='')
data_arg.add_argument('--hgnn_residual', type=int, default=1, help='')
data_arg.add_argument('--distance_shunxu', type=int, default=100, help='')
data_arg.add_argument('--hyper_w_transform_flag', type=int, default=1, help='')


# set for hypergraph conv evaluation
data_arg.add_argument('--vae_flag', type=int, default=1, help='')
data_arg.add_argument('--stddev1', type=float, default=0.01, help='')
data_arg.add_argument('--stddev2', type=float, default=0.01, help='')
data_arg.add_argument('--loss_vae', type=int, default=1, help='')
data_arg.add_argument('--onlyB', type=int, default=1, help='')
data_arg.add_argument('--vae_residual', type=int, default=1, help='')

# hxl adds
data_arg.add_argument('--least_threshold', type=float, default=0.10, help='')
data_arg.add_argument('--trip_table', type=str, default='xian_2m_tr0', help='')

# for mtr dataset
data_arg.add_argument('--hist_range', type=list,choices=[], help='')

data_arg.add_argument('--is_pooling', type=int, default=1, help='')


# temporal flag of scl
data_arg.add_argument('--no_scl', type=int, default=0, help='')
data_arg.add_argument('--mask_weight_no_scl', type=int, default=1, help='')
# spatial flag of scl
data_arg.add_argument('--spt_scl', type=int, default=0, help='')
data_arg.add_argument('--residual', type=int, default=0, help='')
data_arg.add_argument('--congc', type=int, default=1, help='')

data_arg.add_argument('--scl_evl_E', type=int, default=0)
data_arg.add_argument('--scl_evl_T', type=int, default=0)

data_arg.add_argument('--enlarge', type=int, default=50)
data_arg.add_argument('--mx_per_gpu', type=int, default=1)




# for ablation experiments: pooling or not,
# hxl chagned, need to change back
data_arg.add_argument('--is_removing_T', type=int, default=0, help='')
data_arg.add_argument('--is_removing_attn', type=int, default=0, help='')

data_arg.add_argument('--is_sparse', type=int, default=0,help='is_sparse')
data_arg.add_argument('--beta', type=float, default=0.001,help='beta')
data_arg.add_argument('--p_o', type=float, default=0.3,help='p_o')
data_arg.add_argument('--min_relu', type=float, default=0.2,help='min_relu')

data_arg.add_argument('--win_size', type=int, default=1, help='')

data_arg.add_argument('--server_name', type=str, default='xian', help='')

data_arg.add_argument('--temporal_dim', type=int, default=2, help='the number of temporal dimension')
data_arg.add_argument('--temporal_attn', type=ast.literal_eval, default=True, help='false means without temporal attention')

data_arg.add_argument('--hood', type=int, default=2, help='')
data_arg.add_argument('--n_heads_spt', type=int, default=4, help='')

data_arg.add_argument('--n_heads_tmp', type=int, default=4, help='')

data_arg.add_argument('--period_dim', type=int, default=2, help='the number of periodic dimension')

data_arg.add_argument('--n_heads_prd', type=int, default=4, help='')


data_arg.add_argument('--layers', type=int, default=2, help='')



# hxl changes
data_arg.add_argument('--conv', type=str, default='gcnn',
                      choices=['gcnn', 'cnn'], help='')
data_arg.add_argument('--ds_ind', type=int, default=0,
                      choices=[0, 1, 2, 3, 4], help='')
data_arg.add_argument('--normalized', type=bool, default=True, help='')
data_arg.add_argument('--mode', type=str, default='estimation',
                      choices=['estimation', 'prediction'], help='')
# hxl changes
data_arg.add_argument('--target', type=str, default='hist',
                      choices=['hist', 'avg'], help='')
data_arg.add_argument('--sample_rate', type=int, default=30, help='')
data_arg.add_argument('--data_rm', type=float, default=0.6, help='')

data_arg.add_argument('--coarsening_level', type=int, default=4, help='')
data_arg.add_argument('--is_coarsen', type=ast.literal_eval, default=False, choices=[True, False], help='')
train_arg = add_argument_group('Training')
# hxl changed
train_arg.add_argument('--num_epochs', type=int, default=100, help='')
train_arg.add_argument('--stop_win_size', type=int, default=5, help='')
train_arg.add_argument('--stop_early', type=bool, default=True, help='')
train_arg.add_argument('--sub_folder', type=bool, default=False, help='')
# hxl changed
train_arg.add_argument('--batch_size', type=int, default=8, help='')
train_arg.add_argument('--random_seed', type=int, default=123, help='')
train_arg.add_argument('--max_step', type=int, default=1000000, help='')
train_arg.add_argument('--is_train', type=str2bool, default=True, help='')
train_arg.add_argument('--classif_loss', type=str,
                       default='kl', choices=['kl_div', 'l2'], help='')
train_arg.add_argument('--learning_rate', type=float, default=0.001, help='')
train_arg.add_argument('--decay_step', type=int, default=50, help='')
train_arg.add_argument('--decay_rate', type=float, default=0.99, help='')
train_arg.add_argument('--max_grad_norm', type=float, default=-1, help='')
train_arg.add_argument('--optimizer', type=str,
                       default='adam', choices=['adam_wgan', 'adam', 'sgd', 'rmsprop'], help='')
train_arg.add_argument('--checkpoint_secs', type=int, default=300, help='')
train_arg.add_argument('--dropout', type=float, default=0.1, help='')
train_arg.add_argument('--regularization', type=float, default=1e-4, help='')


# Model args
model_arg = add_argument_group('Model')
# hxl changes
model_arg.add_argument('--model_type', type=str, default='hist',
                        choices=['hist', 'avg'], help='')
model_arg.add_argument('--filter', type=str, default='chebyshev5',
                        choices=['chebyshev5', 'conv1', 'conv2','chebyshev5_temp',
                                 'lanczos', 'learn_heat'], help='')
model_arg.add_argument('--brelu', type=str, default='b1relu',
                        choices=['b1relu', 'b2relu'], help='')

model_arg.add_argument('--pool', type=str, default='mpool1',
                        choices=['mpool1', 'mpool2'], help='')

# LSM model args
model_arg.add_argument('--T', type=int, default=1, help='')
# default value for HW dataset,
model_arg.add_argument('--k', type=int, default=10, help='')
model_arg.add_argument('--lamda', type=float, default=pow(2, 3), help='')


model_arg.add_argument('--gamma', type=float, default=2e-5, help='')
model_arg.add_argument('--eval_frequency', type=int, default=20, help='')
model_arg.add_argument('--converge_loss', type=float, default=0.01, help='')

# Hyperparams for graph
graph_arg = add_argument_group('Graph')
graph_arg.add_argument('--feat_in', type=int, default=1,
                       choices=[1, 4, 8], help='')
graph_arg.add_argument('--feat_out', type=int, default=1,
                       choices=[1, 4, 8], help='')
graph_arg.add_argument('--num_kernels', type=list, default=[32, 16])
graph_arg.add_argument('--conv_size', type=list, default=[8, 8])
graph_arg.add_argument('--pool_size', type=list, default=[4, 2])
graph_arg.add_argument('--FC_size', type=list, default=[])

# Miscellaneous (summary write, model reload)
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_step', type=int, default=20, help='')
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--load_path', type=str, default="")
misc_arg.add_argument('--gpu_memory_fraction', type=float, default=1.0)
misc_arg.add_argument('--output_dir', type=str, default='output')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

