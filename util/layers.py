import numpy as np
import tensorflow as tf
import numpy as np

conv1d = tf.layers.conv1d

def attn_head_no_scl(seq, out_sz,weight_mt, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    # traffic attention's update
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        N,M,TB = seq.get_shape()
        N, M, TB = int(N),int(M),int(TB)

        seq_fts = tf.layers.conv1d(seq, TB, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        # coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if weight_mt != None:
            weight_mt = tf.tile(tf.expand_dims(weight_mt, axis=1), [1,M,1])

            eye_m = tf.constant([(np.eye(weight_mt.shape[1])).tolist() for i in range(weight_mt.shape[0])])
            tmp = tf.clip_by_value((weight_mt + eye_m), clip_value_min=0, clip_value_max=1)

            weight_mask = -1e9 * (1.0 - tmp)
            coefs_tmp = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat + weight_mask)

        if weight_mt == None:
            coefs_tmp = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs_tmp = tf.nn.dropout(coefs_tmp, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals_traf = tf.matmul(coefs_tmp, seq_fts)
        ret = tf.contrib.layers.bias_add(vals_traf)

    # residual connection
    if residual:
        if seq.shape[-1] != ret.shape[-1]:
            ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
        else:
            ret = ret + seq

    return activation(ret)  # activation

def attn_head_gc(seq, out_sz,weight_mt, bias_mat,context_f_ac_2, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    # traffic attention's update
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        N,M,TB = seq.get_shape()
        N, M, TB = int(N),int(M),int(TB)

        seq_fts = tf.layers.conv1d(seq, TB, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])

        if weight_mt != None:
            weight_mt = tf.tile(tf.expand_dims(weight_mt, axis=1), [1,M,1])

            eye_m = tf.constant([(np.eye(weight_mt.shape[1])).tolist() for i in range(weight_mt.shape[0])])
            tmp = tf.clip_by_value((weight_mt + eye_m), clip_value_min=0, clip_value_max=1)

            weight_mask = -1e9 * (1.0 - tmp)
            coefs_tmp = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat + weight_mask)

        if weight_mt == None:
            coefs_tmp = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs_tmp = tf.nn.dropout(coefs_tmp, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals_traf = tf.matmul(coefs_tmp, seq_fts)
        ret_traf = tf.contrib.layers.bias_add(vals_traf)

    ret = ret_traf

    # residual connection
    if residual:
        if seq.shape[-1] != ret.shape[-1]:
            ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
        else:
            ret = ret + seq

    return activation(ret)  # activation

def attn_head(seq, out_sz,weight_mt, bias_mat,context_f_ac_2, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    # traffic attention's update
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        N,M,TB = seq.get_shape()
        N, M, TB = int(N),int(M),int(TB)

        seq_fts = tf.layers.conv1d(seq, TB, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        if weight_mt != None:
            weight_mt = tf.tile(tf.expand_dims(weight_mt, axis=1), [1,M,1])

            eye_m = tf.constant([(np.eye(weight_mt.shape[1])).tolist() for i in range(weight_mt.shape[0])])
            tmp = tf.clip_by_value((weight_mt + eye_m), clip_value_min=0, clip_value_max=1)

            weight_mask = -1e9 * (1.0 - tmp)
            coefs_tmp = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat + weight_mask)

        if weight_mt == None:
            coefs_tmp = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs_tmp = tf.nn.dropout(coefs_tmp, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals_traf = tf.matmul(coefs_tmp, seq_fts)
        ret_traf = tf.contrib.layers.bias_add(vals_traf)

    # context attention's update
    with tf.name_scope('context_attn'):
        seq_fts_cont = tf.layers.conv1d(seq, TB, 1, use_bias=False)
        # simplest self-attention possible
        context_f_ac_2 = tf.expand_dims(context_f_ac_2, axis=0)
        f_1_cont = tf.layers.conv1d(context_f_ac_2, 1, 1)
        f_2_cont = tf.layers.conv1d(context_f_ac_2, 1, 1)
        logits_cont = f_1_cont + tf.transpose(f_2_cont, [0, 2, 1])
        coefs_cont = tf.nn.softmax(tf.nn.leaky_relu(logits_cont + bias_mat))
        if coef_drop != 0.0:
            coefs_cont = tf.nn.dropout(coefs_cont, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts_cont = tf.nn.dropout(seq_fts_cont, 1.0 - in_drop)
        vals_cont = tf.matmul(coefs_cont, seq_fts_cont)
        ret_cont = tf.contrib.layers.bias_add(vals_cont)

    ret = tf.concat([ret_traf, ret_cont], axis=-1)

    # residual connection
    if residual:
        if seq.shape[-1] != ret.shape[-1]:
            ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
        else:
            ret = ret + seq

    return activation(ret)  # activation