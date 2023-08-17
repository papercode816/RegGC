import numpy as np
import tensorflow as tf

from util import layers
class attn():
    def inference_no_scl(inputs, attn_drop, ffd_drop,
                  weight_mt, bias_mat, hid_units, n_heads, residual, activation=tf.nn.elu):
        attns = []
        for j in range(n_heads):
            attns.append(layers.attn_head_no_scl(inputs,weight_mt=weight_mt, bias_mat=bias_mat,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
        h_1 = tf.stack(attns, axis=2)

        return h_1

    def inference(inputs, attn_drop, ffd_drop,
                  weight_mt, bias_mat,context_f_ac_2, hid_units, n_heads, residual, activation=tf.nn.elu):
        attns = []
        for j in range(n_heads):
            attns.append(layers.attn_head(inputs,weight_mt=weight_mt, bias_mat=bias_mat,context_f_ac_2=context_f_ac_2,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))

        h_1 = tf.stack(attns, axis=2)

        return h_1

    def inference_gc(inputs, attn_drop, ffd_drop,
                  weight_mt, bias_mat,context_f_ac_2, hid_units, n_heads, residual, activation=tf.nn.elu):
        attns = []
        for j in range(n_heads):

            attns.append(layers.attn_head_gc(inputs,weight_mt=weight_mt, bias_mat=bias_mat,context_f_ac_2=context_f_ac_2,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))

        h_1 = tf.stack(attns, axis=2)

        return h_1
