import tensorflow as tf

def optimizer(cfg, global_step, loss, var_list):

    train = tf.compat.v1.train.AdamOptimizer(cfg.lr,
                                   beta1=cfg.beta1, beta2=cfg.beta2).minimize(loss,
                                   global_step=global_step, var_list=var_list)
    return train