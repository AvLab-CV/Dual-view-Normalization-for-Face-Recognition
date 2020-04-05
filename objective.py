import tensorflow as tf

epsilon = 1e-9
def Perceptual_Loss(Normal_real, Normal_syn, NonNormal_real, NonNormal_syn):
    pool5_p_norm = NonNormal_real[-1] / (tf.norm(NonNormal_real[-1], axis=1, keepdims=True) + epsilon)
    pool5_f_norm = Normal_real[-1] / (tf.norm(Normal_real[-1], axis=1, keepdims=True) + epsilon)
    pool5_gen_p_norm = NonNormal_syn[-1] / (tf.norm(NonNormal_syn[-1], axis=1, keepdims=True) + epsilon)
    pool5_gen_f_norm = Normal_syn[-1] / (tf.norm(Normal_syn[-1], axis=1, keepdims=True) + epsilon)

    feature_distance = 0.5 * (1 - tf.reduce_sum(tf.multiply(pool5_p_norm, pool5_gen_p_norm), [1])) + \
                       0.5 * (1 - tf.reduce_sum(tf.multiply(pool5_f_norm, pool5_gen_f_norm), [1]))
    feature_loss = tf.reduce_mean(feature_distance)

    return feature_loss

def Front_Loss(Input, Tgt):

    front_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(Input / 255. - Tgt / 255.), [1, 2, 3]))

    return front_loss

def Regularation_Loss(vars, cfg):

    reg = tf.contrib.layers.apply_regularization(
        tf.contrib.layers.l2_regularizer(cfg.lambda_reg),
        weights_list=[var for var in vars if 'kernel' in var.name]
    )

    return reg

def Adversarial_Loss(Normal, NonNormal_Syn, Normal_syn):

    d_loss = tf.reduce_mean(tf.add_n(NonNormal_Syn) * 0.5 + tf.add_n(Normal_syn) * 0.5 - tf.add_n(Normal)) / 5
    g_loss = - tf.reduce_mean(tf.add_n(NonNormal_Syn) * 0.5 + tf.add_n(Normal_syn) * 0.5) / 5

    return d_loss, g_loss

def Symmetric_Loss(Normal_syn, NonNormal_syn):

    mirror_p = tf.reverse(NonNormal_syn, axis=[2])
    mirror_f = tf.reverse(Normal_syn, axis=[2])
    sym_distance = 0.5 * tf.reduce_sum(tf.abs(mirror_p / 225. - NonNormal_syn / 255.), [1, 2, 3]) + \
                        0.5 * tf.reduce_sum(tf.abs(mirror_f / 225. - Normal_syn / 255.), [1, 2, 3])
    sym_loss = tf.reduce_mean(sym_distance)

    return sym_loss

def Interpolate(Input, Target):

    alpha = tf.random_uniform((Input.get_shape().as_list()[0], 1, 1, 1), minval=0., maxval=1., )
    inter = Target + alpha * (Input - Target)

    return inter

def Gradient_Penalty(dis_output, interpolate_samples):

    grad = tf.gradients([dis_output], [interpolate_samples])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), [1, 2, 3]))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))

    return gradient_penalty, slopes

