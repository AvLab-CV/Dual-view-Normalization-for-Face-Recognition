import tensorflow as tf

epsilon = 1e-9
def Perceptual_Loss(Fea_list, cfg):

    feature_distance = tf.zeros([cfg.batch_size, 1])
    for features in Fea_list:
        real = features[0][-1] / (tf.norm(features[0][-1], axis=1, keepdims=True) + epsilon)   # [0] for real face
        syn = features[1][-1] / (tf.norm(features[1][-1], axis=1, keepdims=True) + epsilon)   # [1] for synthetic face
        feature_distance = feature_distance + (1 - tf.reduce_sum(tf.multiply(real, syn), [1]))

    feature_distance = feature_distance / len(Fea_list)
    feature_loss = tf.reduce_mean(feature_distance)

    return feature_loss

def Front_Loss(Tgt, Fea_list, cfg):

    front_tmp = tf.zeros([cfg.batch_size, 1])
    for features in Fea_list:
        front_tmp = front_tmp + tf.reduce_sum(tf.abs(Tgt / 255. - features/ 255.), [1, 2, 3])
    front_loss = tf.reduce_mean(front_tmp / len(Fea_list))

    return front_loss

def Regularation_Loss(vars, cfg):

    reg = tf.contrib.layers.apply_regularization(
        tf.contrib.layers.l2_regularizer(cfg.lambda_reg),
        weights_list=[var for var in vars if 'kernel' in var.name]
    )

    return reg

def Adversarial_Loss(Tgt, Syn_list, cfg):

    Syn_tmp = tf.zeros([cfg.batch_size, 1])
    for syn_fea in Syn_list:
        Syn_tmp = Syn_tmp + tf.add_n(syn_fea)
    Syn_tmp = Syn_tmp / len(Syn_list)

    d_loss = tf.reduce_mean(Syn_tmp - tf.add_n(Tgt)) / 2
    g_loss = - tf.reduce_mean(Syn_tmp) / 2

    return d_loss, g_loss

def Symmetric_Loss(Fea_list, cfg):

    sym_distanace = tf.zeros([cfg.batch_size, 1])
    for syn_face in Fea_list:
        mirror_syn_face = tf.reverse(syn_face, axis=[2])
        sym_distanace = sym_distanace + tf.reduce_sum(tf.abs(mirror_syn_face / 225. - syn_face / 255.), [1, 2, 3])
    sym_loss = tf.reduce_mean(sym_distanace / len(Fea_list))

    return sym_loss

def Interpolate(Input, Target):

    alpha = tf.random_uniform((Input.get_shape().as_list()[0], 1, 1, 1), minval=0., maxval=1., )
    inter = Target + alpha * (Input - Target)

    return inter

def Gradient_Penalty(Fea_list, cfg):

    slope_tmp = tf.zeros([cfg.batch_size])

    for inter in Fea_list:
        grad = tf.gradients([inter[0]], [inter[1]])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), [1, 2, 3]))
        slope_tmp = slope_tmp + tf.square(slopes - 1.)
    gradient_penalty = tf.reduce_mean(slope_tmp / len(Fea_list))

    return gradient_penalty

