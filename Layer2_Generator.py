from Pretrained.resnet50 import Resnet50
from utils.ops import *
from utils.objective import *
from utils.optimizer import optimizer
import tensorflow.contrib.slim as slim

epsilon = 1e-9


class Layer2_Generator(object):
    """Class for Face Normalization Model

    This class is for face normalization task, including the following three contributions.

    1. Feature-embedded: Embedding pretrained face recognition model in G.
    2. Attention Mechanism: Attention Discriminator for elaborate image.
    3. Pixel Loss: normal-to-normal transform introduce pixel-wise loss.

    """

    def __init__(self):
        self.graph = tf.compat.v1.get_default_graph()
        self.batch_size = cfg.batch_size

    def build_up(self, source, normal_f, normal_s, src_f_l2, src_s_l2, nml_ff_l2, nml_sf_l2, nml_fs_l2, nml_ss_l2, is_train):
        self.is_train = is_train
        with tf.compat.v1.variable_scope('face_model'):
            self.face_model = Resnet50()
            self.face_model.build()
            print('VGG model built successfully.')

        ##################################################################################################################
        # Use pretrained model(vgg-face) as encoder of Generator （Real to Generated)
        self.feature_src = self.face_model.forward(source, 'src_enc_l2')
        self.feature_f = self.face_model.forward(normal_f, 'normal_f_enc_l2')
        self.feature_s = self.face_model.forward(normal_s, 'normal_s_enc_l2')
        print('Face model output feature shape:', self.feature_src[-1].get_shape())

        # Layer-2 real to generated (Synthesization)
        self.gen_src_f = self.generator_l2_f(self.feature_src)
        self.gen_src_s = self.generator_l2_s(self.feature_src)
        self.gen_ff = self.generator_l2_f(self.feature_f, reuse=True)
        self.gen_ss = self.generator_l2_s(self.feature_s, reuse=True)
        self.gen_sf = self.generator_l2_f(self.feature_s, reuse=True)
        self.gen_fs = self.generator_l2_s(self.feature_f, reuse=True)
        # Layer-2 real to generated (feature extraction)
        self.feature_gen_src_f = self.face_model.forward(self.gen_src_f, 'gen_src_f_real_enc')
        self.feature_gen_src_s = self.face_model.forward(self.gen_src_s, 'gen_src_s_real_enc')
        self.feature_gen_ff = self.face_model.forward(self.gen_ff, 'gen_nml_ff_real_enc')
        self.feature_gen_sf = self.face_model.forward(self.gen_ss, 'gen_nml_sf_real_enc')
        self.feature_gen_fs = self.face_model.forward(self.gen_fs, 'gen_nml_fs_real_enc')
        self.feature_gen_ss = self.face_model.forward(self.gen_sf, 'gen_nml_ss_real_enc')
        print('Feature of Generated Image shape:', self.feature_gen_src_f[-1].get_shape())

        ##################################################################################################################
        # Use pretrained model(vgg-face) as encoder of Generator （Generated to Generated)
        self.feature_gen_src_f_l2 = self.face_model.forward(src_f_l2, 'gen_src_f_syn_enc')
        self.feature_gen_src_s_l2 = self.face_model.forward(src_s_l2, 'gen_src_s_syn_enc')
        self.feature_gen_nml_ff_l2 = self.face_model.forward(nml_ff_l2, 'gen_nml_ff_syn_enc')
        self.feature_gen_nml_sf_l2 = self.face_model.forward(nml_sf_l2, 'gen_nml_sf_syn_enc')
        self.feature_gen_nml_fs_l2 = self.face_model.forward(nml_fs_l2, 'gen_nml_fs_syn_enc')
        self.feature_gen_nml_ss_l2 = self.face_model.forward(nml_ss_l2, 'gen_nml_ss_syn_enc')

        # Layer-2 real to generated (Synthesization)
        self.gen_src_ff_l2 = self.generator_l2_f(self.feature_gen_src_f_l2, reuse=True)   # src -> ff, ss
        self.gen_src_ss_l2 = self.generator_l2_s(self.feature_gen_src_s_l2, reuse=True)
        self.gen_src_sf_l2 = self.generator_l2_f(self.feature_gen_src_s_l2, reuse=True)   # src -> sf, fs
        self.gen_src_fs_l2 = self.generator_l2_s(self.feature_gen_src_f_l2, reuse=True)
        self.gen_fff_l2 = self.generator_l2_f(self.feature_gen_nml_ff_l2, reuse=True)  # fff, ffs
        self.gen_ffs_l2 = self.generator_l2_s(self.feature_gen_nml_ff_l2, reuse=True)
        self.gen_sff_l2 = self.generator_l2_f(self.feature_gen_nml_sf_l2, reuse=True)  # sff, sfs
        self.gen_sfs_l2 = self.generator_l2_s(self.feature_gen_nml_sf_l2, reuse=True)
        self.gen_fsf_l2 = self.generator_l2_f(self.feature_gen_nml_fs_l2, reuse=True)  # fsf, fss
        self.gen_fss_l2 = self.generator_l2_s(self.feature_gen_nml_fs_l2, reuse=True)
        self.gen_ssf_l2 = self.generator_l2_f(self.feature_gen_nml_ss_l2, reuse=True)  # ssf, sss
        self.gen_sss_l2 = self.generator_l2_s(self.feature_gen_nml_ss_l2, reuse=True)




        return [self.gen_src_f, self.gen_src_s]




    def generator_l2_f(self, feature, reuse=False):
        with tf.compat.v1.variable_scope('generator_l2_f', reuse=reuse) as scope:
            norm = bn if (cfg.norm == 'bn') else pixel_norm

            feat28, feat14, feat7, pool5 = feature[0], feature[1], feature[2], feature[3]
            feat7 = tf.nn.relu(norm(conv2d(feat7, 512, 'conv1', kernel_size=1, strides=1), self.is_train, 'norm1'))
            res1_1 = res_block(feat7, 'res1_1', self.is_train, cfg.norm)
            res1_2 = res_block(res1_1, 'res1_2', self.is_train, cfg.norm)
            res1_3 = res_block(res1_2, 'res1_3', self.is_train, cfg.norm)
            res1_4 = res_block(res1_3, 'res1_4', self.is_train, cfg.norm)
            dconv2 = tf.nn.relu(norm(deconv2d(res1_4, 256, 'dconv2', kernel_size=4, strides=2), self.is_train, 'norm2'))
            res2 = res_block(dconv2, 'res2', self.is_train, cfg.norm)
            dconv3 = tf.nn.relu(norm(deconv2d(res2, 128, 'dconv3', kernel_size=4, strides=2), self.is_train, 'norm3'))
            res3 = res_block(dconv3, 'res3', self.is_train, cfg.norm)
            dconv4 = tf.nn.relu(norm(deconv2d(res3, 64, 'dconv4', kernel_size=4, strides=2), self.is_train, 'norm4'))
            res4 = res_block(dconv4, 'res4', self.is_train, cfg.norm)
            dconv5 = tf.nn.relu(norm(deconv2d(res4, 32, 'dconv5', kernel_size=4, strides=2), self.is_train, 'norm5'))
            res5 = res_block(dconv5, 'res5', self.is_train, cfg.norm)
            dconv6 = tf.nn.relu(norm(deconv2d(res5, 32, 'dconv6', kernel_size=4, strides=2), self.is_train, 'norm6'))
            res6 = res_block(dconv6, 'res6', self.is_train, cfg.norm)
            gen = tf.nn.tanh(conv2d(res6, 3, 'pw_conv', kernel_size=1, strides=1))

            return (gen + 1) * 127.5

    def generator_l2_s(self, feature, reuse=False):
        with tf.compat.v1.variable_scope('generator_l2_s', reuse=reuse) as scope:
            norm = bn if (cfg.norm == 'bn') else pixel_norm

            feat28, feat14, feat7, pool5 = feature[0], feature[1], feature[2], feature[3]
            feat7 = tf.nn.relu(norm(conv2d(feat7, 512, 'conv1', kernel_size=1, strides=1), self.is_train, 'norm1'))
            res1_1 = res_block(feat7, 'res1_1', self.is_train, cfg.norm)
            res1_2 = res_block(res1_1, 'res1_2', self.is_train, cfg.norm)
            res1_3 = res_block(res1_2, 'res1_3', self.is_train, cfg.norm)
            res1_4 = res_block(res1_3, 'res1_4', self.is_train, cfg.norm)
            dconv2 = tf.nn.relu(norm(deconv2d(res1_4, 256, 'dconv2', kernel_size=4, strides=2), self.is_train, 'norm2'))
            res2 = res_block(dconv2, 'res2', self.is_train, cfg.norm)
            dconv3 = tf.nn.relu(norm(deconv2d(res2, 128, 'dconv3', kernel_size=4, strides=2), self.is_train, 'norm3'))
            res3 = res_block(dconv3, 'res3', self.is_train, cfg.norm)
            dconv4 = tf.nn.relu(norm(deconv2d(res3, 64, 'dconv4', kernel_size=4, strides=2), self.is_train, 'norm4'))
            res4 = res_block(dconv4, 'res4', self.is_train, cfg.norm)
            dconv5 = tf.nn.relu(norm(deconv2d(res4, 32, 'dconv5', kernel_size=4, strides=2), self.is_train, 'norm5'))
            res5 = res_block(dconv5, 'res5', self.is_train, cfg.norm)
            dconv6 = tf.nn.relu(norm(deconv2d(res5, 32, 'dconv6', kernel_size=4, strides=2), self.is_train, 'norm6'))
            res6 = res_block(dconv6, 'res6', self.is_train, cfg.norm)
            gen = tf.nn.tanh(conv2d(res6, 3, 'pw_conv', kernel_size=1, strides=1))

            return (gen + 1) * 127.5

    def discriminator_l2_f(self, images, reuse=False):
        with tf.compat.v1.variable_scope("discriminator_l2_f", reuse=reuse):
            norm = slim.layer_norm

            images = images / 127.5 - 1
            bs = images.get_shape().as_list()[0]
            face = tf.slice(images, [0, 40, 34, 0], [bs, 150, 156, cfg.channel])  # [40:190,34:190,:]

            with tf.compat.v1.variable_scope("images"):
                with tf.compat.v1.variable_scope('d_conv0'):
                    h0_0 = lrelu(conv2d(images, 32, 'd_conv0', kernel_size=3, strides=2))
                # h0 is (112 x 112 x 32)
                with tf.compat.v1.variable_scope('d_conv1'):
                    h0_1 = lrelu(norm(conv2d(h0_0, 64, 'd_conv1', kernel_size=3, strides=2)))
                # h1 is (56 x 56 x 64)
                with tf.compat.v1.variable_scope('d_conv2'):
                    h0_2 = lrelu(norm(conv2d(h0_1, 128, 'd_conv2', kernel_size=3, strides=2)))
                # h2 is (28 x 28 x 128)
                with tf.compat.v1.variable_scope('d_conv3'):
                    h0_3 = lrelu(norm(conv2d(h0_2, 256, 'd_conv3', kernel_size=3, strides=2)))
                # h3 is (14 x 14 x 256)
                with tf.compat.v1.variable_scope('d_conv4'):
                    h0_4 = lrelu(norm(conv2d(h0_3, 256, 'd_conv4', kernel_size=3, strides=2)))
                # h4 is (7 x 7 x 256)
                with tf.compat.v1.variable_scope('d_fc'):
                    h0_4 = tf.reshape(h0_4, [bs, -1])
                    h0_5 = fullyConnect(h0_4, 1, 'd_fc')
                # h5 is (1)

            with tf.compat.v1.variable_scope("face"):
                with tf.compat.v1.variable_scope('d_conv0'):
                    h4_0 = lrelu(conv2d(face, 32, 'd_conv0', kernel_size=3, strides=2))
                # h0 is (58 x 62 x 32)
                with tf.compat.v1.variable_scope('d_conv1'):
                    h4_1 = lrelu(norm(conv2d(h4_0, 64, 'd_conv1', kernel_size=3, strides=2)))
                # h1 is (29 x 31 x 64)
                with tf.compat.v1.variable_scope('d_conv2'):
                    h4_2 = lrelu(norm(conv2d(h4_1, 128, 'd_conv2', kernel_size=3, strides=2)))
                # h2 is (15 x 16 x 128)
                with tf.compat.v1.variable_scope('d_conv3'):
                    h4_3 = lrelu(norm(conv2d(h4_2, 256, 'd_conv3', kernel_size=3, strides=2)))
                # h3 is (8 x 8 x 256)
                with tf.compat.v1.variable_scope('d_fc'):
                    h4_3 = tf.reshape(h4_3, [bs, -1])
                    h4_4 = fullyConnect(h4_3, 1, 'd_fc')
                # h4 is (1)

            return h0_5, h4_4

    def discriminator_l2_s(self, images, reuse=False):
        with tf.compat.v1.variable_scope("discriminator_l2_s", reuse=reuse):
            norm = slim.layer_norm

            images = images / 127.5 - 1
            bs = images.get_shape().as_list()[0]

            # modified
            face = tf.slice(images, [0, 40, 34, 0], [bs, 150, 156, cfg.channel])  # [40:190,34:190,:]

            with tf.compat.v1.variable_scope("images"):
                with tf.compat.v1.variable_scope('d_conv0'):
                    h0_0 = lrelu(conv2d(images, 32, 'd_conv0', kernel_size=3, strides=2))
                # h0 is (112 x 112 x 32)
                with tf.compat.v1.variable_scope('d_conv1'):
                    h0_1 = lrelu(norm(conv2d(h0_0, 64, 'd_conv1', kernel_size=3, strides=2)))
                # h1 is (56 x 56 x 64)
                with tf.compat.v1.variable_scope('d_conv2'):
                    h0_2 = lrelu(norm(conv2d(h0_1, 128, 'd_conv2', kernel_size=3, strides=2)))
                # h2 is (28 x 28 x 128)
                with tf.compat.v1.variable_scope('d_conv3'):
                    h0_3 = lrelu(norm(conv2d(h0_2, 256, 'd_conv3', kernel_size=3, strides=2)))
                # h3 is (14 x 14 x 256)
                with tf.compat.v1.variable_scope('d_conv4'):
                    h0_4 = lrelu(norm(conv2d(h0_3, 256, 'd_conv4', kernel_size=3, strides=2)))
                # h4 is (7 x 7 x 256)
                with tf.compat.v1.variable_scope('d_fc'):
                    h0_4 = tf.reshape(h0_4, [bs, -1])
                    h0_5 = fullyConnect(h0_4, 1, 'd_fc')
                # h5 is (1)

            with tf.compat.v1.variable_scope("face"):
                with tf.compat.v1.variable_scope('d_conv0'):
                    h4_0 = lrelu(conv2d(face, 32, 'd_conv0', kernel_size=3, strides=2))
                # h0 is (58 x 62 x 32)
                with tf.compat.v1.variable_scope('d_conv1'):
                    h4_1 = lrelu(norm(conv2d(h4_0, 64, 'd_conv1', kernel_size=3, strides=2)))
                # h1 is (29 x 31 x 64)
                with tf.compat.v1.variable_scope('d_conv2'):
                    h4_2 = lrelu(norm(conv2d(h4_1, 128, 'd_conv2', kernel_size=3, strides=2)))
                # h2 is (15 x 16 x 128)
                with tf.compat.v1.variable_scope('d_conv3'):
                    h4_3 = lrelu(norm(conv2d(h4_2, 256, 'd_conv3', kernel_size=3, strides=2)))
                # h3 is (8 x 8 x 256)
                with tf.compat.v1.variable_scope('d_fc'):
                    h4_3 = tf.reshape(h4_3, [bs, -1])
                    h4_4 = fullyConnect(h4_3, 1, 'd_fc')
                # h4 is (1)

            return h0_5, h4_4

    def loss_l2_f(self, normal_f):
            with tf.name_scope('Regularation_Loss'):
                self.reg_gen_f_l2 = Regularation_Loss(self.vars_l2_f, cfg)
                self.reg_dis_f_l2 = Regularation_Loss(self.vars_l2_dis_f, cfg)
                tf.add_to_collection('losses_l2_f', self.reg_gen_f_l2)
                tf.add_to_collection('losses_l2_f', self.reg_dis_f_l2)

            with tf.name_scope('Adversarial_Loss'):
                self.d_loss_f_l2, self.g_loss_f_l2 = Adversarial_Loss(self.dr_f_l2,
                                                                      [self.df1_f_l2, self.df2_f_l2, self.df3_f_l2, self.df4_f_l2,
                                                                       self.df5_f_l2, self.df6_f_l2, self.df7_f_l2, self.df8_f_l2,
                                                                       self.df9_f_l2], cfg)
                tf.add_to_collection('losses_l2_f', self.d_loss_f_l2)
                tf.add_to_collection('losses_l2_f', self.g_loss_f_l2)

            with tf.name_scope('Symmetric_Loss'):
                self.sym_loss_f_l2 = Symmetric_Loss([self.gen_fff_l2, self.gen_fsf_l2, self.gen_sff_l2, self.gen_ssf_l2,
                                                     self.gen_src_ff_l2, self.gen_src_sf_l2, self.gen_ff, self.gen_sf,
                                                     self.gen_src_f], cfg)
                tf.add_to_collection('losses_l2_f', self.sym_loss_f_l2)

            # 7. Total Loss
            with tf.name_scope('Total_Loss'):  #
                self.gen_loss_f_l2 = cfg.lambda_l1 * self.front_loss_l2 + cfg.lambda_fea * self.feature_loss_l2 + \
                                  cfg.lambda_gan * self.g_loss_f_l2 + self.reg_gen_f_l2 + cfg.lambda_sym * self.sym_loss_f_l2
                self.dis_loss_f_l2 = cfg.lambda_gan * self.d_loss_f_l2 + cfg.lambda_gp * self.gradient_penalty_f_l2 + \
                                         self.reg_dis_f_l2

    def loss_l2_s(self, normal_s):
            with tf.name_scope('Regularation_Loss'):
                self.reg_gen_s_l2 = Regularation_Loss(self.vars_l2_s, cfg)
                self.reg_dis_s_l2 = Regularation_Loss(self.vars_l2_dis_s, cfg)
                tf.add_to_collection('losses_l2_s', self.reg_gen_s_l2)
                tf.add_to_collection('losses_l2_s', self.reg_dis_s_l2)

            with tf.name_scope('Adversarial_Loss'):
                self.d_loss_s_l2, self.g_loss_s_l2 = Adversarial_Loss(self.dr_s_l2,
                                                    [self.df1_s_l2, self.df2_s_l2, self.df3_s_l2, self.df4_s_l2,
                                                     self.df5_s_l2, self.df6_s_l2, self.df7_s_l2, self.df8_s_l2,
                                                     self.df9_s_l2], cfg)
                tf.add_to_collection('losses_l2_s', self.d_loss_s_l2)
                tf.add_to_collection('losses_l2_s', self.g_loss_s_l2)

            # 7. Total Loss
            with tf.name_scope('Total_Loss'):  #
                self.gen_loss_s_l2 = cfg.lambda_l1 * self.front_loss_l2 + cfg.lambda_fea * self.feature_loss_l2 + \
                                  cfg.lambda_gan * self.g_loss_s_l2 + self.reg_gen_s_l2
                self.dis_loss_s_l2 = cfg.lambda_gan * self.d_loss_s_l2 + cfg.lambda_gp * self.gradient_penalty_s_l2 + \
                                         self.reg_dis_s_l2


if '__name__' == '__main__':
    net = Layer2_Generator()
