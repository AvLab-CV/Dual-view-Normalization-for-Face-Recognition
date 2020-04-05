# coding: utf-8
# --------------------------------------------------------
# FNM
# Written by Yichen Qian
# --------------------------------------------------------

import tensorflow as tf
from PIL import Image
from config import cfg
from utils import loadData
from Pretrained.resnet50 import Resnet50
from ops import *
from objective import *
import tensorflow.contrib.slim as slim


epsilon = 1e-9

class DVN(object):
  """Class for Face Normalization Model
  
  This class is for face normalization task, including the following three contributions.
  
  1. Feature-embedded: Embedding pretrained face recognition model in G.
  2. Attention Mechanism: Attention Discriminator for elaborate image.
  3. Pixel Loss: front-to-front transform introduce pixel-wise loss.

  """
  def __init__(self):
    self.graph = tf.get_default_graph()
    #with self.graph.as_default():
    self.batch_size = cfg.batch_size
    self.is_train = tf.placeholder(tf.bool, name='is_train')

  def build_up(self, profile, front, front_240):
    """Build up architecture
    
    1. Pretrained face recognition model forward
    2. Decoder from feature to image
    3. Refeed generated image to face recognition model
    4. Feed generated image to Discriminator
    5. Construct 'Grade Penalty' for discriminator
    
    """
    
    #with self.graph.as_default():
    # Construct Template Model (G_enc) to encoder input face
    with tf.variable_scope('face_model'):
      self.face_model = Resnet50()
      self.face_model.build()
      print('VGG model built successfully.')
    
    # Use pretrained model(vgg-face) as encoder of Generator
    self.feature_p = self.face_model.forward(profile,'profile_enc')
    self.feature_f = self.face_model.forward(front, 'front_enc')
    self.feature_f_240 = self.face_model.forward(front_240, 'front_240_enc')
    print('Face model output feature shape:', self.feature_p[-1].get_shape())
    
    # Decoder front face from vgg feature
    self.gen_p_051 = self.decoder_051(self.feature_p)
    self.gen_f_051 = self.decoder_051(self.feature_f, reuse=True)
    self.gen_p_240 = self.decoder_240(self.feature_p)
    self.gen_f_240 = self.decoder_240(self.feature_f_240, reuse=True)


    print('Generator output shape:', self.gen_p_051.get_shape())
    
    # Map texture into features again by VGG  
    self.feature_gen_p_051 = self.face_model.forward(self.gen_p_051,'profile_gen_enc')
    self.feature_gen_f_051 = self.face_model.forward(self.gen_f_051, 'front_gen_enc')
    self.feature_gen_p_240 = self.face_model.forward(self.gen_p_240, 'profile_240_gen_enc')
    self.feature_gen_f_240 = self.face_model.forward(self.gen_f_240, 'front_240_gen_enc')

    # Cycle Term
    self.gen_p_051_cy = self.refiner_051(self.feature_gen_p_051)
    self.gen_f_051_cy = self.refiner_051(self.feature_gen_f_051, reuse=True)
    self.gen_p_240_cy = self.refiner_240(self.feature_gen_p_240)
    self.gen_f_240_cy = self.refiner_240(self.feature_gen_f_240, reuse=True)
    self.feature_gen_f_051_cy = self.face_model.forward(self.gen_f_051_cy, 'front_gen_enc_cy_051')
    self.feature_gen_f_240_cy = self.face_model.forward(self.gen_f_240_cy, 'front_gen_enc_cy_240')
    self.feature_gen_p_051_cy = self.face_model.forward(self.gen_p_051_cy, 'profile_gen_enc_cy_240')
    self.feature_gen_p_240_cy = self.face_model.forward(self.gen_p_240_cy, 'profile_gen_enc_cy_051')
    print('Feature of Generated Image shape:', self.feature_gen_p_051[-1].get_shape())
    
    # (decoders) Construct discriminator between generalized front face and ground truth
    self.dr_051 = self.discriminator_051(front)
    self.df1_051 = self.discriminator_051(self.gen_p_051, reuse=True)
    self.df2_051 = self.discriminator_051(self.gen_f_051, reuse=True)
    self.dr_240 = self.discriminator_240(front_240)
    self.df1_240 = self.discriminator_240(self.gen_p_240, reuse=True)
    self.df2_240 = self.discriminator_240(self.gen_f_240, reuse=True)
    # (refiners) Construct discriminator between generalized front face and ground truth
    self.dr_051_cy = self.discriminator_ref_051(front)
    self.df1_051_cy = self.discriminator_ref_051(self.gen_p_051_cy, reuse=True)
    self.df2_051_cy = self.discriminator_ref_051(self.gen_f_051_cy, reuse=True)
    self.dr_240_cy = self.discriminator_ref_240(front_240)
    self.df1_240_cy = self.discriminator_ref_240(self.gen_p_240_cy, reuse=True)
    self.df2_240_cy = self.discriminator_ref_240(self.gen_f_240_cy, reuse=True)

    # Gradient Penalty #
    with tf.name_scope('gp_051'):
      inter = Interpolate(self.gen_p_051, front)
      d = self.discriminator_051(inter, reuse=True)
      self.gradient_penalty, slopes = Gradient_Penalty(d, inter)
      self.grad4 = tf.reduce_mean(slopes)

    with tf.name_scope('ref_gp_051'):
      inter = Interpolate(self.gen_p_051_cy, front)
      d = self.discriminator_ref_051(inter, reuse=True)
      self.gradient_penalty_051_cy, slopes = Gradient_Penalty(d, inter)

    with tf.name_scope('gp_240'):
      inter = Interpolate(self.gen_p_240, front_240)
      d = self.discriminator_240(inter, reuse=True)
      self.gradient_penalty_240, slopes = Gradient_Penalty(d, inter)
      self.grad4_240 = tf.reduce_mean(slopes)

    with tf.name_scope('ref_gp_240'):
      inter = Interpolate(self.gen_p_240_cy, front_240)
      d = self.discriminator_ref_240(inter, reuse=True)
      self.gradient_penalty_240_cy, slopes = Gradient_Penalty(d, inter)

    # Cycle Consistency term
    with tf.name_scope('CycleConsistency_Loss'):
      self.front_consistency = Perceptual_Loss(self.feature_f, self.feature_gen_f_051_cy, self.feature_p, self.feature_gen_p_051_cy)
      self.profile_consistency = Perceptual_Loss(self.feature_f_240, self.feature_gen_f_240_cy, self.feature_p, self.feature_gen_p_240_cy)
      self.cycle_consistency_loss = self.front_consistency + self.profile_consistency

    
    # Get Viraibles
    all_vars = tf.trainable_variables()
    self.vars_gen_051 = [var for var in all_vars if var.name.startswith('decoder_051')]
    self.vars_gen_240 = [var for var in all_vars if var.name.startswith('decoder_240')]
    self.vars_dis_051 = [var for var in all_vars if var.name.startswith('discriminator_051')]
    self.vars_dis_240 = [var for var in all_vars if var.name.startswith('discriminator_240')]
    # Refiner's pair
    self.vars_ref_051 = [var for var in all_vars if var.name.startswith('refiner_051')]
    self.vars_ref_240 = [var for var in all_vars if var.name.startswith('refiner_240')]
    self.vars_ref_dis_051 = [var for var in all_vars if var.name.startswith('discriminator_ref_051')]
    self.vars_ref_dis_240 = [var for var in all_vars if var.name.startswith('discriminator_ref_240')]

    self.loss_051(front)
    self.loss_240(front_240)
    self.loss_refiner_051(front)
    self.loss_refiner_240(front_240)
             
    # Ops for debug
    with tf.name_scope('Debug'):
      grad1 = tf.gradients([self.feature_loss], [self.gen_p_051])[0]  # feature gradient
      self.grad1 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(grad1), [1,2,3])))
      grad2 = tf.gradients([self.g_loss], [self.gen_p_051])[0]  # generator gradient
      self.grad2 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(grad2), [1,2,3])))
      grad3 = tf.gradients([self.front_loss], [self.gen_f_051])[0]  # L1 front gradient
      self.grad3 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(grad3), [1,2,3])))



    # Summary
    self._summary()  
    
    # Optimizer (1st phase, default settings)
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.train_gen = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2).minimize(self.gen_loss, global_step=self.global_step, var_list=self.vars_gen_051)
    self.train_dis = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2).minimize(self.dis_loss, global_step=self.global_step, var_list=self.vars_dis_051)
    self.train_gen_240 = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2).minimize(self.gen_loss_240, global_step=self.global_step, var_list=self.vars_gen_240)
    self.train_dis_240 = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2).minimize(self.dis_loss_240, global_step=self.global_step, var_list=self.vars_dis_240)

    # Optimizer (2nd phase, generator only)
    self.train_gen_2nd = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2).minimize(self.gen_loss_2nd, global_step=self.global_step, var_list=self.vars_gen_051)
    self.train_gen_240_2nd = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2).minimize(self.gen_loss_240_2nd, global_step=self.global_step, var_list=self.vars_gen_240)

    # Optimizer (refiners)
    self.train_ref_dis = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2).minimize(self.ref_dis_loss_051_cy, global_step=self.global_step, var_list=self.vars_ref_dis_051)
    self.train_ref_dis_240 = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2).minimize(self.ref_dis_loss_240_cy, global_step=self.global_step, var_list=self.vars_ref_dis_240)
    self.train_ref_051 = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2).minimize(self.ref_loss_051, global_step=self.global_step, var_list=self.vars_ref_051)
    self.train_ref_240 = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2).minimize(self.ref_loss_240, global_step=self.global_step, var_list=self.vars_ref_240)
        
  def decoder_051(self, feature, reuse=False):
    with tf.variable_scope('decoder_051', reuse=reuse) as scope:
      norm = bn if(cfg.norm=='bn') else pixel_norm

      feat28,feat14,feat7,pool5 = feature[0],feature[1],feature[2],feature[3]
      feat7 = tf.nn.relu(norm(conv2d(feat7, 512, 'conv1',  kernel_size=1, strides = 1),self.is_train,'norm1'))
      res1_1 = res_block(feat7, 'res1_1',self.is_train, cfg.norm)
      res1_2 = res_block(res1_1, 'res1_2',self.is_train, cfg.norm)
      res1_3 = res_block(res1_2, 'res1_3',self.is_train, cfg.norm)
      res1_4 = res_block(res1_3, 'res1_4',self.is_train, cfg.norm)
      dconv2 = tf.nn.relu(norm(deconv2d(res1_4, 256, 'dconv2',  kernel_size=4, strides = 2),self.is_train,'norm2'))
      res2 = res_block(dconv2, 'res2',self.is_train, cfg.norm)
      dconv3 = tf.nn.relu(norm(deconv2d(res2, 128, 'dconv3', kernel_size=4, strides = 2),self.is_train,'norm3'))
      res3 = res_block(dconv3, 'res3',self.is_train, cfg.norm)
      dconv4 = tf.nn.relu(norm(deconv2d(res3, 64, 'dconv4', kernel_size=4, strides = 2),self.is_train,'norm4'))
      res4 = res_block(dconv4, 'res4',self.is_train, cfg.norm)
      dconv5 = tf.nn.relu(norm(deconv2d(res4, 32, 'dconv5', kernel_size=4, strides = 2),self.is_train,'norm5'))
      res5 = res_block(dconv5, 'res5',self.is_train, cfg.norm)
      dconv6 = tf.nn.relu(norm(deconv2d(res5, 32, 'dconv6', kernel_size=4, strides = 2),self.is_train,'norm6'))
      res6 = res_block(dconv6, 'res6',self.is_train, cfg.norm)
      gen = tf.nn.tanh(conv2d(res6, 3, 'pw_conv', kernel_size=1, strides = 1))
    
      return (gen + 1) * 127.5

  def decoder_240(self, feature, reuse=False):
    with tf.variable_scope('decoder_240', reuse=reuse) as scope:
      norm = bn if (cfg.norm == 'bn') else pixel_norm

      feat28,feat14,feat7,pool5 = feature[0],feature[1],feature[2],feature[3]
      feat7 = tf.nn.relu(norm(conv2d(feat7, 512, 'conv1',  kernel_size=1, strides = 1),self.is_train,'norm1'))
      res1_1 = res_block(feat7, 'res1_1',self.is_train, cfg.norm)
      res1_2 = res_block(res1_1, 'res1_2',self.is_train, cfg.norm)
      res1_3 = res_block(res1_2, 'res1_3',self.is_train, cfg.norm)
      res1_4 = res_block(res1_3, 'res1_4',self.is_train, cfg.norm)
      dconv2 = tf.nn.relu(norm(deconv2d(res1_4, 256, 'dconv2',  kernel_size=4, strides = 2),self.is_train,'norm2'))
      res2 = res_block(dconv2, 'res2',self.is_train, cfg.norm)
      dconv3 = tf.nn.relu(norm(deconv2d(res2, 128, 'dconv3', kernel_size=4, strides = 2),self.is_train,'norm3'))
      res3 = res_block(dconv3, 'res3',self.is_train, cfg.norm)
      dconv4 = tf.nn.relu(norm(deconv2d(res3, 64, 'dconv4', kernel_size=4, strides = 2),self.is_train,'norm4'))
      res4 = res_block(dconv4, 'res4',self.is_train, cfg.norm)
      dconv5 = tf.nn.relu(norm(deconv2d(res4, 32, 'dconv5', kernel_size=4, strides = 2),self.is_train,'norm5'))
      res5 = res_block(dconv5, 'res5',self.is_train, cfg.norm)
      dconv6 = tf.nn.relu(norm(deconv2d(res5, 32, 'dconv6', kernel_size=4, strides = 2),self.is_train,'norm6'))
      res6 = res_block(dconv6, 'res6',self.is_train, cfg.norm)
      gen = tf.nn.tanh(conv2d(res6, 3, 'pw_conv', kernel_size=1, strides = 1))

      return (gen + 1) * 127.5

  def refiner_051(self, feature, reuse=False):
    with tf.variable_scope('refiner_051', reuse=reuse) as scope:
      norm = bn if (cfg.norm == 'bn') else pixel_norm

      feat28,feat14,feat7,pool5 = feature[0],feature[1],feature[2],feature[3]
      feat7 = tf.nn.relu(norm(conv2d(feat7, 512, 'conv1',  kernel_size=1, strides = 1),self.is_train,'norm1'))
      res1_1 = res_block(feat7, 'res1_1',self.is_train, cfg.norm)
      res1_2 = res_block(res1_1, 'res1_2',self.is_train, cfg.norm)
      res1_3 = res_block(res1_2, 'res1_3',self.is_train, cfg.norm)
      res1_4 = res_block(res1_3, 'res1_4',self.is_train, cfg.norm)
      dconv2 = tf.nn.relu(norm(deconv2d(res1_4, 256, 'dconv2',  kernel_size=4, strides = 2),self.is_train,'norm2'))
      res2 = res_block(dconv2, 'res2',self.is_train, cfg.norm)
      dconv3 = tf.nn.relu(norm(deconv2d(res2, 128, 'dconv3', kernel_size=4, strides = 2),self.is_train,'norm3'))
      res3 = res_block(dconv3, 'res3',self.is_train, cfg.norm)
      dconv4 = tf.nn.relu(norm(deconv2d(res3, 64, 'dconv4', kernel_size=4, strides = 2),self.is_train,'norm4'))
      res4 = res_block(dconv4, 'res4',self.is_train, cfg.norm)
      dconv5 = tf.nn.relu(norm(deconv2d(res4, 32, 'dconv5', kernel_size=4, strides = 2),self.is_train,'norm5'))
      res5 = res_block(dconv5, 'res5',self.is_train, cfg.norm)
      dconv6 = tf.nn.relu(norm(deconv2d(res5, 32, 'dconv6', kernel_size=4, strides = 2),self.is_train,'norm6'))
      res6 = res_block(dconv6, 'res6',self.is_train, cfg.norm)
      gen = tf.nn.tanh(conv2d(res6, 3, 'pw_conv', kernel_size=1, strides = 1))

      return (gen + 1) * 127.5

  def refiner_240(self, feature, reuse=False):
    with tf.variable_scope('refiner_240', reuse=reuse) as scope:
      norm = bn if (cfg.norm == 'bn') else pixel_norm

      feat28,feat14,feat7,pool5 = feature[0],feature[1],feature[2],feature[3]
      feat7 = tf.nn.relu(norm(conv2d(feat7, 512, 'conv1',  kernel_size=1, strides = 1),self.is_train,'norm1'))
      res1_1 = res_block(feat7, 'res1_1',self.is_train, cfg.norm)
      res1_2 = res_block(res1_1, 'res1_2',self.is_train, cfg.norm)
      res1_3 = res_block(res1_2, 'res1_3',self.is_train, cfg.norm)
      res1_4 = res_block(res1_3, 'res1_4',self.is_train, cfg.norm)
      dconv2 = tf.nn.relu(norm(deconv2d(res1_4, 256, 'dconv2',  kernel_size=4, strides = 2),self.is_train,'norm2'))
      res2 = res_block(dconv2, 'res2',self.is_train, cfg.norm)
      dconv3 = tf.nn.relu(norm(deconv2d(res2, 128, 'dconv3', kernel_size=4, strides = 2),self.is_train,'norm3'))
      res3 = res_block(dconv3, 'res3',self.is_train, cfg.norm)
      dconv4 = tf.nn.relu(norm(deconv2d(res3, 64, 'dconv4', kernel_size=4, strides = 2),self.is_train,'norm4'))
      res4 = res_block(dconv4, 'res4',self.is_train, cfg.norm)
      dconv5 = tf.nn.relu(norm(deconv2d(res4, 32, 'dconv5', kernel_size=4, strides = 2),self.is_train,'norm5'))
      res5 = res_block(dconv5, 'res5',self.is_train, cfg.norm)
      dconv6 = tf.nn.relu(norm(deconv2d(res5, 32, 'dconv6', kernel_size=4, strides = 2),self.is_train,'norm6'))
      res6 = res_block(dconv6, 'res6',self.is_train, cfg.norm)
      gen = tf.nn.tanh(conv2d(res6, 3, 'pw_conv', kernel_size=1, strides = 1))

      return (gen + 1) * 127.5

    
  def discriminator_051(self, images, reuse=False):

    with tf.variable_scope("discriminator_051", reuse=reuse) as scope:
      norm = slim.layer_norm
      
      images = images / 127.5 - 1
      bs = images.get_shape().as_list()[0]
      face = tf.slice(images, [0,40,34,0], [bs,150,156,cfg.channel])  # [40:190,34:190,:]
      
      with tf.variable_scope("images"):
        with tf.variable_scope('d_conv0'):
          h0_0 = lrelu(conv2d(images, 32, 'd_conv0', kernel_size=3, strides=2))
        # h0 is (112 x 112 x 32)
        with tf.variable_scope('d_conv1'):
          h0_1 = lrelu(norm(conv2d(h0_0, 64, 'd_conv1', kernel_size=3, strides=2)))
        # h1 is (56 x 56 x 64)
        with tf.variable_scope('d_conv2'):
          h0_2 = lrelu(norm(conv2d(h0_1, 128, 'd_conv2', kernel_size=3, strides=2)))
        # h2 is (28 x 28 x 128)
        with tf.variable_scope('d_conv3'):
          h0_3 = lrelu(norm(conv2d(h0_2, 256, 'd_conv3', kernel_size=3, strides=2)))
        # h3 is (14 x 14 x 256)
        with tf.variable_scope('d_conv4'):
          h0_4 = lrelu(norm(conv2d(h0_3, 256, 'd_conv4', kernel_size=3, strides=2)))
        # h4 is (7 x 7 x 256)
        with tf.variable_scope('d_fc'):
          h0_4 = tf.reshape(h0_4, [bs, -1])
          h0_5 = fullyConnect(h0_4, 1, 'd_fc')
        # h5 is (1)

      with tf.variable_scope("face"):
        with tf.variable_scope('d_conv0'):
          h4_0 = lrelu(conv2d(face, 32, 'd_conv0', kernel_size=3, strides=2))
        # h0 is (58 x 62 x 32)
        with tf.variable_scope('d_conv1'):
          h4_1 = lrelu(norm(conv2d(h4_0, 64, 'd_conv1', kernel_size=3, strides=2)))
        # h1 is (29 x 31 x 64)
        with tf.variable_scope('d_conv2'):
          h4_2 = lrelu(norm(conv2d(h4_1, 128, 'd_conv2', kernel_size=3, strides=2)))
        # h2 is (15 x 16 x 128)
        with tf.variable_scope('d_conv3'):
          h4_3 = lrelu(norm(conv2d(h4_2, 256, 'd_conv3', kernel_size=3, strides=2)))
        # h3 is (8 x 8 x 256)
        with tf.variable_scope('d_fc'):
          h4_3 = tf.reshape(h4_3, [bs, -1])
          h4_4 = fullyConnect(h4_3, 1, 'd_fc')
        # h4 is (1)
      
      return h0_5, h4_4

  def discriminator_240(self, images, reuse=False):
    with tf.variable_scope("discriminator_240", reuse=reuse) as scope:
      norm = slim.layer_norm

      images = images / 127.5 - 1
      bs = images.get_shape().as_list()[0]

      # modified
      face = tf.slice(images, [0, 40, 34, 0], [bs, 150, 156, cfg.channel])  # [40:190,34:190,:]

      with tf.variable_scope("images_240"):
        with tf.variable_scope('d_conv0'):
          h0_0 = lrelu(conv2d(images, 32, 'd_conv0', kernel_size=3, strides=2))
        # h0 is (112 x 112 x 32)
        with tf.variable_scope('d_conv1'):
          h0_1 = lrelu(norm(conv2d(h0_0, 64, 'd_conv1', kernel_size=3, strides=2)))
        # h1 is (56 x 56 x 64)
        with tf.variable_scope('d_conv2'):
          h0_2 = lrelu(norm(conv2d(h0_1, 128, 'd_conv2', kernel_size=3, strides=2)))
        # h2 is (28 x 28 x 128)
        with tf.variable_scope('d_conv3'):
          h0_3 = lrelu(norm(conv2d(h0_2, 256, 'd_conv3', kernel_size=3, strides=2)))
        # h3 is (14 x 14 x 256)
        with tf.variable_scope('d_conv4'):
          h0_4 = lrelu(norm(conv2d(h0_3, 256, 'd_conv4', kernel_size=3, strides=2)))
        # h4 is (7 x 7 x 256)
        with tf.variable_scope('d_fc'):
          h0_4 = tf.reshape(h0_4, [bs, -1])
          h0_5 = fullyConnect(h0_4, 1, 'd_fc')
        # h5 is (1)

      with tf.variable_scope("face_240"):
        with tf.variable_scope('d_conv0'):
          h4_0 = lrelu(conv2d(face, 32, 'd_conv0', kernel_size=3, strides=2))
        # h0 is (58 x 62 x 32)
        with tf.variable_scope('d_conv1'):
          h4_1 = lrelu(norm(conv2d(h4_0, 64, 'd_conv1', kernel_size=3, strides=2)))
        # h1 is (29 x 31 x 64)
        with tf.variable_scope('d_conv2'):
          h4_2 = lrelu(norm(conv2d(h4_1, 128, 'd_conv2', kernel_size=3, strides=2)))
        # h2 is (15 x 16 x 128)
        with tf.variable_scope('d_conv3'):
          h4_3 = lrelu(norm(conv2d(h4_2, 256, 'd_conv3', kernel_size=3, strides=2)))
        # h3 is (8 x 8 x 256)
        with tf.variable_scope('d_fc'):
          h4_3 = tf.reshape(h4_3, [bs, -1])
          h4_4 = fullyConnect(h4_3, 1, 'd_fc')
        # h4 is (1)

      return h0_5, h4_4

  def discriminator_ref_051(self, images, reuse=False):

    with tf.variable_scope("discriminator_ref_051", reuse=reuse) as scope:
      norm = slim.layer_norm
      
      images = images / 127.5 - 1
      bs = images.get_shape().as_list()[0]
      face = tf.slice(images, [0,40,34,0], [bs,150,156,cfg.channel])  # [40:190,34:190,:]
      
      with tf.variable_scope("images"):
        with tf.variable_scope('d_conv0'):
          h0_0 = lrelu(conv2d(images, 32, 'd_conv0', kernel_size=3, strides=2))
        # h0 is (112 x 112 x 32)
        with tf.variable_scope('d_conv1'):
          h0_1 = lrelu(norm(conv2d(h0_0, 64, 'd_conv1', kernel_size=3, strides=2)))
        # h1 is (56 x 56 x 64)
        with tf.variable_scope('d_conv2'):
          h0_2 = lrelu(norm(conv2d(h0_1, 128, 'd_conv2', kernel_size=3, strides=2)))
        # h2 is (28 x 28 x 128)
        with tf.variable_scope('d_conv3'):
          h0_3 = lrelu(norm(conv2d(h0_2, 256, 'd_conv3', kernel_size=3, strides=2)))
        # h3 is (14 x 14 x 256)
        with tf.variable_scope('d_conv4'):
          h0_4 = lrelu(norm(conv2d(h0_3, 256, 'd_conv4', kernel_size=3, strides=2)))
        # h4 is (7 x 7 x 256)
        with tf.variable_scope('d_fc'):
          h0_4 = tf.reshape(h0_4, [bs, -1])
          h0_5 = fullyConnect(h0_4, 1, 'd_fc')
        # h5 is (1)

      with tf.variable_scope("face"):
        with tf.variable_scope('d_conv0'):
          h4_0 = lrelu(conv2d(face, 32, 'd_conv0', kernel_size=3, strides=2))
        # h0 is (58 x 62 x 32)
        with tf.variable_scope('d_conv1'):
          h4_1 = lrelu(norm(conv2d(h4_0, 64, 'd_conv1', kernel_size=3, strides=2)))
        # h1 is (29 x 31 x 64)
        with tf.variable_scope('d_conv2'):
          h4_2 = lrelu(norm(conv2d(h4_1, 128, 'd_conv2', kernel_size=3, strides=2)))
        # h2 is (15 x 16 x 128)
        with tf.variable_scope('d_conv3'):
          h4_3 = lrelu(norm(conv2d(h4_2, 256, 'd_conv3', kernel_size=3, strides=2)))
        # h3 is (8 x 8 x 256)
        with tf.variable_scope('d_fc'):
          h4_3 = tf.reshape(h4_3, [bs, -1])
          h4_4 = fullyConnect(h4_3, 1, 'd_fc')
        # h4 is (1)
      
      return h0_5, h4_4

  def discriminator_ref_240(self, images, reuse=False):
    with tf.variable_scope("discriminator_ref_240", reuse=reuse) as scope:
      norm = slim.layer_norm

      images = images / 127.5 - 1
      bs = images.get_shape().as_list()[0]

      # modified
      face = tf.slice(images, [0, 40, 34, 0], [bs, 150, 156, cfg.channel])  # [40:190,34:190,:]

      with tf.variable_scope("images_240"):
        with tf.variable_scope('d_conv0'):
          h0_0 = lrelu(conv2d(images, 32, 'd_conv0', kernel_size=3, strides=2))
        # h0 is (112 x 112 x 32)
        with tf.variable_scope('d_conv1'):
          h0_1 = lrelu(norm(conv2d(h0_0, 64, 'd_conv1', kernel_size=3, strides=2)))
        # h1 is (56 x 56 x 64)
        with tf.variable_scope('d_conv2'):
          h0_2 = lrelu(norm(conv2d(h0_1, 128, 'd_conv2', kernel_size=3, strides=2)))
        # h2 is (28 x 28 x 128)
        with tf.variable_scope('d_conv3'):
          h0_3 = lrelu(norm(conv2d(h0_2, 256, 'd_conv3', kernel_size=3, strides=2)))
        # h3 is (14 x 14 x 256)
        with tf.variable_scope('d_conv4'):
          h0_4 = lrelu(norm(conv2d(h0_3, 256, 'd_conv4', kernel_size=3, strides=2)))
        # h4 is (7 x 7 x 256)
        with tf.variable_scope('d_fc'):
          h0_4 = tf.reshape(h0_4, [bs, -1])
          h0_5 = fullyConnect(h0_4, 1, 'd_fc')
        # h5 is (1)

      with tf.variable_scope("face_240"):
        with tf.variable_scope('d_conv0'):
          h4_0 = lrelu(conv2d(face, 32, 'd_conv0', kernel_size=3, strides=2))
        # h0 is (58 x 62 x 32)
        with tf.variable_scope('d_conv1'):
          h4_1 = lrelu(norm(conv2d(h4_0, 64, 'd_conv1', kernel_size=3, strides=2)))
        # h1 is (29 x 31 x 64)
        with tf.variable_scope('d_conv2'):
          h4_2 = lrelu(norm(conv2d(h4_1, 128, 'd_conv2', kernel_size=3, strides=2)))
        # h2 is (15 x 16 x 128)
        with tf.variable_scope('d_conv3'):
          h4_3 = lrelu(norm(conv2d(h4_2, 256, 'd_conv3', kernel_size=3, strides=2)))
        # h3 is (8 x 8 x 256)
        with tf.variable_scope('d_fc'):
          h4_3 = tf.reshape(h4_3, [bs, -1])
          h4_4 = fullyConnect(h4_3, 1, 'd_fc')
        # h4 is (1)

      return h0_5, h4_4


  def loss_051(self, front):

    with tf.name_scope('loss_051'):
      with tf.name_scope('Perceptual_Loss'):
        self.feature_loss = Perceptual_Loss(self.feature_f, self.feature_gen_f_051, self.feature_p, self.feature_gen_p_051)
        tf.add_to_collection('losses', self.feature_loss)

      with tf.name_scope('Front_Loss'):
        self.front_loss = Front_Loss(front, self.gen_f_051)
        tf.add_to_collection('losses', self.front_loss)

      with tf.name_scope('Regularation_Loss'):
        self.reg_gen = Regularation_Loss(self.vars_gen_051, cfg)
        self.reg_dis = Regularation_Loss(self.vars_dis_051, cfg)
        tf.add_to_collection('losses', self.reg_gen)
        tf.add_to_collection('losses', self.reg_dis)

      with tf.name_scope('Adversarial_Loss'):
        self.d_loss, self.g_loss = Adversarial_Loss(self.dr_051, self.df1_051, self.df2_051)
        tf.add_to_collection('losses', self.d_loss)
        tf.add_to_collection('losses', self.g_loss)

      with tf.name_scope('Symmetric_Loss'):
        self.sym_loss = Symmetric_Loss(self.gen_f_051, self. gen_p_051)
        tf.add_to_collection('losses', self.sym_loss)

      # 7. Total Loss
      with tf.name_scope('Total_Loss'):  #
        self.gen_loss = cfg.lambda_l1 * self.front_loss + cfg.lambda_fea * self.feature_loss + \
                        cfg.lambda_gan * self.g_loss + self.reg_gen
        self.gen_loss_2nd = cfg.lambda_l1 * self.front_loss + cfg.lambda_fea * self.feature_loss + \
                cfg.lambda_gan * self.g_loss + self.reg_gen + cfg.lambda_consistency * self.cycle_consistency_loss
        self.dis_loss = cfg.lambda_gan * self.d_loss + cfg.lambda_gp * self.gradient_penalty + \
                self.reg_dis

  def loss_240(self, front_240):

    with tf.name_scope('loss_240'):
      with tf.name_scope('Perceptual_Loss'):
        self.feature_loss_240 = Perceptual_Loss(self.feature_f_240, self.feature_gen_f_240, self.feature_p, self.feature_gen_p_240)
        tf.add_to_collection('losses_240', self.feature_loss_240)

      with tf.name_scope('Front_Loss'):
        self.front_loss_240 = Front_Loss(self.gen_f_240, front_240)
        tf.add_to_collection('losses_240', self.front_loss_240)

      with tf.name_scope('Regularation_Loss'):
        self.reg_gen_240 = Regularation_Loss(self.vars_gen_240, cfg)
        self.reg_dis_240 = Regularation_Loss(self.vars_dis_240, cfg)
        tf.add_to_collection('losses_240', self.reg_gen_240)
        tf.add_to_collection('losses_240', self.reg_dis_240)

      with tf.name_scope('Adversarial_Loss'):
        self.d_loss_240, self.g_loss_240 = Adversarial_Loss(self.dr_240, self.df1_240, self.df2_240)
        tf.add_to_collection('losses_240', self.d_loss_240)
        tf.add_to_collection('losses_240', self.g_loss_240)

      # 7. Total Loss
      with tf.name_scope('Total_Loss_240'):  #
        self.gen_loss_240 = cfg.lambda_l1 * self.front_loss_240 + cfg.lambda_fea * self.feature_loss_240 + \
                            cfg.lambda_gan * self.g_loss_240 + self.reg_gen_240
        self.gen_loss_240_2nd = cfg.lambda_l1 * self.front_loss_240 + cfg.lambda_fea * self.feature_loss_240 + \
                        cfg.lambda_gan * self.g_loss_240 + self.reg_gen_240 + cfg.lambda_consistency * self.cycle_consistency_loss
        self.dis_loss_240 = cfg.lambda_gan * self.d_loss_240 + cfg.lambda_gp * self.gradient_penalty_240 + \
                        self.reg_dis_240


  def loss_refiner_051(self, front):

    with tf.name_scope('loss_refiner_051'):
      with tf.name_scope('Front_Loss'):
        self.front_loss_051_cy = Front_Loss(self.gen_f_051_cy, front)
        tf.add_to_collection('losses_rf_051', self.front_loss_051_cy)

      with tf.name_scope('Regularation_Loss'):
        self.reg_gen_051_cy = Regularation_Loss(self.vars_ref_051, cfg)
        self.reg_dis_051_cy = Regularation_Loss(self.vars_ref_dis_051, cfg)
        tf.add_to_collection('losses_rf_051', self.reg_gen_051_cy)

      with tf.name_scope('Adversarial_Loss'):
        self.d_loss_051_cy, self.g_loss_051_cy = Adversarial_Loss(self.dr_051_cy, self.df1_051_cy, self.df2_051_cy)
        tf.add_to_collection('losses_rf_051', self.d_loss_051_cy)
        tf.add_to_collection('losses_rf_051', self.g_loss_051_cy)

      with tf.name_scope('Symmetric_Loss'):
        self.sym_loss_051_cy = Symmetric_Loss(self.gen_f_051_cy, self.gen_p_051_cy)
        tf.add_to_collection('losses_rf_051', self.sym_loss_051_cy)

      # 7. Total Loss
      with tf.name_scope('Total_Loss'):  #
        self.ref_loss_051 = cfg.lambda_l1 * self.front_loss_051_cy + cfg.lambda_fea * self.front_consistency + \
                        cfg.lambda_gan * self.g_loss_051_cy + self.reg_gen_051_cy
        self.ref_dis_loss_051_cy = cfg.lambda_gan * self.d_loss_051_cy + cfg.lambda_gp * self.gradient_penalty_051_cy + \
                        self.reg_dis_051_cy

  def loss_refiner_240(self, front_240):

    with tf.name_scope('loss_refiner_240'):
      with tf.name_scope('Front_Loss'):
        self.front_loss_240_cy = Front_Loss(self.gen_f_240_cy, front_240)
        tf.add_to_collection('losses_rf_240', self.front_loss_240_cy)

      with tf.name_scope('Regularation_Loss'):
        self.reg_gen_240_cy = Regularation_Loss(self.vars_ref_240, cfg)
        self.reg_dis_240_cy = Regularation_Loss(self.vars_ref_dis_240, cfg)
        tf.add_to_collection('losses_rf_240', self.reg_gen_240_cy)

      with tf.name_scope('Adversarial_Loss'):
        self.d_loss_240_cy, self.g_loss_240_cy = Adversarial_Loss(self.dr_240_cy, self.df1_240_cy, self.df2_240_cy)
        tf.add_to_collection('losses_rf_240', self.d_loss_240_cy)
        tf.add_to_collection('losses_rf_240', self.g_loss_240_cy)

      # 7. Total Loss
      with tf.name_scope('Total_Loss'):  #
        self.ref_loss_240 = cfg.lambda_l1 * self.front_loss_240_cy + cfg.lambda_fea * self.profile_consistency + \
                        cfg.lambda_gan * self.g_loss_240_cy + self.reg_gen_240_cy
        self.ref_dis_loss_240_cy = cfg.lambda_gan * self.d_loss_240_cy+ cfg.lambda_gp * self.gradient_penalty_240_cy + \
                        self.reg_dis_240_cy

  def _summary(self):
    """Tensorflow Summary"""
    
    train_summary = []
    train_summary.append(tf.summary.scalar('decoder_051/D_Loss', self.d_loss))
    train_summary.append(tf.summary.scalar('decoder_051/G_Loss', self.g_loss))
    train_summary.append(tf.summary.scalar('decoder_051/Front_loss', self.front_loss))
    train_summary.append(tf.summary.scalar('decoder_051/ID_loss', self.feature_loss))
    train_summary.append(tf.summary.scalar('decoder_051/Gradient_Penalty', self.gradient_penalty))
    train_summary.append(tf.summary.scalar('decoder_051/Cycle_Consistency', self.cycle_consistency_loss))

    train_summary.append(tf.summary.scalar('decoder_240/D_Loss', self.d_loss_240))
    train_summary.append(tf.summary.scalar('decoder_240/G_Loss', self.g_loss_240))
    train_summary.append(tf.summary.scalar('decoder_240/Front_loss', self.front_loss_240))
    train_summary.append(tf.summary.scalar('decoder_240/ID_loss', self.feature_loss_240))
    train_summary.append(tf.summary.scalar('decoder_240/Gradient_Penalty', self.gradient_penalty_240))
    train_summary.append(tf.summary.scalar('decoder_240/Cycle_Consistency', self.cycle_consistency_loss))

    # train_summary.append(tf.summary.scalar('refiner_051/D_Loss', self.d_loss_240))
    train_summary.append(tf.summary.scalar('refiner_051/D_Loss', self.ref_dis_loss_051_cy))
    train_summary.append(tf.summary.scalar('refiner_051/G_Loss', self.g_loss_051_cy))
    train_summary.append(tf.summary.scalar('refiner_051/Front_loss', self.front_loss_051_cy))
    train_summary.append(tf.summary.scalar('refiner_051/ID_loss', self.front_consistency))
    train_summary.append(tf.summary.scalar('refiner_051/Gradient_Penalty', self.gradient_penalty_051_cy))

    # train_summary.append(tf.summary.scalar('refiner_051/D_Loss', self.d_loss_240))
    train_summary.append(tf.summary.scalar('refiner_240/D_Loss', self.ref_dis_loss_240_cy))
    train_summary.append(tf.summary.scalar('refiner_240/G_Loss', self.g_loss_240_cy))
    train_summary.append(tf.summary.scalar('refiner_240/Front_loss', self.front_loss_240_cy))
    train_summary.append(tf.summary.scalar('refiner_240/ID_loss', self.profile_consistency))
    train_summary.append(tf.summary.scalar('refiner_240/Gradient_Penalty', self.gradient_penalty_240_cy))


    train_summary.append(tf.summary.scalar('train/grad_feature', self.grad1))
    train_summary.append(tf.summary.scalar('train/grad_D', self.grad2))
    train_summary.append(tf.summary.scalar('train/grad_front', self.grad3))
    self.train_summary = tf.summary.merge(train_summary)
    
    
if '__name__' == '__main__':
  net = WGAN_GP()
