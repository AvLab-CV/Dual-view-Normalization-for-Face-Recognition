#coding: utf-8
import os
import tensorflow as tf
from PIL import Image
from Layer2_Generator import Layer2_Generator
from config import cfg
import numpy as np

def read_img(path, img):
  '''Read test images'''
  img_array = np.array(Image.open(os.path.join(path, img)).resize((224,224)), dtype=np.float32)
  img_array = np.expand_dims(img_array, axis=0)
  return img_array

def save_img(path, img_src, syn_f, syn_s):
  '''Save test images'''
  img_array = np.squeeze(np.concatenate((img_src, syn_f, syn_s), 2).astype(np.uint8))
  Image.fromarray(img_array).save(path)

def main(_):
  if not os.path.exists('{}.meta'.format(cfg.checkpoint_ft)):
    print('Please Input the Valid Path of checkpoint!')
    exit()

  ImgPath = cfg.eval_input_path
  SavePath = cfg.eval_save_path

  if not os.path.exists(SavePath): os.makedirs(SavePath)
   
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_id
  
  cfg.batch_size = 1
  net = Layer2_Generator()
  
  src_set = tf.compat.v1.placeholder(tf.float32, [cfg.batch_size, 224, 224, 3], name='source_l2')
  nml_f_set = tf.compat.v1.placeholder(tf.float32, [cfg.batch_size, 224, 224, 3], name='source_l2')
  nml_s_set = tf.compat.v1.placeholder(tf.float32, [cfg.batch_size, 224, 224, 3], name='source_l2')
  src_f_l2_set = tf.compat.v1.placeholder(tf.float32, [cfg.batch_size, 224, 224, 3], name='source_l2')
  src_s_l2_set = tf.compat.v1.placeholder(tf.float32, [cfg.batch_size, 224, 224, 3], name='source_l2')
  nml_ff_l2_set = tf.compat.v1.placeholder(tf.float32, [cfg.batch_size, 224, 224, 3], name='normal_f_l2')
  nml_sf_l2_set = tf.compat.v1.placeholder(tf.float32, [cfg.batch_size, 224, 224, 3], name='normal_s_l2')
  nml_fs_l2_set = tf.compat.v1.placeholder(tf.float32, [cfg.batch_size, 224, 224, 3], name='normal_f_l2')
  nml_ss_l2_set = tf.compat.v1.placeholder(tf.float32, [cfg.batch_size, 224, 224, 3], name='normal_s_l2')
  is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')
  gen_ref = net.build_up(src_set, nml_f_set, nml_s_set, src_f_l2_set,
                                                  src_s_l2_set, nml_ff_l2_set, nml_sf_l2_set,
                                                  nml_fs_l2_set, nml_ss_l2_set, is_train)
  
  print('Load Finetuned Model Successfully!')
  
  # Train or Test
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver(max_to_keep=0)  #
    saver.restore(sess, cfg.checkpoint_ft)

    for idx, (roots, dirs, files) in enumerate(os.walk(ImgPath)):
      for file in files:

        img_src = read_img(roots, file)
        syn_f, syn_s = sess.run([gen_ref[0], gen_ref[1]], {src_set: img_src, net.is_train: False})  #
        save_img('{}/{}'.format(SavePath, file), img_src, syn_f, syn_s)
        print(file)

if __name__ == "__main__":
  tf.app.run()
  
