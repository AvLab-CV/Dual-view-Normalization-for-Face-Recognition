#coding: utf-8
import os
import tensorflow as tf
from PIL import Image
from DVN import DVN
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
  net = DVN()
  
  source = tf.placeholder(tf.float32, [1, 224, 224, 3], name='source')
  normal_f = tf.placeholder(tf.float32, [1, 224, 224, 3], name='normal_f')
  normal_s = tf.placeholder(tf.float32, [1, 224, 224, 3], name='normal_s')
  net.build_up(source, normal_f, normal_s)
  
  print('Load Finetuned Model Successfully!')
  
  # Train or Test
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config, graph=net.graph) as sess:
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver(max_to_keep=0)  #
    saver.restore(sess, cfg.checkpoint_ft)

    for idx, (roots, dirs, files) in enumerate(os.walk(ImgPath)):
      for file in files:

        img_src = read_img(roots, file)
        syn_f, syn_s = sess.run([net.gen_p_051, net.gen_p_240], {source: img_src, net.is_train: False})  #
        save_img('{}/{}'.format(SavePath, file), img_src, syn_f, syn_s)
        print(file)

if __name__ == "__main__":
  tf.app.run()
  
