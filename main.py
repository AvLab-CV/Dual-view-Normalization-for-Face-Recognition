import os
import tensorflow as tf
from config import cfg
from Layer1_Generator import Layer1_Generator
from Layer2_Generator import Layer2_Generator
from utils.utils import loadData
import datetime
import shutil

def main(_):

  # Environment Setting
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_id
  
  cfg.results = '{}/{}'.format(cfg.results, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
  cfg.checkpoint = '{}/{}'.format(cfg.checkpoint, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
  cfg.summary_dir = '{}/{}'.format(cfg.summary_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
  
  if not os.path.exists(cfg.results): os.makedirs(cfg.results)
  if not os.path.exists(cfg.checkpoint): os.makedirs(cfg.checkpoint)
  if not os.path.exists(cfg.summary_dir): os.makedirs(cfg.summary_dir)
  shutil.copy('./main.py', './{}/main.py'.format(cfg.checkpoint))
  shutil.copy('./config.py', './{}/config.py'.format(cfg.checkpoint))
  shutil.copy('./Layer1_Generator.py', './{}/Layer1_Generator.py'.format(cfg.checkpoint))
  shutil.copy('./Layer2_Generator.py', './{}/Layer2_Generator.py'.format(cfg.checkpoint))
  
  # Construct Networks
  data_feed = loadData(batch_size=cfg.batch_size, train_shuffle=True)  # False
  source, normal_f, normal_s, num_batch = data_feed.get_train()

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

  net1 = Layer1_Generator()
  train_op, gen, loss = net1.build_up(source, normal_f, normal_s, is_train)
  net2 = Layer2_Generator()
  train_op_ref, gen_ref, loss_ref = net2.build_up(src_set, nml_f_set, nml_s_set, src_f_l2_set,
                                                  src_s_l2_set, nml_ff_l2_set, nml_sf_l2_set,
                                                  nml_fs_l2_set, nml_ss_l2_set, is_train)

  # Train or Test
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    # load pretrained with different name tensorflow
    # Start Thread
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.compat.v1.train.Saver(max_to_keep=0)  #
    if cfg.is_pretrained:
      var = tf.global_variables()
      var_to_restore = [val for val in var if 'decoder_f' in val.name or 'decoder_s' in val.name or 'discriminator_f' in val.name or 'discriminator_s' in val.name]
      saver = tf.compat.v1.train.Saver(var_to_restore)
      saver.restore(sess, cfg.checkpoint_ft)
      print('Load Finetuned Model Successfully!')

    if cfg.is_resume:
      sess.run(tf.global_variables_initializer())
      saver.restore(sess, cfg.checkpoint_ft)
      print('Resumed Model Successfully!')

    writer = tf.compat.v1.summary.FileWriter(cfg.summary_dir, sess.graph)

    for epoch in range(cfg.epoch):
      for step in range(num_batch):


        ## Train Layer-1 generators
        if (step < 25 and epoch == 0):
          critic = 25
        else:
          critic = cfg.critic
        for i in range(critic):
            _ = sess.run(train_op[2], {is_train:True})  # Train front-view discriminator
            _ = sess.run(train_op[3], {is_train:True})  # Train front-view discriminator
        _ = sess.run(train_op[0], {is_train: True})  # Train front-view generator
        _, fl, sl, ftl, dl, gl, summary = sess.run([train_op[1], loss[0], loss[1], loss[2], loss[3], loss[4], loss[5]], {is_train: True})  # Train side-view generator
        print('{}-{}, Fea Loss:{:.4f}, Sym Loss:{:.4f}, Reconst Loss:{:.4f}, D Loss:{:.4f}, G Loss:{:.4f}'
              .format(epoch, step, fl*cfg.lambda_fea, sl*cfg.lambda_sym, ftl*cfg.lambda_l1, dl, gl))

        ## Train Layer-2 generators
        def Argument():
          src_f_l2, src_s_l2, nml_ff_l2, nml_ss_l2, nml_sf_l2, nml_fs_l2, src, nml_f, nml_s = sess.run([gen[0], gen[1], gen[2], gen[3],
                                                                                                        gen[4], gen[5], gen[6], gen[7],
                                                                                                        gen[8]],  {is_train: False})  # Train front-view generator
          InputAugument = {src_set: src, nml_f_set: nml_f, nml_s_set: nml_s, src_f_l2_set: src_f_l2,
                           src_s_l2_set: src_s_l2, nml_ff_l2_set: nml_ff_l2, nml_sf_l2_set: nml_sf_l2,
                           nml_fs_l2_set: nml_fs_l2, nml_ss_l2_set: nml_ss_l2, is_train: True}
          return InputAugument
        if (step < 25 and epoch == 0):
          critic = 25
        else:
          critic = cfg.critic
        for i in range(critic):
            _ = sess.run(train_op_ref[2], Argument())  # Train front-view discriminator
            _ = sess.run(train_op_ref[3], Argument())  # Train front-view discriminator
        _ = sess.run(train_op_ref[0], Argument())  # Train front-view generator
        _, fl, sl, ftl, dl, gl, summary = sess.run([train_op_ref[1], loss_ref[0], loss_ref[1],
                                                    loss_ref[2], loss_ref[3], loss_ref[4], loss_ref[5]], Argument())  # Train side-view generator
        print('Layer-2, epoch {}- step {}, Fea Loss:{:.4f}, Sym Loss:{:.4f}, Reconst Loss:{:.4f}, D Loss:{:.4f}, G Loss:{:.4f}'
              .format(epoch, step, fl*cfg.lambda_fea, sl*cfg.lambda_sym, ftl*cfg.lambda_l1, dl, gl))


        # Save Model and Summary and Test
        if(step % cfg.save_freq == 0):
          writer.add_summary(summary, epoch*num_batch + step)
          print("Saving Model....")
          if cfg.is_pretrained:
            saver_pre = tf.train.Saver()
            saver_pre.save(sess, os.path.join(cfg.checkpoint, 'ck-epoch{}-step{}'.format(epoch, step)))
          else:
            saver.save(sess, os.path.join(cfg.checkpoint, 'ck-epoch{}-step{}'.format(epoch, step)))  #

          for i in range(int(800/cfg.batch_size)):
            te_profile, te_front = data_feed.get_test_batch(cfg.batch_size)
            images_f, images_s = sess.run([gen[0], gen[1]], {source: te_profile, is_train: False}) #
            data_feed.save_images(images_f, 'f', epoch, step)
            data_feed.save_images(images_s, 's', epoch, step)
            images_f_ref, images_s_ref = sess.run([gen_ref[0], gen_ref[1]], {src_set: te_profile, is_train: False})  #
            data_feed.save_images(images_f_ref, '2nd_f', epoch, step)
            data_feed.save_images(images_s_ref, '2nd_s', epoch, step)

    
    # Close Threads
    coord.request_stop()
    coord.join(threads)
    
    
if __name__ == "__main__":
  tf.compat.v1.app.run()
