# coding: utf-8
# --------------------------------------------------------
# FNM
# Written by Yichen Qian
# --------------------------------------------------------

import tensorflow as tf

flags = tf.app.flags

############################
#    hyper parameters      #
############################

# For hyper parameters
flags.DEFINE_float('lambda_l1', 0.001, 'weight of the loss for L1 texture loss') # 0.001
flags.DEFINE_float('lambda_sym', 0, 'weight of the loss for symmetry loss') # 0.001
flags.DEFINE_float('lambda_consistency', 0.1, 'weight of the loss for consistency loss') # 0.001
flags.DEFINE_float('lambda_fea', 3500, 'weight of the loss for face model feature loss')
flags.DEFINE_float('lambda_reg', 1e-5, 'weight of the loss for L2 regularitaion loss')
flags.DEFINE_float('lambda_gan', 1, 'weight of the loss for gan loss')
flags.DEFINE_float('lambda_gp', 10, 'weight of the loss for gradient penalty on parameter of D')

# For training/validation (Path)
flags.DEFINE_integer('dataset_size', 277336, 'number of non-normal face set')
flags.DEFINE_string('source_path', '../../Database', 'dataset path')  # casia_aligned_250_250_jpg
flags.DEFINE_string('source_list', '../DataList/FNM_CASIA_TrainList_FOCropped_250_250_84.txt', 'train source list')
flags.DEFINE_string('normal_f_path', '../../Database', 'front-view data path')
flags.DEFINE_string('normal_f_list', '../DataList/FNM_MPIE_TrainList_FOCropped_250_250_84_woill_0_337.txt', 'train front-view normal list')
flags.DEFINE_string('normal_s_path', '../../Database', 'side-view data path')
flags.DEFINE_string('normal_s_list', '../DataList/FNM_MPIE_TrainList_FOCropped_190deg_250_250_84_woill6_9_0_337.txt', 'train side-view normal list')
flags.DEFINE_string('test_path', '../../Database', 'test data path')
flags.DEFINE_string('test_list', '../DataList/FNM_CASIATEST_TrainList_FOCropped_250_250_84.txt', 'test list')

# For evaluation (Path)
flags.DEFINE_string('eval_input_path', './Eval/Face_Cropped', 'evaluation data path')
flags.DEFINE_string('eval_save_path', './Eval/Face_Syn', 'evaluation data path')


# Option
flags.DEFINE_boolean('is_train', True, 'train or test')
flags.DEFINE_boolean('is_finetune', False, 'finetune') # False, True
flags.DEFINE_string('face_model', './Pretrained/resnet50.npy', 'face model path')
flags.DEFINE_string('checkpoint', 'checkpoint', 'checkpoint directory')
flags.DEFINE_string('summary_dir', 'log', 'logs directory')
flags.DEFINE_string('checkpoint_ft', './Pretrained/DVN/ck-epoch2-step0', 'finetune or test checkpoint path')
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_integer('epoch', 10, 'epoch')
flags.DEFINE_integer('critic', 1, 'number of D training times')
flags.DEFINE_integer('save_freq', 8500, 'the frequency of saving model')
flags.DEFINE_float('lr', 1e-4, 'base learning rate')
flags.DEFINE_float('beta1', 0., 'beta1 momentum term of adam')
flags.DEFINE_float('beta2', 0.9, 'beta2 momentum term of adam')
flags.DEFINE_float('stddev', 0.02, 'stddev for W initializer')
flags.DEFINE_boolean('use_bias', False, 'whether to use bias')
flags.DEFINE_string('norm', 'bn', 'normalize function for G')
flags.DEFINE_string('results', 'results_valid', 'path for saving results') #

############################
#   environment setting    #
############################
flags.DEFINE_string('device_id', '0', 'device id')
flags.DEFINE_integer('ori_height', 250, 'original height of source images')
flags.DEFINE_integer('ori_width', 250, 'original width of source images')
flags.DEFINE_integer('height', 224, 'height of images') # do not modified
flags.DEFINE_integer('width', 224, 'width of images') # do not modified
flags.DEFINE_integer('channel', 3, 'channel of images')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing examples')


cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
