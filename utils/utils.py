import os
import numpy as np
import tensorflow as tf
from PIL import Image
from config import cfg


class loadData(object):

    def __init__(self, batch_size=20, train_shuffle=True):
        self.batch_size = batch_size
        self.source = np.loadtxt(cfg.source_list, dtype='str', delimiter=',')
        self.normal_f = np.loadtxt(cfg.normal_f_list, dtype='str', delimiter=',')
        self.normal_s = np.loadtxt(cfg.normal_s_list, dtype='str', delimiter=',')

        if (train_shuffle):
            assert len(self.normal_f) == len(self.normal_s)
            p = np.random.permutation(len(self.normal_f))
            np.random.shuffle(self.source)
            self.normal_f = self.normal_f[p]
            self.normal_s = self.normal_s[p]

        self.test_list = np.loadtxt(cfg.test_list, dtype='str', delimiter=',')  #
        self.test_index = 0

        # Crop Box: left, upper, right, lower
        self.crop_box = [(cfg.ori_width - cfg.width) / 2, (cfg.ori_height - cfg.height) / 2,
                         (cfg.ori_width + cfg.width) / 2, (cfg.ori_height + cfg.height) / 2]
        assert Image.open(os.path.join(cfg.source_path, self.source[0])).size == \
               (cfg.ori_width, cfg.ori_height)

    def get_train(self):
        """Get train images

        Train images will be horizontal-flipped, center-cropped and adjust brightness randomly.

        return:
          profile (tf.tensor): profile of identity A
          front (tf.tensor): front face of identity B
        """
        def Concate_List(path, list):
            final_list = ['{}/{}'.format(path, img) for img in list if '{}/{}'.format(path, img)]
            if not len(final_list)==len(list):
                print('Some data missed !')
                exit()
            return final_list

        def Open_File(list):
            files = tf.train.string_input_producer(list, shuffle=False)  #
            _, value = tf.WholeFileReader().read(files)
            value = tf.image.decode_jpeg(value, channels=cfg.channel)
            value = tf.cast(value, tf.float32)
            return value

        def Adjust_src(value):
            value = tf.image.resize_images(value, [cfg.ori_height, cfg.ori_width])
            value = tf.image.random_brightness(value, max_delta=20.)
            value = tf.clip_by_value(value, clip_value_min=0., clip_value_max=255.)
            value = tf.image.random_flip_left_right(value)
            value = tf.random_crop(value, [cfg.height, cfg.width, cfg.channel])
            return value

        def Adjust_nml(value):
            value = tf.image.random_brightness(value, max_delta=20.)
            value = tf.clip_by_value(value, clip_value_min=0., clip_value_max=255.)
            value = tf.image.resize_images(value, [cfg.height, cfg.width])
            return value

        with tf.name_scope('data_feed'):
            source_list = Concate_List(cfg.source_path, self.source)
            normal_f_list = Concate_List(cfg.normal_f_path, self.normal_f)
            normal_s_list = Concate_List(cfg.normal_s_path, self.normal_s)

            source_value = Open_File(source_list)
            normal_f_value = Open_File(normal_f_list)
            normal_s_value = Open_File(normal_s_list)

            # Flip, crop and adjust brightness of image
            source_value = Adjust_src(source_value)
            normal_f_value = Adjust_nml(normal_f_value)
            normal_s_value = Adjust_nml(normal_s_value)

            steps_per_epoch = np.ceil(len(self.source) / cfg.batch_size).astype(np.int32)

            source, normal_f, normal_s = tf.train.shuffle_batch([source_value, normal_f_value, normal_s_value],
                                                                batch_size=self.batch_size,
                                                                num_threads=8,
                                                                capacity=32 * self.batch_size,
                                                                min_after_dequeue=self.batch_size * 16,
                                                                allow_smaller_final_batch=False)
            return source, normal_f, normal_s, steps_per_epoch

    def get_train_batch(self):
        """Get train images by preload

        return:
          trX: training profile images
          trY: training front images
        """
        trX = np.zeros((self.batch_size, cfg.height, cfg.width, cfg.channel), dtype=np.float32)
        trY = np.zeros((self.batch_size, cfg.height, cfg.width, cfg.channel), dtype=np.float32)
        for i in range(self.batch_size):
            try:
                trX[i] = self.read_image(self.profile[i + self.train_index], flip=True)
                trY[i] = self.read_image(self.front[i + self.train_index], flip=True)
            except:
                self.train_index = -i
                trX[i] = self.read_image(self.profile[i + self.train_index], flip=True)
                trY[i] = self.read_image(self.front[i + self.train_index], flip=True)
        self.train_index += self.batch_size
        return trX, trY

    def get_test_batch(self, batch_size=cfg.batch_size):
        """Get test images by batch

        args:
          batch_size: size of test scratch
        return:
          teX: testing profile images
          teY: testing front images, same as profile images
        """
        teX = np.zeros((batch_size, cfg.height, cfg.width, cfg.channel), dtype=np.float32)
        teY = np.zeros((batch_size, cfg.height, cfg.width, cfg.channel), dtype=np.float32)
        for i in range(batch_size):
            try:
                teX[i] = self.read_image(os.path.join(cfg.test_path, self.test_list[i + self.test_index]))
                teY[i] = self.read_image(os.path.join(cfg.test_path, self.test_list[i + self.test_index]))
            except:
                print("Test Loop at %d!" % self.test_index)
                self.test_index = -i
                teX[i] = self.read_image(os.path.join(cfg.test_path, self.test_list[i + self.test_index]))
                teY[i] = self.read_image(os.path.join(cfg.test_path, self.test_list[i + self.test_index]))
        self.test_index += batch_size
        return teX, teY

    def read_image(self, img, flip=False):
        """Read image

        Read a image from image path, and crop to target size
        and random flip horizontally

        args:
          img: image path
        return:
          img: data matrix from image
        """
        img = Image.open(img)
        if (img.mode == 'L' and cfg.channel == 3):
            img = img.convert('RGB')
        if flip and np.random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # if cfg.crop:
        #  img = img.crop(self.crop_box)
        img = img.resize((cfg.width, cfg.height))
        img = np.array(img, dtype=np.float32)
        if (cfg.channel == 1):
            img = np.expand_dims(img, axis=2)
        return img

    def save_images(self, imgs, imgs_name, epoch, step):
        """Save images

        args:
          imgs: images in shape of [BatchSize, Weight, Height, Channel], must be normalized to [0,255]
          epoch: epoch number
        """
        imgs = imgs.astype('uint8')  # inverse_transform
        if (cfg.channel == 1):
            imgs = imgs[:, :, :, 0]
        img_num = imgs.shape[0]
        test_size = self.test_list.shape[0]
        save_path = cfg.results + '/epoch-{}-{}_{}'.format(epoch, step, imgs_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for i in range(imgs.shape[0]):
            try:
                # img_name = self.test_list[i + self.test_index - img_num].split('/')[-1]
                tmp_img = self.test_list[i + self.test_index - img_num].split('/')
                img_name = '{}_{}'.format(tmp_img[2], tmp_img[3])
            except:
                img_name = self.test_list[test_size + i + self.test_index - img_num].split('/')[-1]
            Image.fromarray(imgs[i]).save(os.path.join(save_path, img_name))

    def cancate_path(self, path, img):

        tmp = img.split('/')
        for idx in range(len(tmp) - 1):
            path = os.path.join(path, tmp[idx])
        if not os.path.exists(path):
            os.makedirs(path)
        return os.path.join(path, tmp[-1])


    def save_test_images(self, imgs, imgs_name):
        """Save images

        args:
          imgs: images in shape of [BatchSize, Weight, Height, Channel], must be normalized to [0,255]
          epoch: epoch number
        """
        imgs = imgs.astype('uint8')  # inverse_transform
        if (cfg.channel == 1):
            imgs = imgs[:, :, :, 0]
        img_num = imgs.shape[0]
        test_size = self.test_list.shape[0]
        for i in range(imgs.shape[0]):
            try:
                img_name = self.cancate_path(imgs_name, self.test_list[i + self.test_index - img_num])
            except:
                img_name = self.test_list[test_size + i + self.test_index - img_num].split('/')[-1]
            Image.fromarray(imgs[i]).save(img_name)



