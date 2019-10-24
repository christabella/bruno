from collections import defaultdict

import numpy as np

import utils_conditional


class ShapenetConditionalNPDataIterator():
    """
    seq_len: how many different poses (y) to sample
    batch_size: [usually 4] 
    """
    def __init__(self,
                 seq_len,
                 batch_size,
                 set='train',
                 rng=None,
                 should_dequantise=True):
        # Images, labels, and angles [N, 2].
        # Images: (len(images), image_height, image_width, image_channels)
        self.x, self.y, self.info = utils_conditional.load_shapenet(set)

        self.n_samples = len(self.x)
        self.img_shape = self.x.shape[1:]

        self.classes = np.unique(self.y)
        print(set, self.classes)
        self.y2idxs = defaultdict(list)
        for i in range(self.n_samples):
            # Dictionary of class_idx : [indices from this class]
            self.y2idxs[self.y[i]].append(i)

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = rng
        self.should_dequantise = should_dequantise
        self.set = set

        print(set, 'dataset size:', self.x.shape)
        print(set, 'classes', self.classes)
        print(set, 'min, max', np.min(self.x), np.max(self.x))
        print(set, 'nsamples', self.n_samples)
        print('--------------')

    def process_angle(self, x):
        angle_rad = x * np.pi / 180
        return np.sin(angle_rad), np.cos(angle_rad)

    def deprocess_angle(self, x):
        """Accepts x1, x2 which are sin(angle) and cos(angle) where angle is in radians.

        Returns angle in degrees.
        """
        x1, x2 = x
        angle = np.arctan2(x1, x2) * 180 / np.pi
        angle += 360 if angle < 0 else 0
        return angle

    def get_label_size(self):
        """Return (seq_len, info_size) since self.info is [N, info_size]
        where info_size==2 for shapenet.
        """
        return (self.seq_len, self.info.shape[-1])

    def get_observation_size(self):
        return (self.seq_len, ) + self.img_shape

    def generate(self, rng=None, noise_rng=None):
        rng = self.rng if rng is None else rng
        noise_rng = self.rng if noise_rng is None else noise_rng

        while True:
            x_batch = np.zeros((
                self.batch_size,
                self.seq_len,
            ) + self.img_shape,
                               dtype='float32')
            y_batch = np.zeros((self.batch_size, ) + self.get_label_size(),
                               dtype='float32')

            for i in range(self.batch_size):
                c = rng.choice(self.classes)
                img_idxs = rng.choice(self.y2idxs[c],
                                      size=self.seq_len,
                                      replace=True)

                for j in range(self.seq_len):
                    x_batch[i, j] = self.x[img_idxs[j]]
                    y_batch[i, j] = self.info[img_idxs[j]]
            if self.should_dequantise:
                x_batch += noise_rng.uniform(size=x_batch.shape)
            yield x_batch, y_batch

    def generate_each_digit(self,
                            rng=None,
                            noise_rng=None,
                            random_classes=True):
        """
        Generator yielding, for each label (e.g. digit or type of chair/plane),
        batches of sequences of randomly-chosen images and indices.
        """
        rng = self.rng if rng is None else rng
        noise_rng = self.rng if noise_rng is None else noise_rng
        if random_classes:
            rng.shuffle(self.classes)
        for c in self.classes:
            # x_batch has shape (batch size, seq len, img height, img width, channels)
            # y_batch has shape (batch size, seq len, angle_dims)
            # where angle_dims = 2 (sin and cos)
            x_batch = np.zeros((
                self.batch_size,
                self.seq_len,
            ) + self.img_shape,
                               dtype='float32')
            y_batch = np.zeros((self.batch_size, ) + self.get_label_size(),
                               dtype='float32')
            # Populate x batch with #(seq_len) random images
            for i in range(self.batch_size):
                img_idxs = rng.choice(self.y2idxs[c],
                                      size=self.seq_len,
                                      replace=False)

                for j in range(self.seq_len):
                    x_batch[i, j] = self.x[img_idxs[j]]
                    y_batch[i, j] = self.info[img_idxs[j]]
            # To make good use of modelling densities, the Real NVP has to
            # treat its inputs as instances of a continuous random variable.
            # Integer pixel values in x are dequantised by adding uniform noise
            if self.should_dequantise:
                x_batch += noise_rng.uniform(size=x_batch.shape)
            yield c, x_batch, y_batch
