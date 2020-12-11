import torch
from torchvision import transforms
import numpy as np

from torchvision.transforms import functional as F


class RandomHorizontalFlip(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, label=False):
        if not label:
            self.current_p = torch.rand(1)

        if self.current_p < self.p:
            img = F.hflip(img)
        return img


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Resize(object):
    def __init__(self, output_size, interpolation=2):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.interpolation = interpolation
    
    def __call__(self, image, label=False):
        d, h, w = image.shape
        if isinstance(self.output_size, int):
            new_d, new_h, new_w = self.output_size, self.output_size, self.output_size
        else:
            new_d, new_h, new_w = self.output_size

        resize = transforms.Resize(size=(new_h, new_w), interpolation=self.interpolation)
        image = resize(image)

        image = torch.transpose(image, 0, 2)
        resize = transforms.Resize(size=(new_h, new_d), interpolation=self.interpolation)
        image = resize(image)

        image = torch.transpose(image, 0, 2)

        return image


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size
    
    def _set_indices(self, d_idx, h_idx, w_idx):
        self.d_idx = d_idx
        self.h_idx = h_idx
        self.w_idx = w_idx

    def __call__(self, image, label=False):
        d, h, w = image.shape[-3:]
        new_d, new_h, new_w = self.output_size

        if not label: 
            d_idx = np.random.randint(0, d - new_d)
            h_idx = np.random.randint(0, h - new_h)
            w_idx = np.random.randint(0, w - new_w)
            self._set_indices(d_idx, h_idx, w_idx)

            image = image[:, self.d_idx: self.d_idx + new_d,
                        self.h_idx: self.h_idx + new_h,
                        self.w_idx: self.w_idx + new_w]

        else:
            image = image[:, self.d_idx: self.d_idx + new_d,
                        self.h_idx: self.h_idx + new_h,
                        self.w_idx: self.w_idx + new_w]

        return image


class RandomAffineWithLabel(transforms.RandomAffine):
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=0, fillcolor=0):
        super(RandomAffineWithLabel, self).__init__(degrees, translate=None, scale=None, shear=None, resample=0, fillcolor=0)
    
    def forward(self, image, label=False):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """

        img_size = image.shape
        image = image.reshape(-1, *image.shape[-3:])

        if not label:
            self.ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
            image = F.affine(image, *self.ret, resample=self.resample, fillcolor=self.fillcolor)
        else:
            image = F.affine(image, *self.ret, resample=self.resample, fillcolor=self.fillcolor)
    
        image = image.reshape(*img_size)

        return image


class ToTensor():
    def __call__(self, pic, label=False):
        return F.to_tensor(pic)


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label=False):
        for t in self.transforms:
            img = t(img, label)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string