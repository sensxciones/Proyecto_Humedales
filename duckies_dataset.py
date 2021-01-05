import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            w, h = image.size
        if isinstance(self.output_size, int):
            if h < w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        if isinstance(image, np.ndarray):
            img = transform.resize(image, (new_h, new_w))
        else:
            tr_ = transforms.Resize(((new_h, new_w)))
            img = tr_(image)

        return img


class DuckieDataset(Dataset):
    def __init__(self, root, train = True, transform = None):
      patos = os.listdir(root)
      patos.sort()
      self.class_to_idx = {}   
      for i, pato in enumerate(patos):
        self.class_to_idx[pato] = i                                                              
      self.classes = set(self.class_to_idx.keys())
      # self.root = directorio donde está el dataset ('./train100x100/')
      self.root = root
      # self.transform = serie de transformaciones que se aplica a cada imagen
      self.transform = transform
      self.train = train
      self.data = list()
      self.targets = list()
      
      p = int(len(patos)*0.8)
      if self.train:
        patos = patos[:p]
      else:
        patos = patos[p:]
        
      for cl in patos:
          for img in os.listdir(root + '/' + cl):
              self.data.append(root + '/' + cl + '/' + img)
              self.targets.append(self.class_to_idx[cl])
      #self.data = np.array(self.data)
      self.targets = torch.Tensor(self.targets)

    
    def __getitem__(self, index):
        data_ = self.data[index]
        label_ = self.targets[index]

        img = Image.open(data_).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return (img, label_)

    def __len__(self):
        return len(self.data)