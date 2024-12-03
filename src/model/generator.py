from typing import Union, Tuple
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms 

class SiameseOnlineGenerator(tf.keras.utils.Sequence):
  
    def __init__(self, df: pd.DataFrame,
                 batch_size: int=32, 
                 img_width: int=224,
                 img_height: int=224,
                 shuffle: bool=True,
                 supervised: bool=False) -> None:
        self._batch_size = batch_size
        self._img_width = img_width
        self._img_height = img_height
        self._supervised = supervised
        self._df = df.copy()
        self._index = np.arange(len(df))
        self._shuffle = shuffle
        self.on_epoch_end()
        return
        
    def __len__(self) -> int:
        return -(-len(self._df)//self._batch_size)
    
    def on_epoch_end(self) -> None:
        if self._shuffle:
            np.random.shuffle(self._index)
        return
    
    def _random_crop(self, im: Image.Image) -> Image.Image:
        '''
        Perform random data augmentation on the image
        '''
        # random flip
        if np.random.choice(2) == 0:
            im = im.transpose(method=Image.FLIP_LEFT_RIGHT)
        if np.random.choice(2) == 0:
            im = im.transpose(method=Image.FLIP_TOP_BOTTOM)
        # random rotation between -90 and 90 degrees
        deg = np.random.randint(-90, 91)
        if deg != 0:
            im = im.rotate(deg)
        # random zoom from 0.9 to 1.1
        zoom = np.random.uniform(0.9, 1.1)
        w, h = im.size
        im = im.resize((int(w*zoom), int(h*zoom)))
        w, h = im.size
        # random translation between -20 and 20 pixels
        iw, ih = self._img_width, self._img_height
        dx = np.random.randint(-20, 21)
        dy = np.random.randint(-20, 21)
        x0 = (w - iw)/2 + dx
        y0 = (h - ih)/2 + dy
        im_cropped = im.crop((x0, y0, x0+iw, y0+ih))
        # color jitter: brightness, contrast, saturation, hue
        transform = transforms.ColorJitter( 
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2) 
        im_cropped = transform(im_cropped)
        return im_cropped

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        i_start = idx*self._batch_size
        i_end = (idx+1)*self._batch_size
        batch_index = self._index[i_start:i_end]
        imgs = []
        labels = []
        df = self._df
        for i in batch_index:
            if self._supervised:
                img = self.load_image(df.iloc[i]['path'])
                label = df['label'].cat.codes.iloc[i]
                imgs.append(self._random_crop(img))
                labels.append(label)
                continue
            # for each batch, generate two images for each file
            img = self.load_image(df.iloc[i]['path'])
            label = df.iloc[i].name # row index
            imgs.append(self._random_crop(img))
            labels.append(label)
            imgs.append(self._random_crop(img))
            labels.append(label)
        imgs = np.array([np.asarray(img)/1. for img in imgs])
        labels = np.array(labels)
        return imgs, labels
    
    def load_image(self, file: Union[str, Image.Image], 
                   center_crop: bool=False) -> Image.Image:
        '''
        load image
        '''
        if isinstance(file, str):
            img = tf.keras.preprocessing.image.load_img(file, 
                                                color_mode='rgb',
                                                keep_aspect_ratio=True,
                                                target_size=None)
        else:
            img = file
        # color normalize by gray world 
        img = tf.keras.preprocessing.image.img_to_array(img).astype(float)
        img = (img * (img.mean() / img.mean(axis=(0, 1)))).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        # center crop
        if center_crop:
            w, h = img.size
            iw, ih = self._img_width, self._img_height
            img = img.crop(((w - iw)/2, (h - ih)/2, 
                            (w - iw)/2 + iw, (h - ih)/2 + ih))
        return img
    
class SupervisedGenerator(tf.keras.utils.Sequence):
    def __init__(self, df: pd.DataFrame,
                 encoder_name: str='efficientnetv2b0',
                 batch_size: int=32, 
                 img_width: int=224,
                 img_height: int=224,
                 shuffle: bool=True,
                 is_validate: bool=False) -> None:
        self._batch_size = batch_size
        self._img_width = img_width
        self._img_height = img_height
        self._encoder_name = encoder_name
        self._df = df.copy()
        self._index = np.arange(len(df))
        self._shuffle = shuffle
        self._is_validate = is_validate
        self.on_epoch_end()
        return
        
    def __len__(self) -> int:
        return -(-len(self._df)//self._batch_size)
    
    def on_epoch_end(self) -> None:
        if self._shuffle:
            np.random.shuffle(self._index)
        return
    
    def _random_crop(self, im: Image.Image) -> Image.Image:
        '''
        Perform random data augmentation on the image
        '''
        # random flip
        if np.random.choice(2) == 0:
            im = im.transpose(method=Image.FLIP_LEFT_RIGHT)
        if np.random.choice(2) == 0:
            im = im.transpose(method=Image.FLIP_TOP_BOTTOM)
        # random rotation between -90 and 90 degrees
        deg = np.random.randint(-90, 91)
        if deg != 0:
            im = im.rotate(deg)
        # random zoom from 0.9 to 1.1
        zoom = np.random.uniform(0.9, 1.1)
        w, h = im.size
        im = im.resize((int(w*zoom), int(h*zoom)))
        w, h = im.size
        # random translation between -20 and 20 pixels
        iw, ih = self._img_width, self._img_height
        dx = np.random.randint(-20, 21)
        dy = np.random.randint(-20, 21)
        x0 = (w - iw)/2 + dx
        y0 = (h - ih)/2 + dy
        im_cropped = im.crop((x0, y0, x0+iw, y0+ih))
        # color jitter: brightness, contrast, saturation, hue
        transform = transforms.ColorJitter( 
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2) 
        im_cropped = transform(im_cropped)
        return im_cropped

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        i_start = idx*self._batch_size
        i_end = (idx+1)*self._batch_size
        batch_index = self._index[i_start:i_end]
        imgs = []
        labels = []
        df = self._df
        if self._encoder_name == "efficientnetv2b0":
            rescale = 1.
        for i in batch_index:
            if self._is_validate:
                img = self.load_image(df.iloc[i]['path'], center_crop=True)
            else:
                img = self.load_image(df.iloc[i]['path'])        
                img = self._random_crop(img)
            imgs.append(img)
            label = df['label'].cat.codes.iloc[i] # label is catelogical
            labels.append(label)
        imgs = np.array([np.asarray(img)/rescale for img in imgs])
        labels = np.array(labels)
        # convert to one-hot encoding
        # labels = tf.keras.utils.to_categorical(labels, num_classes=len(df['label'].cat.categories))
        return imgs, labels
    
    def load_image(self, file: Union[str, Image.Image], 
                   center_crop: bool=False) -> Image.Image:
        '''
        load image
        '''
        if isinstance(file, str):
            img = tf.keras.preprocessing.image.load_img(file, 
                                                color_mode='rgb',
                                                keep_aspect_ratio=True,
                                                target_size=None)
        else:
            img = file
        # color normalize by gray world 
        img = tf.keras.preprocessing.image.img_to_array(img).astype(float)
        img = (img * (img.mean() / img.mean(axis=(0, 1)))).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        # center crop
        if center_crop:
            w, h = img.size
            iw, ih = self._img_width, self._img_height
            img = img.crop(((w - iw)/2, (h - ih)/2, 
                            (w - iw)/2 + iw, (h - ih)/2 + ih))
        return img
   
def get_generator(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                  opt: dict) -> tuple:
    np.random.seed(0)
    opt = opt.copy()
    name = opt['name']
    del opt['name']
    if name == 'siameseonline':
        train_generator = SiameseOnlineGenerator(train_df, **opt)
        val_generator = SiameseOnlineGenerator(val_df, **opt)
    elif name == 'supervised':
        train_generator = SupervisedGenerator(train_df, **opt)
        val_generator = SupervisedGenerator(val_df, **opt, 
                                            shuffle=False,
                                            is_validate=True)
    else:
        raise ValueError(f"Generator {name} not implemented")
    return train_generator, val_generator
    