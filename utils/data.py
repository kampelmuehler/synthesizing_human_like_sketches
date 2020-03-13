from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter, ToPILImage, RandomAffine
import torch
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
from PIL import Image, ImageOps
import numpy as np
import random


class SketchyDB(Dataset):
    """ class representing the Sketchy Database """

    def __init__(self, root_dir, image_type='image', sketch_type='sketch', sketch_difficulties=(1, 2, 3, 4),
                 filter_erroneous=False, filter_context=False, filter_ambiguous=False, filter_pose=False,
                 split='train'):
        """
        :param root_dir: root directory of the dataset
        :param image_type: which image augmentation to use ('image' or 'bbox'):

                'image' : image is non-uniformly scaled to 256x256
                'bbox'  : image bounding box scaled to 256x256 with
                          an additional +10% on each edge; note
                          that due to position within the image,
                          sometimes the object is not centered

        :param sketch_type: which sketch augmentation to use ('sketch', 'centered_scaled',
                'centered_bbox', 'centered_bbox_scaled_least', 'centered_bbox_scaled_most'
                or 'centered_bbox_scaled_nonuniform'):

                'sketch'                          : sketch canvas is rendered to 256x256 such that it
                                                    undergoes the same scaling as the paired photo
                'centered_scaled'                 : sketch is centered and uniformly scaled
                                                    such that its greatest dimension (x or y)
                                                    fills 78% of the canvas (roughly the same
                                                    as in Eitz 2012 sketch data set)
                'centered_bbox'                   : sketch is translated such that it is
                                                    centered on the object bounding box
                'centered_bbox_scaled_least'      : sketch is centered on bounding box and
                                                    is uniformly scaled such that one dimension
                                                    (x or y; whichever requires the least amount
                                                    of scaling) fits within the bounding box
                'centered_bbox_scaled_most'       : sketch is centered on bounding box and
                                                    is uniformly scaled such that one dimension
                                                    (x or y; whichever requires the most amount
                                                    of scaling) fits within the bounding box
                'centered_bbox_scaled_nonuniform' : sketch is centered on bounding box and
                                                    is non-uniformly scaled such that it
                                                    completely fits within the bounding box

        :param sketch_difficulties: between 1 and 4, list of difficulties to use
        :param filter_erroneous: filter sketches tagged as erroneous
        :param filter_context: filter sketches tagged as containing context
        :param filter_ambiguous: filter sketches tagged as ambiguous
        :param filter_pose: filter sketches tagged with an incorrect pose
        """
        image_folders = {'image': 'tx_000000000000',
                         'bbox': 'tx_000100000000'}
        sketch_folders = {'sketch': 'tx_000000000000',
                          'centered_scaled': 'tx_000100000000',
                          'centered_bbox': 'tx_000000000010',
                          'centered_bbox_scaled_least': 'tx_000000000110',
                          'centered_bbox_scaled_most': 'tx_000000001010',
                          'centered_bbox_scaled_nonuniform': 'tx_000000001110'}
        self.image_dir = join(root_dir, 'photo', image_folders[image_type])
        self.sketch_dir = join(root_dir, 'sketch', sketch_folders[sketch_type])

        # image statistics calculated for whole dataset
        self.image_statistics = ((118.87253346 / 255., 114.55559207 / 255., 101.7765648 / 255.),
                                 (55.22226547 / 255., 53.83240694 / 255., 54.51108792 / 255.))  # mean, std

        self.image_transform = Compose([ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                                        ToTensor(),
                                        Normalize(self.image_statistics[0],
                                                  self.image_statistics[1])])

        self.sketch_transform = Compose([ToTensor()])

        database_dir = join(root_dir, 'info/stats.csv')
        df = pd.read_csv(database_dir, usecols=['CategoryID', 'Category', 'ImageNetID', 'SketchID', 'Difficulty',
                                                'Error?', 'Context?', 'Ambiguous?', 'WrongPose?'])

        categories = df[['Category', 'CategoryID']].drop_duplicates(subset='CategoryID')
        self.class_names = list(categories.Category)
        self.IDtolabel = dict(zip(categories.CategoryID, categories.Category))
        self.labeltoID = dict(zip(categories.Category, categories.CategoryID))

        # filter data
        df = df[df['Difficulty'].isin(sketch_difficulties)]
        if filter_erroneous is True:
            df = df[df['Error?'] == 0]
        if filter_context is True:
            df = df[df['Context?'] == 0]
        if filter_ambiguous is True:
            df = df[df['Ambiguous?'] == 0]
        if filter_pose is True:
            df = df[df['WrongPose?'] == 0]

        df_unique = df.drop_duplicates(subset='ImageNetID')

        np.random.seed(1337)
        msk = np.random.rand(len(df_unique)) < 0.9
        df_train = df.loc[df['ImageNetID'].isin(df_unique[msk]['ImageNetID'])]
        df_test = df.loc[df['ImageNetID'].isin(df_unique[~msk]['ImageNetID'])]
        df_test = df_test.drop_duplicates(subset='ImageNetID')  # only return a single 'sketch' per image for testing
        assert (split in ['train', 'test']), f'unknown split {split}'
        if split == 'train':
            self.data = df_train
        else:
            self.data = df_test

        self.num_samples = len(self.data.index)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        hflip = random.random() < 0.5
        image_path = join(self.image_dir, self.data.iloc[idx]['Category'].replace(' ', '_'),
                          f'{self.data.iloc[idx]["ImageNetID"]}.jpg')

        image = Image.open(image_path).resize((224, 224))

        if hflip is True:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image = self.image_transform(image).float()

        sketch_path = join(self.sketch_dir, self.data.iloc[idx]['Category'].replace(' ', '_'),
                           f'{self.data.iloc[idx]["ImageNetID"]}-{self.data.iloc[idx]["SketchID"]}.png')

        sketch = Image.open(sketch_path).resize((224, 224))

        if hflip is True:
            sketch = sketch.transpose(Image.FLIP_LEFT_RIGHT)

        sketch = ImageOps.invert(sketch)
        sketch = np.array(sketch)[..., 0][..., np.newaxis]
        sketch = self.sketch_transform(sketch).float()

        return {'image': image,
                'sketch': sketch,
                'image_path': image_path,
                'sketch_path': sketch_path,
                'hflip': hflip,
                'label': self.data.iloc[idx]['Category'],
                'labelID': self.data.iloc[idx]['CategoryID'],
                'imageID': self.data.iloc[idx]['ImageNetID']}


class UnnormImage(Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self):
        mean = torch.as_tensor((118.87253346 / 255., 114.55559207 / 255., 101.7765648 / 255.))
        std = torch.as_tensor((55.22226547 / 255., 53.83240694 / 255., 54.51108792 / 255.))
        std_inv = 1 / std
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


class UnnormSketch(Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self):
        mean = torch.as_tensor((244.97762159614328 / 255.,))
        std = torch.as_tensor((46.71412621304228 / 255.,))
        self.std_inv = 1 / std
        self.mean_inv = -mean * self.std_inv
        super().__init__(mean=self.mean_inv, std=self.std_inv)

    def __call__(self, tensor):
        return (tensor - self.mean_inv) / self.std_inv


if __name__ == '__main__':
    dataset_train = SketchyDB('../data',
                              image_type='bbox',
                              sketch_type='centered_scaled',
                              filter_erroneous=True,
                              filter_context=True,
                              filter_ambiguous=True,
                              filter_pose=True)
    dataset_test = SketchyDB('../data',
                             image_type='bbox',
                             sketch_type='centered_scaled',
                             filter_erroneous=True,
                             filter_context=True,
                             filter_ambiguous=True,
                             filter_pose=True,
                             split='test')
