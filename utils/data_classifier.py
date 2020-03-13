from torchvision.transforms import ToTensor, Compose, RandomAffine
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
from PIL import Image, ImageOps
import numpy as np
import random


class SketchyDB(Dataset):
    """ class representing the Sketchy Database """

    def __init__(self, root_dir, sketch_type='sketch', sketch_difficulties=(1, 2, 3, 4),
                 filter_erroneous=False, filter_context=False, filter_ambiguous=False, filter_pose=False,
                 train=True, split=0.8, sketch_augmentation=False, seed=1337):
        """
        :param root_dir: root directory of the dataset

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
        :param train: train or test set
        :param split: train/test ratio
        :param sketch_augmentation: whether to augment sketches
        """
        sketch_folders = {'sketch': 'tx_000000000000',
                          'centered_scaled': 'tx_000100000000',
                          'centered_bbox': 'tx_000000000010',
                          'centered_bbox_scaled_least': 'tx_000000000110',
                          'centered_bbox_scaled_most': 'tx_000000001010',
                          'centered_bbox_scaled_nonuniform': 'tx_000000001110'}

        self.sketch_dir = join(root_dir, 'sketch', sketch_folders[sketch_type])

        self.sketch_augmentation = sketch_augmentation

        if sketch_augmentation is True:
            self.sketch_augmentation_transform = Compose([RandomAffine(10,
                                                                       translate=(0.08, 0.08),
                                                                       scale=(0.9, 1.1),
                                                                       fillcolor=255,
                                                                       shear=10)])

        self.sketch_transform = Compose([ToTensor()])

        database_dir = join(root_dir, 'info/stats.csv')
        df = pd.read_csv(database_dir, usecols=['CategoryID', 'Category', 'ImageNetID', 'SketchID', 'Difficulty',
                                                'Error?', 'Context?', 'Ambiguous?', 'WrongPose?'])
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

        self.seed = seed
        self.train = train
        self.data = df
        self.num_samples = len(self.data.index)
        self.indices = list(range(self.num_samples))
        random.Random(self.seed).shuffle(self.indices)
        if self.train is True:
            self.indices = self.indices[:int(len(self.indices) * split)]
        else:
            self.indices = self.indices[int(len(self.indices) * split):]
        self.num_samples = len(self.indices)
        self.num_classes = df['CategoryID'].nunique()

    def __len__(self):
        return self.num_samples * 2 if self.train is True else self.num_samples # (*2 for flipping every single sample)

    def __getitem__(self, idx):
        hflip = False
        if self.train:
            idx = idx // 2
            hflip = (idx % 2 == 0)
        idx = self.indices[idx]

        sketch_path = join(self.sketch_dir, self.data.iloc[idx]['Category'].replace(' ', '_'),
                           f'{self.data.iloc[idx]["ImageNetID"]}-{self.data.iloc[idx]["SketchID"]}.png')
        sketch = Image.open(sketch_path).resize((224, 224))
        if self.sketch_augmentation is True:
            sketch = self.sketch_augmentation_transform(sketch)
        if hflip is True:
            sketch = sketch.transpose(Image.FLIP_LEFT_RIGHT)
        sketch = ImageOps.invert(sketch)
        sketch = np.array(sketch)[..., 0:1]
        sketch = self.sketch_transform(sketch).float()

        return {'sketch': sketch, 'label': self.data.iloc[idx]['CategoryID']}


if __name__ == '__main__':
    dataset = SketchyDB('../data')
