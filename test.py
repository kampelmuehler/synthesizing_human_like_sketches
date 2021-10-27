from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter
from torchvision.utils import save_image
import argparse
import torch
from PIL import Image
from models.sketch_generator import SketchGenerator
import os.path

labeltoID = {'airplane': 0,
             'alarm clock': 1,
             'ant': 2,
             'ape': 3,
             'apple': 4,
             'armor': 5,
             'axe': 6,
             'banana': 7,
             'bat': 8,
             'bear': 9,
             'bee': 10,
             'beetle': 11,
             'bell': 12,
             'bench': 13,
             'bicycle': 14,
             'blimp': 15,
             'bread': 16,
             'butterfly': 17,
             'cabin': 18,
             'camel': 19,
             'candle': 20,
             'cannon': 21,
             'car (sedan)': 22,
             'castle': 23,
             'cat': 24,
             'chair': 25,
             'chicken': 26,
             'church': 27,
             'couch': 28,
             'cow': 29,
             'crab': 30,
             'crocodilian': 31,
             'cup': 32,
             'deer': 33,
             'dog': 34,
             'dolphin': 35,
             'door': 36,
             'duck': 37,
             'elephant': 38,
             'eyeglasses': 39,
             'fan': 40,
             'fish': 41,
             'flower': 42,
             'frog': 43,
             'geyser': 44,
             'giraffe': 45,
             'guitar': 46,
             'hamburger': 47,
             'hammer': 48,
             'harp': 49,
             'hat': 50,
             'hedgehog': 51,
             'helicopter': 52,
             'hermit crab': 53,
             'horse': 54,
             'hot-air balloon': 55,
             'hotdog': 56,
             'hourglass': 57,
             'jack-o-lantern': 58,
             'jellyfish': 59,
             'kangaroo': 60,
             'knife': 61,
             'lion': 62,
             'lizard': 63,
             'lobster': 64,
             'motorcycle': 65,
             'mouse': 66,
             'mushroom': 67,
             'owl': 68,
             'parrot': 69,
             'pear': 70,
             'penguin': 71,
             'piano': 72,
             'pickup truck': 73,
             'pig': 74,
             'pineapple': 75,
             'pistol': 76,
             'pizza': 77,
             'pretzel': 78,
             'rabbit': 79,
             'raccoon': 80,
             'racket': 81,
             'ray': 82,
             'rhinoceros': 83,
             'rifle': 84,
             'rocket': 85,
             'sailboat': 86,
             'saw': 87,
             'saxophone': 88,
             'scissors': 89,
             'scorpion': 90,
             'sea turtle': 93,
             'seagull': 91,
             'seal': 92,
             'shark': 94,
             'sheep': 95,
             'shoe': 96,
             'skyscraper': 97,
             'snail': 98,
             'snake': 99,
             'songbird': 100,
             'spider': 101,
             'spoon': 102,
             'squirrel': 103,
             'starfish': 104,
             'strawberry': 105,
             'swan': 106,
             'sword': 107,
             'table': 108,
             'tank': 109,
             'teapot': 110,
             'teddy bear': 111,
             'tiger': 112,
             'tree': 113,
             'trumpet': 114,
             'turtle': 115,
             'umbrella': 116,
             'violin': 117,
             'volcano': 118,
             'wading bird': 119,
             'wheelchair': 120,
             'windmill': 121,
             'window': 122,
             'wine bottle': 123,
             'zebra': 124}

parser = argparse.ArgumentParser(description='Synthesizing human-like sketches')

parser.add_argument('--img_path', default='test_images/fish.png', type=str, help='input to model')
parser.add_argument('--label', default='fish', type=str, help='label of input to model')
parser.add_argument('--checkpoint', default='pretrained_models/sketch_generator.pt', type=str, help='input to model')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--num_classes', default=125, type=int, help='number of classes in dataset')
parser.add_argument('--run_name', default='test-run', type=str, help='name of the rollout')
args = parser.parse_args()

assert (args.label in labeltoID.keys()), f'unknown input label {args.label}'


def load_image(img_path):
    image_statistics = ((118.87253346 / 255., 114.55559207 / 255., 101.7765648 / 255.),
                        (55.22226547 / 255., 53.83240694 / 255., 54.51108792 / 255.))  # mean, std

    image_transform = Compose([ToTensor(),
                               Normalize(image_statistics[0], image_statistics[1])])

    image_in = Image.open(img_path).resize((224, 224)).convert("RGB")
    image = image_transform(image_in).float()
    return image.view(1, 3, 224, 224)


device = torch.device(f"cuda:{args.gpu}")
checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
model = SketchGenerator()
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to(device)

image = load_image(args.img_path)
image = image.to(device)
labels_onehot = torch.zeros(1, args.num_classes).to(device)
labels_onehot[:, labeltoID[args.label]] = 1.
sketch = model(image, labels=labels_onehot)
sketch = sketch.cpu()
sketch = 1-sketch 
name, ext = os.path.splitext(args.img_path)
save_image(sketch, f"{name}_processed{ext}")
