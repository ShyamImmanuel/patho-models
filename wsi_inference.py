import argparse
import torch
import numpy as np
import matplotlib.cm as cm
from skimage.filters import gaussian
from xml.etree import ElementTree as ET
from PIL import Image, ImageDraw
from skimage.color import rgb2hsv
from skimage.transform import resize
Image.MAX_IMAGE_PIXELS = None
import utils
import networks
import cv2
from PIL import Image, ImageDraw
import os
import sys
sys.path.append(r'C:\Users\shyam.i.albert\Documents\python-models\metastasis-detection\ASAP-1.8\bin')
try:
    import _multiresolutionimageinterface as mir
except ImportError:
    print("ASAP 1.8 not installed.")


parser = argparse.ArgumentParser()
parser.add_argument('-file', default='C:/Users/shyam.i.albert/Documents/python-models/metastasis-detection/images/test01.png', help="Path to whole slide image TIF from Camelyon dataset.")
parser.add_argument('-level', default=2, type=int, help="Magnification 0=40x, 1=20x, 2=10x, ...")
parser.add_argument('-saved_model', default='C:/Users/shyam.i.albert/Documents/python-models/metastasis-detection/models/model-00100.pth', help="Path to trained model")
parser.add_argument('-save_path', default='figures', help="Path to save figures")
parser.add_argument('-window_shape', default=96, type=int, help="Size of sliding window")
parser.add_argument('-device', default=0, type=int)
args = parser.parse_args()
print(args)


# CUDA device
device = "cuda:{}".format(args.device) if torch.cuda.is_available() else 'cpu'


# Load the trained model
model = networks.CamelyonClassifier().to(device)
model.load_state_dict(torch.load(args.saved_model))
model.eval()


class TIFReader:

    def __init__(self, file, level):

        self.file = file
        # self.reader = mir.MultiResolutionImageReader()
        # self.mr_image = self.reader.open(self.file)
        # self.level = level
        self.image = Image.open(self.file)
        self.level = 1  # JPG images don't have multiple levels, so we use a default level

    def get_shape(self):
        # X, Y
        return self.image.size
        # return self.mr_image.getLevelDimensions(self.level)

    def load_patch(self, x, y, width, height):
        # ds = self.mr_image.getLevelDownsample(self.level)
        # image_patch = self.mr_image.getUCharPatch(int(x * ds), int(y * ds), width, height, self.level)
        # return image_patch
        
        # Crop a patch from the image
        patch = self.image.crop((x, y, x + width, y + height))
        return np.array(patch)

    def load_image(self):
        # assert self.level >= 2
        # shape = self.get_shape()
        # return self.load_patch(0, 0, shape[0], shape[1])
        
        # Load the entire image as a NumPy array
        return np.array(self.image)

    def create_mask(self, xml_file):

        # Create blank mask
        # mask = Image.new(mode='L', size=self.get_shape())
        # draw = ImageDraw.Draw(mask)
        # ds = self.mr_image.getLevelDownsample(self.level)

        # Create a blank mask
        width, height = self.get_shape()
        mask = Image.new(mode='L', size=(width, height))
        draw = ImageDraw.Draw(mask)

        # Parse XML
        tree = ET.parse(xml_file)

        AnnotationDict = {}
        for children in tree.iter('Annotations'):
            for annotation in children.iter('Annotation'):
                XY = []
                if annotation.get('PartOfGroup') in ['Tumor', '_0', '_1'] :
                    for coordinates in annotation.iter('Coordinate'):
                        XY.append((float(coordinates.get('X')), float(coordinates.get('Y'))))
                    draw.polygon(XY, fill=255)
                    AnnotationDict[annotation.get('Name')] = XY
                else:
                    print(annotation.get('PartOfGroup'))

        return mask

    @staticmethod
    def segment_tissue(image):
        resized = image[::16, ::16, :].copy()
        hsv = rgb2hsv(resized)
        return resize(hsv[:, :, 1], image.shape[:2], mode='constant', cval=0, anti_aliasing=False)


def predict_tumor_regions(wsi, tissue_mask, windows):

    # Initialize with zeros
    tumor = np.zeros(wsi.shape[:2])

    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            print("Predicting window {:07d} of {:07d}".format(i * windows.shape[1] + j, windows.size), end='\r')

            # [x1, y1, x2, y2]
            bbox = windows[i, j, :].reshape(-1)

            # Tissue mask patch
            mask_patch = tissue_mask[bbox[1]:bbox[3], bbox[0]: bbox[2]]

            if mask_patch.mean() > 0.075:

                # Select patch from window
                wsi_patch = np.expand_dims(wsi[bbox[1]:bbox[3], bbox[0]: bbox[2], :].copy(), axis=0)

                # Convert to tensor
                wsi_tensor = torch.from_numpy(wsi_patch).permute(0, 3, 1, 2).float().to(device) / 255.

                # Inference
                tumor[bbox[1]:bbox[3], bbox[0]: bbox[2]] = model(wsi_tensor).squeeze().item()

    print("Predicting window {:07d} of {:07d}".format(windows.size, windows.size))

    return gaussian(tumor, preserve_range=True)


def main():

    # Create TIF reader
    reader = TIFReader(args.file, args.level)

    # Load image and create tissue mask
    print("Loading image...", end='\r')
    wsi = reader.load_image()
    print("Loading image...Done.")

    # Get the sliding window and padding params
    windows, padding = utils.sliding_window(wsi.shape, (args.window_shape, args.window_shape))

    # Pad the WSI
    wsi_padded = np.pad(wsi, ((0, padding['y']), (0, padding['x']), (0, 0)), mode='constant', constant_values=255)

    print("Segmenting tissue...", end='\r')
    tissue_mask = reader.segment_tissue(wsi_padded)
    print("Segmenting tissue...Done.")

    # Prediction
    tumor_map = predict_tumor_regions(wsi_padded, tissue_mask, windows)

    # Save
    # Assuming args.file contains the full file path
    file_name = os.path.basename(args.file).split('.')[0]  # Extract base name without extension
    np.save('figures/{}.npy'.format(file_name), tumor_map)
    # np.save('figures/{}.npy'.format(args.file.split('.')[0]), tumor_map)
    cmapper = cm.get_cmap('plasma')
    # colorized = Image.fromarray(np.uint8(cmapper(np.clip(tumor_map, 0, 1)) * 255))
    # colorized.save('figures/{}.tif'.format(args.file.split('.')[0]))
    # Save tumor map as .tif
    # cmapper = cm['plasma']  
    # Updated to use the new Matplotlib colormap API
    colorized = Image.fromarray(np.uint8(cmapper(np.clip(tumor_map, 0, 1)) * 255))
    colorized.save(f'figures/{file_name}.tif')

if __name__ == '__main__':
    main()
