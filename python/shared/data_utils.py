import SimpleITK as sitk
import numpy as np
import random
from PIL import Image
import os


class LggHggGenerator:

    def __init__(self, image_generator, seg_generator, image_dir, seg_dir,
                 seed=42):
        self.image_generator = image_generator
        self.seg_generator = seg_generator
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.seed = seed
        self.params = {
            'shuffle': False,
            'classes': ['LGG', 'HGG'],
            'color_mode': "rgb",
            'target_size': (240, 240),
            'class_mode': 'sparse',
            'seed': self.seed
        }

    def get_image_generator(self, batch_size=16, shuffle=False):
        return self.image_generator.flow_from_directory(batch_size=batch_size,
                                                        directory=self.image_dir,
                                                        classes=['LGG', 'HGG'],
                                                        color_mode="rgb",
                                                        target_size=(240, 240),
                                                        class_mode='sparse',
                                                        shuffle=shuffle,
                                                        seed=self.seed)

    def get_seg_generator(self, batch_size=16, shuffle=False):
        return self.seg_generator.flow_from_directory(batch_size=batch_size,
                                                      directory=self.seg_dir,
                                                      classes=['LGG', 'HGG'],
                                                      color_mode="grayscale",
                                                      target_size=(240, 240),
                                                      class_mode='sparse',
                                                      shuffle=shuffle,
                                                      seed=self.seed)

    def get_image_seg_generator(self, batch_size=16):
        return zip(self.get_image_generator(batch_size),
                   self.get_seg_generator(batch_size))


# helper functions to process BraTs dataset into 2D slices

def process_data(directory, postfixes, saving_directory, seg_directory=None,
                 interval=(0, 1), slices_gap=1, num_of_samples=20):
    file_names = os.listdir(directory)
    random.seed(5)
    random.shuffle(file_names)
    file_names = file_names[int(len(file_names) * interval[0]): int(
        len(file_names) * interval[1])]
    iterator = 0
    for name in file_names:
        #       read images
        images = [sitk.ReadImage(os.path.join(directory, name, name + postfix),
                                 sitk.sitkFloat32)
                  for postfix in postfixes
                  ]
        image_seg = sitk.ReadImage(
            os.path.join(directory, name, name + '_seg.nii.gz'))
        #       transform image to n-dim numpy arrays
        vols = [sitk.GetArrayFromImage(image) for image in images]
        vol_seg = sitk.GetArrayFromImage(image_seg)
        #       get slices with labels from image
        imgs_slices = [slice_images(vol, vol_seg, 50, 130, slices_gap) for vol
                       in vols]
        slices_seg = slice_images(vol_seg, vol_seg, 50, 130, slices_gap)
        #       get slice index with biggest tumor segmentation
        index = index_of_largest_segmentation(slices_seg)
        low_index = max(0, index - num_of_samples // 2)
        high_index = min(len(slices_seg), index + num_of_samples // 2 + 1)
        imgs_slices = [vol[low_index:high_index] for vol in imgs_slices]
        slices_seg = slices_seg[low_index:high_index]
        #       save images
        for idx, (images, segmentation) in enumerate(
                zip(zip(*imgs_slices), slices_seg)):
            if segmentation[1] == 0:
                continue
            iterator += 1
            only_images = [img for (img, label) in images]
            normalized_images = [normalize(s, vol) for s, vol in
                                 zip(only_images, vols)]

            image_name = name + '_' + str(idx) + '.png'
            save_image_rgb(normalized_images,
                           os.path.join(saving_directory, image_name))
            if seg_directory:
                save_image_gray(segmentation[0],
                                os.path.join(seg_directory, image_name))
            print(os.path.join(saving_directory, image_name))
        print(iterator, ' generated images')


def process_data_as_numpy(directory, postfixes, interval=(0, 1), slices_gap=1,
                          num_of_samples=20, ):
    file_names = os.listdir(directory)
    random.seed(5)
    random.shuffle(file_names)
    file_names = file_names[int(len(file_names) * interval[0]): int(
        len(file_names) * interval[1])]
    iterator = 0
    X = None
    X_seg = None

    for name in file_names:
        #       read images
        images = [sitk.ReadImage(os.path.join(directory, name, name + postfix),
                                 sitk.sitkFloat32)
                  for postfix in postfixes
                  ]
        image_seg = sitk.ReadImage(
            os.path.join(directory, name, name + '_seg.nii.gz'))
        #       transform image to n-dim numpy arrays
        vols = [sitk.GetArrayFromImage(image) for image in images]
        vol_seg = sitk.GetArrayFromImage(image_seg)
        #       get slices with labels from image
        imgs_slices = [slice_images(vol, vol_seg, 50, 130, slices_gap) for vol
                       in vols]
        slices_seg = slice_images(vol_seg, vol_seg, 50, 130, slices_gap)
        #       get slice index with biggest tumor segmentation
        index = index_of_largest_segmentation(slices_seg)
        low_index = max(0, index - num_of_samples // 2)
        high_index = min(len(slices_seg), index + num_of_samples // 2 + 1)
        imgs_slices = [vol[low_index:high_index] for vol in imgs_slices]
        slices_seg = slices_seg[low_index:high_index]
        #       save images
        for idx, (images, segmentation) in enumerate(
                zip(zip(*imgs_slices), slices_seg)):
            if segmentation[1] == 0:
                continue
            iterator += 1
            only_images = [img for (img, label) in images]
            normalized_images = [standardize(s, vol) for s, vol in
                                 zip(only_images, vols)]
            if X is None:
                X = np.array([np.dstack(normalized_images)])
                X_seg = np.array([segmentation[0]])
            else:
                X = np.append([np.dstack(normalized_images)], X, axis=0)
                X_seg = np.append([segmentation[0]], X_seg, axis=0)

        print(iterator, ' generated images')
    return X, X_seg


def slice_images(image_vol, image_seg_vol, from_z, to_z, step):
    return [(image_vol[z], get_label(image_seg_vol[z])) for z in
            range(from_z, to_z, step)]


def normalize(s, image):
    return s / image.max() * 255


def standardize(s, image):
    return (s - image[image > 0].mean()) / image[image > 0].std()


def get_label(vol):
    return 1 if any(vol.flatten() > 0) else 0


def save_image_rgb(image, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    vol = np.dstack(image)
    vol = vol.astype('uint8')
    im = Image.fromarray(vol)
    im.save(filename)


def save_image_gray(image, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    vol = image.astype('uint8')
    im = Image.fromarray(vol)
    im.save(filename)


def index_of_largest_segmentation(segmentations):
    return np.argmax([np.sum(x.flatten() > 0) for x, y in segmentations])