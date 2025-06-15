import numpy as np
import PIL.Image as Image
import cv2 as cv
import os

import torch
import cv2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import shutil

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def tensor_to_images(tensor, device):
    tensor = tensor[:, 0:3, :, :]
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    result = []
    for i in range(tensor.shape[0]):
        current_img = np.transpose((tensor[i, :, :, :].cpu().detach().numpy() * 255).astype(np.uint8), (1, 2, 0))
        result.append(current_img)
    return result

def show_image(image, size=None, name=None):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    if name is None:
        name = 'Image'

    if size is not None:
        image = cv.resize(image, size)

    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
DEFAULT_MASK_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

def image_pil_resize(image: Image, size: tuple):
    resized_image = image.resize(size, resample=Image.Resampling.LANCZOS)
    return resized_image

def resize_cv_image(image: np.ndarray, size: tuple):
    resized_image = cv.resize(image, size, interpolation=cv.INTER_LANCZOS4)
    return resized_image

def transform_pil_to_cv2(image: Image):
    return cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

def get_outpaint_mask(mask_size=256, border=32):
    mask = np.zeros((mask_size, mask_size, 3), dtype=np.uint8)
    border = border
    mask[border:mask_size-border, border:mask_size-border, :] = 1
    return mask

def prepare_image_for_outpaint(image):
    """:return np.ndarray"""

    def resize_image_for_outpaint(image: np.ndarray):
        standard_dimensions = (192, 192)
        image = cv.resize(image, standard_dimensions, interpolation=cv.INTER_LANCZOS4)
        return image

    if isinstance(image, Image.Image):
        image = transform_pil_to_cv2(image)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    image = resize_image_for_outpaint(image)
    masked_image = get_outpaint_mask()

    masked_image[32:-32, 32:-32, :] = image
    return masked_image

def prepare_image_for_inpaint(image, coordinates):
    """:return np.ndarray"""

    def resize_image_for_inpaint(image: np.ndarray):
        standard_dimensions = (256, 256)
        image = cv.resize(image, standard_dimensions, interpolation=cv.INTER_LANCZOS4)
        return image


    if isinstance(image, Image.Image):
        image = transform_pil_to_cv2(image)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    x1, y1, x2, y2 = coordinates
    mask = np.ones(image.shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 0

    image = image * mask
    mask = mask * 255

    return image, mask

def transform_image_to_tensor(image: np.ndarray):
    image_tensor = DEFAULT_TRANSFORM(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def transform_tensor_to_cv2_image(image):
    image = tensor_to_images(image, device=DEFAULT_DEVICE)[0]
    return image

def transform_tensor_to_pil_image(image: torch.Tensor):
    image = tensor_to_images(image, device=DEFAULT_DEVICE)[0]
    image = Image.fromarray(image)
    return image



