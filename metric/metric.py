"""
Metrics for comparing two images or two set of images.
"""

from PIL import Image
import os
import numpy as np
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import subprocess
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def fid(images1_path, images2_path, batch_size=1, dims=None, cuda=0, num_workers=0):
    """
    Calculating fid between two datasets.
    :param images1_path: path of the ground true images.
    :param batch_size: batch size
    :param images2_path: path of the test (fake) images.
    :param dims: size of the features
    :param num_workers: number of workers.
    :param cuda: The device used to calculate fid. cuda=0 by default to use GPU. If you want to use cpu,
    do cuda='cpu' instead.
    """
    batch_size_param = f"--batch-size={batch_size}"
    if isinstance(cuda, int):
        device_param = f"--device=cuda:{cuda}"
    else:
        device_param = f"--device=cpu"
    if dims is not None:
        if dims not in [64, 192, 768, 2048]:
            raise ValueError('Choice for dims only from 64, 192, 768, 2048')
        dims_param = f"--dims={dims}"
        command = ["python", "-m", "pytorch_fid", images1_path, images2_path, batch_size_param, dims_param,
                   device_param, f"--num-workers={num_workers}"]
    else:
        command = ["python", "-m", "pytorch_fid", images1_path, images2_path, batch_size_param, device_param,
                   f"--num-workers={num_workers}"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        fid_val_str = result.stdout.decode('utf-8')
        float_numbers = re.findall(r'\d+(\.\d+)?', fid_val_str)
        return float(float_numbers[0])
    except IndexError:
        return result


def metrics(img1_path, img2_path):
    """
    Calculating metrics comparing two different images. The metrics used are
    mean square error, peak signal noise ratio and structural similarity.

    :param img1_path: path of ground true image in jpg or png format.
    :param img2_path: path of test(fake) image in jpg or png format.
    :return: A dict form including the metric.
    """
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    # img2 = img2.resize(img1.size)
    # print(img1.shape, img2.shape)
    img1_gray = img1.convert('L')
    img2_gray = img2.convert('L')
    img2_gray = img2_gray.resize(img1_gray.size)
    img1_arr = np.asarray(img1)
    img2_arr = np.asarray(img2)
    mse = mean_squared_error(img1_arr, img2_arr)
    psnr = peak_signal_noise_ratio(img1_arr, img2_arr)
    img1_gray_arr, img2_gray_arr = np.array(img1_gray), np.array(img2_gray)
    ssim = structural_similarity(img1_gray_arr, img2_gray_arr, data_range=255)

    res = {'mse': mse, 'psnr': psnr, 'ssim': ssim}

    return res


def metrics_dataset(images1_path, images2_path, if_fid=False, batch_size=1, dims=None, cuda=0, num_workers=0):
    """
    Calculating metrics to show similarity between two dataset:
    average of mean square error, peak signal, noise ratio, and
    frechet_distance.
    :param images1_path: Path to the folder of the images.
    :param images2_path: Path to the folder of the images.
    :param if_fid: whether to calculate fid.
    :param batch_size: batch size used in fid.
    :param dims: size of the features
    :param cuda: The device used to calculate fid. cuda=0 by default to use GPU. If you want to use cpu,
    do cuda='cpu' instead.
    :return: Result of the metrics.
    """
    images1 = os.listdir(images1_path)
    images2 = os.listdir(images2_path)
    mse = []
    psnr = []
    ssim = []
    for img1, img2 in zip(images1, images2):
        res = metrics(images1_path + './' + img1, images2_path + './' + img2)
        mse.append(res['mse'])
        psnr.append(res['psnr'])
        ssim.append(res['ssim'])

    if if_fid:
        fid_val = fid(images1_path, images2_path, batch_size=batch_size, dims=dims, cuda=cuda, num_workers=num_workers)
    else:
        fid_val = False
    result = {
        'mse': np.mean(np.array(mse)),
        'psnr': np.mean(np.array(psnr)),
        'ssim': np.mean(np.array(ssim)),
        'fid': fid_val
    }

    return result


# if __name__ == "__main__":
#     fold_path1 = './fold1'
#     fold_path2 = './fold2'
#     print(fid(fold_path1, fold_path2))
