#!/usr/bin/python
# from msilib.schema import Class
import shutil
import time
import numpy as np
import scipy.ndimage
import scipy.misc
from scipy import ndimage
import argparse
import os
import imageio.v2 as imageio
import pathlib
from multiprocessing import Pool


class NormalMapGenerator:
    def __init__(self, _path):
        self.__path = _path
        self.__max_value = 30
        self.__offset = 2
        self.__check_dic()

    def __check_dic(self):
        file_path = pathlib.PurePath(self.__path)
        file_name = pathlib.PurePath(self.__path).stem
        # print("self.__path=" + self.__path)
        # print("create folder:" + str(file_path.parent) + str(file_name))
        if not os.path.exists(str(file_path.parent) + "/" + str(file_name)):
            # print("create folder:" + str(file_path.parent) + "/" + str(file_name))
            os.mkdir(str(file_path.parent) + "/" + str(file_name))
        else:
            shutil.rmtree(str(file_path.parent) + "/" + str(file_name))
            os.mkdir(str(file_path.parent) + "/" + str(file_name))

    def __smooth_gaussian(self, im, sigma):

        if sigma == 0:
            return im

        im_smooth = im.astype(float)
        kernel_x = np.arange(-3 * sigma, 3 * sigma + 1).astype(float)
        kernel_x = np.exp((-(kernel_x**2)) / (2 * (sigma**2)))

        im_smooth = scipy.ndimage.convolve(im_smooth, kernel_x[np.newaxis])

        im_smooth = scipy.ndimage.convolve(im_smooth, kernel_x[np.newaxis].T)

        return im_smooth

    def gradient(self, im_smooth):

        gradient_x = im_smooth.astype(float)
        gradient_y = im_smooth.astype(float)

        kernel = np.arange(-1, 2).astype(float)
        kernel = -kernel / 2

        gradient_x = scipy.ndimage.convolve(gradient_x, kernel[np.newaxis])
        gradient_y = scipy.ndimage.convolve(gradient_y, kernel[np.newaxis].T)

        return gradient_x, gradient_y

    def __sobel(self, im_smooth):
        gradient_x = im_smooth.astype(float)
        gradient_y = im_smooth.astype(float)

        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        gradient_x = scipy.ndimage.convolve(gradient_x, kernel)
        gradient_y = scipy.ndimage.convolve(gradient_y, kernel.T)

        return gradient_x, gradient_y

    def __compute_normal_map(self, gradient_x, gradient_y, intensity=1):

        width = gradient_x.shape[1]
        height = gradient_x.shape[0]
        max_x = np.max(gradient_x)
        max_y = np.max(gradient_y)

        max_value = max_x

        if max_y > max_x:
            max_value = max_y

        normal_map = np.zeros((height, width, 3), dtype=np.float32)

        intensity = 1 / intensity

        strength = max_value / (max_value * intensity)

        normal_map[..., 0] = gradient_x / max_value
        normal_map[..., 1] = gradient_y / max_value
        normal_map[..., 2] = 1 / strength

        norm = np.sqrt(
            np.power(normal_map[..., 0], 2)
            + np.power(normal_map[..., 1], 2)
            + np.power(normal_map[..., 2], 2)
        )

        normal_map[..., 0] /= norm
        normal_map[..., 1] /= norm
        normal_map[..., 2] /= norm

        normal_map *= 0.5
        normal_map += 0.5

        return normal_map

    def __create_out_put_file(self, _path, blur, detail_scale):
        file_path = _path[: _path.rfind("/")]
        file_name = os.path.splitext(_path)[0][
            os.path.splitext(_path)[0].rfind("/") + 1 :
        ]
        file_format = os.path.splitext(_path)[-1]
        return f"{file_path}/{ file_name}/{file_name}_normal_map_blur_{blur}_detail_scale_{detail_scale}{file_format}"

    def _generate_normal_map(self, blur=1, detail_scale=10):
        output_file = self.__create_out_put_file(self.__path, blur, detail_scale)
        im = imageio.imread(self.__path)

        if im.ndim == 3:
            im_grey = np.zeros((im.shape[0], im.shape[1])).astype(float)
            im_grey = im[..., 0] * 0.3 + im[..., 1] * 0.6 + im[..., 2] * 0.1
            im = im_grey
        im_smooth = self.__smooth_gaussian(im, blur)
        sobel_x, sobel_y = self.__sobel(im_smooth)
        normal_map = self.__compute_normal_map(sobel_x, sobel_y, detail_scale)

        imageio.imwrite(output_file, normal_map)

    def _generate_normal_map_pool(self, index):
        blur = round(index / self.__max_value)
        detail_scale = index % self.__max_value + 1
        # if detail_scale % self.__max_value != 0:
        #     return
        # if blur%3==0 and detail_scale%3==0:
        # print(f"blur={blur}, detail_scale={detail_scale}")
        output_file = self.__create_out_put_file(self.__path, blur, detail_scale)
        im = imageio.imread(self.__path)

        if im.ndim == 3:
            im_grey = np.zeros((im.shape[0], im.shape[1])).astype(float)
            im_grey = im[..., 0] * 0.3 + im[..., 1] * 0.6 + im[..., 2] * 0.1
            im = im_grey
        im_smooth = self.__smooth_gaussian(im, blur)
        sobel_x, sobel_y = self.__sobel(im_smooth)
        normal_map = self.__compute_normal_map(sobel_x, sobel_y, detail_scale)
        imageio.imwrite(output_file, normal_map)


def main():
    parser = argparse.ArgumentParser(description="Compute normal map of an image")
    parser.add_argument("input_file", type=str, help="input image path")
    args = parser.parse_args()

    start_time = time.time()
    nmg = NormalMapGenerator(args.input_file)
    with Pool() as p:
        p.map(nmg._generate_normal_map_pool, [blur for blur in range(0, 300, 2)])
    print("time taken: ", time.time() - start_time)
    # for blur in range(1, 30,3):
    #     for detail_scale in range(1, 30,3):
    #         print(f"blur={blur}, detail_scale={detail_scale}")
    #         nmg._generate_normal_map(blur, detail_scale)
    # nmg._generate_normal_map()


if __name__ == "__main__":
    main()
