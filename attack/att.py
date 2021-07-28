# various attacks on the watermark
import cv2
import numpy as np


def cut_att_height(input_filename, output_file_name, ratio=0.80):
    # cut in height
    input_img = cv2.imread(input_filename)
    input_img_shape = input_img.shape
    height = int(input_img_shape[0] * ratio)

    cv2.imwrite(output_file_name, input_img[:height, :, :])


def cut_att_width(input_filename, output_file_name, ratio=0.80):
    # cut in width
    input_img = cv2.imread(input_filename)
    input_img_shape = input_img.shape
    width = int(input_img_shape[1] * ratio)

    cv2.imwrite(output_file_name, input_img[:, :width, :])


def resize_att(input_filename, output_file_name, out_shape=(500, 500)):
    # resize
    input_img = cv2.imread(input_filename)
    output_img = cv2.resize(input_img, dsize=out_shape)
    cv2.imwrite(output_file_name, output_img)


def bright_att(input_filename, output_file_name, ratio=0.75):
    # brightness adjustment
    # ratio should more than 0
    # ratio>1: brighter ratio<1: darker
    input_img = cv2.imread(input_filename)
    output_img = input_img * ratio
    output_img[output_img > 255] = 255
    cv2.imwrite(output_file_name, output_img)


def shelter_att(input_filename, output_file_name, ratio=0.1, n=3):
    # shelter
    # n is the number of the blocks
    # ratio is the size of the shelter
    input_img = cv2.imread(input_filename)
    input_img_shape = input_img.shape
    output_img = input_img.copy()
    for i in range(n):
        tmp = np.random.rand() * (1 - ratio)  # 随机选择一个地方，1-ratio是为了防止溢出
        start_height, end_height = int(tmp * input_img_shape[0]), int((tmp + ratio) * input_img_shape[0])
        tmp = np.random.rand() * (1 - ratio)
        start_width, end_width = int(tmp * input_img_shape[1]), int((tmp + ratio) * input_img_shape[1])

        output_img[start_height:end_height, start_width:end_width, :] = 0

    cv2.imwrite(output_file_name, output_img)


def salt_pepper_att(input_filename, output_file_name, ratio=0.01):
    # salt and pepper attack
    input_img = cv2.imread(input_filename)
    input_img_shape = input_img.shape
    output_img = input_img.copy()
    for i in range(input_img_shape[0]):
        for j in range(input_img_shape[1]):
            if np.random.rand() < ratio:
                output_img[i, j, :] = 255
    cv2.imwrite(output_file_name, output_img)


def rot_att(input_filename, output_file_name, angle=1):
    # rotate
    input_img = cv2.imread(input_filename)
    rows, cols, _ = input_img.shape
    M = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=angle, scale=1)
    output_img = cv2.warpAffine(input_img, M, (cols, rows))
    cv2.imwrite(output_file_name, output_img)
