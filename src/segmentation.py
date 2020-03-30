import numpy as np


def split_mn(split_str):
    m, n = split_str.split('x')
    return int(m), int(n)


def create_image_block(image, row_number, col_number):
    block_row = np.array_split(image, row_number, axis=0)  # 垂直方向切割，得到很多横向长条
    print(image.shape)
    img_blocks = []
    for block in block_row:
        block_col = np.array_split(block, col_number, axis=1)  # 水平方向切割，得到很多图像块
        img_blocks += [block_col]
    return img_blocks


def image_concat(block_image):
    # rows number of each block
    block_m = block_image[0][0].shape[0]
    # cols number of each block
    block_n = block_image[0][0].shape[1]
    # the concatenated image
    reunion_image = np.zeros((len(block_image) * block_m, len(block_image[0]) * block_n, 3), np.uint8)
    for i_b in range(len(block_image)):
        for j_b in range(len(block_image[0])):
            reunion_image[i_b * block_m:(i_b + 1) * block_m, j_b * block_n:(j_b + 1) * block_n, :] = block_image[i_b][j_b]
    return reunion_image

