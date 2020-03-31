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
    horizontal_slice = list()
    for i_b in range(len(block_image)):
        # concatenate each block in horizontal direction
        horizontal_slice.append(np.concatenate(block_image[i_b], axis=1))
        reunion_image = np.concatenate(horizontal_slice, axis=0)
    return reunion_image

#
# def image_concat(block_image):
#     # rows number of each block
#     block_m = block_image[0][0].shape[0]#
#     # cols number of each block
#     block_n = block_image[0][0].shape[1]#
#     # the height of original image
#     reunion_image_h = 0
#     for i_b in range(len(block_image)):
#         reunion_image_h += block_image[i_b][0].shape[0]
#     # the width of original image
#     reunion_image_w = 0
#     for j_b in range(len(block_image[0])):
#         reunion_image_w += block_image[0][j_b].shape[1]
#     # initialize the concatenated image
#     reunion_image = np.zeros((reunion_image_h, reunion_image_w, 3), np.uint8)
#     # the first column of the image
#
#     for i_b in range(len(block_image)):
#         row_idx += block_image[0][0].shape[0]
#         reunion_image[0:block_image[0][0].shape[0], 0:block_image[0][0].shape[1], :] = block_image[0][0]
#
#     row_idx1 = 0
#     col_idx1 = 0
#     for i_b in range(1, len(block_image)):
#         for j_b in range(1, len(block_image[0])):
#             row_idx1 += block_image[i_b - 1][j_b - 1].shape[0]
#             row_idx2 = col_idx1 + block_image[i_b - 1][j_b].shape[0]
#             col_idx1 += block_image[i_b - 1][j_b - 1].shape[1]
#             col_idx2 = row_idx1 + block_image[i_b][j_b - 1].shape[1]
#             reunion_image[row_idx1:row_idx2, col_idx1:col_idx2, :] = block_image[i_b][j_b]
#     return reunion_image

