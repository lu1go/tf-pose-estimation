import argparse
import logging
import time
import pdb

import segmentation
import common
import cv2
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    # parser.add_argument('--image', type=str, default='/Users/ildoonet/Downloads/me.jpg')
    parser.add_argument('--image', type=str, default='./images/apink2.jpg')
    # parser.add_argument('--model', type=str, default='mobilenet_320x240', help='cmu / mobilenet_320x240')
    parser.add_argument('--model', type=str, default='mobilenet_thin_432x368', help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--split', type=str, default='4x6', help='split the original image into m by n blocks')
    args = parser.parse_args()

    m, n = segmentation.split_mn(args.split)
    w, h = model_wh(args.model)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
    # image is of regularized size
    image = common.read_imgfile(args.image, w, h)
    t = time.time()

    image_block = segmentation.create_image_block(image, m, n)

    humans_block = []
    for i in range(m):
        humans_block.append([])
        for j in range(n):
            humans_block[i].append(e.inference(image_block[i][j]))

    # humans = e.inference(image)
    # pdb.set_trace()
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    # img is of original size
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    img_block = segmentation.create_image_block(img, m, n)

    # #####################################################
    # for i in range(m):
    #     for j in range(n):
    #         print(np.shape(img_block[i][j]))

    # pdb.set_trace()
    for i in range(m):
        for j in range(n):
            # draw humans uses original image but the humans are inferred using normalized image
            img_block[i][j] = TfPoseEstimator.draw_humans(img_block[i][j], humans_block[i][j], imgcopy=False)

    # image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    # pdb.set_trace()
    # concatenate the img blocks
    restored_img = segmentation.image_concat(img_block)
    cv2.imshow('reconnected pose estimation result', restored_img)
    # cv2.imshow('tf-pose-estimation result', image)
    cv2.waitKey()
    # cv2.imwrite('t1pt2.jpg', image)

    img_name = args.image.split('/')[-1]
    img_name = 'result_' + img_name
    cv2.imwrite(img_name, restored_img)

    import sys
    sys.exit(0)

    logger.info('3d lifting initialization.')
    poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')

    image_h, image_w = image.shape[:2]
    standard_w = 640
    standard_h = 480

    pose_2d_mpiis = []
    visibilities = []
    for human in humans:
        pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
        pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])
        visibilities.append(visibility)

    pose_2d_mpiis = np.array(pose_2d_mpiis)
    visibilities = np.array(visibilities)
    transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)
    pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)
    # /zym pose_3d为(1,3,17)的矩阵代表17个关键点的三维坐标
    print('zym{}'.format(pose_3d))

    import matplotlib.pyplot as plt

    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Result')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # show network output
    a = fig.add_subplot(2, 2, 2)
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(heatmap.shape[1], heatmap.shape[0])), alpha=0.5)
    tmp = np.amax(e.heatMat, axis=2)
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    tmp2 = e.pafMat.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    a = fig.add_subplot(2, 2, 3)
    a.set_title('Vectormap-x')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 2, 4)
    a.set_title('Vectormap-y')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    for i, single_3d in enumerate(pose_3d):
        plot_pose(single_3d)
    plt.show()

    pass
