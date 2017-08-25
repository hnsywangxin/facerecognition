# coding: utf-8
import cv2
from scipy import misc
import tensorflow as tf
import facenet
import numpy as np
from align_dlib import AlignDlib
from matplotlib import pyplot as plt

with tf.Graph().as_default() as graph:
    with tf.Session() as sess:
        # Load the model
        facenet.load_model('./model/20170512-110547.pb')

        # Get input and output tensors
        images_placeholder = graph.get_tensor_by_name("input:0")
        embeddings = graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

face_predictor_path = './model/shape_predictor_68_face_landmarks.dat'
align = AlignDlib(face_predictor_path)
landmarkIndices = AlignDlib.OUTER_EYES_AND_NOSE

def readimg(img_path):
    img = misc.imread(img_path, mode='RGB')

    img = misc.imresize(img, (160, 160))
    img = facenet.prewhiten(img)
    img = np.expand_dims(img, axis=0)

    return img

def get_embedding(img_path, coord):
    img = misc.imread(img_path, mode='RGB')

    # judge alignment
    #aligned = align.align(160, img, [0, 0, img.shape[1], img.shape[0]], landmarkIndices=landmarkIndices)

    #misc.imsave(img_path.split('.')[0]+'_aligned.jpg', aligned)
    img = img[coord[1]:coord[3], coord[0]:coord[2]]
    img = misc.imresize(img, (160,160))
    # save_path = '/Users/ngxin/ngxin/facerecognition/server/test/test{}.jpg'.format(str(img_path[-10:]))
    # cv2.imwrite(save_path, img)



    img = facenet.prewhiten(img)
    img = np.expand_dims(img, axis=0)

    # aligned = facenet.prewhiten(aligned)
    # aligned = np.expand_dims(aligned, axis=0)


    # Run forward pass to calculate embeddings
    feed_dict = {images_placeholder: img, phase_train_placeholder: False}
    emb = sess.run(embeddings, feed_dict=feed_dict)

    # Run forward pass to calculate embeddings
    # feed_dict_aligned = {images_placeholder: aligned, phase_train_placeholder: False}
    # emb_aligned = sess.run(embeddings, feed_dict=feed_dict_aligned)

    #return emb.ravel(), emb_aligned.ravel()
    return emb.ravel()

# # for test
# import os
# from time import time
# def main(dir_path):
#     img_all = os.listdir(dir_path)
#     for f in img_all:
#         start = time()
#         embedding_result = get_embedding(os.path.join(dir_path, f))
#         print time() - start
#         print embedding_result
#
# main('./data')

# embedding_result = get_embedding('/Users/ngxin/ngxin/facerecognition/client/my_faces/acbc32c974db20170824171539795646330.jpg', [0,0,276,321])
# print embedding_result.shape