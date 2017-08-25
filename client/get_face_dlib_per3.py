# -*- coding:utf-8 -*-
import cv2
import datetime
import dlib
import os
import json
import requests
import time
import numpy as np
from face_landmark_detection import faceLandmarkDetection

import sys
import random
# from align_dlib import AlignDlib
from scipy.misc import imsave

from sftp_upload import sftp_upload

output_dir = '/Users/ngxin/ngxin/facerecognition/client/my_faces'

# host = '121.69.75.194'  #
# port = 22  # 端口
# username = 'wac'  # 用户名
# password = '8112whz'  # 密码
# local = '/Users/ngxin/facerecognition/client/face_recognition/my_faces'
# remote = '/home/wac/ngxin/ftp_upload/'


# face_predictor_path = './model/shape_predictor_68_face_landmarks.dat'
# align = AlignDlib(face_predictor_path)
# landmarkIndices = AlignDlib.OUTER_EYES_AND_NOSE

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#获取mac地址

def get_mac_address():
    import uuid
    node = uuid.getnode()
    mac = uuid.UUID(int=node).hex[-12:]
    return mac

TRACKED_POINTS = (17,21,22,26,36,39,42,45,31,35,48,54,57,8)
my_detector = faceLandmarkDetection(
    '/Users/ngxin/ngxin/facerecognition/server/model/shape_predictor_68_face_landmarks.dat')
my_face_detector = dlib.get_frontal_face_detector()

camera = cv2.VideoCapture(0)
index = 0
scale = 2
font = cv2.FONT_HERSHEY_SIMPLEX
text_list = []
coord_list = []
while True:
    print('Being processed picture %s' % index)
    frame, img = camera.read()
    size = img.shape
    cv2.imshow('face', img)
    if (index % 3 == 0):
        text_list = []
        coord_list = []

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img, (gray_img.shape[1]/scale, gray_img.shape[0]/scale))
        dets = my_face_detector(gray_img)
        detector_scores = my_face_detector.run(gray_img)[1]
        for i, d in enumerate(dets):
            add = (d.right() - d.left()) / 4

            x1 = d.left() - add
            x2 = d.right() + add
            y1 = d.top() - add
            y2 = d.bottom() + add
            x1 = x1 if x1 > 0 else 0
            x2 = x2 if x2 < gray_img.shape[1] else gray_img.shape[1]
            y1 = y1 if y1 > 0 else 0
            y2 = y2 if y2 < gray_img.shape[0] else gray_img.shape[0]


            face = gray_img[y1:y2, x1:x2]

            detector_score = []
            detector_score.append(detector_scores[i])
            cur_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            mac_addr = get_mac_address()
            path_init = output_dir + '/' + mac_addr + cur_time+str(index) + str(i) + '.jpg'
            path_aligned = output_dir + '/' + mac_addr + cur_time + str(index) + str(i) + '_init.jpg'
            image_points = my_detector.returnLandmarks(gray_img, d.left(), d.top(), d.right(), d.bottom(),
                                                       points_to_return=TRACKED_POINTS)
            for point in image_points:
                cv2.circle(img, (int(point[0])*scale, int(point[1])*scale), 2, (0, 0, 255), -1)

            model_points = np.array([
                (6.825897, 6.760612, 4.402142),  # Nose tip
                (1.330353, 7.122144, 6.903745),
                (-1.330353, 7.122144, 6.903745),  # Chin
                (-6.825897, 6.760612, 4.402142),  # Left eye left corner
                (5.311432, 5.485328, 3.987654),  # Right eye right corne
                (1.789930, 5.393625, 4.413414),  # Left Mouth corner
                (-1.789930, 5.393625, 4.413414),
                (-5.311432, 5.485328, 3.987654),
                (2.005628, 1.409845, 6.165652),
                (-2.005628, 1.409845, 6.165652),
                (2.774015, -2.080775, 5.048531),
                (-2.774015, -2.080775, 5.048531),
                (0.000000, -3.116408, 6.097667),
                (0.000000, -7.415691, 4.070434)])
            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
            dist_coeffs = np.zeros((5, 1))
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs, flags=cv2.CV_ITERATIVE)
            rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)
            pose_mat = np.hstack((rotation_matrix, translation_vector))
            socore_result = cv2.decomposeProjectionMatrix(pose_mat)[-1]
            # print socore_result[0], socore_result[1], socore_result[2]
            # print '----------------' + str(i)

            reprojectsrc = np.array([
                (10.0, 10.0, 10.0),
                (10.0, 10.0, -10.0),
                (10.0, -10.0, -10.0),
                (10.0, -10.0, 10.0),
                (-10.0, 10.0, 10.0),
                (-10.0, 10.0, -10.0),
                (-10.0, -10.0, -10.0),
                (-10.0, -10.0, 10.0)
            ])
            for p in image_points:
                cv2.circle(img, (int(p[0])*scale, int(p[1])*scale), 3, (0, 0, 255), -1)
            #
            # if detector_score[0] <1.0 or (result[0]>10 or result[0<-10]):
            #     # rect = cv2.rectangle(img, (x1 * scale, y1 * scale), (x2 * scale, y2 * scale), (0, 0, 255), 2)
            #     # cv2.putText(img, 'welcome', (x1 * scale, y1 * scale), font, 3, (0, 255, 255), 2)
            #     rect = cv2.rectangle(img, (x1 , y1 ), (x2 , y2 ), (0, 0, 255), 2)
            #     #cv2.putText(img, 'welcome'+str(result[0])[1:4]+'\n'+str(result[1])[1:4]+'\n'+str(result[2])[1:4], (x1 , y1 ), font, 3, (0, 255, 255), 2,lineType=4)
            #     cv2.putText(img, 'welcome',(x1 , y1 ), font, 3, (0, 255, 255), 2)
            #     #cv2.putText(img, 'x:'+str(result[0])[1:5], (50 , 50 ), font, 2, (0, 255, 255), 2)
            #     cv2.putText(img, 'x:'+str(result[0])[1:6], (50 , 50 ), font, 2, (0, 255, 255), 2)
            #     cv2.putText(img, 'y:'+str(result[1])[1:6], (50 , 100 ), font, 2, (0, 255, 255), 2)
            #     cv2.putText(img, 'z:'+str(result[2])[1:6], (50 , 150 ), font, 2, (0, 255, 255), 2)
            #     continue
            if detector_score[0] > 0.8 and -20<socore_result[0]<20 and -20<socore_result[1]<20 and -20<socore_result[2]<20:
                cv2.imwrite(path_init, face)


            # cv2.imwrite(path_aligned, gray_img[d.top():d.bottom(), d.left():d.right()])
            #sftp_upload(host, port, username, password, local, remote)#上传所有图片？
                new_coor_x1 = d.left()-x1
                new_coor_x2 = d.right()-x1

                new_coor_y1 = d.top()-y1
                new_coor_y2 = d.bottom()-y1

                # cv2.imwrite('./result.jpg', face[new_coor_y1:new_coor_y2, new_coor_x1:new_coor_x2])

                message_search = {"id": "weibo",
                                  "pics": [{
                                               "path": path_init,
                                               # "path_aligned": path_aligned,
                                                "coord":[new_coor_x1,new_coor_y1,new_coor_x2,new_coor_y2],
                                               "id": cur_time, "consume_history": "True"}]
                                  }
                temp = json.dumps(message_search)
                payloadfiles = {'files': temp}

                start = time.time()
                r = requests.post("http://0.0.0.0:3006/query", data=payloadfiles)#为什么返回值为200
                result = json.loads(r.text)

                text = ''
                if result['result'] != 'error':
                    text = result['tag'][cur_time]

                cv2.putText(img, text, (x1*scale, y1*scale), font, 3, (0, 255, 255), 2)
                # cv2.putText(img, 'x:' + str(socore_result[0])[1:6], (50, 50), font, 2, (0, 255, 255), 2)
                # cv2.putText(img, 'y:' + str(socore_result[1])[1:6], (50, 100), font, 2, (0, 255, 255), 2)
                # cv2.putText(img, 'z:' + str(socore_result[2])[1:6], (50, 150), font, 2, (0, 255, 255), 2)

                rect = cv2.rectangle(img, (x1*scale, y1*scale), (x2*scale, y2*scale), (0, 0, 255), 2)

                text_list.append(text)
                #coord_list.append([x1*scale, y1*scale, x2*scale, y2*scale])
                coord_list.append([x1*scale, y1*scale, x2*scale, y2*scale])
                print time.time() - start
                print r.text
            else:
                rect = cv2.rectangle(img, (x1*scale, y1*scale), (x2*scale, y2*scale), (0, 0, 255), 2)

                text = ''
                text_list.append(text)
                coord_list.append([x1*scale, y1*scale, x2*scale, y2*scale])
                #     #cv2.putText(img, 'welcome'+str(result[0])[1:4]+'\n'+str(result[1])[1:4]+'\n'+str(result[2])[1:4], (x1 , y1 ), font, 3, (0, 255, 255), 2,lineType=4)
                cv2.putText(img, '',(x1*scale , y1*scale ), font, 3, (0, 255, 255), 2)
                # cv2.putText(img, 'x:'+str(socore_result[0])[1:6], (50 , 50 ), font, 2, (0, 255, 255), 2)
                # cv2.putText(img, 'y:'+str(socore_result[1])[1:6], (50 , 100 ), font, 2, (0, 255, 255), 2)
                # cv2.putText(img, 'z:'+str(socore_result[2])[1:6], (50 , 150 ), font, 2, (0, 255, 255), 2)
                continue
    else:
        for i, text in enumerate(text_list):
            cv2.putText(img, text, (coord_list[i][0], coord_list[i][1]), font, 3, (0, 255, 255), 2)
            # cv2.putText(img, 'x:' + str(socore_result[0])[1:6], (50, 50), font, 2, (0, 255, 255), 2)
            # cv2.putText(img, 'y:' + str(socore_result[1])[1:6], (50, 100), font, 2, (0, 255, 255), 2)
            # cv2.putText(img, 'z:' + str(socore_result[2])[1:6], (50, 150), font, 2, (0, 255, 255), 2)

            rect = cv2.rectangle(img, (coord_list[i][0], coord_list[i][1]), (coord_list[i][2], coord_list[i][3]), (0, 0, 255), 2)


    cv2.imshow('face', img)
    index += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
