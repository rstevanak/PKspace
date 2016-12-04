import glob
import xml.etree.ElementTree

import cv2
import math
import numpy as np
import os

def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
# subor = 'meh/2012-12-07_17_02_25'
# cely = cv2.imread(subor + '.jpg')
# cv2.imshow("cely", cely)
# otoceny=rotate_about_center(cely,-71)
# cv2.imshow("otoceny",otoceny)
percent=0
counter=0
for subor in glob.glob(os.path.join(os.getcwd() + '\meh', '*.jpg')):
    cely = cv2.imread(subor)
    e = xml.etree.ElementTree.parse(subor[:-4] + '.xml')
    root = e.getroot()
    cv2.imshow("cely", cely)
    for space in root:
        counter+=1
        if space.get('occupied')=='1':
            percent+=1
print(percent/counter*100)