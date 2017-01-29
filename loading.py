import numpy as np
import cv2
import xml.etree.ElementTree
import os
import glob
import json

import sys


def getparkingspaces(src, tabroot):
    finsize = (80, 80)
    allspaces = []
    answers = []
    for space in tabroot:
        center = int(space[0][0].get('x')), int(space[0][0].get('y'))
        h, w = int(space[0][1].get('h')), int(space[0][1].get('w'))

        M = cv2.getRotationMatrix2D(center, int(space[0][2].get('d')), 1)
        rows, cols, tmp = src.shape
        dst = cv2.warpAffine(src, M, (cols, rows))
        roi = dst[int(center[1] - h / 2):int(center[1] + h / 2), int(center[0] - w / 2):int(center[0] + w / 2)]
        fin = cv2.resize(roi, finsize).flatten()
        answers.append(space.get('occupied', 1))
        allspaces.append(fin)
    return allspaces, answers


def getparkingspaces2(src, tabroot):
    finsize = (80, 80)
    allspaces = []
    answers = []
    for space in tabroot:
        center = int(space[0][0].get('x')), int(space[0][0].get('y'))
        h, w = int(space[0][1].get('h')), int(space[0][1].get('w'))
        circ=np.math.sqrt(h * h + w * w) /2
        croppped= src[max(0,int(center[1] - circ)):int(center[1] + circ),
                  max(0, int(center[0] - circ)):int(center[0] + circ)]
        rows, cols = croppped.shape[:2]
        center=(rows/2,cols/2)
        M = cv2.getRotationMatrix2D(center, int(space[0][2].get('d')), 1)
        dst = cv2.warpAffine(croppped, M, (cols, rows))

        roi = dst[int(center[1] - h / 2):int(center[1] + h / 2), int(center[0] - w / 2):int(center[0] + w / 2)]
        fin = cv2.resize(roi, finsize).flatten()
        answers.append(space.get('occupied', 1))
        allspaces.append(fin)
    return allspaces, answers


def getadmanspaces(src, desc):
    finsize = (80, 80)
    allspaces = []
    answers = []
    for space in desc.get('spots'):
        points=space.get('points')
        angle=space.get('rotation')
        xsum,ysum=0,0
        for x,y in points:
            xsum+=x
            ysum+=y
        center = np.ceil(xsum/len(points)),np.ceil(ysum/len(points))
        minx,miny=sys.maxsize,sys.maxsize
        maxx,maxy=0,0
        for x,y in points:
            xrotated=((x - center[0]) * np.math.cos(angle)) - ((y - center[1]) * np.sin(angle)) + center[0]
            yrotated=((x - center[0]) * np.math.sin(angle)) + ((y - center[1]) * np.cos(angle)) + center[1]
            minx, miny = max(min(minx, int(xrotated)), 0), max(min(miny, int(yrotated)), 0)
            maxx, maxy = max(maxx, int(xrotated)), max(maxy, int(yrotated))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rows, cols, tmp = src.shape
        rotated = cv2.warpAffine(src, M, (cols, rows))
        roi = rotated[minx:maxx, miny:maxy]
        fin = cv2.resize(roi, finsize).flatten()
        allspaces.append(fin)

        answers.append(space.get('occupied', -1))
    return allspaces, answers


def loadfrom(folder,adman=False):
    if os.path.exists(os.path.join(os.path.curdir,folder,"saved.npz")):
        with open(os.path.join(os.path.curdir,folder,"saved.npz"),'rb') as f:
            loaded=np.load(f)
            allparkinglots, allanswers=loaded['arr_0'],loaded['arr_1']
        return allparkinglots,allanswers
    allparkinglots = []
    allanswers = []
    for file in glob.glob(os.path.join(os.path.curdir, folder, '*.jpg')):
        img = cv2.imread(file)
        if adman:
            desc=json.load(file[:-4] + '.json')
            spaces,answers = getadmanspaces(img, desc)
        else:
            e = xml.etree.ElementTree.parse(file[:-4] + '.xml')
            root = e.getroot()
            spaces, answers = getparkingspaces2(img, root)
        allparkinglots += spaces
        allanswers += answers
    allparkinglots = np.array(allparkinglots)
    allanswers = np.array(allanswers)
    with open(os.path.join(os.path.curdir,folder,"saved.npz"),'wb') as f:
        np.savez(f,allparkinglots,allanswers)
    return allparkinglots, allanswers


def rate(folder):
    for file in glob.glob(os.path.join(os.getcwd(),folder, '*.jpg')):
        img = cv2.imread(file)
        e = xml.etree.ElementTree.parse(file[:-4] + '.xml')
        root = e.getroot()
        for space in root:
            while not space.get('occupied'):
                contour = []
                for i in range(4):
                    # print(space[1][i].attrib, end='')
                    contour.append([int(space[1][i].get('x')), int(space[1][i].get('y'))])
                pts = np.array(contour, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, (0, 255, 0))
                cv2.imshow("Not rated space", img)
                inp = cv2.waitKey(0)
                cv2.destroyWindow("Not rated space")
                if inp == ord('y'):
                    space.set('occupied', '1')
                elif inp == ord('n'):
                    space.set('occupied', '0')
                img = cv2.imread(file)
                e.write(file[:-4] + '.xml', xml_declaration=True)
    print("All spaces rated now")
