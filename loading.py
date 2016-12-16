import numpy as np
import cv2
import xml.etree.ElementTree
import os
import glob


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

def loadfrom(folder):
    allparkinglots = []
    allanswers = []
    for file in glob.glob(os.path.join(os.getcwd() + '\\'+folder, '*.jpg')):
        img = cv2.imread(file)
        e = xml.etree.ElementTree.parse(file[:-4] + '.xml')
        root = e.getroot()
        spaces, answers = getparkingspaces(img, root)
        allparkinglots += spaces
        allanswers += answers
    allparkinglots = np.array(allparkinglots)
    allanswers = np.array(allanswers)
    return allparkinglots, allanswers
def rate(folder):
    for file in glob.glob(os.path.join(os.getcwd() + '\\' + folder, '*.jpg')):
        img = cv2.imread(file)
        e = xml.etree.ElementTree.parse(file[:-4] + '.xml')
        root = e.getroot()
        for space in root:
            while not space.get('occupied'):
                contour=[]
                for i in range(4):
                    # print(space[1][i].attrib, end='')
                    contour.append([int(space[1][i].get('x')), int(space[1][i].get('y'))])
                pts = np.array(contour, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, (0,255,0))
                cv2.imshow("Not rated space", img)
                inp = cv2.waitKey(0)
                cv2.destroyWindow("Not rated space")
                if inp==ord('y'):
                    space.set('occupied','1')
                elif inp==ord('n'):
                    space.set('occupied','0')
                img = cv2.imread(file)
                e.write(file[:-4]+'.xml',xml_declaration=True)
    print("ohodnotene vsetky")