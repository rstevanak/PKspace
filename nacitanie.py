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

        # print("\nMiesto")
        # print(space.attrib)
        # print("  Rotacia", end='\n   ')
        # for i in range(3):
        #     print(space[0][i].attrib, end='')
        #
        # print("\n  Obrys", end="\n   ")
        contour = []
        for i in range(4):
            # print(space[1][i].attrib, end='')
            contour.append([int(space[1][i].get('x')), int(space[1][i].get('y'))])

        center = int(space[0][0].get('x')), int(space[0][0].get('y'))
        h, w = int(space[0][1].get('h')), int(space[0][1].get('w'))

        M = cv2.getRotationMatrix2D(center, int(space[0][2].get('d')), 1)
        rows, cols, tmp = src.shape
        dst = cv2.warpAffine(src, M, (cols, rows))

        roi = dst[int(center[1] - h / 2):int(center[1] + h / 2), int(center[0] - w / 2):int(center[0] + w / 2)]


        # cv2.circle(src, center, 2, (255, 0, 0), -1)
        #
        # farbaobrysu = (0, 0, 255) if space.get('occupied', 0) == '1' else(0, 255, 0)
        # pts = np.array(contour, np.int32)
        # pts = pts.reshape((-1, 1, 2))
        # cv2.polylines(src, [pts], True, farbaobrysu)

        # cv2.imshow("PK c." + space.get('id'), roi)
        # cv2.imshow("src", src)
        #
        # cv2.waitKey(0)
        # cv2.destroyWindow("PK c." + space.get('id'))

        fin = cv2.resize(roi, finsize).flatten()
        # if not space.get('occupied'):
        #     cv2.imshow("Neohodnotene",roi)
        #     inp=cv2.waitKey(0)
        #     cv2.destroyWindow("Neohodnotene")
        #     print(space.get('id'))
        #
        #     if inp=='y':
        #         print("je")
        #     elif inp=='n':
        #         print("nie je")

        answers.append(space.get('occupied', -1))
        allspaces.append(fin)
    return allspaces, answers


def loadfrommeh():
    allparkinglots = []
    allanswers = []
    for file in glob.glob(os.path.join(os.getcwd() + '\meh', '*.jpg')):
        img = cv2.imread(file)
        e = xml.etree.ElementTree.parse(file[:-4] + '.xml')
        root = e.getroot()
        spaces, answers = getparkingspaces(img, root)
        allparkinglots += spaces
        allanswers += answers
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # for space in vsetky:
        #     cv2.imshow("PK c.", space[0])
        #
        #     cv2.waitKey(0)
        #     cv2.destroyWindow("PK c.")
    allparkinglots = np.array(allparkinglots)
    allanswers = np.array(allanswers)
    # print(allparkinglots)
    # print(allanswers)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return allparkinglots, allanswers
