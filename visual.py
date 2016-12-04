import glob
import os
import numpy as np
import cv2
import xml.etree.ElementTree


def visualizefrommeh(answered):
    counter=-1
    for file in glob.glob(os.path.join(os.getcwd() + '\meh', '*.jpg')):
        img = cv2.imread(file)
        e = xml.etree.ElementTree.parse(file[:-4] + '.xml')
        root = e.getroot()
        for space in root:
            counter += 1
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

            centercolor = (0, 0, 255) if int(space.get('occupied', 0)) else(0, 255, 0)
            cv2.circle(img, center, 2, centercolor, -1)

            if answered.get(counter):
                contourcolor = (0, 0, 255) if int(answered.get(counter, 0)) else(0, 255, 0)
            else:
                contourcolor = (0, 255, 255)
            pts = np.array(contour, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, contourcolor)
        cv2.imshow("PKspaces", img)
        cv2.waitKey(0)
