import glob
import json
import os
import numpy as np
import cv2
import xml.etree.ElementTree


def visualizefrom(folder,answered):
    counter=-1
    for file in glob.glob(os.path.join(os.path.curdir,folder, '*.jpg')):
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

            if answered.get(counter):
                if space.get('occupied'):
                    if not int(space.get('occupied')) == int(answered.get(counter)):
                        centercolor =(0, 0, 255)
                        cv2.circle(img, center, 2, centercolor, -1)
                contourcolor = (0, 0, 255) if answered.get(counter)=="1" else(0, 255, 0)
            else:
                contourcolor = (0, 255, 255)
            pts = np.array(contour, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, contourcolor)
        cv2.imshow("PKspaces", img)
        cv2.waitKey(0)
def visualizefromadman(folder,answered):
    counter = -1
    for file in glob.glob(os.path.join(os.path.curdir, folder, '*.jpg')):
        img = cv2.imread(file)
        desc = json.load(file[:-4] + '.json')
        for space in desc.get('spots'):
            counter += 1
            contour = []
            xsum,ysum=0,0
            for x,y in space.get('points'):
                contour.append([int(x), int(y)])
                xsum += int(x)
                ysum += int(y)
            center = np.math.ceil(xsum/len(space.get('points'))), np.math.ceil(ysum/len(space.get('points')))

            if answered.get(counter):
                if space.get('occupied'):
                    if not int(space.get('occupied')) == int(answered.get(counter)):
                        centercolor = (0, 0, 255)
                        cv2.circle(img, center, 2, centercolor, -1)
                contourcolor = (0, 0, 255) if answered.get(counter) == "1" else(0, 255, 0)
            else:
                contourcolor = (0, 255, 255)
            pts = np.array(contour, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, contourcolor)
        cv2.imshow("PKspaces", img)
        cv2.waitKey(0)