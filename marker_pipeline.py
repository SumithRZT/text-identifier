import numpy as np
import cv2
from sklearn.externals import joblib

clf = joblib.load('text_marker.pkl')

img = cv2.imread('/Users/sumithkrishna/PycharmProjects/TextPatchExtractor/test_data/test3.jpg', 0)
# img = cv2.pyrDown(img)

box_list = []
shp = list(img.shape)
for x in range(0, int(shp[0]) - 15, 8):
    for y in range(0, int(shp[1]) - 15, 8):
        box = (y, x, y + 15, x + 15)
        box_list.append(box)

img_patches = [img[abs(box[1]):abs(box[3]), abs(box[0]):abs(box[2])] for box in
                 box_list]

print(len(box_list))
flattened = []
for im in img_patches:
    flattened.append(im.flatten())

pred = clf.predict(flattened)

count = 0
black = np.zeros((15, 15), dtype=int)
for i in pred:
    if i == 1:
        # cv2.imwrite("/Users/sumithkrishna/PycharmProjects/TextPatchExtractor/predicted/"+str(count)+".jpg", img_patches[count])
        box = box_list[count]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
        img[abs(box[1]):abs(box[3]), abs(box[0]):abs(box[2])] = black
    count += 1

imgray = img

ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

new_contours = []

for c in contours:
    shp = np.asarray(c).shape

    if shp[0] is not 1 or shp[1] is not 1:
        new_contours.append(c)

from PatchExtractor import nms

new = cv2.imread('/Users/sumithkrishna/PycharmProjects/TextPatchExtractor/test_data/test3.jpg', 0)
rect_list = []
for contour in contours[1:]:
    arr = np.squeeze(np.asarray(contour), axis=1)

    rect_list.append([min(arr[:, 0:-1])[0], min(arr[:, 1:])[0], max(arr[:, 0:-1])[0], max(arr[:, 1:])[0]])
    # cv2.rectangle(new, (min(arr[:, 0:-1]), min(arr[:, 1:])), (max(arr[:, 0:-1]), max(arr[:, 1:])), (0, 0, 255), 3)
localized = nms.non_max_suppression(np.asarray(rect_list))
for l in localized:
     cv2.rectangle(new, (l[0], l[1]), (l[2], l[3]), (0, 0, 200), 3)
cv2.imwrite('result_new.jpg', new)



