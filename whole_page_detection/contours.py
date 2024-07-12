# Import preprocessors
import os
import cv2
import numpy as np

# Read image
dir = os.path.abspath(os.path.dirname(__file__))
im = cv2.imread(dir+'test_image_processed.jpg')

# Add padding around the original image
pad = 5
h, w = im.shape[:2]
im2 = ~(np.ones((h+pad*2, w+pad*2, 3), dtype=np.uint8))
im2[pad:pad+h, pad:pad+w] = im[:]
im = im2

# Blur it to remove noise
im = cv2.GaussianBlur(im, (5, 5), 5)

# Gray and B/W version
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
bw = cv2.threshold(im_gray, 200, 255, cv2.THRESH_BINARY)[1]

# Find contours and sort them by position
cnts, _ = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnts.sort(key=lambda x: cv2.boundingRect(x)[0])

# Draw contours on the original image
im_contours = im.copy()
cv2.drawContours(im_contours, cnts, -1, (0, 255, 0), 2)

# Find and save blocks
s1, s2 = w / 2, w / 10
i = 0
x2, y2, w2, h2 = 0, 0, 0, 0
for cnt in cnts:
    x, y, w, h = cv2.boundingRect(cnt)
    if (w + h < s1 and w + h > s2) and (i == 0 or (x2 + w2) < x):
        i += 1
        cv2.imwrite(dir + '/_' + str(i) + ".jpg", im[y:y + h, x:x + w])
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
    x2, y2, w2, h2 = x, y, w, h

# Save the processed images
cv2.imwrite(dir + '/out.png', im)
cv2.imwrite(dir + '/out_bw.png', bw)
cv2.imwrite(dir + '/out_contours.png', im_contours)

# Display the image with contours
cv2.imshow('Contours', im_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()