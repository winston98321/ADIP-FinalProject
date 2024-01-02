#img= cv2.imread('img/aurora_1.jpg') 

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

img= cv2.imread('img/aurora_2.jpg')

img = cv2.resize(img, (500, 500))
Z = np.float32(img.reshape((-1,3)))
db = DBSCAN(eps=5, min_samples=1).fit(Z[:,:2])

plt.imshow(np.uint8(db.labels_.reshape(img.shape[:2])))
plt.show()

'''
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

Z = np.float32(img.reshape((-1,3)))

# Define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.5)
ret, label, center = cv2.kmeans(Z, 5, None, criteria, 6, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make the original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = np.uint8(label.reshape(img.shape[:2]))
res2.shape

plt.imshow(res2)
plt.show()'''
