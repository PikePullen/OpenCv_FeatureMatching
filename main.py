import numpy as np
import matplotlib.pyplot as plt
import cv2

def display_img(img, cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    plt.imshow(img, cmap=cmap)
    plt.show()

reeses = cv2.imread('../DATA/reeses_puffs.png',0)
# display_img(reeses)

cereals = cv2.imread('../DATA/many_cereals.jpg',0)
# display_img(cereals)

"""
Bruteforce matching
This is a really poor matching method, results are poor
"""

# orb = cv2.ORB_create()
# keypoints1, descriptors1 = orb.detectAndCompute(reeses, None)
# keypoints2, descriptors2 = orb.detectAndCompute(cereals, None)

# bruteForce = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bruteForce.match(descriptors1,descriptors2)

"""distance, less distance it has, the better of a match"""
# single_match = matches[0]
# print(single_match.distance)
#
# matches = sorted(matches, key=lambda x:x.distance)
#
# reeses_matches = cv2.drawMatches(reeses,keypoints1,cereals,keypoints2,matches[:25],None,flags=2)
# display_img(reeses_matches)

"""
Sift matching
this is a little bit better version of brute force
"""

# sift = cv2.xfeatures2d.SIFT_create()
# sift = cv2.SIFT_create()
# kp1, d1 = sift.detectAndCompute(reeses, None)
# kp2, d2 = sift.detectAndCompute(cereals, None)
#
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(d1,d2,k=2)
# print(len(matches))
#
# # less distance is better match
# # begin ratio match match1 < 74% of match 2
# good = []
# for match1,match2 in matches:
#     """
#     IF MATCH1 DISTANCE IS LESS THAN 75% OF MATCH 2 DISTANCE
#     THEN DESCRIPTOR WAS A GOOD MATCH
#     """
#     if match1.distance < .75*match2.distance:
#         good.append([match1])
#
# print(len(good))
#
# reeses_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,good,None,flags=2)
# display_img(reeses_matches)

"""
FLANN
faster, but not as good matching
"""

sift = cv2.SIFT_create()
kp1, d1 = sift.detectAndCompute(reeses, None)
kp2, d2 = sift.detectAndCompute(cereals, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(d1,d2,k=2)
matchesMask = [[0,0] for i in range(len(matches))] #added this instead of using 'good' below, uses index insteas

# good = []
# for match1,match2 in matches:
#     if match1.distance < .7*match2.distance:
#         good.append([match1])

for i,(match1,match2) in enumerate(matches):
    if match1.distance < .7*match2.distance:
        matchesMask[i] = [1,0]

# flags=2 only shows matches, flags=0 is better
draw_params = dict(matchColor=(0,255,0), singlePointColor=(255,0,0), matchesMask=matchesMask, flags=0)
flann_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,matches,None,**draw_params)
display_img(flann_matches)

# good = []
# for match1,match2 in matches:
#     if match1.distance < .7*match2.distance:
#         good.append([match1])

# flann_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,good,None,flags=0)
# display_img(flann_matches)