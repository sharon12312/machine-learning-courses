import skimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from sklearn.feature_extraction import image
from skimage import data, io, filters
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import peak_local_max

img_stork = Image.open('../images/bird.jpg')

# plt.figure(figsize=(12, 6))
# plt.imshow(img_stork)
# plt.title('Original Image')
# plt.show()

print(img_stork.format)
print(img_stork.mode)
print(img_stork.size)

# convert to gray scale image
gs_image = img_stork.convert(mode='L')

# plt.figure(figsize=(12, 6))
# plt.imshow(gs_image)
# plt.title('Gray Scale Image')
# plt.show()

gs_image.thumbnail((200, 200))  # 200, 127 (~127: ratio stays the same)
print(gs_image.size)

# plt.figure(figsize=(12, 6))
# plt.imshow(gs_image)
# plt.title('Thumbnail Image')
# plt.show()

img_resize = img_stork.resize((300, 300))  # 300, 300
print(img_resize.size)

# plt.figure(figsize=(12, 6))
# plt.imshow(img_resize)
# plt.title('Resize Image')
# plt.show()

# flip the image: horizontal and vertical
hoz_flip = img_stork.transpose(Image.FLIP_LEFT_RIGHT)
ver_flip = img_stork.transpose(Image.FLIP_TOP_BOTTOM)

# plt.figure(figsize=(12, 6))
# plt.imshow(hoz_flip)
# plt.title('Horizontal Flip')
# plt.figure(figsize=(12, 6))
# plt.imshow(ver_flip)
# plt.title('Vertical Flip')
# plt.show()

# convert to an array format
img_stork_arr = np.array(img_stork.resize((500, 320)))
print(type(img_stork_arr))
print(img_stork_arr.shape)  # (height, width, dimensional[RGB]) => (320, 500, 3)

# plt.figure(figsize=(6, 6))
# plt.imshow(img_stork_arr, cmap='gray')
# plt.title('Numpy Image')
# plt.show()

img_stork_arr_gray = rgb2gray(img_stork_arr)
print(img_stork_arr_gray.shape)  # (height, width, [1]) => (320, 500)

# plt.figure(figsize=(6, 6))
# plt.imshow(img_stork_arr_gray, cmap='gray')  # rgb2gray() function did convert the image to gray
# plt.title('Gray Image')
# plt.show()

# anti_aliasing=True : reduce the disturation that might be resized lower resolution image
stork_resized = resize(img_stork_arr, (224, 224), anti_aliasing=True)
print(stork_resized.shape)

# plt.figure(figsize=(6, 6))
# plt.imshow(stork_resized, cmap='gray')
# plt.title('Resized Image')
# plt.show()

# -----------------------------

patches = image.extract_patches_2d(img_stork_arr, (64, 64))
# (total patches, height, width, [RGB]) : (112309, 64, 64, 3)
# 112309 = (320 - 64 + 1) * (500 - 64 + 1)  image array => (320, 500)
print(patches.shape)

# plt.imshow(patches[70450])
# plt.show()

image = data.coins()  # import an image of coins
print(image)

# plt.figure(figsize=(12, 6))
# io.imshow(image)
# plt.show()

# used to find boundaries of objects in images
edges = filters.sobel(image)
print(edges)

# plt.figure(figsize=(12, 6))
# plt.title('Edge Detection', fontsize=20)
# io.imshow(edges)
# plt.show()

# coordinates = peak_local_max(image, min_distance=20)
# fig, axes = plt.subplots(figsize=(12, 6))
# axes.imshow(image, cmap=plt.cm.gray)
# axes.plot(coordinates[:, 1], coordinates[:, 0], 'r*')
# axes.set_title('Peak local max', fontsize=20)
# plt.show()

# -----------------------------

pisa1 = cv2.imread('../images/leaning_pisa_1.jpg')
print(pisa1.shape)

# plt.figure(figsize=(10, 10))
# plt.imshow(pisa1)
# plt.show()

gray1 = cv2.cvtColor(pisa1, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()  # SIFT = Scale Invariant Feature Transform
keypoinst1 = sift.detect(gray1, None)  # key points = what is interesting in the image
pisa1 = cv2.drawKeypoints(gray1, keypoinst1, outImage=None)
cv2.imwrite('sift_pisa_keypoints.jpg', pisa1)

# sift_pisa_keypoints = io.imread('sift_pisa_keypoints.jpg')
# plt.figure(figsize=(10, 10))
# plt.imshow(sift_pisa_keypoints, cmap='gray')
# plt.show()

pisa_rich1 = cv2.drawKeypoints(gray1, keypoinst1, outImage=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('rich_sift_pisa_keypoints.jpg', pisa_rich1)

# rich_sift_keypoints = io.imread('rich_sift_pisa_keypoints.jpg')
# plt.figure(figsize=(10, 10))
# plt.imshow(rich_sift_keypoints, cmap='gray')
# plt.show()

kp1, des1 = sift.compute(gray1, keypoinst1)  # key points & description

pisa2 = cv2.imread('../images/leaning_pisa_2.jpg')
gray2 = cv2.cvtColor(pisa2, cv2.COLOR_BGR2GRAY)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # brute force matcher
matches = bf.match(des1, des2)  # brute force comparison

# N_MATCHES = 20
# match_img = cv2.drawMatches(gray1, kp1, gray2, kp2, matches[:N_MATCHES], gray2.copy(), flags=0)
# plt.figure(figsize=(20, 10))
# plt.imshow(match_img)
# plt.show()

# ------------------------------

