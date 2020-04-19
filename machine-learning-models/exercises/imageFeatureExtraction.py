import numpy as np
import matplotlib.pyplot as plt
import cv2

img_path = './images/dog.jpg'
image = cv2.imread(img_path)
print(image.shape)

# plt.imshow(image)
# plt.show()

resized_image_feature_vector = cv2.resize(image, (32,32))
print(resized_image_feature_vector.shape)

# plt.imshow(resized_image_feature_vector)
# plt.show()

# convert to 1D array
resized_flattened_image_feature_vector = resized_image_feature_vector.flatten()
print(resized_flattened_image_feature_vector)
print(len(resized_flattened_image_feature_vector))  # height * width * color dimensional (32 * 32 * 3 = 3072)

image_grayscale = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
print(image_grayscale.shape)

# plt.imshow(image_grayscale)
# plt.show()

expanded_image_grayscale = np.expand_dims(image_grayscale, axis=2)
print(expanded_image_grayscale.shape)
