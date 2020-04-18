import matplotlib.pyplot as plt
import pytesseract
from PIL import Image, ImageFilter

quote_img = Image.open('../images/motivational_quote.jpg')
# print(quote_img.format)
# print(quote_img.mode)
# print(quote_img.size)

# plt.figure(figsize=(12, 6))
# plt.title('Original Image')
# plt.imshow(quote_img)
# plt.show()

# pull the text from the image
text = pytesseract.image_to_string(quote_img, lang='eng')
print(text)

play_img = Image.open('../images/dandruce.png').convert('RGB')
text = pytesseract.image_to_string(play_img, lang='eng')
print(text)

# blur the image
img_blur = play_img.filter(ImageFilter.GaussianBlur(1))
text = pytesseract.image_to_string(img_blur, lang='eng')
print(text)

img_blur_more = play_img.filter(ImageFilter.GaussianBlur(1.75))
text = pytesseract.image_to_string(img_blur_more, lang='eng')
print(text)

# cannot convert image to string on a flipped image
img_flip = quote_img.transpose(Image.FLIP_TOP_BOTTOM)
# plt.figure(figsize=(12, 6))
# plt.title('Flip Image')
# plt.imshow(img_flip)
# plt.show()
text_flip = pytesseract.image_to_string(img_flip, lang='eng')
print(text_flip)