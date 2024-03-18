import matplotlib.pyplot as plt

from PIL import Image

path_to_annotation = "./annotations/trimaps/"
path_to_image = "./images/"

annotation = Image.open(path_to_annotation + "Abyssinian_1.png")
plt.subplot(1,2,1)
plt.title("annotation")
plt.imshow(annotation)

image = Image.open(path_to_image + "Abyssinian_1.jpg")
plt.subplot(1,2,2)
plt.title("image")
plt.imshow(image)

plt.show()

