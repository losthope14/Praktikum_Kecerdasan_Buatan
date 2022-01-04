from PIL import Image

image = Image.open('DSC_0044.JPG')

print(image.format)
print(image.mode)
print(image.size)

image.show()
