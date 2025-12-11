from PIL import Image
import os

os.makedirs("test_images", exist_ok=True)
Image.new('RGB', (512, 512), color='red').save('test_images/img1.jpg')
Image.new('RGB', (512, 512), color='blue').save('test_images/img2.jpg')
print("Created dummy images.")
