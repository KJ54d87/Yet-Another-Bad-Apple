import sys
from PIL import Image

SIZE = int(sys.argv[1])
IMAGE_NUMBER = sys.argv[2]

with Image.open(f"frames/out{IMAGE_NUMBER}.png") as img:
    img2 = img.resize((SIZE, SIZE))
    img2 = img2.rotate(-90)
    img2 = img2.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    img2.save(f"frames_small/out{IMAGE_NUMBER}.png")