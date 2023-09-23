from rembg import remove
from PIL import Image

input_ = 'DSCF3721.jpg'
output_ = 'removed.png'

input = Image.open(input_)
output = remove(input)
output.save(output_)
