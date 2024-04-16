from rembg import remove
from PIL import Image
import io

# Load your image
input_path = '../../../datasets_stamp/valid_stamp/stamp_0003.jpg'
output_path = 'output_image.png'

with open(input_path, 'rb') as i:
    input_image = i.read()

# Remove background
output_image = remove(input_image)

# Save or display the output
img = Image.open(io.BytesIO(output_image))
img.save(output_path)
img.show()