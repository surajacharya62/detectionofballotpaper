from PIL import Image, ImageDraw
import cv2
import os


def add_red_circle_to_object(img_paths):
    # Open the image
    symbols = []
    for image in img_paths:
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)).convert("RGBA")
        width, height = img.size

        # Calculate the position and size of the circle
        circle_x0 = (width - min(width, height)) // 2
        circle_y0 = (height - min(width, height)) // 2
        circle_x1 = circle_x0 + min(width, height)
        circle_y1 = circle_y0 + min(width, height)
        line_width = 2 # Thickness of the circle line

        # Create a mask for the circle
        mask = Image.new('L', (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse([circle_x0, circle_y0, circle_x1, circle_y1], fill=255, outline=255)

        # Create a white background image
        white_bg = Image.new('RGBA', (width, height), (255, 255, 255, 255))

        # Composite the original image with the white background, using the circle mask
        img_with_bg = Image.composite(img, white_bg, mask)

        # Draw a red circle on the composite image
        draw = ImageDraw.Draw(img_with_bg)
        draw.ellipse([circle_x0 + line_width // 2, circle_y0 + line_width // 2, circle_x1 - line_width // 2, circle_y1 - line_width // 2], outline="red", width=line_width)

        # Save the resulting image
        symbols.append(img_with_bg)

    return symbols
 

image_paths = [
    '../../../datasets_symbol_train/nepal_communist_party_maobadi_kendra/train/hammer_scythe_0096.png'
    # 'path_to_your_second_image.png',
    # Add paths to all other images you want to process
]


original_img_path = '../../../datasets_symbol_train/nepal_communist_party_maobadi_samajwadi/train'
    # Replace with your original image path
output_img_path = '../../../datasets_symbol_train/nepal_communist_party_maobadi_samajwadi/results'


def load_image(files):
    symbols = []
    for filename in os.listdir(files):
        # print(filename)
        if filename.endswith(('.jepg','png','.jpg')):
            path = os.path.join(files, filename)
            print(path)
            img = cv2.imread(path)
            symbols.append(img)
    return symbols



images = load_image(original_img_path)
print(len(images))


new_images = add_red_circle_to_object(images)

for i, image in enumerate(new_images,1):
    output_filename = f'woman_man_{i:04}.png'
    output_path = os.path.join(output_img_path, output_filename)
    image.save(output_path)
