import cv2
import os
import random
from PIL import Image, ImageDraw, ImageFont

def load_symbols(folder_path):
    """
    
    """
    
    all_party_symbols = []
    symbols = []
    for folder in os.listdir(folder_path):
        sub_folder = os.path.join(folder_path, folder) 
        # sub_folder = folder_path + folder
        if os.path.isdir(sub_folder):
            
            for filename in os.listdir(sub_folder):
                if filename.endswith(('.jepg','png','.jpg')):
                    img = cv2.imread(os.path.join(sub_folder, filename))
                    symbols.append(img)
            if symbols:
                
                image = random.choice(symbols)
                all_party_symbols.append(image)        
                symbols.clear()
        else:
            print("invalid directory")
    
    return all_party_symbols

def create_ballot(rows, columns, candidates, symbols, symbol_size, top_text, header_text, signature_text, ballot_size, margins=(100, 100, 50, 50),line_height=30,header_box_size=(1000, 100)):
    # Load Nepali font
    print("create_ballot")
    font_path = '../font/Noto_Sans_Devanagari/devanagari.ttf'
    nepali_font = ImageFont.truetype(font_path, 24)

    # Margins: top, bottom, left, right
    mt, mb, ml, mr = margins

    # Calculate cell size
    cell_width = symbol_size[0] * 2  # Twice the symbol width for symbol and space
    cell_height = max(symbol_size[1], ballot_size[1] // rows)  # At least as tall as the symbol

    # Adjusted size for ballot paper
    adjusted_width = ml + mr + cell_width * columns
    adjusted_height = mt + mb + cell_height * rows
    adjusted_size = (adjusted_width, adjusted_height)

    # Initialize a blank image
    ballot_paper = Image.new('RGB', adjusted_size, (255, 255, 255))
    draw = ImageDraw.Draw(ballot_paper)

    # Add top text
    y_text = mt
    for line in top_text:
        draw.text((ml, y_text), line, font=nepali_font, fill=(0, 0, 0))
        y_text += line_height 

    # Draw header box in the middle
    header_box_left = (adjusted_width - header_box_size[0]) // 2
    header_box_top = y_text + 20
    header_box_right = header_box_left + header_box_size[0]
    header_box_bottom = header_box_top + header_box_size[1]
    draw.rectangle([header_box_left, header_box_top, header_box_right, header_box_bottom], outline=(0, 0, 0), width=2)

    # Add header text in the box, centered
    header_text_y = header_box_top + (header_box_size[1] - len(header_text) * line_height) // 2
    for line in header_text:
        text_x = header_box_left + (header_box_size[0] - len(line) * 24) // 2  # Rough width estimation
        draw.text((text_x, header_text_y), line, font=nepali_font, fill=(0, 0, 0))
        header_text_y += line_height


   # Update top margin to include text height
    mt = header_box_bottom + 20

    # Place symbols in the grid
    for i in range(candidates):
        row = i // columns
        col = i % columns
        if row < rows:
            symbol = random.choice(symbols)
            symbol_pil = Image.fromarray(cv2.cvtColor(symbol, cv2.COLOR_BGR2RGB))
            x_pos = ml + col * cell_width  # Place symbol at the left side of the cell
            # Center the symbol vertically within the cell
            y_pos = mt + row * cell_height + (cell_height - symbol_size[1]) // 2
            ballot_paper.paste(symbol_pil, (x_pos, y_pos))

    # Draw grid lines
    for row in range(1, rows):
        y = mt + row * cell_height
        draw.line([(ml, y), (adjusted_size[0] - mr, y)], fill=(0, 0, 0), width=1)
    for col in range(1, columns):
        x = ml + col * cell_width
        draw.line([(x, mt), (x, adjusted_size[1] - mb)], fill=(0, 0, 0), width=1)

    

    # Draw grid outline
    draw.rectangle([ml, mt, adjusted_size[0] - mr, adjusted_size[1] - mb], outline=(0, 0, 0), width=2)
    # Calculate the end of the grid to place the signature line
    grid_end_y = mt + rows * cell_height

    # Ensure there is space for the signature line, adjust if necessary
    if grid_end_y + line_height + 20 > adjusted_height - mb:
        adjusted_height = grid_end_y + line_height + 20 + mb
        ballot_paper = Image.new('RGB', (adjusted_width, adjusted_height), (255, 255, 255))
        draw = ImageDraw.Draw(ballot_paper)
        # Redraw everything as the image size has changed
        # [Redraw the top text, header box, symbols, and grid lines]

    # Add signature line at the bottom, outside the grid
    signature_y = grid_end_y + 20  # Position below the grid
    draw.text((ml, signature_y), signature_text, font=nepali_font, fill=(0, 0, 0))

    return ballot_paper

# Example usage
folder_path = '../../../datasets'
symbols = load_symbols(folder_path)
print(len(symbols))
symbol_size = (189, 189)  # Symbol size (width, height)
top_text = ["नेपाली टेक्स्ट १", "नेपाली टेक्स्ट २", "..."]  # Replace with actual text
header_text = ["हेडर टेक्स्ट १", "हेडर टेक्स्ट २", "..."]
signature_text = "मतदान अधिकृतको  दस्तखत"
ballot_paper = create_ballot(10, 8, 25, symbols, symbol_size, top_text, header_text, signature_text, ballot_size=(4000, 3000), margins=(100, 100, 50, 50))
ballot_paper.show()  # Or save using ballot_paper.save('ballot_paper.jpg')
