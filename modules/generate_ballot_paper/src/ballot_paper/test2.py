import cv2
import os
import random
from PIL import Image, ImageDraw, ImageFont
import math


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
            print("invalid directory path")
    
    return all_party_symbols

# def create_ballot(columns, candidates, symbols, symbol_size, margins):
#     number_of_symbols = len(symbols)

#     if candidates > number_of_symbols:
#         print("Number of candidates exceed the number of available symbols")
#         return None

#     # Margins: top, bottom, left, right
#     mt, mb, ml, mr = margins

#     # Dynamically calculate the number of rows needed based on the number of candidates and columns
#     rows = math.ceil(candidates / columns)

#     # Adjust cell size to match the symbol size
#     cell_width = symbol_size[0] * 2  # Adjust as needed for width
#     cell_height = symbol_size[1]     # Use symbol height for cell height

#     # Adjusted size for ballot paper
#     adjusted_width = ml + mr + cell_width * columns
#     adjusted_height = mt + mb + cell_height * rows  # Height based on the number of rows needed

#     # Initialize a blank image
#     ballot_paper = Image.new('RGB', (adjusted_width, adjusted_height), (255, 255, 255))
#     draw = ImageDraw.Draw(ballot_paper)

#     # Place symbols in the grid
#     for i in range(candidates):
#         row = i // columns
#         col = i % columns
#         symbol = symbols[i % len(symbols)]
#         symbol_pil = Image.fromarray(cv2.cvtColor(symbol, cv2.COLOR_BGR2RGB))
#         x_pos = ml + col * cell_width + (cell_width - symbol_size[0]) // 2  # Center horizontally
#         y_pos = mt + row * cell_height  # Place at top of cell
#         ballot_paper.paste(symbol_pil, (x_pos, y_pos))

#     # Draw grid lines
#     for row in range(rows):
#         y = mt + row * cell_height
#         draw.line([(ml, y), (adjusted_width - mr, y)], fill=(0, 0, 0), width=2)
#     for col in range(columns):
#         x = ml + col * cell_width
#         draw.line([(x, mt), (x, adjusted_height - mb)], fill=(0, 0, 0), width=2)

#     # Draw grid outline
#     draw.rectangle([ml, mt, adjusted_width - mr, adjusted_height - mb], outline=(0, 0, 0), width=2)

#     return ballot_paper


def create_ballot(columns, candidates, symbols, symbol_size, margins, cell_padding):
    number_of_symbols = len(symbols)

    if candidates > number_of_symbols:
        print("Number of candidates exceed the number of available symbols")
        return None

    # Margins: top, bottom, left, right
    mt, mb, ml, mr = margins

    # Calculate rows and columns
    rows = math.ceil(candidates / columns)
    print(rows)
    last_row_symbols = candidates % columns if candidates % columns != 0 else columns
    print(last_row_symbols)

    pt, pb, pl = cell_padding

    # Adjust cell size
    cell_width = (symbol_size[0] + pl) * 2
    cell_height = symbol_size[1] + pt + pb

    # Adjusted size for ballot paper
    adjusted_width = ml + mr + cell_width * columns
    adjusted_height = mt + mb + cell_height * rows

    # Initialize a blank image
    ballot_paper = Image.new('RGB', (adjusted_width, adjusted_height), (255, 255, 255))
    draw = ImageDraw.Draw(ballot_paper)

    # Place symbols in the grid
    for i in range(candidates):
        row = i // columns
        col = i % columns
        symbol = symbols[i % len(symbols)]
        symbol_pil = Image.fromarray(cv2.cvtColor(symbol, cv2.COLOR_BGR2RGB))
        x_pos = ml + col * cell_width + pl
        y_pos = mt + row *  cell_height + pt
        ballot_paper.paste(symbol_pil, (x_pos, y_pos))

    # Draw grid lines 
    for row in range(rows + 1):
        y = mt + row * cell_height
        x_end = ml + (last_row_symbols if row == rows else columns) * cell_width
        draw.line([(ml, y), (x_end, y)], fill=(0, 0, 0), width=2)

    # Draw vertical lines for each column
    for col in range(columns + 1):  # +1 to draw the closing line of the grid
        x = ml + col * cell_width
        y_end = mt + cell_height * (rows if col < last_row_symbols + 1 or rows == 1 else rows - 1)
        draw.line([(x, mt), (x, y_end)], fill=(0, 0, 0), width=2)
  

    return ballot_paper


# Example usage
folder_path = '../../../datasets'
symbols = load_symbols(folder_path)
symbol_size = (189, 189)  # Symbol size (width, height)
ballot_paper = create_ballot(4, 15, symbols, symbol_size, margins=(300, 300, 200, 200), cell_padding=(30,30,30))
if ballot_paper:
    ballot_paper.save('../../generated_ballot_paper/ballot_paper1.jpg')
else:
    print("Please provide valid candidates")
