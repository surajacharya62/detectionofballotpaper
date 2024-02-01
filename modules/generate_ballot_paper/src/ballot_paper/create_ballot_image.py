import cv2
import os
import random
from PIL import Image, ImageDraw

def load_symbols(folder_path):
    symbols = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder_path, filename))
            symbols.append(img)
    return symbols 

def create_ballot(rows, columns, candidates, symbols, symbol_size, ballot_size, margins):
    number_of_symbols = len(symbols)

    if candidates <= (rows*columns) and candidates <= number_of_symbols:

        # Margins: top, bottom, left, right
        mt, mb, ml, mr = margins

        random.shuffle(symbols)

        # Calculate cell size
        cell_width = symbol_size[0] * 3  # Twice the symbol width for symbol and space
        cell_height = max(symbol_size[1], ballot_size[1] // rows)  # At least as tall as the symbol

        # Adjusted size for ballot paper
        adjusted_width = ml + mr + cell_width * columns
        adjusted_height = mt + mb + cell_height * rows
        adjusted_size = (adjusted_width, adjusted_height)

        # Initialize a blank image
        ballot_paper = Image.new('RGB', adjusted_size, (255, 255, 255))
        draw = ImageDraw.Draw(ballot_paper)

        # Place symbols in the grid
        for i in range(candidates):
            row = i // columns
            col = i % columns
            if row < rows:
                symbol = symbols[i]
                symbol_pil = Image.fromarray(cv2.cvtColor(symbol, cv2.COLOR_BGR2RGB))
                x_pos = ml + col * cell_width + 30  # Place symbol at the left side of the cell
                # Center the symbol vertically within the cell
                y_pos = mt + row * cell_height + (cell_height - symbol_size[1]) // 2
                ballot_paper.paste(symbol_pil, (x_pos, y_pos))

        # Draw grid lines
        for row in range(1, rows):
            y = mt + row * cell_height 
            draw.line([(ml, y), (adjusted_size[0] - mr, y)], fill=(0, 0, 0), width=20)
        for col in range(1, columns):
            x = ml + col * cell_width
            draw.line([(x, mt), (x, adjusted_size[1] - mb)], fill=(0, 0, 0), width=20)

        # Draw grid outline
        draw.rectangle([ml, mt, adjusted_size[0] - mr, adjusted_size[1] - mb], outline=(0, 0, 0), width=20)

        return ballot_paper   
    else:
       
       print("Number of candidates exceed")

# Example usage
folder_path = '../../../datasets/nepal_communist_party_maobadi_kendra'
symbols = load_symbols(folder_path)
symbol_size = (189, 189)  # Symbol size (width, height)
ballot_paper = create_ballot(10, 6, 10, symbols, symbol_size, ballot_size=(2000, 3000), margins=(500, 500, 200, 200))
if ballot_paper:
    ballot_paper.save('../../generated_ballot_paper/ballot_paper1.jpg')
    # ballot_paper.show()  # Or save using ballot_paper.save('ballot_paper.jpg')
else:
    print("Please provide valid candidates")
