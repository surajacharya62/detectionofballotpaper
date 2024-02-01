import cv2
import os
import random
from PIL import Image, ImageDraw


def load_symbols(folder_path):
    """
    
    """
    # symbols = []
    # for filename in os.listdir(folder_path):
    #     if filename.endswith(('.png', '.jpg', '.jpeg')):
    #         img = cv2.imread(os.path.join(folder_path, filename))
    #         symbols.append(img) 
    # return symbols 

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


def create_ballot(rows, columns, candidates, symbols, symbol_size, ballot_size, margins):
    """
    """

    number_of_symbols = len(symbols)

    if candidates <= (rows*columns) and candidates <= number_of_symbols:

        # Margins: top, bottom, left, right
        mt, mb, ml, mr = margins

        # shuffle the symbols to select it randomly
        random.shuffle(symbols)

        # Calculate cell size
        cell_width = symbol_size[0] * 2  # Twice the symbol width for symbol and space
        cell_height = max(symbol_size[1], ballot_size[1] // rows)  # At least as tall as the symbol
        
        
        # Adjusted size for ballot paper
        adjusted_width = ml + mr + cell_width * columns
        adjusted_height = mt + mb + cell_height * rows
        adjusted_size = (adjusted_width, adjusted_height)

        # create a blank image
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
        print("Number of candidates exceeded.")


def place_stamp(ballot, stamp, symbol_size, cell_width, cell_height, rows, columns, margins, is_valid):
    """
    
    """
    mt, mb, ml, mr = margins
    stamp_width, stamp_height = stamp.size

    # Randomly choose a cell to place stamp in grid cell
    row = random.randint(0, rows - 1)
    col = random.randint(0, columns - 1)
    print(row,col)

    if is_valid:
        # Placing the stamp entirely within the cell
        x = ml + col * cell_width + random.randint(0, cell_width - stamp_width)
        y = mt + row * cell_height + random.randint(0, cell_height - stamp_height)
        print(x,y)
    else:
        # Placing stamp in a such way that it touches grid lines or overlaps in two symbols.       
        x = ml + col * cell_width - random.randint(0, stamp_width // 2)
        y = mt + row * cell_height - random.randint(0, stamp_height // 2)

    ballot.paste(stamp, (x, y), stamp)


def create_stamped_ballots(folder_path, stamp_path, rows, columns, candidates,
                            symbol_size, ballot_size, margins):
    """
    
    """
    symbols = load_symbols(folder_path)
    stamp = Image.open(stamp_path).convert("RGBA")

    # calls create_ballot function for creating ballot paper with the symbols included.
    ballot_paper = create_ballot(rows, columns, candidates, symbols, symbol_size, ballot_size, margins)
    cell_width = symbol_size[0] * 3
    # print(symbol_size[2])

    # alls place_stamp function for placing the stamp for valid vote
    valid_ballot = ballot_paper.copy()
    place_stamp(valid_ballot, stamp, symbol_size, cell_width, symbol_size[1], rows, columns, margins, is_valid=True)

    # Place stamp for invalid vote
    invalid_ballot = ballot_paper.copy()
    place_stamp(invalid_ballot, stamp, symbol_size, cell_width, symbol_size[1], rows, columns, margins, is_valid=False)

    return valid_ballot, invalid_ballot


folder_path = '../../../datasets'
stamp_path = 'newstamp.png'  # Path to your stamp image
symbol_size = (189, 189)  # Symbol size (width, height)
valid_ballot, invalid_ballot = create_stamped_ballots(folder_path, stamp_path, 
                                                      10, 6, 29, symbol_size,
                                                        (2000, 3000),
                                                          (200, 200, 200, 200))

# Save the ballot papers
valid_ballot.save('../../generated_ballot_paper/valid_ballot_paper.jpg')
invalid_ballot.save('../../generated_ballot_paper/invalid_ballot_paper.jpg')
