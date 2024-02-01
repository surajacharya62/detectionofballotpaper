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
            print("invalid directory")
    
    return all_party_symbols

def draw_top_text(draw, top_text, start_y, ml, line_height, nepali_font, line_spacing=80):
    y_text = start_y
    for line in top_text:
        draw.text((ml, y_text), line, font=nepali_font, fill=(0, 0, 0))
        y_text += line_height + line_spacing
    return y_text

def draw_header_box(draw, header_text, start_y, width, ml, mr, line_height, nepali_font, line_spacing=25, line_width=20, padding=30, extra_height=20):
    # Adjust the estimation factor for text width
    estimation_factor = 27  # Increase this factor based on your font
    line_spacing = 35
    line_height = 45
    
    max_text_width = 0
    for line in header_text:
        estimated_text_width = len(line) * estimation_factor
        max_text_width = max(max_text_width, estimated_text_width)

    # Calculate header box width based on the maximum text width with padding
    header_box_width = max_text_width + 2 * padding
    header_box_width = min(header_box_width, width - ml - mr)

    # Center the header box within the width of the image
    header_box_left = ml + (width - ml - mr - header_box_width) // 2
    header_box_top = start_y
    header_box_height = len(header_text) * (line_height + line_spacing) + extra_height
    header_box_bottom = header_box_top + header_box_height

    # Draw the header box
    draw.rectangle([header_box_left, header_box_top, header_box_left + header_box_width, header_box_bottom], outline=(0, 0, 0), width=line_width)

    # Center the text within the header box
    text_start_y = header_box_top + (extra_height // 2)
    for line in header_text:
        estimated_text_width = len(line) * estimation_factor
        text_x = header_box_left + (header_box_width - estimated_text_width) // 2
        draw.text((text_x, text_start_y), line, font=nepali_font, fill=(0, 0, 0))
        text_start_y += line_height + line_spacing

    return header_box_bottom




def draw_symbols(ballot_paper, symbols, start_y, columns, rows, ml, cell_width, cell_height, symbol_size, num_candidates, left_gap=30,right_gap=1):
    # Shuffle the symbols list for random placement
    random_symbols = random.sample(symbols, len(symbols))   
    
    
    if num_candidates <= len(symbols):
        # Place the symbols in the grid
        for i in range(min(num_candidates, rows * columns)):
            row = i // columns
            col = i % columns 

         

            symbol = random_symbols[i]          
            symbol_pil = Image.fromarray(cv2.cvtColor(symbol, cv2.COLOR_BGR2RGB))


            # Calculate position with left and right gaps
            
            x_pos = ml + col * cell_width + left_gap 
            y_pos = start_y + row * cell_height + (cell_height - symbol_size[1]) // 2           
            # y_pos = start_y + row * cell_height + 30    

           
            
                      
            symbol_pil = symbol_pil.resize((symbol_size[0], symbol_size[1]))    
            # symbol_pil_resized_height = min(symbol_size[1], vertical_space)
            # print(symbol_pil_resized_height)
            # y_pos = start_y + row * cell_height + 30 + (vertical_space - symbol_pil_resized_height) // 2


            ballot_paper.paste(symbol_pil, (x_pos, y_pos))
    else:
        print("Please ensure number of candidates should not exceed.")

def draw_grid(draw, start_y, rows, columns, ml, mr, cell_height, cell_width, adjusted_height, adjusted_width,candidates, line_width=1):
    # Draw horizontal lines for each row
    rows = math.ceil(candidates / columns)
    last_row_symbols = candidates % columns if candidates % columns != 0 else columns
    adjusted_cell_height = cell_height 
    actual_rows = math.ceil(candidates / columns)  # Actual number of rows needed   

    for row in range(actual_rows ):  # +1 to draw the line below the last row
        y = start_y + row * adjusted_cell_height
        
        # For the last row or second last row with fewer symbols, draw up to the last symbol
        # if row == actual_rows and last_row_symbols < columns:
        #     print(str(row)+ '==' + str(actual_rows))
        #     x_end = ml + last_row_symbols * cell_width
           
        # else:
            # Full width for all other rows
        x_end = ml + columns * cell_width
           
        draw.line([(ml, y), (x_end, y)], fill=(0, 0, 0), width=line_width)
   
    # Draw vertical lines for each column
    for col in range(columns + 1):
        x = ml + col * cell_width
        # For columns beyond the last symbol in the last row, adjust the line end
        y_end = start_y + (actual_rows - 1 if col >= last_row_symbols else actual_rows) * adjusted_cell_height
        draw.line([(x, start_y), (x, y_end)], fill=(0, 0, 0), width=line_width)

    #Ensure the bottom right corner of the last symbol's cell is closed if needed
    if last_row_symbols < columns:
        x = ml + last_row_symbols * cell_width
        y = start_y + (actual_rows - 1) * adjusted_cell_height
        draw.line([(x, y), (x, y + adjusted_cell_height)], fill=(0, 0, 0), width=line_width)  # Vertical
        draw.line([(x, y + adjusted_cell_height), (ml, y + adjusted_cell_height)], fill=(0, 0, 0), width=line_width)  # Horizontal
        
   

    # for row in range(actual_rows + 1):  # +1 to draw the line below the last row
    #     y = start_y + row * cell_height
    #     if row < actual_rows - 1 or (row == actual_rows - 1 and last_row_symbols == columns):
    #         # Full width for all but the last row
    #         x_end = ml + columns * cell_width
    #     else:
    #         # For the last row or second last row with fewer symbols, draw up to the last symbol
    #         x_end = ml + last_row_symbols * cell_width
    #     draw.line([(ml, y), (x_end, y)], fill=(0, 0, 0), width=line_width)

    # # Draw vertical lines for each column
    # for col in range(columns + 1):
    #     x = ml + col * cell_width
    #     # For columns beyond the last symbol in the last row, adjust the line end
    #     y_end = start_y + (actual_rows if col < last_row_symbols else actual_rows - 1) * cell_height
    #     draw.line([(x, start_y), (x, y_end)], fill=(0, 0, 0), width=line_width)

   

  

def create_ballot(rows, columns, candidates, symbols, symbol_size, top_text, header_text, signature_text, ballot_size, margins, line_height=30, header_box_size=(1200, 200)):
    # Load Nepali font
    # font_path = '../font/Noto_Sans_Devanagari/extrabold.ttf'

    font_path = '../font/Times New Roman/times new roman bold.ttf'
    nepali_font = ImageFont.truetype(font_path, 60)
    
    # columns = min(columns, candidates)
    # rows = math.ceil(candidates / columns)

    # Margins: top, bottom, left, right
    mt, mb, ml, mr = margins 

    # Calculate cell size
    cell_width = symbol_size[0] * 2 # Twice the symbol width for symbol and space
   
    cell_height = max(symbol_size[1], ballot_size[1] // rows)  # At least as tall as the symbol
    
    # Adjusted size for ballot paper 
    adjusted_width = ml + mr + cell_width * columns
    adjusted_height = mt + mb + cell_height * rows
    adjusted_size = (adjusted_width, adjusted_height)
    
    
    # Initialize a blank image
    ballot_paper = Image.new('RGB', adjusted_size, (255, 255, 255))
    draw = ImageDraw.Draw(ballot_paper)

    # Draw top text
    y_text_end = draw_top_text(draw, top_text, mt, ml, line_height, nepali_font)

    # Draw header box and text
    header_box_bottom = draw_header_box(draw, header_text, y_text_end, adjusted_width, ml, mr, line_height, nepali_font)

    # Print out header box bottom for debugging
    

    # Update top margin to include text height and the desired gap (e.g., 250 pixels) after the header box
    mt_updated = header_box_bottom + 100

  

    # Ensure mt_updated is within the image bounds
    if mt_updated < adjusted_height - mb:
        # Place symbols in the grid, starting from the updated margin
        actual_candidates = min(candidates, rows * columns)  
        draw_symbols(ballot_paper, symbols[:actual_candidates], mt_updated, columns, rows, ml, cell_width, cell_height, symbol_size, candidates)

        # Draw grid lines within the symbol area, starting from the updated margin
        grid_end_y = mt_updated + rows * cell_height
        draw_grid(draw, mt_updated, rows, columns, ml, mr, cell_height, cell_width, grid_end_y, adjusted_width,candidates, line_width=20)

        average_char_width = 20  # This is an estimated average width for each character

        # Calculate the position for the signature line at the bottom
        signature_y = grid_end_y + 250 # Position below the grid
        if signature_y + line_height <= adjusted_height - mb:
            signature_text_full = signature_text
            estimated_text_width = len(signature_text_full) * average_char_width
            text_x = (adjusted_width - estimated_text_width) // 2
            draw.text((text_x, signature_y), signature_text_full, font=nepali_font, fill=(0, 0, 0))

        else:
            # Resize the image to fit the signature line if necessary
            adjusted_height = signature_y + line_height + 50 + mb
            ballot_paper = Image.new('RGB', (adjusted_width, adjusted_height), (255, 255, 255))
            draw = ImageDraw.Draw(ballot_paper) 

            # Redraw all elements on the resized ballot paper
            y_text_end = draw_top_text(draw, top_text, mt, ml, line_height, nepali_font)
            header_box_bottom = draw_header_box(draw, header_text, y_text_end, adjusted_width, ml, mr, line_height, nepali_font, header_box_size)
            mt_updated = header_box_bottom + 200
            draw_symbols(ballot_paper, symbols[:actual_candidates], mt_updated, columns, rows, ml, cell_width, cell_height, symbol_size, candidates)
            draw_grid(draw, mt_updated, rows, columns, ml, mr, cell_height, cell_width, grid_end_y, adjusted_width,candidates,line_width=20)

            # Centered signature text on the resized ballot paper
            signature_text_full = signature_text
            estimated_text_width = len(signature_text_full) * average_char_width
            text_x = (adjusted_width - estimated_text_width) // 2
            draw.text((text_x, signature_y), signature_text_full, font=nepali_font, fill=(0, 0, 0))

   
        return ballot_paper

    else:
        print("Insufficient space for the grid layout")




def place_stamp(ballot, stamp, symbol_size, cell_width, cell_height, rows, columns, margins, is_valid):
    """
    
    """
    mt, mb, ml, mr = margins
    stamp_width, stamp_height = stamp.size

    # Randomly choose a cell to place stamp in grid cell
    row = random.randint(0, rows - 1)
    col = random.randint(0, columns - 1)
    

    if is_valid:
        # Placing the stamp entirely within the cell
        x = ml + col * cell_width + random.randint(0, cell_width - stamp_width)
        y = mt + row * cell_height + random.randint(0, cell_height - stamp_height)
        
    else:
        # Placing stamp in a such way that it touches grid lines or overlaps in two symbols.       
        x = ml + col * cell_width - random.randint(0, stamp_width // 2)
        y = mt + row * cell_height - random.randint(0, stamp_height // 2)

    ballot.paste(stamp, (x, y), stamp)


def create_stamped_ballots(folder_path, stamp_path, rows, columns, candidates,
                            symbol_size,top_text, header_text, signature_text, ballot_size, margins):
    """
    
    """
    symbols = load_symbols(folder_path)
    stamp = Image.open(stamp_path).convert("RGBA")

 
    # calls create_ballot function for creating ballot paper with the symbols included.
    ballot_paper = create_ballot(rows, columns, candidates, symbols, symbol_size, top_text, header_text, signature_text, 
                                 ballot_size, margins,
                                 line_height=30,header_box_size=(1000, 100))

    cell_width = symbol_size[0] * 3 
    # print(symbol_size[2])

    # alls place_stamp function for placing the stamp for valid vote
    valid_ballot = ballot_paper.copy()
    place_stamp(valid_ballot, stamp, symbol_size, cell_width, symbol_size[1], rows, columns, margins, is_valid=True)

    # Place stamp for invalid vote
    invalid_ballot = ballot_paper.copy()
    place_stamp(invalid_ballot, stamp, symbol_size, cell_width, symbol_size[1], rows, columns, margins, is_valid=False)

    return valid_ballot, invalid_ballot


# # Example usage
# folder_path = '../../../datasets'
# symbols = load_symbols(folder_path)
# symbol_size = (189, 189)  # Symbol size (width, height)
# top_text = ["मनुष्यों : प्रदेश न १ ", "जिल्ला :", "प्रतिनिधि सभा निर्वाचन ","प्रतिनिधि सभा निर्वाचन ", "जिल्ला :", "प्रतिनिधि सभा निर्वाचन ","प्रतिनिधि सभा निर्वाचन "]  # Replace with actual text
# header_text = ["प्रदेश सभा सदस्य निर्वाचन, २०७९", "समानुपातिक निर्वाचन प्रणालीको मतपत्र", "[एउटा कोठाभित्र मात्र मतसङ्केत गर्नुहोस् े]"]
# signature_text = "मतदान अधिकृतको  दस्तखत :................."
# ballot_paper = create_ballot(10, 5, 10, symbols, symbol_size, top_text, header_text,
#                               signature_text, ballot_size=(5500, 3000), margins=(200, 200, 100, 100))
# ballot_paper.show()  # Or save using ballot_paper.save('ballot_paper.jpg')


folder_path = '../../../datasets'
stamp_path = 'newstamp.png'  # Path to your stamp image
symbol_size = (189, 189)  # Symbol size (width, height)
# top_text = ["प्रदेश : बागमती ", "जिल्ला : ....", "प्रतिनिधी  सभा निर्वाचन  क्षेत्र नं. : ....","प्रदेश सभा निर्वाचन  क्षेत्र न. : ...", "गाउँपालिका/नगरपालिका : ....",
#              "वडा न. : .... ","मतदाता क्रमसंख्या : ..."]  # Replace with actual text
# header_text = ["प्रदेश सभा सदस्य निर्वाचन, २०७९", "समानुपातिक निर्वाचन प्रणालीको मतपत्र", "[एउटा कोठाभित्र मात्र मतसङ्केत गर्नुहोस् े]"]
# signature_text = "मतदान अधिकृतको  दस्तखत : ................."

top_text = ["Province : Bagmati", "District: Kathmandu", "House of Representatives Constituency No. : ...", 
            "Provincial Assembly Constituency No. : ...","Rural/Municipality : ...","Ward No. : ...","Voters No. : ..."]  # Replace with actual text
header_text = ["State Assembly Member Election", " Ballots of Proportional Electoral System", " ..."]
signature_text = "Voting Athorized Signature : ......"
valid_ballot, invalid_ballot = create_stamped_ballots(folder_path, stamp_path,
                                                       7,5 , 23, symbol_size,
                                                      top_text, header_text, signature_text, 
                                                      ballot_size=(595, 842),
                                                        margins=(300, 300, 200, 200))

# Save the ballot papers
valid_ballot.save('../../generated_ballot_paper/valid_ballot_paper.jpg')
invalid_ballot.save('../../generated_ballot_paper/invalid_ballot_paper.jpg')

