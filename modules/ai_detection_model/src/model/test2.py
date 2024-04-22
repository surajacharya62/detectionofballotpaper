import matplotlib.pyplot as plt
import matplotlib.patches as patches

# def reconstruct_grid_cells( margins, ballot_size, symbol_size, rows, columns):
#     mt, mb, ml, mr = margins
#     ballot_width, ballot_height = ballot_size
#     symbol_width, symbol_height = symbol_size

#     # Calculate cell size
#     cell_width = symbol_width * 2
#     cell_height = max(symbol_height, ballot_height // rows)

#     # Initialize grid cells list
#     grid_cells = []

#     # Calculate the starting y-coordinate of the grid
#     header_box_bottom = mt  # Assuming the top margin includes the header
#     grid_start_y = header_box_bottom + 100  # Additional offset for the header box

#     # Calculate the starting x-coordinate of the grid
#     grid_start_x = ml

#     # Generate grid cells based on calculated dimensions
#     for row_idx in range(rows):
#         for col_idx in range(columns):
#             x1 = grid_start_x + col_idx * cell_width
#             y1 = grid_start_y + row_idx * cell_height
#             x2 = x1 + cell_width
#             y2 = y1 + cell_height
#             grid_cells.append((x1, y1, x2, y2))

#     return grid_cells   

def reconstruct_grid_cells( margins, ballot_size, symbol_size, rows, columns):
    mt, mb, ml, mr = margins
    ballot_width, ballot_height = ballot_size
    symbol_width, symbol_height = symbol_size

    # Calculate cell size based on the symbol size being half the width of the cell
    cell_width = symbol_width * 2  # Ensure each cell is double the width of the symbol
    cell_height = symbol_height    # Height of the cell matches the height of the symbol

    # Initialize grid cells list
    grid_cells = []

    # Calculate the starting y-coordinate of the grid
    header_box_bottom = mt
    grid_start_y = header_box_bottom  # Additional offset for the header box

    # Calculate the starting x-coordinate of the grid
    grid_start_x = ml

    # Generate grid cells based on calculated dimensions
    for row_idx in range(rows):
        for col_idx in range(columns):
            x1 = grid_start_x + col_idx * cell_width
            y1 = grid_start_y + row_idx * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            grid_cells.append((x1, y1, x2, y2))

    return grid_cells


def visualize_grid_and_stamps(grid_cells, stamp_box, img_size=(2668, 3413)):
    """
    Visualizes grid cells and a single stamp placement.

    :param grid_cells: List of tuples, each representing a cell (x1, y1, x2, y2).
    :param stamp_box: Tuple representing the stamp's bounding box (x1, y1, x2, y2).
    :param img_size: Size of the background image (width, height) for scaling the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 20))  # Adjust the size as needed to fit your display

    # Create a rectangle for each grid cell
    for cell in grid_cells:
        cell_rect = patches.Rectangle((cell[0], cell[1]), cell[2] - cell[0], cell[3] - cell[1],
                                      linewidth=1, edgecolor='red', facecolor='none', linestyle='-')
        ax.add_patch(cell_rect)

    # Add the stamp box
    stamp_rect = patches.Rectangle((stamp_box[0], stamp_box[1]), stamp_box[2] - stamp_box[0], stamp_box[3] - stamp_box[1],
                                   linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
    ax.add_patch(stamp_rect)

    # Set the limits of the plot and invert the y-axis (to match image coordinates)
    ax.set_xlim(0, img_size[0])
    ax.set_ylim(img_size[1], 0)
    ax.set_aspect('equal')
    plt.show()

# Define the grid cells and the stamp box
grid_cells = [(200, 1660, 578, 1849), (578, 1660, 956, 1849), (956, 1660, 1334, 1849), (1334, 1660, 1712, 1849), (1712, 1660, 2090, 1849), 
              (2090, 1660, 2468, 1849), (200, 1849, 578, 2038), (578, 1849, 956, 2038), (956, 1849, 1334, 2038), (1334, 1849, 1712, 2038), 
              (1712, 1849, 2090, 2038), (2090, 1849, 2468, 2038), (200, 2038, 578, 2227), (578, 2038, 956, 2227), (956, 2038, 1334, 2227), (1334, 2038, 1712, 2227), (1712, 2038, 2090, 2227), (2090, 2038, 2468, 2227), (200, 2227, 578, 2416), (578, 2227, 956, 2416), (956, 2227, 1334, 2416), (1334, 2227, 1712, 2416), (1712, 2227, 2090, 2416), (2090, 2227, 2468, 2416), (200, 2416, 578, 2605), (578, 2416, 956, 2605), (956, 2416, 1334, 2605), (1334, 2416, 1712, 2605), (1712, 2416, 2090, 2605), (2090, 2416, 2468, 2605), (200, 2605, 578, 2794), (578, 2605, 956, 2794), (956, 2605, 1334, 2794), (1334, 2605, 1712, 2794), (1712, 2605, 2090, 2794), (2090, 2605, 2468, 2794), (200, 2794, 578, 2983),
               (578, 2794, 956, 2983), (956, 2794, 1334, 2983), (1334, 2794, 1712, 2983), (1712, 2794, 2090, 2983), (2090, 2794, 2468, 2983)]


# image_0005.jpg # Stamp bounding box [x1, y1, x2, y2]
stamp_box = [1151.8798, 2711.558 , 1275.8987, 2863.8665]

margins = (1560, 300, 200, 200)  # top, bottom, left, right margins
ballot_size = (2668, 3413)  # width, height of the ballot paper
symbol_size = (189, 189)  # width, height of the symbols
rows = 7  # Number of symbol rows
columns = 6  # Number of symbol columns  
candidates = 42
grid_cells = reconstruct_grid_cells(margins, ballot_size, symbol_size, rows, columns) 


# Call the visualization function
visualize_grid_and_stamps(grid_cells, stamp_box)
