from PIL import Image
import os
from collections import Counter
import cv2
import numpy as np
import json

# load overall map from path
def load_img(path):
    if not os.path.exists(path):
        # try:
        #     # If the file does not exist, create a blank image
        #     img = Image.new('RGBA', (1, 1), (255, 255, 255, 0))
        #     img.save(path)
        #     return Image.open(path)
        # except:
        raise FileNotFoundError(f"The file at {path} does not exist.")
    return Image.open(path)

def save_pos(top_left_coords, output_map_path):
    """Saves the top-left coordinates of the stitched image to a json file with filename as key."""

    # Extract the filename from the output map path
    filename = os.path.basename(output_map_path)

    # Define the JSON file path
    json_file_path = os.path.join(os.path.dirname(output_map_path), "positions.json")

    # Load existing data if the JSON file exists
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
    else:
        data = {}

    # Update the data with the new top-left coordinates
    data[filename] = top_left_coords

    # Save the updated data back to the JSON file
    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    return True

def save_img(img, imgdir, img_name = "output.png"):
    maps_dir = os.path.join(os.getcwd(), 'maps')
    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)

    # if img is not Image.Image, switch cv2 BGR2RGB for saving
    if isinstance(img, np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[-1] == 3 else cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    # Overwrite the file if img_name already exists
    if os.path.exists(os.path.join(imgdir, img_name)):
        # print(f"Overwriting existing file: {img_name}")
        img.save(os.path.join(imgdir, img_name)) if isinstance(img, Image.Image) else cv2.imwrite(os.path.join(imgdir, img_name), img)
    else:
        if "Map" in img_name:
            # Check for existing map files in the directory
            existing_maps = [f for f in os.listdir(maps_dir) if f.startswith("Map_") and f.endswith(".png")]
            if existing_maps:
                # Extract the highest map number
                max_map_num = max(int(f.split('_')[1].split('.')[0]) for f in existing_maps if f.split('_')[1].split('.')[0].isdigit())
                img_name = f"Map_{max_map_num + 1}.png"
            else:
                # Start with Map_1 if no maps exist
                img_name = "Map_1.png"
        # print(f"Saving new file: {img_name}")            
        img.save(os.path.join(imgdir, img_name)) if type(img) == Image.Image else cv2.imwrite(os.path.join(maps_dir, img_name), img)
    return os.path.join(imgdir, img_name)

# crop image to game area
def crop_to_game_area(image, game_size = (966, 970)):
    colormap = {"dialogue" :(248, 248, 248, 255), "outside_black": (0, 0, 0, 255), "outside_white": (243, 243, 243, 255)}
    # crop edges if colour is outside_black or outside_white
    l, r, t, b = 0, 0, 0, 0
    width, height = image.size
    thresh = 150 #pixels to exceed to be counted as outside
    while sum([image.getpixel((x, t)) == colormap["outside_black"] or image.getpixel((x, t)) == colormap["outside_white"] for x in range(width)]) >thresh:
        t += 1
    while sum([image.getpixel((x, height - b - 1)) == colormap["outside_black"] or image.getpixel((x, height - b - 1)) == colormap["outside_white"] or image.getpixel((x, height - b - 1)) in [(19,19,19,255),(18,18,18,255),(17,17,17,255)] for x in range(width)])>thresh or sum([image.getpixel((x, height - b - 1)) == (117, 117, 117, 255) for x in range(width)])>993:
        b += 1
    while sum([image.getpixel((l, y)) == colormap["outside_black"] or image.getpixel((l, y)) == colormap["outside_white"] for y in range(height)])>thresh:
        l += 1
    while sum([image.getpixel((width - r - 1, y)) == colormap["outside_black"] or image.getpixel((width - r - 1, y)) == colormap["outside_white"] for y in range(height)])>thresh:
        r += 1
    # print(Counter([image.getpixel((x, height - b - 2)) for x in range(width)]))
    # print(sum([image.getpixel((x, height - b - 1)) == colormap["outside_black"] or image.getpixel((x, height - b - 1)) == colormap["outside_white"] or image.getpixel((x, height - b - 1)) == (23,23,23,255) for x in range(width)]))
    # print(l, r, t, b)

    #crop image by l r t b
    image = image.crop((l, t, width - r, height - b))
    # print(image.size, game_size)
    # print("Check image_size is correct: ", image.size == game_size)
    return image

# check image for dialogue (if dialogue, ignore)
def contains_dialogue(image):
    colormap = {"dialogue" :(248, 248, 248, 255), "outside_black": (0, 0, 0, 255), "outside_white": (243, 243, 243, 255)}
    width, height = image.size
    bottom_side = [image.getpixel((x, height - 1)) for x in range(width)]  # Bottom side
    if any([pixel == colormap["dialogue"] for pixel in bottom_side]):
        ### dialogue present, return True

        # b = 0
        # while any([image.getpixel((x, height - b - 1)) == colormap["dialogue"] for x in range(width)]):
        #     b += 1
        # print(height, b, b/height)
        # image = image.crop((0, 0, width, height - b))
        return (image, True)
    else:
        #no dialogue present
        return (image, False)
     #check image for dialogue at bottom

    # Apply Canny Edge Detection
    big_map_edges = cv2.Canny(big_map_gray, 50, 200)
    patch_edges = cv2.Canny(patch_gray, 50, 200)
    print(big_map_edges.shape, patch_edges.shape)
    # # Resize patch_gray to match cropped image size for comparison
    resized_patch_gray = cv2.resize(patch_edges, (big_map_edges.shape[1], int(big_map_edges.shape[1] * patch_edges.shape[0] / patch_edges.shape[1])))
    print("resized shape: ", resized_patch_gray.shape)

    # Calculate padding for top/bottom (height) and left/right (width)
    top_padding = (big_map_edges.shape[0] - resized_patch_gray.shape[0]) // 2
    bottom_padding = big_map_edges.shape[0] - resized_patch_gray.shape[0] - top_padding
    left_padding = (big_map_edges.shape[1] - resized_patch_gray.shape[1]) // 2
    right_padding = big_map_edges.shape[1] - resized_patch_gray.shape[1] - left_padding

    # Pad the resized patch with zeroes (black padding) to match the target size
    padded_patch_gray = cv2.copyMakeBorder(resized_patch_gray, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=0)

    # Stack them horizontally
    combined_image = np.hstack((big_map_edges, padded_patch_gray))

    cv2.imshow("Canny images", combined_image)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convertPILToCV2(img, color = False, map = False):
    img_np = np.array(img) if type(img) != np.ndarray else img
    if map:
        if img_np.shape[-1] == 4:
            img_np_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY) if not color else img_np #cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR) # Convert to grayscale if color
        else:
            img_np_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if not color else img_np #cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # Convert to grayscale if color
    else:
        if img_np.shape[-1] == 4:
            img_np_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY) if not color else cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB) #cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR) # Convert to grayscale if color
        else:
            img_np_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if not color else img_np #cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # Convert to grayscale if color
    return img_np_cv

def initialise_map(image_cv, save_path, map_name):
    """Initializes a new map by cropping out the player and replacing with black."""
    # image_cv = convertPILToCV2(image, color=True)
    # Replace the player area with black
    image_cv[402:402+107, 430:430+107] = 0
    # Save the modified image as a new map
    save_img(image_cv, save_path, map_name)
    return image_cv

def split_into_patches(img, grid_size=(9, 10)):
    """Splits an image into grid of (h, w) and returns patch sizes and grid size."""
    img_h, img_w = img.shape[0:2]
    gh, gw = grid_size
    ph, pw = img_h // gh, img_w // gw  # Patch height and width
    ph2, pw2 = img_h - ph * (gh-1), img_w - pw * (gw-1)  # Last row/column patch sizes

    return [ph, ph2], [pw, pw2], grid_size

# return image with coordinates, takes in global player coordinate
def make_coordinates_global(img, grid_size=(9, 10), font_scale=0.5, color=(0, 0, 255), 
                     thickness=1, rescaleimg=1, player_position=(4, 4), top_left_position = (0,0)):
    """Overlays coordinate labels on the image based on the given grid size and global player position."""
    base_patch_size = 16*rescaleimg
    img = cv2.resize(img, (0, 0), fx=rescaleimg, fy=rescaleimg, interpolation=cv2.INTER_NEAREST)
    h_sizes, w_sizes, (gh, gw) = split_into_patches(img, grid_size) if img.shape[0] == 144 and img.shape[1] == 160 \
        else [base_patch_size, base_patch_size+img.shape[0]%base_patch_size], [base_patch_size, base_patch_size+img.shape[1]%16], (img.shape[0]//base_patch_size, img.shape[1]//base_patch_size)
    
    ph, ph2 = h_sizes
    pw, pw2 = w_sizes

    # Initialize offsets with default values
    offset_x, offset_y = 0, 0

    if player_position != (4,4):
        # Global offset: image center (4, 4) is where the player is locally
        offset_x = player_position[0] - 4
        offset_y = player_position[1] - 4
    elif top_left_position != (0,0):
        # Global offset: top left (0, 0) is the top left grid coordinates
        offset_x = top_left_position[0] - 0
        offset_y = top_left_position[1] - 0

    for i in range(gh):
        for j in range(gw):
            y = i * ph if i < gh - 1 else img.shape[0] - ph2
            x = j * pw if j < gw - 1 else img.shape[1] - pw2
            
            # Calculate the center of the grid cell
            center_x = x + (pw // 2 if j < gw - 1 else pw2 // 2)
            center_y = y + (ph // 2 if i < gh - 1 else ph2 // 2)
            
            # Global coordinate label
            global_x = j + offset_x
            global_y = i + offset_y
            label = f"({global_x},{global_y})"
            
            cv2.putText(img, label, (center_x - 25, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, color, thickness, cv2.LINE_AA)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (0, 0), fx=1/rescaleimg, fy=1/rescaleimg, interpolation=cv2.INTER_NEAREST)
    return img

# return image with coordinates
def make_coordinates(img, grid_size=(9, 10), font_scale=0.5, color=(0, 0, 255), thickness=1, rescaleimg = 1):
    """Overlays coordinate labels on the image based on the given grid size."""
    img = cv2.resize(img, (0, 0), fx=rescaleimg, fy=rescaleimg, interpolation=cv2.INTER_NEAREST)
    h_sizes, w_sizes, (gh, gw) = split_into_patches(img, grid_size)
    
    ph, ph2 = h_sizes
    pw, pw2 = w_sizes
    
    for i in range(gh):
        for j in range(gw):
            y = i * ph if i < gh - 1 else img.shape[0] - ph2
            x = j * pw if j < gw - 1 else img.shape[1] - pw2
            
            # Calculate the center of the grid cell
            center_x = x + (pw // 2 if i < gh - 1 else pw2 // 2)
            center_y = y + (ph // 2 if j < gw - 1 else ph2 // 2)
            
            label = f"({j},{i})"
            cv2.putText(img, label, (center_x - 20, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, color, thickness, cv2.LINE_AA)
    return img



def generate_player_mask(screenshot):
    """
    based on screenshot mask out player and sprites
    """
    mask = np.zeros(screenshot.shape, dtype=np.uint8)
    #set all to 1
    mask[:, :] = 1
    #set player to 0: player position = 430, 445 to 430 + 107, 445 + 107
    mask[402-107:402+1, 430-107:430] = 0
    return mask

def determine_displacement(array1, array2, grid_size=(9, 10), thresh = 0.8, offset=1, player_position=(4,4)):
    """
    Determines the best displacement direction by cropping array2 to its inner grid
    and matching it to array1. Calculates the necessary canvas size for stitching
    and determines the relative positions of array1 and array2.

    Parameters:
        array1 (numpy array): The base map to which array2 will be matched.
        array2 (numpy array): The image to be matched and stitched to array1.
        grid_size (tuple): The grid size used for splitting the image (default is (9, 10)).
        thresh (float): The threshold for match confidence (default is 0.8).

    Returns:
        stitched_array_size (tuple): The dimensions of the canvas required to merge array1 and array2 (height, width).
        array1_coord (tuple): The top-left coordinates of array1 on the canvas (y, x).
        array2_coord (tuple): The top-left coordinates of array2 on the canvas (y, x).
    """
    # Split array2 into patches to calculate patch dimensions
    patch_heights, patch_widths, grid_size = split_into_patches(array2, (grid_size[0]+(1-offset)*2, grid_size[1]+(1-offset)*2))
    ph, ph2 = patch_heights
    pw, pw2 = patch_widths
    gh, gw = grid_size

    # Crop array2 to its inner grid (remove first and last row/column)
    inner_array2 = array2[ph:-ph2, pw:-pw2]

    best_coordinates = None
    player_mask = np.zeros(inner_array2.shape, dtype=np.uint8)
    #set all to 1
    player_mask[:, :] = 1
    #set player to 0: player position = 430, 445 to 430 + 107, 445 + 107
    player_mask[402-(ph*offset):403-(ph*(offset-1)), 430-(pw*offset):430-(pw*(offset-1))] = 0
    # player_mask[402-(107*offset):402+1-(107*offset-1), 430-(107*offset):430-(107*offset-1)] = 0    
    # player_mask = generate_player_mask(inner_array2)

    ## Visualize inner_array2 with the mask applied
    # masked_inner_array2 = cv2.bitwise_and(inner_array2, inner_array2, mask=player_mask)
    # cv2.imshow(f"Inner Array2 with Mask{offset}", masked_inner_array2)

    # Use cv2.matchTemplate to find the best match for inner_array2 in array1
    result = cv2.matchTemplate(array1, inner_array2, cv2.TM_CCOEFF_NORMED, mask = player_mask)
    _, max_val, _, max_loc= cv2.minMaxLoc(result)

    if max_val > thresh:
        # Determine the best direction based on the location of the match
        best_coordinates = max_loc  # Top-left corner of the best match

        # not sure why but this is [y(height), x(width)]
        array2_coord = [best_coordinates[1]-(ph*offset), best_coordinates[0]-(pw*offset)]

        expandy = expandx = 0
        # if any coord in array2 outside array1 bounds, expand canvas.
        if array2_coord[0] < 0 or array2_coord[0]+array2.shape[0]+(ph*2*(offset-1)) > array1.shape[0]:
            expandy = -array2_coord[0] if array2_coord[0] < 0 else array2_coord[0]+array2.shape[0]+(ph*2*(offset-1))-array1.shape[0]
        if array2_coord[1] < 0 or array2_coord[1]+array2.shape[1]+(pw*2*(offset-1)) > array1.shape[1]:
            expandx = -array2_coord[1] if array2_coord[1] < 0 else array2_coord[1]+array2.shape[1]+(pw*2*(offset-1))-array1.shape[1]

        # #final coord
        new_height = array1.shape[0]+expandy
        new_width = array1.shape[1]+expandx

        array1_h = array1_w = 0
        #find where to place array1
        if array2_coord[0] < 0:
            array1_h = -array2_coord[0]
            array2_coord[0]  = 0
        if array2_coord[1] < 0:
            array1_w = -array2_coord[1]
            array2_coord[1]  = 0

        grid_y = array2_coord[0] // ph
        grid_x = array2_coord[1] // pw

        global_x = player_position[0] - 4
        global_y = player_position[1] - 4

        canvas_top_left_x = global_x - max(0, grid_x)
        canvas_top_left_y = global_y - max(0, grid_y)

        return (new_height, new_width), (array1_h, array1_w), array2_coord, (canvas_top_left_x, canvas_top_left_y)
    else:
        #try again with smaller area match
        # print(f"failed to match inner_array2+{offset}")
        while offset < 2:
            offset += 1
            return determine_displacement(array1, inner_array2, offset=offset)
        # print("No match found with confidence above threshold.")
        return None, None, None, None

def stitch_images(canvas, array1_coord, array2_coord, array1, array2):
    """
    Stitches two images together based on their relative coordinates and positions.
    This function takes two image arrays and their respective coordinates, calculates
    the necessary dimensions for a new canvas to accommodate both images, and overlays
    the second image onto the first at the specified position. Missing rows or columns
    are filled with black pixels.
    Args:
        canvas (numpy.ndarray): The base canvas where the images will be stitched.
        array1_coord (tuple): The coordinates of the top-left corner of the first image
                              on the canvas (x, y).
        array2_coord (tuple): The coordinates of the top-left corner of the second image
                              relative to the first image (x, y).
        array1 (numpy.ndarray): The first image array to be stitched.
        array2 (numpy.ndarray): The second image array to be stitched.
    Returns:
        numpy.ndarray: A new array representing the stitched image with both input
                       images combined.
    """

    # Determine number of channels from array1
    channels = 3 if len(array1[0][0]) == 3 else 4
    new_height, new_width = canvas

    #make sure canvas fits
    new_height = max(new_height, array1.shape[0]+array1_coord[0], array2.shape[0]+array2_coord[0])
    new_width = max(new_width, array1.shape[1]+array1_coord[1], array2.shape[1]+array2_coord[1])

    stitched_array = np.zeros((new_height, new_width, channels), dtype=array1.dtype)

    # Create a mask for the player area in array2
    player_mask = np.zeros(array2.shape[:2], dtype=np.uint8)  # Ensure mask matches the first two dimensions of array2
    player_mask[:, :] = 1
    pm_offset = 0 
    player_mask[402-pm_offset:402+108+pm_offset, 430-pm_offset:430+107+pm_offset] = 0

    # Add array 2 only where mask is pass-through using cv2.bitwise_and
    masked_array2 = cv2.bitwise_and(array2, array2, mask=player_mask)
    # print("array2 shape", array2.shape, "array1 shape", array1.shape)
    # print("st shape", stitched_array[array2_coord[0]:array2_coord[0] + array2.shape[0], array2_coord[1]:array2_coord[1] + array2.shape[1]].shape, "masked2 shape", masked_array2.shape)

    # cv2.imshow("array2", array2)

    stitched_array[array2_coord[0]:array2_coord[0] + array2.shape[0], array2_coord[1]:array2_coord[1] + array2.shape[1]] = masked_array2

    # cv2.imshow("stitch after array2", stitched_array)

    # Fill in stitched_array using array1 where it's within array1_coord and both are black
    stitched_region = stitched_array[array1_coord[0]:array1_coord[0]+array1.shape[0], array1_coord[1]:array1_coord[1]+array1.shape[1]]
    # Create a mask for continuous black rectangular regions in stitched_array
    gray_stitched = cv2.cvtColor(stitched_region, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    _, binary_mask = cv2.threshold(gray_stitched, 5, 255, cv2.THRESH_BINARY_INV)  # Binary mask for non-black regions
    # binary_mask = cv2.adaptiveThreshold(gray_stitched, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernelint = 3
    kernel = np.ones((kernelint,kernelint), np.uint8)
    binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    # cv2.imshow("binmask", binary_mask)
    mask_stitched = np.zeros_like(gray_stitched, dtype=bool)  # Initialize mask

    print_stitched = stitched_region.copy()
    print_stitched_array1 = array1.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Get bounding rectangle
        # print(x,y,w,h)
        #draw rectangles
        cv2.rectangle(print_stitched, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(print_stitched_array1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Check if the width or height is greater than a certain threshold (e.g., 50 pixels)
        if (w > 90 and w < 150) and (h > 90 and h < 150) or (w > 900 or h > 900):  # Ensure it's a continuous rectangular region with width or height > 50
            kernelint = 1
            y1 = y-kernelint if y-kernelint > 0 else 0
            y2 = y+h+kernelint if y+h+kernelint < stitched_region.shape[0] else stitched_region.shape[0]
            x1 = x-kernelint if x-kernelint > 0 else 0
            x2 = x+w+kernelint if x+w+kernelint < stitched_region.shape[1] else stitched_region.shape[1]
            mask_stitched[y1:y2, x1:x2] = True  # Example fix: complete the slicing operation
            # mask_stitched[y:y+h, x:x+w] = True  # Example fix: complete the slicing operation
    # cv2.imshow("rectangles", print_stitched)
    # cv2.imshow("rectangles_on_array1", print_stitched_array1)
    stitched_region[mask_stitched] = array1[mask_stitched]
    # cv2.imshow("stitch after array1", stitched_region)

    stitched_array[array1_coord[0]:array1_coord[0]+array1.shape[0], array1_coord[1]:array1_coord[1]+array1.shape[1]] = stitched_region



    #one more time just for the player
    # Create a mask for the player area in array1
    player_mask_arr1 = np.zeros(array1.shape[:2], dtype=np.uint8)  # Ensure mask matches the first two dimensions of array2
    player_mask_arr1[:, :] = 1
    pm_offset = 0 
    player_mask_arr1[402-pm_offset:402+108+pm_offset, 430-pm_offset:430+107+pm_offset] = 0

    # Add array 1 only where mask is pass-through using cv2.bitwise_and
    masked_array1 = cv2.bitwise_and(array1, array1, mask=player_mask_arr1)

    stitched_array[array1_coord[0]:array1_coord[0]+array1.shape[0], array1_coord[1]:array1_coord[1]+array1.shape[1]] = masked_array1
    # cv2.imshow("round2", stitched_array)
    # Fill in stitched_array using array1 where it's within array1_coord and both are black
    stitched_region = stitched_array[array2_coord[0]:array2_coord[0] + array2.shape[0], array2_coord[1]:array2_coord[1] + array2.shape[1]]
    # Create a mask for continuous black rectangular regions in stitched_array
    gray_stitched = cv2.cvtColor(stitched_region, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    _, binary_mask = cv2.threshold(gray_stitched, 5, 255, cv2.THRESH_BINARY_INV)  # Binary mask for non-black regions
    # binary_mask = cv2.adaptiveThreshold(gray_stitched, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernelint = 3
    kernel = np.ones((kernelint,kernelint), np.uint8)
    binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    mask_stitched = np.zeros_like(gray_stitched, dtype=bool)  # Initialize mask
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Get bounding rectangle
        # Check if the width or height is greater than a certain threshold (e.g., 50 pixels)
        if (w > 90 and w < 150) and (h > 90 and h < 150):
            kernelint = 1
            y1 = y-kernelint if y-kernelint > 0 else 0
            y2 = y+h+kernelint if y+h+kernelint < stitched_region.shape[0] else stitched_region.shape[0]
            x1 = x-kernelint if x-kernelint > 0 else 0
            x2 = x+w+kernelint if x+w+kernelint < stitched_region.shape[1] else stitched_region.shape[1]
            mask_stitched[y1:y2, x1:x2] = True  # Example fix: complete the slicing operation
    stitched_region[mask_stitched] = array2[mask_stitched]
    stitched_array[array2_coord[0]:array2_coord[0] + array2.shape[0], array2_coord[1]:array2_coord[1] + array2.shape[1]] = stitched_region

    return stitched_array

def extract_number(filename):
    """Extracts the numeric part X from 'YYYYX.png' using isdigit()."""
    parts = filename.split('.')  # Split at '.'
    if len(parts) > 1:  # Check if there are two parts
        return int("".join(filter(str.isdigit, parts[0])))  # Extract the number
    return float('inf')  # If no valid number, push to the end

def get_sorted_images(directory):
    """
    Retrieves image filenames from a directory and sorts them based on the numeric value X in 'House_X.png'.
    
    Parameters:
        directory (str): Path to the directory containing the images.
        
    Returns:
        sorted_files (list): List of sorted file paths.
    """
    # files = [f for f in os.listdir(directory) if f.startswith("House") and f.endswith(".png")]
    files = [f for f in os.listdir(directory) if f.endswith(".png")]

    # Sort files based on the extracted number X
    sorted_files = sorted(files, key=extract_number)

    # Return full paths
    return [os.path.join(directory, f) for f in sorted_files]

def resize_image(cv_img, scale_percent = 50):
    # Resize the image to a smaller size (e.g., 50% of the original size)
    width = int(cv_img.shape[1] * scale_percent / 100)
    height = int(cv_img.shape[0] * scale_percent / 100)
    res = cv2.resize(cv_img, (width, height))
    return res

# directory = os.path.join(os.getcwd(),'mapstitching_incomplete/images')
# sorted_images = get_sorted_images(directory)


# # Stitch images
# directions = []
# startind= 0
# initial = load_img(sorted_images[startind])
# cropped_initial = crop_to_game_area(initial)
# initial_img, has_dialogue = contains_dialogue(cropped_initial)
# while has_dialogue:
#     print(f"{sorted_images[0].split('/')[-1]} has dialogue, deleting.")
#     sorted_images.pop(startind)
#     initial = load_img(sorted_images[startind])
#     cropped_initial = crop_to_game_area(initial)
#     initial_img, has_dialogue = contains_dialogue(cropped_initial)
# crop_image_cv = convertPILToCV2(initial_img)
# crop_color = convertPILToCV2(initial_img, color=True)
# # curr_patches = split_into_patches(crop_image_cv, grid_size=(9, 10))
# cv2.imshow(f"original {sorted_images[startind].split('/')[-1]}", resize_image(crop_color))

# for img_path in sorted_images[startind+1:]:
#     next_img = load_img(img_path)
#     cropped_next = crop_to_game_area(next_img)
#     next_img_pil, next_has_dialogue = contains_dialogue(cropped_next)
#     if next_has_dialogue:
#         print(f"skipping {img_path}")
#         continue
#     else:
#         next_crop_image_cv = convertPILToCV2(next_img_pil)
#         next_color = convertPILToCV2(next_img_pil, color=True)
#         # next_patches = split_into_patches(next_crop_image_cv, grid_size=(9, 10))
#         canvas, array1_coord, array2_coord = determine_displacement(crop_image_cv, next_crop_image_cv)
#         print(f"canvas size: {canvas}, array1_coord: {array1_coord}, array2_coord: {array2_coord}")
#         # print(f"Best direction: {direction}")
#         # directions.append(direction)
#     if canvas is None:
#         print(f"New image {img_path.split('/')[-1]} due to no match.")
#         # Save the stitched image to the /maps directory
#         save_img(crop_color, os.path.join(os.getcwd(),'mapstitching_incomplete/maps'), f"Map_1.png")

#         crop_image_cv, crop_color = next_crop_image_cv, next_color
#         # cv2.imshow(f"new image {img_path.split('/')[-1]}", resize_image(crop_color))
#     else:
#         stitched_image_mid = stitch_images(canvas, array1_coord, array2_coord, crop_color, next_color)
#         cv2.imshow(f"merged {img_path.split('/')[-1]}", resize_image(stitched_image_mid))
#         crop_color = stitched_image_mid

# stitched_image = stitch_images(canvas, array1_coord, array2_coord, crop_color, next_color)
# save_img(crop_color, os.path.join(os.getcwd(),'mapstitching_incomplete/maps'), f"Map_1.png")
# cv2.imshow("final merged", resize_image(stitched_image_mid))
# cv2.waitKey(0)
# cv2.destroyAllWindows()