from Mapstitching.processimages import (load_img, save_img, convertPILToCV2, 
                           resize_image, contains_dialogue, crop_to_game_area, 
                           stitch_images, determine_displacement, 
                           get_sorted_images, initialise_map)
import os
import cv2
import numpy as np
import argparse
import sys
from PIL import Image

def run_imagestitching(image_input):
    """Main function that handles image input either as path or numpy array"""
    if isinstance(image_input, str):
        # If image_input is a string (path), load the image
        try:
            initial = load_img(image_input)
        except:
            try:
                img_path = os.path.join(os.getcwd(), image_input)
                initial = load_img(img_path)
            except:
                print(f"Error: Unable to load image from {image_input}")
                sys.exit(1)
    elif isinstance(image_input, np.ndarray):
        # If image_input is already a numpy array, use it directly as pil image
        initial = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        # If image_input is a PIL image, use it directly
        initial = image_input
    else:
        print("Error: Invalid input type. Expected image path (str) or image as numpy array.")
        return "INVALIDINPUT", initial
        sys.exit(1)

    # Define directories for images and maps
    directory = os.path.join(os.getcwd(), 'mapstitching_partial/images')
    maps_directory = os.path.join(os.getcwd(), 'Maps')

    # crop to area if not directly taken from pyboy
    initial = crop_to_game_area(initial)  #<----------------- COMMENT OUT IF NOT NEEDED

    # Check if image is invalid (dialogue, battle, etc.)
    initial_img, has_dialogue = contains_dialogue(initial)
    if has_dialogue:
        print("Image contains dialogue, skipping.")
        return "DIALOGUE", initial
    # Other special screens, currently not used
    # elif has_battle:
    #     print("Image contains battle, skipping.")
    #     return "BATTLE"

    # Convert the initial image to OpenCV format
    crop_image_cv = convertPILToCV2(initial_img)
    crop_color = convertPILToCV2(initial_img, color=True)

    # If maps directory exists, check for existing match
    if os.path.exists(maps_directory):
        # Get a list of existing maps
        existing_maps = get_sorted_images(maps_directory) #<----------- directory should contain only YYY_[INT].png files
    # Process each map for matches
    match_found = False
    stitched_image_mid = None
    for map_path in existing_maps:
        # Load and crop the next image
        next_img = load_img(map_path)

        # Convert the next image to OpenCV format
        map_cv = convertPILToCV2(next_img)
        map_color = convertPILToCV2(next_img, color=True)

        # Determine displacement between the map and the next image
        canvas, array1_coord, array2_coord = determine_displacement(map_cv, crop_image_cv)
        if canvas is not None:
            print(f"Match found with {map_path.split('/')[-1]}")
            # Stitch the images together and save the updated map
            stitched_image_mid = stitch_images(canvas, array1_coord, array2_coord, map_color, crop_color)
            output_map_path = save_img(stitched_image_mid, maps_directory, map_path.split('/')[-1])
            match_found = True
            break
    
    if stitched_image_mid is not None:
        return output_map_path, stitched_image_mid

    # If no match is found, initialize a new map
    if not match_found:
        print(f"New map for {img_path.split('/')[-1]} due to no match.")
        crop_color = initialise_map(crop_color, maps_directory, f"Map_{len(existing_maps) + 1}.png")
        # Add the new map to the list of existing maps
        existing_maps.append(os.path.join(maps_directory, f"Map_{len(existing_maps) + 1}.png"))

        return os.path.join(maps_directory, f"Map_{len(existing_maps) + 1}.png"), crop_color


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Process an image for analysis.")
    
    # Argument for input image
    parser.add_argument(
        'image_input', 
        type=str, 
        help="Path to the input image file or import the function to another script to directly pass in a NumPy array"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # If the input is 'numpy', use a sample image or a NumPy array to test
    if args.image_input.lower() == 'numpy':
        # Example: create a dummy NumPy array (e.g., 100x100 image of zeros)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        run_imagestitching(image)
    else:
        # Process the image from the provided path
        run_imagestitching(args.image_input)