import base64
import json
import os
import time
from datetime import datetime

import pyautogui
import requests
from google.cloud import vision
from scripts.env_loader import load_api_key
from utils.create_screenshot_folder import create_screenshots_folder

from taskgen import Agent, Function

# from scripts.actions import greet_user, store_name

api_key = load_api_key()


def take_screenshot(shared_variables, x: int):
    screenshots_folder = create_screenshots_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_filename = f"screenshot_{timestamp}.png"
    screenshot_path = os.path.join(screenshots_folder, screenshot_filename)
    screenshot = pyautogui.screenshot()
    screenshot.save(screenshot_path)
    return {"Output": f"Screenshot saved at: {screenshot_path}"}


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def describe_image_openai_vision_api(shared_variables, image_path):
    # Encode the image to base64
    base64_image = encode_image(image_path)

    # Set up the headers with the API key
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    # Prepare the payload with the base64 image
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze the image and return 'image_type = computer_screen' if it depicts a computer screen, 'image_type = picture' if it's a regular image, or 'image_type = None' if the image is neither a computer screen nor a regular image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    # Send the request to OpenAI API
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    content = response.json()["choices"][0]["message"]["content"]
    shared_variables["image_type"] = content

    # Return the response
    return {"Output": content}


def analyze_for_objects_within_image_with_google_vision_api(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Perform both object localization and label detection
    response = client.annotate_image(
        {
            "image": image,
            "features": [
                {"type_": vision.Feature.Type.OBJECT_LOCALIZATION},
                {"type_": vision.Feature.Type.LABEL_DETECTION},
            ],
        }
    )

    objects = response.localized_object_annotations
    labels = response.label_annotations

    # Prepare output with bounding boxes and labels
    output = []
    for object_ in objects:
        vertices = [
            (vertex.x, vertex.y) for vertex in object_.bounding_poly.normalized_vertices
        ]
        # Find matching labels by checking if object name is in label descriptions
        object_labels = [
            label.description
            for label in labels
            if label.description.lower() in object_.name.lower()
        ]
        output.append(
            {
                "object": object_.name,
                "bounding_box": vertices,
                "labels": object_labels if object_labels else [object_.name],
            }
        )

    # Store output in folder
    output_path = os.path.join(os.path.dirname(image_path), "output_objects.json")
    with open(output_path, "w") as output_file:
        json.dump(output, output_file)
    return {"Output": output}


def analyze_for_text_within_image_with_google_vision_api(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Perform object localization, label detection, and text detection
    response = client.annotate_image(
        {
            "image": image,
            "features": [
                {"type_": vision.Feature.Type.OBJECT_LOCALIZATION},
                {"type_": vision.Feature.Type.LABEL_DETECTION},
                {"type_": vision.Feature.Type.TEXT_DETECTION},
            ],
        }
    )

    objects = response.localized_object_annotations
    labels = response.label_annotations
    texts = response.text_annotations

    # Prepare output with bounding boxes, labels, and detected text
    output = []
    for object_ in objects:
        vertices = [
            (vertex.x, vertex.y) for vertex in object_.bounding_poly.normalized_vertices
        ]
        object_labels = [
            label.description
            for label in labels
            if label.description.lower() in object_.name.lower()
        ]
        output.append(
            {
                "object": object_.name,
                "bounding_box": vertices,
                "labels": object_labels if object_labels else [object_.name],
            }
        )

    for text in texts:
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        output.append(
            {
                "text": text.description,
                "bounding_box": vertices,
            }
        )

    return {"Output": output}


def click_bounding_box(shared_variables, bounding_box):
    # Calculate the center coordinates of the bounding box
    x = (bounding_box[0][0] + bounding_box[1][0]) // 2
    y = (bounding_box[0][1] + bounding_box[2][1]) // 2

    # Click at the center of the bounding box
    pyautogui.click(x, y)

    return {"Output": f"Clicked the bounding box at coordinates ({x}, {y})"}


# Define functions for the agent
fn_list = [
    Function(
        fn_description="Clicks the screen at coordinates <x: int> and <y: int>",
        external_fn=click_bounding_box,
        output_format={"Output": "Action status"},
    ),
    Function(
        fn_description="Takes a screenshot and saves it to a file <x: int>",
        external_fn=take_screenshot,
        output_format={"Output": "Screenshot save path"},
    ),
    Function(
        fn_description="Describes the image <image_path: str>",
        external_fn=describe_image_openai_vision_api,
        output_format={"Output": "Description"},
    ),
    Function(
        fn_description="If image_type = picture, analyze the objects within the image <image_path: str>",
        external_fn=analyze_for_objects_within_image_with_google_vision_api,
        output_format={"Output": "Bounding boxes"},
    ),
    Function(
        fn_description="If image_type = computer screen, analyze the text within the image <image_path: str>",
        external_fn=analyze_for_text_within_image_with_google_vision_api,
        output_format={"Output": "Bounding boxes"},
    ),
]

# Create the agent
agent = Agent(
    "Screen Interaction Agent",
    "Performs screen interaction actions",
    shared_variables={"image_type": []},
).assign_functions(fn_list)

# see the auto-generated names of your functions :)
agent.list_functions()


# Testing the google vision api
# output = agent.run(
#     "Use the google vision api '/Users/brianlim/coding/taskgen/contrib/web_browser_implementation/screenshots/dog_bike_car.jpg' "
# )

# sleep for 5 seconds to allow the screenshot to be taken
time.sleep(3)

# Run the agent
output = agent.run("Take a screenshot, describe the image, and analyze it.")


print(f"Agent: {output}")
