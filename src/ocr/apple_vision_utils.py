import string
import Quartz
import Vision
from Cocoa import NSURL
from IPython.display import Image, display
from PIL import Image, ImageDraw, ImageFont
from typing import List

# Pre-compute special characters
SPECIAL_CHARS = set(string.punctuation)

def extract_text(image_path):
    """
    Perform OCR on an image using Apple's Vision framework and return recognized text.
    """
    #  creates a URL object for the image file.
    image_url = NSURL.fileURLWithPath_(image_path)
    # creates an image source from the URL.
    image_source = Quartz.CGImageSourceCreateWithURL(image_url, None)
    # extracts the image from the image source.
    cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)

    # Create a request handler
    # initializes a request handler with the image
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})

    # Creates a text recognition request
    request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(None)

    # Perform the request
    handler.performRequests_error_([request], None)

    # Extract the recognized text
    observations = request.results()
    recognized_text = []
    for observation in observations:
        recognized_text.append(observation.topCandidates_(1)[0].string())
    # print(observations)
    
    return recognized_text

def extract_text_bboxes(image_path):
    """
    Returns:
        recognized_text: List[str]
        bounding_boxes:  List[Vision.VNRectangleObservation.boundingBox] (normalized CGRects)
    """
    # Load the image
    image_url = NSURL.fileURLWithPath_(image_path)
    image_source = Quartz.CGImageSourceCreateWithURL(image_url, None)
    cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)

    # Create a request handler
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})

    # Create a text recognition request
    request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(None)

    # Perform the request
    handler.performRequests_error_([request], None)

    # Extract the recognized text and bounding boxes
    observations = request.results()
    recognized_text = []
    bounding_boxes = []
    for observation in observations:
        recognized_text.append(observation.topCandidates_(1)[0].string())
        bounding_boxes.append(observation.boundingBox())  # normalized CGRect

    return recognized_text, bounding_boxes

# Modify font here
font = cv2.FONT_HERSHEY_TRIPLEX
font_scale = 2 # font size
font_thickness = 2
color_code = (255, 0, 0)
# Define the font and size for the text
try:
    font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 60)  # Adjust the font path and size as needed
except IOError:
    font = ImageFont.load_default()


def display_text_bboxes(image_path):
    # Load the image
    image_url = NSURL.fileURLWithPath_(image_path)
    image_source = Quartz.CGImageSourceCreateWithURL(image_url, None)
    cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)

    # Create a request handler
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})

    # Create a text recognition request
    request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(None)

    # Perform the request
    handler.performRequests_error_([request], None)

    # Extract the recognized text and bounding boxes
    observations = request.results()
    recognized_text = []
    bounding_boxes = []
    for observation in observations:
        recognized_text.append(observation.topCandidates_(1)[0].string())
        bounding_boxes.append(observation.boundingBox())

    # Load the image using PIL
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Draw bounding boxes and text on the image
    for text, bbox in zip(recognized_text, bounding_boxes):
        # Convert bounding box coordinates
        width, height = image.size
        left = bbox.origin.x * width
        top = (1 - bbox.origin.y - bbox.size.height) * height
        right = left + bbox.size.width * width
        bottom = top + bbox.size.height * height

        # Draw the bounding box
        draw.rectangle([left, top, right, bottom], outline="red", width=2)

        # Draw the text
        draw.text((left, top), text, fill="red", font=font)

    # # Save or display the image
    # image.show()
    display(image)

def get_text_at_click(image_path: str, click_x: float, click_y: float, padding: int = 10) -> List[str]:
    """
    click_x, click_y are in original image pixel coordinates (origin top-left).
    Returns list of recognized strings whose bounding boxes contain (or are within padding of) the click.
    """
    recognized_text, bounding_boxes = extract_text_bboxes(image_path)

    image = Image.open(image_path)
    width, height = image.size

    selected_texts = []

    for text, bbox in zip(recognized_text, bounding_boxes):
        # bbox is normalized with origin at bottom-left
        left = bbox.origin.x * width
        top = (1 - bbox.origin.y - bbox.size.height) * height
        right = left + bbox.size.width * width
        bottom = top + bbox.size.height * height

        if (left - padding <= click_x <= right + padding and
            top - padding <= click_y <= bottom + padding):
            selected_texts.append(text)

    return selected_texts


def highlight_click_text_bbox(image_path: str,
                                         click_x: float = None,
                                         click_y: float = None,
                                         padding: int = 10):
    """
    Draws bounding boxes and recognized text.
    Optionally draws the click position and highlights boxes that match the click.
    """
    recognized_text, bounding_boxes = extract_text_bboxes(image_path)

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for text, bbox in zip(recognized_text, bounding_boxes):
        left = bbox.origin.x * width
        top = (1 - bbox.origin.y - bbox.size.height) * height
        right = left + bbox.size.width * width
        bottom = top + bbox.size.height * height

        # by default, red
        outline_color = "red"

        # if we have a click, highlight the bbox if it contains/near click
        if click_x is not None and click_y is not None:
            if (left - padding <= click_x <= right + padding and
                top - padding <= click_y <= bottom + padding):
                outline_color = "green"

        draw.rectangle([left, top, right, bottom], outline=outline_color, width=2)
        draw.text((left, top), text, fill=outline_color, font=font)

    # draw click point if provided
    if click_x is not None and click_y is not None:
        r = 10
        draw.ellipse([click_x - r, click_y - r, click_x + r, click_y + r],
                     outline="blue", width=3)

    display(image)

def display_click_text_bbox(image_path: str,
                            click_x: float = None,
                            click_y: float = None,
                            padding: int = 10):
    """
    Displays ONLY the bounding box and text that correspond to the click.
    All other text/bboxes are ignored.
    """
    # OCR results
    recognized_text, bounding_boxes = extract_text_bboxes(image_path)

    # Load image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size

    matched = False  # Track if any bbox matched the click

    for text, bbox in zip(recognized_text, bounding_boxes):

        # Convert Vision bbox to image coordinates
        left = bbox.origin.x * width
        top = (1 - bbox.origin.y - bbox.size.height) * height
        right = left + bbox.size.width * width
        bottom = top + bbox.size.height * height

        # Show ONLY the bbox that contains the click
        if click_x is not None and click_y is not None:
            if (left - padding <= click_x <= right + padding and
                top - padding <= click_y <= bottom + padding):

                matched = True

                # Draw ONLY this bbox (green)
                draw.rectangle([left, top, right, bottom],
                               outline="red", width=3)

                # Draw ONLY this text
                draw.text((left, top), text, fill="red", font=font)

                break  # stop after drawing the matched bbox

    # Optional: draw click point
    if click_x is not None and click_y is not None:
        r = 10
        draw.ellipse([click_x - r, click_y - r, click_x + r, click_y + r],
                     outline="blue", width=3)

    if not matched:
        print("No OCR text found at this click location.")

    display(image)



### 

def get_ocr_observations(image_path):
    """
    Perform OCR using Apple's Vision framework and return raw text observations.

    Parameters:
    - image_path (str): Path to the image file.

    Returns:
    - observations (list): List of VNRecognizedTextObservation objects.
    """
    # Create a URL object for the image file
    image_url = NSURL.fileURLWithPath_(image_path)
    
    # Create an image source from the URL
    image_source = Quartz.CGImageSourceCreateWithURL(image_url, None)
    
    # Extract the CGImage from the image source
    cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)

    # Initialize a Vision image request handler with the CGImage
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})

    # Create a text recognition request (no custom completion handler needed here)
    request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(None)

    # Perform the text recognition request
    handler.performRequests_error_([request], None)

    # Retrieve the results (text observations) from the request
    return request.results()


def extract_recognized_text(observations):
    """
    Extract the top recognized text string from each VNRecognizedTextObservation.

    Parameters:
    - observations (list): List of VNRecognizedTextObservation objects returned by Apple's Vision framework.

    Returns:
    - recognized_text (list): A list of strings, each representing the top recognized text from an observation.
    """
    # Initialize an empty list to store the recognized text strings
    recognized_text = []

    # Iterate through each observation in the list
    for observation in observations:
        # Get the top candidate(s) for recognized text from the observation
        # topCandidates_(1) returns a list with the best match (highest confidence)
        top_candidate = observation.topCandidates_(1)

        # Check if a top candidate exists and is not None
        if top_candidate and top_candidate[0]:
            # Extract the string from the top candidate and append it to the result list
            recognized_text.append(top_candidate[0].string())

    # Return the list of recognized text strings
    return recognized_text


def process_text(value: str, special_chars: list = SPECIAL_CHARS) -> str:
    """
    Processes the input text by extracting the first three words and checking for special characters.

    Args:
        value (str): The input text to be processed.
        special_chars (list): A list of special characters to check for in the text.

    Returns:
        str: The processed text, split at the first occurrence of a special character if found.
    """
    # Extract the first 3 words from the input text
    words = value.split()[:3]
    text = ' '.join(words)
    
    # Check for the presence of any special character in the extracted text
    split_char = next((char for char in special_chars if char in text), None)
    
    # Split the text at the first occurrence of the special character if found, otherwise return the text
    return text.split(split_char)[0] if split_char else text


# Function to remove characters after the special character
def remove_after_special_char(value: str, special_chars: list = SPECIAL_CHARS) -> str:
    # Check for the presence of any special character in the extracted text
    split_char = next((char for char in special_chars if char in value), None)
    
    # Split the text at the first occurrence of the special character if found, otherwise return the text
    return value.split(split_char)[0] if split_char else value


def split_text(text: str) -> List[str]:
    """
    Splits the input text into all possible contiguous subphrases.

    Args:
        text (str): The input text to be split.

    Returns:
        List[str]: A list of all possible contiguous subphrases.

    Eg: text = "Hello world" will result in 'Hello', 'world', 'Hello world']
    """
    # Split the text into individual words
    words = text.split()
    # Determine the maximum length of subphrases
    max_length = len(words)
    # Initialize the result list to store subphrases
    result = []

    # Generate subphrases of varying lengths
    for length in range(1, max_length + 1):
        for i in range(len(words) - length + 1):
            # Join the words to form a subphrase and add to the result list
            result.append(' '.join(words[i:i + length]))

    return result


def draw_text_on_image(image_path, text_bbox_list):
    """
    Draw recognized text and bounding boxes on the image.

    Parameters:
    - image_path (str): Path to the image file.
    - text_bbox_list (list): List of tuples (text, bounding_box), where each bounding_box is in normalized coordinates.
    """
    # Open the image using PIL
    image = Image.open(image_path)

    # Create a drawing context to modify the image
    draw = ImageDraw.Draw(image)

    # Get the dimensions of the image
    width, height = image.size

    # Try to load a specific font; fall back to default if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 60)  # Use a large font size for visibility
    except:
        font = ImageFont.load_default()  # Fallback to default font if Arial is not found

    # Iterate over each recognized text and its bounding box
    for text, bbox in text_bbox_list:
        # Convert normalized bounding box coordinates (0–1) to pixel values
        left = bbox.origin.x * width
        top = (1 - bbox.origin.y - bbox.size.height) * height  # Flip y-axis and adjust for height
        right = left + bbox.size.width * width
        bottom = top + bbox.size.height * height

        # Draw a red rectangle around the recognized text
        draw.rectangle([left, top, right, bottom], outline="red", width=2)

        # Draw the recognized text slightly above the bounding box
        draw.text((left, top - 60), text, fill="red", font=font)

    # Display the modified image with bounding boxes and text
    display(image)


def apple_vision_framework_ocr_text(image_path):
    # Load the image
    #  creates a URL object for the image file.
    image_url = NSURL.fileURLWithPath_(image_path)
    # creates an image source from the URL.
    image_source = Quartz.CGImageSourceCreateWithURL(image_url, None)
    # extracts the image from the image source.
    cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)

    # Create a request handler
    # initializes a request handler with the image
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})

    # Creates a text recognition request
    request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(None)

    # Perform the request
    handler.performRequests_error_([request], None)

    # Extract the recognized text
    observations = request.results()
    recognized_text = []
    for observation in observations:
        recognized_text.append(observation.topCandidates_(1)[0].string())
    # print(observations)
    
    return recognized_text


# Modify font here
font = cv2.FONT_HERSHEY_TRIPLEX
font_scale = 2 # font size
font_thickness = 2 
color_code = (255, 0, 0)
# Define the font and size for the text
try:
    font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 60)  # Adjust the font path and size as needed
except IOError:
    font = ImageFont.load_default() 
        

def apple_vision_framework_ocr_text_bbox(image_path):
    # Load the image
    image_url = NSURL.fileURLWithPath_(image_path)
    image_source = Quartz.CGImageSourceCreateWithURL(image_url, None)
    cg_image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)

    # Create a request handler
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})

    # Create a text recognition request
    request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(None)

    # Perform the request
    handler.performRequests_error_([request], None)

    # Extract the recognized text and bounding boxes
    observations = request.results()
    recognized_text = []
    bounding_boxes = []
    for observation in observations:
        recognized_text.append(observation.topCandidates_(1)[0].string())
        bounding_boxes.append(observation.boundingBox())

    # Load the image using PIL
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Draw bounding boxes and text on the image
    for text, bbox in zip(recognized_text, bounding_boxes):
        # Convert bounding box coordinates
        width, height = image.size
        left = bbox.origin.x * width
        top = (1 - bbox.origin.y - bbox.size.height) * height
        right = left + bbox.size.width * width
        bottom = top + bbox.size.height * height

        # Draw the bounding box
        draw.rectangle([left, top, right, bottom], outline="red", width=2)

        # Draw the text
        draw.text((left, top), text, fill="red", font=font)

    # # Save or display the image
    # image.show()
    display(image)













