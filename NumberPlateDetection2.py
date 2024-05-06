import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
def hough_line_detection(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray.jpg', gray)

    # Define a kernel for the morphological operation
    kernel = np.ones((3,3),np.uint8)

    # Perform dilation
    dilated_image = cv2.dilate(gray, kernel, iterations=1)
    cv2.imwrite('dilated.jpg', dilated_image)

    # Perform erosion
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
    cv2.imwrite('eroded.jpg', eroded_image)

    _, binary_image = cv2.threshold(eroded_image, 210, 255, cv2.THRESH_BINARY)
    cv2.imwrite('binary.jpg', binary_image)

    # Apply edge detection using the Canny edge detector
    edges = cv2.Canny(binary_image, 100, 200)
    cv2.imwrite('edges.jpg', edges)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(pytesseract.image_to_string(thresh))
    # Perform Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi/180, 20)
    # Array to store rho, theta, and number of votes of the lines
    lines_info = []

    # Draw detected lines on the original image
    if lines is not None:
        for line in lines:
            rho, theta = line[0]  # Extract rho and theta
            votes = line[0][0]    # Extract votes (number of votes)
            lines_info.append((rho, theta, votes))

    # Remove similar lines with theta close to 0 or 90 degrees
    unique_lines_info = []
    for rho1, theta1, votes1 in lines_info:
        is_similar = False

        if ( abs(theta1 - (np.pi)/2) < 2*np.pi/180):
            theta1 = (np.pi)/2
        elif(abs(theta1) < 2*np.pi/180 or abs(theta1 - np.pi) < 2*np.pi/180):
            theta1 = 0
        else:
            is_similar = True

        for rho2, theta2, _ in unique_lines_info:
            if abs(rho1 - rho2) < 10 :
                is_similar = True
                break

        if  (0 > votes1):
            #is_similar = True
            continue

        if not is_similar:
            unique_lines_info.append((rho1, theta1, votes1))

    # Draw unique lines on the original image
    for rho, theta, _ in unique_lines_info:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Calculate the endpoints of the line
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        # Draw the line on the original image
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite('image.jpg', image)

    return image, unique_lines_info


def rectangle(unique_lines_info, crop_left, crop_top):
    # Filter lines by theta value
    lines_by_theta = {0: [], 1: []}
    for rho, theta, _ in unique_lines_info:
        if abs(theta) == 0:
            lines_by_theta[0].append((rho, theta))
        else:
            lines_by_theta[1].append((rho, theta))

    if lines_by_theta[0] and lines_by_theta[1]:  # Check if there are any lines detected
        top_left_x = min(lines_by_theta[0])[0] + crop_left
        top_left_y = min(lines_by_theta[1])[0] + crop_top
        bottom_right_x = max(lines_by_theta[0])[0] + crop_left
        bottom_right_y = max(lines_by_theta[1])[0] + crop_top
    else:
        # If no lines detected, return None for all coordinates
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = None, None, None, None

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y




# Function for processing a single image
def process_image(image_path):
    # Read image
    image = cv2.imread(image_path)
    cv2.imwrite('input.jpg', image)

    resize_factor = max(1, image.shape[0]//800)
    image = cv2.resize(image, (image.shape[1]//resize_factor, image.shape[0]//resize_factor))

    # Crop to the middle 50% of the image both vertically and horizontally
    height, width = image.shape[:2]
    crop_width = int(width * 0.5)
    crop_height = int(height * 0.5)
    crop_left = int((width - crop_width) / 2)
    crop_top = int((height - crop_height) / 2)
    cropped_image = image[crop_top:crop_top+crop_height, crop_left:crop_left+crop_width]
    cv2.imwrite('cropped.jpg',cropped_image)
    # Apply Hough line detection on the cropped image
    result, lines_info = hough_line_detection(cropped_image.copy())

    image_copy = image.copy()

    # Overlay detected lines on the original image
    image[crop_top:crop_top+crop_height, crop_left:crop_left+crop_width] = result

    # Find rectangle information
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = rectangle(lines_info, crop_left, crop_top)

    # Draw rectangle on original image if coordinates are valid
    if top_left_x is not None and top_left_y is not None and bottom_right_x is not None and bottom_right_y is not None:
        # Convert coordinates to integers
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = map(int, [top_left_x, top_left_y, bottom_right_x, bottom_right_y])
        print("Drawing rectangle with coordinates:", top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        final = cv2.rectangle(image_copy, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
        cv2.imwrite('final.jpg', final)
    else:
        final = image_copy  # If coordinates are None, use the original image
        cv2.imwrite('final.jpg', final)
    return final

# Process all images in the input folder
input_folder = "input_images_small"
output_folder = "output_images_small"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        image_path = os.path.join(input_folder, filename)
        # Process the image
        processed_image = process_image(image_path)
        # Save the processed image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, processed_image)
