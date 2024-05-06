import os
import zipfile
import cv2
import imutils
import numpy as np

# Extract images from the zip file
def extract_images(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def convert_to_jpg(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        # Check if the file exists and is a valid image
        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
            # Check if the image was loaded successfully
            if image is not None:
                # Write the image to the output folder with JPG extension
                output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.jpg")
                cv2.imwrite(output_path, image)
            else:
                print(f"Error: Unable to load image '{filename}'.")
        else:
            print(f"Error: File '{filename}' not found.")


# Function to order points for perspective transformation
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Function for perspective transformation
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# detector on each image and save the results
def detector(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if not filename.endswith(".jpg"):
            continue
        img_path = os.path.join(input_folder, filename)
        print(f"Processing image: {img_path}")
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Unable to load image '{filename}'. Skipping...")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('gray2.jpg',gray)
            bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
            cv2.imwrite('bfilter2.jpg',bfilter)
            edged = cv2.Canny(bfilter, 30, 200)
            cv2.imwrite('edged2.jpg',edged)
            keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contourss = imutils.grab_contours(keypoints)
            contours = sorted(contourss, key=cv2.contourArea, reverse=True)[:10]
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 10, True)
                if len(approx) == 4:
                    # Check aspect ratio
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if aspect_ratio >= 2.5 and aspect_ratio <= 5:  # Adjust this range as needed
                        pts = approx.reshape(4, 2)
                        warped = four_point_transform(gray, pts)
                        cv2.imwrite('warped2.jpg',warped)
                        cv2.polylines(img, [pts], True, (0,255,0), 3)
                        
            output_path = os.path.join(output_folder, filename)
            # Save original and output images side by side
            original_img = cv2.imread(img_path)
            cv2.imwrite('final2.jpg',img)
            combined_img = np.concatenate((original_img, img), axis=1)
            cv2.imwrite(os.path.join(output_folder, f"combined_{filename}"), combined_img)
        except Exception as e:
            print(f"Error processing image '{filename}': {e}")
            continue


# Define input and output directories
zip_file = "input_images.zip"
extract_to = "extracted_images"
input_images_folder = "input_images_small"
output_folder = "output_images_small"

# Extract images from the zip file
#extract_images(zip_file, extract_to)

# Make sure input and output folders exist
#os.makedirs(input_images_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Convert images to JPG format
#convert_to_jpg(extract_to, input_images_folder)

# Perform detection on each image and save the results
detector(input_images_folder, output_folder)



print("detection process completed and images saved.")
