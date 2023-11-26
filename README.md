# sheet_grader

# Optical Mark Recognition (OMR) Grader

This Python script performs Optical Mark Recognition (OMR) on an image of a multiple-choice exam sheet. It detects the contours of the exam sheet, extracts the document, identifies the regions of interest, and evaluates the answers based on a predefined answer key.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Imutils

Install the required libraries using:

```bash
pip install opencv-python numpy imutils
```
## Usage
1. Clone this repository
```bash
git clone https://github.com/matinmrv/sheet_grader.git
```
2. Navigate to the project directory:
```bash
cd /sheet_grader
```
3. Run the script
```bash
python sheet_grader.py --image /path/to/the/image.jpg
```

## Script Overview
- `show_images` : Function to display images using OpenCV.
- `preprocess_image`: Resize and make a copy of the original image.
- `find_image_contours`: Find contours in the resized image.
- `get_rect_cnts`: Identify rectangles in the image contours.
- `get_top_down_document`: Perform a perspective transform to get a top-down view of the document.
- `find_document_contours`: Find contours in the document image.
- `get_question_mask`: Create a mask for the region of interest (question area).
- `thresholding`: Apply thresholding to the masked image.
- `split_image`: Split the thresholded image into individual question boxes.
- `evaluation`: Evaluate the user's answers based on a predefined answer key.

## Configuration
* `height and width`: Dimensions for resizing the image.
* `questions` and answers: Number of questions and answer choices.
* `ans_dict`: Dictionary mapping question indices to correct answer indices.

## Output
The script will display images, such as contours, document view, question box, and thresholded images. The final output is the evaluation score.
