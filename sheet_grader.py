import numpy as np
import cv2
from imutils.perspective import four_point_transform
import argparse


def show_images(titles, images, wait=True):

    for (title, image) in zip(titles, images):
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def preprocess_image(args, width=800, height=1000):
    img = cv2.imread(args["image"])
    img = cv2.resize(img, (width, height))
    img_copy = img.copy()
    return img, img_copy

def find_image_contours(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edge_img = cv2.Canny(blur_img, 10, 70)
    contours, _ = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours

def get_rect_cnts(contours):
    rect_cnts = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            rect_cnts.append(approx)
    rect_cnts = sorted(rect_cnts, key=cv2.contourArea, reverse=True)
    
    return rect_cnts

def get_top_down_document(img, rect_cnts):
    document = four_point_transform(img, rect_cnts[0].reshape(4, 2))
    doc_copy = document.copy()
    return document, doc_copy

def find_document_contours(document):
    gray_doc = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)
    blur_doc = cv2.GaussianBlur(gray_doc, (5, 5), 0)
    edge_doc = cv2.Canny(blur_doc, 10, 70)
    contours, _ = cv2.findContours(edge_doc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_question_mask(document, doc_copy, biggest_cnt):
    x, y = biggest_cnt[0][0][0] + 8, biggest_cnt[0][0][1] + 8
    x_W, y_H = biggest_cnt[2][0][0] + 8, biggest_cnt[2][0][1] + 8 
    mask =  np.zeros((document.shape[0], document.shape[1]), np.uint8)
    cv2.rectangle(mask, (x, y), (x_W, y_H), (255, 255, 255), -1) 
    masked = cv2.bitwise_and(doc_copy, doc_copy, mask=mask)
    masked = masked[y:y_H, x:x_W]
    return masked, mask

def thresholding(masked):
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
    return thresh

def split_image(image, questions, answers):
    r = len(image) // questions * questions
    c = len(image[0]) // answers * answers
    image = image[:r, :c]
    rows = np.vsplit(image, questions)
    boxes = []
    for row in rows:
        cols = np.hsplit(row, answers)
        for box in cols:
            boxes.append(box)
    return boxes

def evaluation(boxes, ans_dict, questions, answers):
    score = 0
    for i in range(0, questions):
        user_answer = None
        
        for j in range(answers):
            pixels = cv2.countNonZero(boxes[j + i * 5])
            if user_answer is None or pixels > user_answer[1]:
                user_answer = (j, pixels)
            
        if ans_dict[i] == user_answer[0]:
            score += 1

    score = (score / questions) * 100

    return score

def main():

    height = 1000
    width = 800
    questions = 5
    answers = 5
    ans_dict = {0:0, 1:3, 2:1, 3:4, 4:0}

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    args = vars(ap.parse_args())

    img, img_copy = preprocess_image(args)

    image_contours = find_image_contours(img)

    # image contours on the original image
    cv2.drawContours(img, image_contours, -1, (0, 255, 0), 3)
    show_images(['image'], [img]) 

    # get the biggest rectange contour, to find the document
    rect_cnts = get_rect_cnts(image_contours)

    # bird_view on the document
    document, doc_copy = get_top_down_document(img_copy, rect_cnts)

    #document's contour on the original image and the document bird eye view
    cv2.drawContours(img_copy, rect_cnts, -1, (0, 255, 0), 3)
    show_images(['image', 'document'], [img_copy, document])

    document_contours = find_document_contours(document)
    rect_cnts = get_rect_cnts(document_contours)
    biggest_cnt = rect_cnts[0]

    # question box on document
    cv2.drawContours(document, biggest_cnt, -1, (0, 255, 0), 3)
    show_images(['document'], [document])

    masked, mask = get_question_mask(document, doc_copy, biggest_cnt)
    show_images(['masked'], [masked])

    # thresholded question box
    thresh = thresholding(masked)
    show_images(['thresh'], [thresh])

    boxes = split_image(thresh, questions, answers)

    score = evaluation(boxes, ans_dict, questions, answers)

    print(score)


if __name__ == "__main__":
    main()
