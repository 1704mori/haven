import numpy as np
import cv2

class Bot:
    def __init__(self):
        self.template = cv2.imread("indicator.png", cv2.IMREAD_GRAYSCALE)
        self.template = cv2.Canny(self.template, 50, 200)
        self.threshold = 0.8

    def draw_contours(self, roi):
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lower_gray = np.array([171], dtype=np.uint8)
        upper_gray = np.array([171], dtype=np.uint8)
        gray_mask = cv2.inRange(hsv_roi, lower_gray, upper_gray)

        contours, _ = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if contours:
            # Find the largest contour based on area
            largest_contour = max(contours, key=cv2.contourArea)

            # Draw the largest contour
            cv2.drawContours(roi, [largest_contour], -1, (0, 0, 255), 2)

        return contours
    
    def draw_indicator(self, frame, top_left, bottom_right):
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        region_top = max(0, top_left[1] - 15)
        region_bottom = min(frame.shape[0], bottom_right[1] + 15)
        region_left = max(0, top_left[0] - 100)
        region_right = min(frame.shape[1], bottom_right[0] + 100)

        roi = frame[region_top:region_bottom, region_left:region_right].copy()
        contours = self.draw_contours(roi)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            contour_box = (x, y, x + w, y + h)

            indicator_box = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

            # Calculate aspect ratios
            indicator_aspect_ratio = (indicator_box[2] - indicator_box[0]) / (indicator_box[3] - indicator_box[1])
            contour_aspect_ratio = w / h
            

            abs_ratio = abs(indicator_aspect_ratio - contour_aspect_ratio)
            if abs_ratio >= .8 and abs_ratio <= 1.2:
                print("overlap")
                
        else:
            print("No contours found.")

        cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("contours", 300, 30)
        cv2.imshow("contours", roi)

    def process_image(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.Canny(frame_gray, 50, 200)

        h, w = frame_gray.shape
        result = cv2.matchTemplate(frame_gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        if result[max_loc[1], max_loc[0]] > self.threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + self.template.shape[1], top_left[1] + self.template.shape[0])

            self.draw_indicator(frame, top_left, bottom_right)
