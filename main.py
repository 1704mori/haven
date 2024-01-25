import mss
import numpy as np
import cv2

template = cv2.imread("indicator.png", cv2.IMREAD_GRAYSCALE)
# load the template image and apply Canny edge detection (optional)
template = cv2.Canny(template, 50, 200)

# create a window for the main frame
# cv2.namedWindow("bot", cv2.WINDOW_NORMAL)
# cv2.moveWindow("bot", 1920, 0)  # adjust the window position as needed

while True:
    # capture the screen using mss
    screenshot = mss.mss()
    screen = screenshot.grab({
        "top": 0,
        "left": 0,
        "width": 1920,
        "height": 1080,
    })

    # convert the screen capture to a numpy array
    frame = np.array(screen)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply Canny edge detection to the frame (optional)
    frame_gray = cv2.Canny(frame_gray, 50, 200)

    # preserve the aspect ratio while resizing the template
    h, w = frame_gray.shape

    # perform template matching
    result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # if the match confidence is above a certain threshold, draw a rectangle around the template
    threshold = 0.8  # adjust this threshold as needed
    if result[max_loc[1], max_loc[0]] > threshold:
        top_left = max_loc
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # define a region around the matched template area in a 300x30 area
        region_top = max(0, top_left[1] - 15)  # adjust as needed
        region_bottom = min(h, bottom_right[1] + 15)  # adjust as needed
        region_left = max(0, top_left[0] - 150)  # adjust as needed
        region_right = min(w, bottom_right[0] + 150)  # adjust as needed

        # extract the region of interest (ROI) around the defined region
        roi = frame[region_top:region_bottom, region_left:region_right].copy()

        # detect green mask in the defined region
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv_roi, (36, 25, 25), (70, 255, 255))

        # draw a rectangle around the defined region
        # cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (0, 0, 255), 2)

        contours = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # resize the roi to be 300x30 and create a new window
        cv2.namedWindow("roi", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("roi", 300, 30)
        cv2.imshow("roi", roi)

    # resize the main frame window
    # cv2.resizeWindow("bot", 1920, 1080)
    
    # display the modified frame in the "bot" window
    # cv2.imshow("bot", frame)
    # cv2.setWindowProperty("bot", cv2.WND_PROP_TOPMOST, 1)

    # check for the 'q' key to quit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
