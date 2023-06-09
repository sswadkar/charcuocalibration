import numpy
import cv2
import glob
import datetime
import copy


# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 12
CHARUCOBOARD_COLCOUNT = 9
SQUARE_LENGTH = 0.020574 # meters
MARKER_LENGTH = 0.015912 # meters
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)

# Create constants to be passed into OpenCV and cv2.aruco methods
CHARUCO_BOARD = cv2.aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=ARUCO_DICT)

# Create the arrays and variables we'll use to store info like corners and IDs from images processed
corners_all = [] # Corners discovered in all images processed
ids_all = [] # cv2.aruco ids corresponding to corners discovered
image_size = None # Determined at runtime

def process_frame(image) -> None:
        # Get image size
        _imsize = (image.shape[0], image.shape[1])

        # Detect tags
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, ARUCO_DICT)
        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(image, corners)

            # Find Charuco corners
            (retval, charuco_corners, charuco_ids) = cv2.aruco.interpolateCornersCharuco(
                corners, ids, image, CHARUCO_BOARD)
            if retval:
                cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

cam = cv2.VideoCapture(0)

cv2.namedWindow("Charuco Calibration")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    display_frame = copy.deepcopy(frame)
    process_frame(display_frame)
    cv2.imshow("Charuco Calibration", display_frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "./calibration_pictures/calibration_img{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

images = glob.glob('./calibration_pictures/*.*')

# Loop through images glob'ed
for iname in images:
    # Open the image
    img = cv2.imread(iname)
    # Grayscale the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find cv2.aruco markers in the query image
    corners, ids, _ = cv2.aruco.detectMarkers(
            image=gray,
            dictionary=ARUCO_DICT)

    # Outline the cv2.aruco markers found in our query image
    img = cv2.aruco.drawDetectedMarkers(
            image=img, 
            corners=corners)

    # Get charuco corners and ids from detected cv2.aruco markers
    response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=CHARUCO_BOARD)

    # If a Charuco board was found, let's collect image/corner points
    # Requiring at least 20 squares
    if response > 20:
        # Add these corners and ids to our calibration arrays
        corners_all.append(charuco_corners)
        ids_all.append(charuco_ids)
        
        # Draw the Charuco board we've detected to show our calibrator the board was properly detected
        img = cv2.aruco.drawDetectedCornersCharuco(
                image=img,
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids)
       
        # If our image size is unknown, set it now
        if not image_size:
            image_size = gray.shape[::-1]
    
        # Reproportion the image, maxing width or height at 1000
        proportion = max(img.shape) / 1000.0
        img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))

    else:
        print("Not able to detect a charuco board in image: {}".format(iname))

# Make sure at least one image was found
if len(images) < 1:
    # Calibration failed because there were no images, warn the user
    print("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")
    # Exit for failure
    exit()

# Make sure we were able to calibrate on at least one charucoboard by checking
# if we ever determined the image size
if not image_size:
    # Calibration failed because we didn't see any charucoboards of the PatternSize used
    print("Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
    # Exit for failure
    exit()

# Now that we've seen all of our images, perform the camera calibration
# based on the set of points we've discovered
(retval, camera_matrix, distortion_coefficients, rvecs, tvecs) = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)
    
if retval:
    calibration_store = cv2.FileStorage("calibration.json", cv2.FILE_STORAGE_WRITE)
    calibration_store.write("calibration_date", str(datetime.datetime.now()))
    calibration_store.write("camera_resolution", image_size)
    calibration_store.write("camera_matrix", camera_matrix)
    calibration_store.write("distortion_coefficients", distortion_coefficients)
    calibration_store.release()
    print("Calibration finished")
else:
    print("ERROR: Calibration failed")
