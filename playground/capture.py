import cv2
cap = cv2.VideoCapture("../labeled/1.hevc")
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
n=0
while cap.isOpened():
    n+=1
    ret, frame = cap.read()
    if not ret:
        print('cant read?')
        break
    
    mask = object_detector.apply(frame)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame', frame)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # area = cv2.contourArea(contour)
        # if area
        cv2.drawContours(frame, [contour], -1, (0, 255,0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) == ord('q'):
        break
print(n)
cap.release()
cv2.destroyAllWindows()
