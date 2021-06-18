import cv2

#Creating a cascading classifier object
cars_cascade = cv2.CascadeClassifier("C:\\Users\\subha\\docs\\Desktop\\Coursera\\Python\\work_place\\cars.xml")

#Reading an image as it is
img = cv2.imread("C:\\Users\\subha\\docs\\Desktop\\Coursera\\Python\\work_place\\frames\\20.png")

#Converting it to gray scale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Finding coordinates of faces in image
cars = cars_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)

print(type(cars))
print(cars)

for x, y, w, h in cars:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

resized = cv2.resize(img, (600,600))
cv2.imshow("Gray",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
