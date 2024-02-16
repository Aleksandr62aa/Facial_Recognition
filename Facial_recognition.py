# importing the cv2 library
import cv2


def img_imshow(img):
    cv2.imshow('Foto', img)
    cv2.waitKey(0)
# cv2.imshow('Cat Dog', img[0:100, 0:150]) # обрезка img

# print(img.shape) # img shape (h, w, 3)
# cv2.waitKey(5000)
    


# loading the Haar Cascade algorithm file into alg variable
alg = r"haarcascade\haarcascade_frontalface_default.xml"

# # passing the algorithm to OpenCV
haar_cascade = cv2.CascadeClassifier(alg)

# # loading the image path into file_name variable
# file_name = r'<INSERT YOUR IMAGE NAME HERE> for eg-> IMAGE\Persons1.jpg'
file_name = r'IMAGE\TN3.jpg'

# # reading the image
img = cv2.imread(file_name, 0)

# # creating a black and white version of the image
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# # detecting the faces
faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100))

# # for each face detected
for i, size in enumerate(faces):
    x, y, w, h = size[0],size[1], size[2],size[3]
    # crop the image to select only the face
    cropped_image = img[y : y + h, x : x + w]
    img_imshow(cropped_image)
#     # loading the target image path into target_file_name variable
    # target_file_name = '<INSERT YOUR OUTPUT FACE IMAGE NAME HERE> for eg-> Facial_recognition\cropped_imag.jpg'
    target_file_name = f'Facial_recognition\cropped_imag_{i}.jpg'
    cv2.imwrite(target_file_name, cropped_image)

  

    