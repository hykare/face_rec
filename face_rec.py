import cv2 as cv
image_color = cv.imread('./bla2.webp')
# image_color = cv.imread('./people.jpg'    )
image_gray = cv.cvtColor(image_color, cv.COLOR_BGR2GRAY)

mickey = cv.imread('./mickey.png', cv.IMREAD_UNCHANGED)
mickey_h = mickey.shape[0]
mickey_w = mickey.shape[1]

# scale = 0.16
# scaled_h = int(mickey_h * scale)
# scaled_w = int(mickey_w * scale)
# mickey = cv.resize(mickey, (scaled_w,scaled_h))

face_cascade = cv.CascadeClassifier('./haarcascade_frontalface_alt.xml')
detected_faces = face_cascade.detectMultiScale(image_gray)

for (col, row, w, h) in detected_faces:
    # cv.rectangle(image_color, (col, row), (col + w, row + h), (255,255,0), 2)
    scale = 1.9 * h / mickey_h
    h_m = int(mickey_h * scale)
    w_m = int(mickey_w * scale)
    temp_mickey = cv.resize(mickey, (w_m, h_m))

    y_offset = row + int(h/2 - h_m/1.5);
    # if y_offset < 0:
    #     y_offset = 0
    x_offset = col + int(w/2 - w_m/2);
    y1, y2 = y_offset, y_offset + temp_mickey.shape[0]
    x1, x2 = x_offset, x_offset + temp_mickey.shape[1]

    cutoff = 0
    if y1 < 0:
        cutoff = -y1
        y1 = 0

    alpha_mickey = temp_mickey[:,:,3] / 255.0
    alpha_people = 1.0 - alpha_mickey

    for color in range(0, 3):
        print("y1 ", y1)
        print("cutoff ", cutoff)
        mickey_part = alpha_mickey[cutoff:,:] * temp_mickey[cutoff:,:,color]
        img_part = alpha_people[cutoff:,:] * image_color[y1:y2, x1:x2, color]
        image_color[y1:y2, x1:x2, color] = mickey_part + img_part
        # image_color[y1:y2, x1:x2, color] = (alpha_mickey[cutoff:,:]*temp_mickey[cutoff:,:,color] + alpha_people * image_color[y1:y2, x1:x2, color])




cv.imshow('Image', image_color)
cv.waitKey(0)
cv.destroyAllWindows()