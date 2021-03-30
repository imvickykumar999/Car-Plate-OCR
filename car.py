import numpy as np
import cv2, time

def number_plate(inputimage = 'test.jpg'):
    confidenceThreshold = 0.5
    NMSThreshold = 0.3

    modelConfiguration = 'files/yolov3.cfg'
    modelWeights = 'files/yolov3.weights'

    # inputimage = 'download.jpg'
    # inputimage = input('\nEnter Image path : ')

    labelsPath = 'files/coco.names'
    count=0

    labels = open(labelsPath).read().strip().split('\n')
    np.random.seed(10)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    image = cv2.imread(inputimage)
    (H, W) = image.shape[:2]

    layerName = net.getLayerNames()
    layerName = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)
    net.setInput(blob)
    layersOutputs = net.forward(layerName)

    boxes = []
    confidences = []
    classIDs = []

    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidenceThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY,  width, height) = box.astype('int')
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)
    if(len(detectionNMS) > 0):
        for i in detectionNMS.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            detected = labels[classIDs[i]]
            print(detected)

            if detected == 'car':
                count+=1
                a,b,c,d = x,y,w,h

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print('\nThere are ', count, ' Cars, visible in frame...')

    # cv2.imshow('Image', image)
    cv2.imwrite('image/croped car.jpeg', image[b:b+d, a:a+c])
    # cv2.waitKey(0)

    # time.sleep(5)
    # print('\nWaiting for 5 seconds...')

    from vicky import crop_plate as crop
    loop = crop.plate()

    from vicky import vicksocr as vix
    text = vix.ocr(loop)

    file = open('data.txt','w')
    file.write(text)
    file.close()

    with open("data.txt", "r") as myfile:
        data = myfile.read().splitlines()

    a = [len(i) for i in data]
    i = a.index(max(a))
    return data[i].strip()

# print(data)

# inputimage = 'test.jpg'
# number_plate(inputimage)
