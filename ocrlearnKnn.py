import cv2
import numpy as np


def initKnn():
    knn = cv2.ml.KNearest_create()
    img = cv2.imread('digits.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
    train = np.array(cells).reshape(-1,400).astype(np.float32)
    trainLabel = np.repeat(np.arange(10),500)
    return knn, train, trainLabel

def updateKnn(knn, train, trainLabel, newData=None, newDataLabel=None):
    if (newData is not None) and (newDataLabel is not None):
        print(train.shape, newData.shape)
        newData = newData.reshape(-1,400).astype(np.float32)
        train = np.vstack((train,newData))
        trainLabel = np.hstack((trainLabel,newDataLabel))
    knn.train(train,cv2.ml.ROW_SAMPLE,trainLabel)
    return knn, train, trainLabel

def findRoi(frame, thresValue):
    rois = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.dilate(gray,None,iterations=2)
    gray2 = cv2.erode(gray2,None,iterations=2)
    edges = cv2.absdiff(gray,gray2)
    x = cv2.Sobel(edges,cv2.CV_16S,1,0)
    y = cv2.Sobel(edges,cv2.CV_16S,0,1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    ret, ddst = cv2.threshold(dst,thresValue,255,cv2.THRESH_BINARY)
    im, contours, hierarchy = cv2.findContours(ddst,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 20:
            rois.append((x,y,w,h))
    return rois, edges

def findDigit(knn, roi, thresValue):
    ret, th = cv2.threshold(roi, thresValue, 255, cv2.THRESH_BINARY)
    th = cv2.resize(th,(20,20))
    out = th.reshape(-1,400).astype(np.float32)
    ret, result, neighbours, dist = knn.findNearest(out, k=5)
    return int(result[0][0]), th

def concatenate(images):
    n = len(images)
    output = np.zeros(20*20*n).reshape(-1,20)
    for i in range(n):
        output[20*i:20*(i+1),:] = images[i]
    return output

knn, train, trainLabel = initKnn()
knn, train, trainLabel = updateKnn(knn, train, trainLabel)
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# width = 426
# height = 480
videoFrame = cv2.VideoWriter('frame.avi',cv2.VideoWriter_fourcc('M','J','P','G'),25,(int(width)*2,int(height)),True)
count = 0
while True:
    ret, frame = cap.read()
    frame = frame[:,:width]
    rois, edges = findRoi(frame, 50)
    digits = []
    for r in rois:
        x, y, w, h = r
        digit, th = findDigit(knn, edges[y:y+h,x:x+w], 50)
        digits.append(cv2.resize(th,(20,20)))
        cv2.rectangle(frame, (x,y), (x+w,y+h), (153,153,0), 2)
        cv2.putText(frame, str(digit), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,0,255), 2)
    newEdges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    newFrame = np.hstack((frame,newEdges))
    cv2.imshow('frame', newFrame)
    videoFrame.write(newFrame)
    key = cv2.waitKey(1) & 0xff
    if key == ord(' '):
        break
    elif key == ord('x'):
        Nd = len(digits)
        output = concatenate(digits)
        showDigits = cv2.resize(output,(60,60*Nd))
        cv2.imshow('digits', showDigits)
        cv2.imwrite(str(count)+'.png', showDigits)
        count += 1
        if cv2.waitKey(0) & 0xff == ord('e'):
            pass
        print('input the digits(separate by space):')
        numbers = input().split(' ')
        Nn = len(numbers)
        if Nd != Nn:
            print('update KNN fail!')
            continue
        try:
            for i in range(Nn):
                numbers[i] = int(numbers[i])
        except:
            continue
        knn, train, trainLabel = updateKnn(knn, train, trainLabel, output, numbers)
        print('update KNN, Done!')

print('Numbers of trained images:',len(train))
print('Numbers of trained image labels', len(trainLabel))
cap.release()
cv2.destroyAllWindows()
