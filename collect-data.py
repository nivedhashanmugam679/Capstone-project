import cv2
import numpy as np
import os

# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    os.makedirs("data/train/0")
    os.makedirs("data/train/1")
    os.makedirs("data/train/2")
    os.makedirs("data/train/3")
    os.makedirs("data/train/4")
    os.makedirs("data/train/5")
    os.makedirs("data/train/6")
    os.makedirs("data/train/7")
    os.makedirs("data/train/8")
    os.makedirs("data/train/9")
    os.makedirs("data/train/A")
    os.makedirs("data/train/B")
    os.makedirs("data/train/C")
    os.makedirs("data/train/D")
    os.makedirs("data/train/E")   
    os.makedirs("data/train/F")
    os.makedirs("data/train/G")
    os.makedirs("data/train/H")
    os.makedirs("data/train/I")
    os.makedirs("data/train/J")
    os.makedirs("data/train/K")
    os.makedirs("data/train/L")
    os.makedirs("data/train/M")
    os.makedirs("data/train/N")
    os.makedirs("data/train/O")
    os.makedirs("data/train/P")
    os.makedirs("data/train/Q")
    os.makedirs("data/train/R")
    os.makedirs("data/train/S")
    os.makedirs("data/train/T")
    os.makedirs("data/train/U")
    os.makedirs("data/train/V")
    os.makedirs("data/train/W")
    os.makedirs("data/train/X")
    os.makedirs("data/train/Y")
    os.makedirs("data/train/Z")
    os.makedirs("data/test/0")
    os.makedirs("data/test/1")
    os.makedirs("data/test/2")
    os.makedirs("data/test/3")
    os.makedirs("data/test/4")
    os.makedirs("data/test/5")
    os.makedirs("data/test/6")
    os.makedirs("data/test/7")
    os.makedirs("data/test/8")
    os.makedirs("data/test/9")
    os.makedirs("data/test/A")
    os.makedirs("data/test/B")
    os.makedirs("data/test/C")
    os.makedirs("data/test/D")
    os.makedirs("data/test/E")
    os.makedirs("data/test/F")
    os.makedirs("data/test/G")
    os.makedirs("data/test/H")
    os.makedirs("data/test/I")
    os.makedirs("data/test/J")
    os.makedirs("data/test/K")
    os.makedirs("data/test/L")
    os.makedirs("data/test/M")
    os.makedirs("data/test/N")
    os.makedirs("data/test/O")
    os.makedirs("data/test/P")
    os.makedirs("data/test/Q")
    os.makedirs("data/test/R")
    os.makedirs("data/test/S")
    os.makedirs("data/test/T")
    os.makedirs("data/test/U")
    os.makedirs("data/test/V")
    os.makedirs("data/test/W")
    os.makedirs("data/test/X")
    os.makedirs("data/test/Y")
    os.makedirs("data/test/Z")

# Train or test 
mode = 'train'
directory = 'data/'+mode+'/'

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Getting count of existing images
    count = {'zero': len(os.listdir(directory+"/0")),
             'one': len(os.listdir(directory+"/1")),
             'two': len(os.listdir(directory+"/2")),
             'three': len(os.listdir(directory+"/3")),
             'four': len(os.listdir(directory+"/4")),
             'five': len(os.listdir(directory+"/5"))
             'six': len(os.listdir(directory+"/6")) 
             'seven': len(os.listdir(directory+"/7"))
             'eight': len(os.listdir(directory+"/8"))
             'nine': len(os.listdir(directory+"/9"))
             'A': len(os.listdir(directory+"/A"))
             'B': len(os.listdir(directory+"/B"))
             'C': len(os.listdir(directory+"/C"))
             'D': len(os.listdir(directory+"/D"))
             'E': len(os.listdir(directory+"/E"))
             'F': len(os.listdir(directory+"/F"))
             'G': len(os.listdir(directory+"/G"))
             'H': len(os.listdir(directory+"/H"))
             'I': len(os.listdir(directory+"/I"))
             'J': len(os.listdir(directory+"/J"))
             'K': len(os.listdir(directory+"/K"))
             'L': len(os.listdir(directory+"/L"))
             'M': len(os.listdir(directory+"/M"))
             'N': len(os.listdir(directory+"/N"))
             'O': len(os.listdir(directory+"/O"))
             'P': len(os.listdir(directory+"/P"))
             'Q': len(os.listdir(directory+"/Q"))
             'R': len(os.listdir(directory+"/R"))
             'S': len(os.listdir(directory+"/S"))
             'T': len(os.listdir(directory+"/T"))
             'U': len(os.listdir(directory+"/U"))
             'V': len(os.listdir(directory+"/V"))
             'W': len(os.listdir(directory+"/W"))
             'X': len(os.listdir(directory+"/X"))
             'Y': len(os.listdir(directory+"/Y"))
             'Z': len(os.listdir(directory+"/Z"))

    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ZERO : "+str(count['zero']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ONE : "+str(count['one']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "TWO : "+str(count['two']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "THREE : "+str(count['three']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "FOUR : "+str(count['four']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "FIVE : "+str(count['five']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64)) 
 
    cv2.imshow("Frame", frame)
    
    #_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    #kernel = np.ones((1, 1), np.uint8)
    #img = cv2.dilate(mask, kernel, iterations=1)
    #img = cv2.erode(mask, kernel, iterations=1)
    # do the processing after capturing the image!
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", roi)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory+'0/'+str(count['zero'])+'.jpg', roi)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'1/'+str(count['one'])+'.jpg', roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory+'2/'+str(count['two'])+'.jpg', roi)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory+'3/'+str(count['three'])+'.jpg', roi)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory+'4/'+str(count['four'])+'.jpg', roi)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory+'5/'+str(count['five'])+'.jpg', roi)
    
cap.release()
cv2.destroyAllWindows()
"""
d = "old-data/test/0"
newd = "data/test/0"
for walk in os.walk(d):
    for file in walk[2]:
        roi = cv2.imread(d+"/"+file)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imwrite(newd+"/"+file, mask)     
"""
