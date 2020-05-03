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
    os.makedirs("data/train/A1")
    os.makedirs("data/train/B1")
    os.makedirs("data/train/C1")
    os.makedirs("data/train/D1")
    os.makedirs("data/train/E1")   
    os.makedirs("data/train/F1")
    os.makedirs("data/train/G1")
    os.makedirs("data/train/H1")
    os.makedirs("data/train/I1")
    os.makedirs("data/train/J1")
    os.makedirs("data/train/K1")
    os.makedirs("data/train/L1")
    os.makedirs("data/train/M1")
    os.makedirs("data/train/N1")
    os.makedirs("data/train/O1")
    os.makedirs("data/train/P1")
    os.makedirs("data/train/Q1")
    os.makedirs("data/train/R1")
    os.makedirs("data/train/S1")
    os.makedirs("data/train/T1")
    os.makedirs("data/train/U1")
    os.makedirs("data/train/V1")
    os.makedirs("data/train/W1")
    os.makedirs("data/train/X1")
    os.makedirs("data/train/Y1")
    os.makedirs("data/train/Z1")
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
    os.makedirs("data/test/A1")
    os.makedirs("data/test/B1")
    os.makedirs("data/test/C1")
    os.makedirs("data/test/D1")
    os.makedirs("data/test/E1")
    os.makedirs("data/test/F1")
    os.makedirs("data/test/G1")
    os.makedirs("data/test/H1")
    os.makedirs("data/test/I1")
    os.makedirs("data/test/J1")
    os.makedirs("data/test/K1")
    os.makedirs("data/test/L1")
    os.makedirs("data/test/M1")
    os.makedirs("data/test/N1")
    os.makedirs("data/test/O1")
    os.makedirs("data/test/P1")
    os.makedirs("data/test/Q1")
    os.makedirs("data/test/R1")
    os.makedirs("data/test/S1")
    os.makedirs("data/test/T1")
    os.makedirs("data/test/U1")
    os.makedirs("data/test/V1")
    os.makedirs("data/test/W1")
    os.makedirs("data/test/X1")
    os.makedirs("data/test/Y1")
    os.makedirs("data/test/Z1")

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
             'a': len(os.listdir(directory+"/A1"))
             'b': len(os.listdir(directory+"/B1"))
             'c': len(os.listdir(directory+"/C1"))
             'd': len(os.listdir(directory+"/D1"))
             'e': len(os.listdir(directory+"/E1"))
             'f': len(os.listdir(directory+"/F1"))
             'g': len(os.listdir(directory+"/G1"))
             'h': len(os.listdir(directory+"/H1"))
             'i': len(os.listdir(directory+"/I1"))
             'j': len(os.listdir(directory+"/J1"))
             'k': len(os.listdir(directory+"/K1"))
             'l': len(os.listdir(directory+"/L1"))
             'm': len(os.listdir(directory+"/M1"))
             'n': len(os.listdir(directory+"/N1"))
             'o': len(os.listdir(directory+"/O1"))
             'p': len(os.listdir(directory+"/P1"))
             'q': len(os.listdir(directory+"/Q1"))
             'r': len(os.listdir(directory+"/R1"))
             's': len(os.listdir(directory+"/S1"))
             't': len(os.listdir(directory+"/T1"))
             'u': len(os.listdir(directory+"/U1"))
             'v': len(os.listdir(directory+"/V1"))
             'w': len(os.listdir(directory+"/W1"))
             'x': len(os.listdir(directory+"/X1"))
             'y': len(os.listdir(directory+"/Y1"))
             'z': len(os.listdir(directory+"/Z1"))

    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ZERO : "+str(count['zero']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ONE : "+str(count['one']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "TWO : "+str(count['two']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "THREE : "+str(count['three']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "FOUR : "+str(count['four']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "FIVE : "+str(count['five']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "SIX : "+str(count['six']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "SEVEN : "+str(count['seven']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "EIGHT : "+str(count['eight']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "NINE : "+str(count['nine']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "A : "+str(count['a']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "B : "+str(count['b']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "C : "+str(count['c']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "D : "+str(count['d']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "E : "+str(count['e']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "F : "+str(count['f']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "G : "+str(count['g']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "H : "+str(count['h']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "I : "+str(count['i']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "J : "+str(count['j']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "K : "+str(count['k']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "L : "+str(count['l']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "M : "+str(count['m']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "N : "+str(count['n']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "O : "+str(count['o']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "P : "+str(count['p']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "Q : "+str(count['q']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "R : "+str(count['r']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "S : "+str(count['s']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "T : "+str(count['t']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "U : "+str(count['u']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "V : "+str(count['v']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "W : "+str(count['w']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "X : "+str(count['x']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "Y : "+str(count['y']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "Z : "+str(count['z']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    

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
    if interrupt & 0xFF == ord('6'):
        cv2.imwrite(directory+'6/'+str(count['six'])+'.jpg', roi)
    if interrupt & 0xFF == ord('7'):
        cv2.imwrite(directory+'7/'+str(count['seven'])+'.jpg', roi)
    if interrupt & 0xFF == ord('8'):
        cv2.imwrite(directory+'8/'+str(count['eight'])+'.jpg', roi)
    if interrupt & 0xFF == ord('9'):
        cv2.imwrite(directory+'9/'+str(count['nine'])+'.jpg', roi)
    if interrupt & 0xFF == ord('A1'):
        cv2.imwrite(directory+'A1/'+str(count['a'])+'.jpg', roi)
    if interrupt & 0xFF == ord('B1'):
        cv2.imwrite(directory+'B1/'+str(count['b'])+'.jpg', roi)
    if interrupt & 0xFF == ord('C1'):
        cv2.imwrite(directory+'C1/'+str(count['c'])+'.jpg', roi)
    if interrupt & 0xFF == ord('D1'):
        cv2.imwrite(directory+'D1/'+str(count['d'])+'.jpg', roi)
    if interrupt & 0xFF == ord('E1'):
        cv2.imwrite(directory+'E1/'+str(count['e'])+'.jpg', roi)
    if interrupt & 0xFF == ord('F1'):
        cv2.imwrite(directory+'F1/'+str(count['f'])+'.jpg', roi)
    if interrupt & 0xFF == ord('G1'):
        cv2.imwrite(directory+'G1/'+str(count['g'])+'.jpg', roi)
    if interrupt & 0xFF == ord('H1'):
        cv2.imwrite(directory+'H1/'+str(count['h'])+'.jpg', roi)
    if interrupt & 0xFF == ord('I1'):
        cv2.imwrite(directory+'I1/'+str(count['i'])+'.jpg', roi)
    if interrupt & 0xFF == ord('J1'):
        cv2.imwrite(directory+'J1/'+str(count['j'])+'.jpg', roi)
    if interrupt & 0xFF == ord('K1'):
        cv2.imwrite(directory+'K1/'+str(count['k'])+'.jpg', roi)
    if interrupt & 0xFF == ord('L1'):
        cv2.imwrite(directory+'L1/'+str(count['l'])+'.jpg', roi)
    if interrupt & 0xFF == ord('M1'):
        cv2.imwrite(directory+'M1/'+str(count['m'])+'.jpg', roi)
    if interrupt & 0xFF == ord('N1'):
        cv2.imwrite(directory+'N1/'+str(count['n'])+'.jpg', roi)
    if interrupt & 0xFF == ord('O1'):
        cv2.imwrite(directory+'O1/'+str(count['o'])+'.jpg', roi)
    if interrupt & 0xFF == ord('P1'):
        cv2.imwrite(directory+'P1/'+str(count['p'])+'.jpg', roi)
    if interrupt & 0xFF == ord('Q1'):
        cv2.imwrite(directory+'Q1/'+str(count['q'])+'.jpg', roi)
    if interrupt & 0xFF == ord('R1'):
        cv2.imwrite(directory+'R1/'+str(count['r'])+'.jpg', roi)
    if interrupt & 0xFF == ord('S1'):
        cv2.imwrite(directory+'S1/'+str(count['s'])+'.jpg', roi)
    if interrupt & 0xFF == ord('T1'):
        cv2.imwrite(directory+'T1/'+str(count['t'])+'.jpg', roi)
    if interrupt & 0xFF == ord('U1'):
        cv2.imwrite(directory+'U1/'+str(count['u'])+'.jpg', roi)
    if interrupt & 0xFF == ord('V1'):
        cv2.imwrite(directory+'V1/'+str(count['v'])+'.jpg', roi)
    if interrupt & 0xFF == ord('W1'):
        cv2.imwrite(directory+'W1/'+str(count['w'])+'.jpg', roi)
    if interrupt & 0xFF == ord('X1'):
        cv2.imwrite(directory+'X1/'+str(count['x'])+'.jpg', roi)
    if interrupt & 0xFF == ord('Y1'):
        cv2.imwrite(directory+'Y1/'+str(count['y'])+'.jpg', roi)
    if interrupt & 0xFF == ord('Z1'):
        cv2.imwrite(directory+'Z1/'+str(count['z'])+'.jpg', roi)
    
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
