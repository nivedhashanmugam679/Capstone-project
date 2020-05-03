import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os

# Loading the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE', 6: 'SIX', 7: 'SEVEN, 8: 'EIGHT', 9: 'NINE', A1: 'A', B1: 'B', C1: 'C', D1: 'D', E1: 'E', F1: 'F', G1: 'G', H1: 'H', I1: 'I', J1: 'J', K1: 'K', L1: 'L', M1: 'M', N1: 'N', O1: 'O', P1: 'P', Q1: 'Q', R1: 'R', S1: 'S', T1: 'T', U1: 'U', V1: 'V', W1: 'W', X1: 'X', Y1: 'Y', Z1: 'Z' }

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Got this from collect-data.py
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
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (64, 64)) 
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
    prediction = {'ZERO': result[0][0], 
                  'ONE': result[0][1], 
                  'TWO': result[0][2],
                  'THREE': result[0][3],
                  'FOUR': result[0][4],
                  'FIVE': result[0][5]}
                  'SIX': result[0][6]}
                  'SEVEN': result[0][7]}
                  'EIGHT': result[0][8]}
                  'NINE': result[0][9]}
                  'A': result[0][A1]}
                  'B': result[0][B1]}
                  'C': result[0][C1]}
                  'D': result[0][D1]}
                  'E': result[0][E1]}
                  'F': result[0][F1]}
                  'G': result[0][G1]}
                  'H': result[0][H1]}
                  'I': result[0][I1]}
                  'J': result[0][J1]}
                  'K': result[0][K1]}
                  'L': result[0][L1]}
                  'M': result[0][M1]}
                  'N': result[0][N1]}
                  'O': result[0][O1]}
                  'P': result[0][P1]}
                  'Q': result[0][Q1]}
                  'R': result[0][R1]}
                  'S': result[0][S1]}
                  'T': result[0][T1]}
                  'U': result[0][U1]}
                  'V': result[0][V1]}
                  'W': result[0][W1]}
                  'X': result[0][X1]}
                  'Y': result[0][Y1]}
                  'Z': result[0][Z1]}

    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)    
    cv2.imshow("Frame", frame)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()
