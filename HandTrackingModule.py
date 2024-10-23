# """
# Hand Tracking Module
# By: Murtaza Hassan
# Modify by Baptiste Bédouret
# """

import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, static_image_mode = False, max_num_hands = 2, model_complexity = 1, min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
        self.mode = static_image_mode
        self.maxHands = max_num_hands
        self.modelComplexity = model_complexity
        self.detectionCon = min_detection_confidence
        self.trackCon = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        # Définir les spécifications de dessin
        self.handLandmarkStyle = self.mpDraw.DrawingSpec(color=(0, 100, 0), thickness=8)  # Couleur et épaisseur des landmarks
        self.handConnectionStyle = self.mpDraw.DrawingSpec(color=(50, 205, 50), thickness=4)  # Couleur et épaisseur des connecteurs

    def findHands(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        image, 
                        handLms, 
                        self.mpHands.HAND_CONNECTIONS, 
                        self.handLandmarkStyle,  
                        self.handConnectionStyle 
                    )
        return image

    def findPosition(self, image, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmList


def main():

    # Ouvrir la caméra 
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = HandDetector()

    while True:
        # Lire une image de la caméra
        success, image = cap.read()
        image = detector.findHands(image)
        lmList = detector.findPosition(image, draw=False)
        #if len(lmList) != 0:
            #print(lmList[0]) # affiche les coordonnées du premier point de repère


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Afficher du texte sur l'image
        cv2.putText(image,  f'fps:{str(int(fps))}', (10, 40), cv2.FONT_HERSHEY_SCRIPT_COMPLEX , 1, (255, 0, 0), 3)  

        # Afficher l'image capturée
        cv2.imshow("Flux caméra", image)

        #Sortir de la boucle si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer les ressources et fermer les fenêtres
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
