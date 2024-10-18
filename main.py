import cv2
import time
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0
detector = htm.HandDetector()

while True:
    # Lire une image de la caméra
    success, image = cap.read()
    image = detector.findHands(image)
    lmList = detector.findPosition(image, draw=False)
    # if len(lmList) != 0:
    #     print(lmList[0]) # affiche les coordonnées du premier point de repère


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Afficher du texte sur l'image
    cv2.putText(image,  f'fps:{str(int(fps))}', (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 3)  

    # Afficher l'image capturée
    cv2.imshow("Flux caméra", image)

    #Sortir de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()