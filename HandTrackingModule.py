# """
# Hand Tracking Module
# By: Murtaza Hassan
# Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
# Website: https://www.computervision.zone
# """


import cv2

# Ouvrir la caméra (index 0 pour la caméra par défaut)
cap = cv2.VideoCapture(0)

while True:
    # Lire une image de la caméra
    success, image = cap.read()
    
    # Vérifier si la capture a réussi
    if not success:
        print("Erreur: Impossible de lire l'image de la caméra")
        break

    # Afficher l'image capturée
    cv2.imshow("Flux caméra", image)

    # Sortir de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
