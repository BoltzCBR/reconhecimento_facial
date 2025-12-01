import os
import cv2
import numpy as np

TRAINER_DIR = "trainer"


def reconhecer():
    """
    Carrega o modelo treinado e faz reconhecimento facial em tempo real
    usando a webcam. Mostra o nome da pessoa por cima da face.
    """
    # Verificar se o modelo já foi treinado
    caminho_modelo = os.path.join(TRAINER_DIR, "trainer.yml")
    caminho_labels = os.path.join(TRAINER_DIR, "labels.npy")

    if not os.path.exists(caminho_modelo) or not os.path.exists(caminho_labels):
        print("Ainda não existe modelo treinado. Corre primeiro o treinar.py")
        return

    # Carregar o modelo LBPH
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(caminho_modelo)

    # Carregar mapeamento de IDs para nomes (dicionário)
    nomes = np.load(caminho_labels, allow_pickle=True).item()

    # Classificador de face para detetar rostos no frame
    detector_face = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Erro ao aceder à webcam.")
        return

    print("Reconhecimento em tempo real. Carrega em 'q' para sair.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Não foi possível ler frame da webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detetar faces neste frame
        faces = detector_face.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            roi = gray[y : y + h, x : x + w]

            # O recognizer devolve id e "confiança" (quanto menor, melhor)
            pred_id, confianca = recognizer.predict(roi)

            if confianca < 80:
                nome = nomes.get(pred_id, "Desconhecido")
            else:
                nome = "Desconhecido"

            # Desenhar retângulo à volta da cara
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Mostrar o nome por cima da cara
            cv2.rectangle(frame, (x, y - 30), (x + w, y), (0, 255, 0), cv2.FILLED)
            cv2.putText(
                frame,
                f"{nome}",
                (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Reconhecimento Facial", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Programa encerrado.")


if __name__ == "__main__":
    reconhecer()
