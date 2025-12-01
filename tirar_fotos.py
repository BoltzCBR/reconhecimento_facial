import cv2
import os

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)


def tirar_fotos():
    
    nome = input("Escreve o nome da pessoa (sem espaços): ").strip()

    # Criar pasta específica para essa pessoa
    pasta_pessoa = os.path.join(DATASET_DIR, nome)
    os.makedirs(pasta_pessoa, exist_ok=True)

    # Abrir webcam (0 = webcam padrão)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Erro ao aceder à webcam.")
        return

    # Carregador do classificador de faces do OpenCV
    detector_face = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    numero_foto = 0
    total = 100  # quantas fotos queremos captar

    print("A captar imagens... carrega em 'q' para parar.")

    while numero_foto < total:
        ret, frame = cam.read()
        if not ret:
            print("Não foi possível ler frame da webcam.")
            break

        # Converter para cinzento para facilitar a deteção
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detetar faces na imagem
        faces = detector_face.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            # Desenhar retângulo para feedback visual
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Recortar apenas a região da face
            rosto = gray[y : y + h, x : x + w]

            # Guardar foto no disco
            caminho_foto = os.path.join(pasta_pessoa, f"{nome}_{numero_foto}.jpg")
            cv2.imwrite(caminho_foto, rosto)

            numero_foto += 1
            print(f"Foto {numero_foto}/{total} gravada em: {caminho_foto}")

            # Só guardamos uma face por frame
            break

        # Mostrar imagem ao utilizador
        cv2.putText(
            frame,
            f"Fotos: {numero_foto}/{total}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Captura de fotos", frame)

        # Permitir sair mais cedo com 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Captura terminada.")


if __name__ == "__main__":

    tirar_fotos()

