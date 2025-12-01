import os
import cv2
import numpy as np

# Pastas principais
DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"
os.makedirs(TRAINER_DIR, exist_ok=True)


def treinar():
    """
    Lê as imagens do dataset, extrai as faces e treina o modelo LBPH.
    Guarda o modelo e o mapeamento de IDs para nomes na pasta trainer/.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector_face = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = []     # aqui vamos guardar as imagens das caras (em cinzento)
    ids = []       # aqui o ID numérico correspondente a cada face
    nomes = {}     # dicionário id -> nome da pessoa
    proximo_id = 0

    # Percorrer cada subpasta em dataset (uma pasta por pessoa)
    for nome_pessoa in os.listdir(DATASET_DIR):
        pasta_pessoa = os.path.join(DATASET_DIR, nome_pessoa)
        if not os.path.isdir(pasta_pessoa):
            continue

        # Se ainda não temos ID para esta pessoa, criamos um novo
        if nome_pessoa not in nomes.values():
            nomes[proximo_id] = nome_pessoa
            pessoa_id = proximo_id
            proximo_id += 1
        else:
            # encontrar ID já registado para esta pessoa
            pessoa_id = [i for i, n in nomes.items() if n == nome_pessoa][0]

        # Percorrer as imagens dessa pessoa
        for ficheiro in os.listdir(pasta_pessoa):
            if not ficheiro.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            caminho_imagem = os.path.join(pasta_pessoa, ficheiro)
            imagem = cv2.imread(caminho_imagem)
            if imagem is None:
                print("Aviso: não foi possível ler:", caminho_imagem)
                continue

            gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

            # Detetar face na imagem (só usamos a primeira encontrada)
            rostos = detector_face.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in rostos:
                roi = gray[y : y + h, x : x + w]
                faces.append(roi)
                ids.append(pessoa_id)
                break

    if len(faces) == 0:
        print("Nenhuma face encontrada. Tira primeiro as fotos com tirar_fotos.py")
        return

    print(f"A treinar modelo com {len(faces)} exemplos de rosto...")

    recognizer.train(faces, np.array(ids))

    # Guardar o modelo treinado
    recognizer.write(os.path.join(TRAINER_DIR, "trainer.yml"))

    # Guardar o dicionário id -> nome
    np.save(os.path.join(TRAINER_DIR, "labels.npy"), nomes)

    print("Treino concluído.")
    print("Modelo guardado em trainer/trainer.yml")
    print("Labels guardados em trainer/labels.npy")


if __name__ == "__main__":
    treinar()   