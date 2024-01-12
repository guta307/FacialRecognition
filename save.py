import cv2
import dlib
import numpy as np
import pickle
import os
from tkinter import Tk, simpledialog

def add_faces_to_database(folder_path, person_name, database_path='database.pkl'):
    face_detector = dlib.get_frontal_face_detector()
    face_recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Carregar base de dados existente
    try:
        with open(database_path, 'rb') as f:
            database = pickle.load(f)
    except FileNotFoundError:
        database = []

    # Listar todos os arquivos na pasta
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for image_path in image_paths:
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray_image)
        
        if len(faces) != 1:
            print(f"A imagem {image_path} deve conter exatamente uma face.")
            continue

        face = faces[0]
        shape_landmarks = shape_predictor(image, face)
        shape_descriptor = face_recognizer.compute_face_descriptor(image, shape_landmarks)

        # Adicionar novo descritor
        database.append((person_name, shape_descriptor))

    # Salvar a base de dados atualizada
    with open(database_path, 'wb') as f:
        pickle.dump(database, f)

    print(f"Faces de {person_name} adicionadas à base de dados.")

# Exemplo de uso
folder_path = '/home/gustavo/Pictures/Renata'
person_name = 'Renata'
add_faces_to_database(folder_path, person_name)

