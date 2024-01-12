#!/home/gustavo/anaconda3/envs/facial/bin/python
import os
os.environ['DISPLAY'] = ':0'
from xvfbwrapper import Xvfb
import cv2
import dlib
import numpy as np
import pickle
import requests
import time
from tkinter import Tk, simpledialog
from gtts import gTTS
vdisplay = Xvfb()
vdisplay.start()


def add_faces_to_database(folder_path, person_name, database_path='/home/gustavo/Documents/projetos/personal/FacialRecognition/database.pkl'):
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

def load_database(database_path='/home/gustavo/Documents/projetos/personal/FacialRecognition/database.pkl'):
    try:
        with open(database_path, 'rb') as f:
            database = pickle.load(f)
    except FileNotFoundError:
        print("Arquivo de banco de dados não encontrado!")
        database = []
    return database


def ask_for_new_person():
    root = Tk()
    root.withdraw()
    person_name = simpledialog.askstring("Input", "Enter the name of the new person:")
    folder_path = simpledialog.askstring("Input", "Enter the path of the folder containing the person's pictures:")
    root.destroy()
    return person_name, folder_path

def recognize_face_from_camera(database, tolerance=0.6):
    face_detector = dlib.get_frontal_face_detector()
    face_recognizer = dlib.face_recognition_model_v1('/home/gustavo/Documents/projetos/personal/FacialRecognition/dlib_face_recognition_resnet_model_v1.dat')
    shape_predictor = dlib.shape_predictor('/home/gustavo/Documents/projetos/personal/FacialRecognition/shape_predictor_68_face_landmarks.dat')
    
    cap = cv2.VideoCapture(0)
    
    last_recognized_name = None # Variável para guardar a última face reconhecida
    recognition_pause_until = 0
    while True:
        current_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível capturar o quadro da câmera. Verifique se a câmera está conectada e funcionando corretamente.")
            break
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if current_time >= recognition_pause_until:
            faces = face_detector(gray_image)

            for face in faces:
                shape_landmarks = shape_predictor(frame, face)
                shape_descriptor = face_recognizer.compute_face_descriptor(frame, shape_landmarks)

                recognized_name = "Desconhecido"
                min_distance = tolerance

                for name, saved_descriptor in database:
                    distance = np.linalg.norm(np.array(saved_descriptor) - np.array(shape_descriptor))
                    if distance < min_distance:
                        min_distance = distance
                        recognized_name = name

                if recognized_name == "Desconhecido":
                    person_name, folder_path = ask_for_new_person()
                    if person_name and folder_path:  # If the user provided both a name and a folder path
                        add_faces_to_database(folder_path, person_name)

                if recognized_name != last_recognized_name and recognized_name != "Desconhecido":
                 welcome_message = f"Bem vindo, {recognized_name}!"
    
                 # Cria um objeto gTTS com a mensagem de boas-vindas
                 tts = gTTS(welcome_message, lang='pt-br')

                 # Salva a mensagem de boas-vindas em um arquivo mp3
                 tts.save("welcome_message.mp3")

                 # Reproduz o arquivo mp3
                 os.system("mpg321 welcome_message.mp3")
                 # Executar a função aqui, se a face reconhecida for diferente da anterior
                 last_recognized_name = recognized_name
                 url = 'https://www.virtualsmarthome.xyz/url_routine_trigger/activate.php?trigger=b9c97636-d8ca-4cf4-a49d-e9ce8ff67539&token=660635b7-5b52-4098-b8e7-546536400265&response=json'
                 headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
                 response = requests.get(url, headers=headers)
                       
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    vdisplay.stop()
# Carregar banco de dados
database_path = '/home/gustavo/Documents/projetos/personal/FacialRecognition/database.pkl'
database = load_database(database_path)

# Iniciar reconhecimento
recognize_face_from_camera(database)
