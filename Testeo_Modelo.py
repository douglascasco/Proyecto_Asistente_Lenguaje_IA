import sys
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyttsx3
import threading

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

class VideoWidget(QtWidgets.QLabel):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window  # Referencia al MainWindow
        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Cargar modelo
        model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = model_dict['model']

        # Configuración de pyttsx3 para síntesis de voz
        self.engine = pyttsx3.init()
        self.last_character = None  # Almacena el último carácter predicho
        self.is_speech_enabled = True  # Control de voz activada/desactivada

        self.setMinimumSize(640, 480)
        self.setMaximumSize(1024, 768)

    def play_audio(self, text):
        """Función que reproduce el audio solo si el texto está habilitado."""
        if self.is_speech_enabled:  # Reproduce solo si la voz está habilitada
            self.engine.say(text)
            self.engine.runAndWait()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Reflejar el video horizontalmente para mejor precisión en la predicción
        #frame = cv2.flip(frame, 1)

        # Procesamiento de MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Recolección de datos para el modelo
                data_aux = []
                x_ = []
                y_ = []

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                if self.model is not None:  # Solo predice si el modelo está cargado
                    prediction = self.model.predict([np.asarray(data_aux)])
                    predicted_character = prediction[0]

                    # Reproduce el audio solo si el carácter es diferente al anterior
                    if predicted_character != self.last_character:
                        self.last_character = predicted_character
                        threading.Thread(target=self.play_audio, args=(predicted_character,)).start()

                    # Dibujar resultado en el video
                    #cv2.putText(frame, predicted_character, (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 4, (30, 144, 255), 2)

                    # Muestra el carácter predicho en el QLabel (interfaz de PyQt)
                    self.main_window.label_prediccion.setText(f"Predicción: {predicted_character}")

        # Convertir imagen para PyQt
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.setPixmap(QtGui.QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reconocimiento de Gestos")
        self.setGeometry(100, 100, 800, 600)

        # Label para mostrar la predicción actual
        self.label_prediccion = QtWidgets.QLabel("Predicción:")
        self.label_prediccion.setFont(QtGui.QFont("Arial", 50))
        self.label_prediccion.setAlignment(QtCore.Qt.AlignCenter)

        # Widget de video
        self.video_widget = VideoWidget(main_window=self)
        self.video_widget.setAlignment(QtCore.Qt.AlignCenter)  # Centrar el video

        # Botones y menú desplegable en la parte inferior
        self.btn_Salir = QtWidgets.QPushButton("Salir")
        self.btn_Silenciar = QtWidgets.QPushButton("Silenciar")
        
        # Menú desplegable
        self.combo_opciones = QtWidgets.QComboBox()
        self.combo_opciones.addItems(["ABECEDARIO", "NUMEROS"])

        # Eventos de botones
        self.btn_Salir.clicked.connect(self.Salir_funcion)
        self.combo_opciones.currentTextChanged.connect(self.opcion_seleccionada)
        self.btn_Silenciar.clicked.connect(self.toggle_speech)

        # Layouts
        layout_principal = QtWidgets.QVBoxLayout()
        layout_principal.addWidget(self.video_widget, alignment=QtCore.Qt.AlignCenter)  # Centrar el video en el layout
        layout_principal.addWidget(self.label_prediccion, alignment=QtCore.Qt.AlignCenter)

        layout_botones = QtWidgets.QHBoxLayout()
        layout_botones.addWidget(self.btn_Salir)
        layout_botones.addWidget(self.combo_opciones)
        layout_botones.addWidget(self.btn_Silenciar)
        layout_principal.addLayout(layout_botones)

        self.setLayout(layout_principal)

    def Salir_funcion(self):
        sys.exit(app.exec_())
        print("Botón 'Salir' presionado")

    def opcion_seleccionada(self, opcion):
        # Cambia el modelo según la opción seleccionada
        if opcion == "ABECEDARIO":
            model_dict = pickle.load(open('./model.p', 'rb'))
            self.video_widget.model = model_dict['model']
            print("Modelo de Abecedario cargado.")
        elif opcion == "NUMEROS":
            model_dict = pickle.load(open('./model_numerico.p', 'rb'))
            self.video_widget.model = model_dict['model']
            print("Modelo de Números cargado.")

    def toggle_speech(self):
        """Alterna entre habilitar/deshabilitar la voz."""
        self.video_widget.is_speech_enabled = not self.video_widget.is_speech_enabled
        if self.video_widget.is_speech_enabled:
            self.btn_Silenciar.setText("Silenciar")
        else:
            self.btn_Silenciar.setText("Activar Voz")

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
