import sys
import json
import socket
import whisper
import re
import struct
import numpy as np
import sounddevice as sd
import soundfile as sf
import pyaudio as aud
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from gtts import gTTS
from langdetect import detect
from pydub import AudioSegment
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QTextEdit, QHBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os

BUFFER = 1024 * 2
FMT = aud.paInt16
CHANNELS = 1
FREQ = 48000

audio = aud.PyAudio()
stream = audio.open(
    format=FMT,
    channels=CHANNELS,
    rate=FREQ,
    input=True,
    output=True,
    frames_per_buffer=BUFFER
)

def bersihkan_teks(text):
    try:
        filtered_response = re.sub(r'\*\s*', '', text)
        filtered_response = re.sub(r'^(\d+)\.\s*', r'\1. ', filtered_response, flags=re.MULTILINE)
        filtered_response = re.sub(r' +', ' ', filtered_response)
        filtered_response = re.sub(r'\n{3,}', '\n\n', filtered_response)
        return filtered_response.strip()
    except Exception as e:
        return "Terjadi kesalahan saat memproses teks."

class SpectrumWorker(QThread):
    update_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.audio = aud.PyAudio()
        self.stream = self.audio.open(
            format=FMT,
            channels=CHANNELS,
            rate=FREQ,
            input=True,
            frames_per_buffer=BUFFER
        )
    
    def run(self):
        while self.running:
            data = self.stream.read(BUFFER, exception_on_overflow=False)
            data_int = np.frombuffer(data, dtype=np.int16)
            self.update_signal.emit(data_int)
    
    def stop(self):
        self.running = False
        self.quit()
        self.wait()

class SpeechRecognitionWorker(QThread):
    text_signal = pyqtSignal(str)
    
    def __init__(self, whisper_model):
        super().__init__()
        self.whisper_model = whisper_model
    
    def run(self):
        try:
            duration = 5
            sample_rate = 48000
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
            sd.wait()
            temp_audio_file = "temp_audio.wav"
            sf.write(temp_audio_file, audio_data, sample_rate)
            
            result = self.whisper_model.transcribe(temp_audio_file)
            text = result["text"].strip()
            
            self.text_signal.emit(text if text else "❌ Tidak ada suara yang terdeteksi.")
        except Exception as e:
            self.text_signal.emit(f"❌ Error saat menangkap suara: {e}")

class RAGServerWorker(QThread):
    response_signal = pyqtSignal(str)
    
    def __init__(self, server_host, server_port, prompt):
        super().__init__()
        self.server_host = server_host
        self.server_port = server_port
        self.prompt = prompt
    
    def run(self):
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.server_host, self.server_port))
            
            request_data = json.dumps({"prompt": self.prompt}, ensure_ascii=False)
            client_socket.sendall(request_data.encode('utf-8') + b'\n')
            
            data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk

            response_json = json.loads(data.decode('utf-8', errors='ignore'))
            response_text = response_json.get("response", "").strip()
            
            self.response_signal.emit(response_text)
        except Exception as e:
            self.response_signal.emit(f"⚠️ Error: {e}")
        finally:
            client_socket.close()

class TextToSpeechWorker(QThread):
    finished = pyqtSignal()
    
    def __init__(self, text):
        super().__init__()
        self.text = text
    
    def run(self):
        try:
            lang = 'id' if detect(self.text) == 'id' else 'en'
            tts = gTTS(text=self.text, lang=lang)
            tts.save("temp_tts.mp3")
            
            audio = AudioSegment.from_mp3("temp_tts.mp3")
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            samples /= np.iinfo(audio.array_type).max
            
            sd.play(samples, audio.frame_rate)
            sd.wait()
            os.remove("temp_tts.mp3")
            self.finished.emit()
        except Exception as e:
            print(f"❌ Error saat menghasilkan suara: {e}")

class RAGVoiceAssistantApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.rag_server_host = "codex.petra.ac.id"
        self.rag_server_port = 50001
        self.whisper_model = whisper.load_model("small", device="cuda")
        self.spectrum_worker = SpectrumWorker()
        self.spectrum_worker.update_signal.connect(self.updateSpectrum)
    
    def initUI(self):
        self.setWindowTitle("RAG AI Chatbot with Spectrum")
        self.setGeometry(100, 100, 800, 400)
        
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setVisible(False)  # Hide spectrum canvas initially
        self.x = np.arange(0, BUFFER, 1)
        self.line, = self.ax.plot(self.x, np.random.rand(BUFFER), 'r')
        self.ax.set_ylim(-60000, 60000)
        self.ax.set_xlim(0, BUFFER)
        self.ax.axis("off")
        
        self.label = QLabel("Tekan tombol untuk mulai berbicara:", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.recordButton = QPushButton("🎤 Mulai Bicara", self)
        self.recordButton.clicked.connect(self.startListening)
        self.responseText = QTextEdit(self)
        self.responseText.setReadOnly(True)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.label)
        layout.addWidget(self.recordButton)
        layout.addWidget(self.responseText)
        self.setLayout(layout)
    
    def updateSpectrum(self, data):
        self.line.set_ydata(data)
        self.canvas.draw()
    
    def startListening(self):
        # Show spectrum animation when starting to listen
        self.canvas.setVisible(True)
        self.spectrum_worker.running = True
        self.spectrum_worker.start()
        self.recordButton.setEnabled(False)
        self.label.setText("🎧 Mendengarkan...")
        self.sr_worker = SpeechRecognitionWorker(self.whisper_model)
        self.sr_worker.text_signal.connect(self.processSpeech)
        self.sr_worker.start()
    
    def processSpeech(self, text):
        # Hide spectrum animation when processing text
        self.spectrum_worker.stop()
        self.canvas.setVisible(False)
        self.label.setText("Tekan tombol untuk mulai berbicara:")
        self.recordButton.setEnabled(True)
        if text:
            self.responseText.setText(f"📝 Anda berkata: {text}\n\n🔄 Memproses...")
            self.getAIResponse(text)
    
    def getAIResponse(self, text):
        self.rag_worker = RAGServerWorker(self.rag_server_host, self.rag_server_port, text)
        self.rag_worker.response_signal.connect(self.displayResponse)
        self.rag_worker.start()
    
    def displayResponse(self, response):
        # Keep spectrum animation hidden when displaying response
        self.responseText.setText(response)
        self.tts_worker = TextToSpeechWorker(response)
        self.tts_worker.finished.connect(self.onTTSFinished)
        self.tts_worker.start()
    
    def onTTSFinished(self):
        # Don't restart spectrum animation after TTS
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RAGVoiceAssistantApp()
    window.show()
    sys.exit(app.exec_())