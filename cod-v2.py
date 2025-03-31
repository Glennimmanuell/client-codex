import sys
import json
import socket
import whisper
import re
from gtts import gTTS
from langdetect import detect
from pydub import AudioSegment
import sounddevice as sd
import numpy as np
import soundfile as sf
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QTextEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os

def bersihkan_teks(text):
    try:
        filtered_response = re.sub(r'\*\s*', '', text)
        filtered_response = re.sub(r'^(\d+)\.\s*', r'\1. ', filtered_response, flags=re.MULTILINE)
        filtered_response = re.sub(r' +', ' ', filtered_response)
        filtered_response = re.sub(r'\n{3,}', '\n\n', filtered_response)
        return filtered_response.strip()
    except Exception as e:
        return "Terjadi kesalahan saat memproses teks."

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

            try:
                response_json = json.loads(data.decode('utf-8', errors='ignore'))
            except json.JSONDecodeError as e:
                self.response_signal.emit(f"⚠️ Error dalam parsing JSON: {e}")
                return

            response_text = response_json.get("response", "").strip()
            sources = response_json.get("sources", [])

            formatted_response = f"{response_text}\n\n📚 Sumber:\n" + "\n".join(sources) if sources else response_text
            self.response_signal.emit(formatted_response)

        except Exception as e:
            self.response_signal.emit(f"⚠️ Error: {e}")
        finally:
            client_socket.close()

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

class TextToSpeechWorker(QThread):
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
        except Exception as e:
            print(f"❌ Error saat menghasilkan suara: {e}")

class RAGVoiceAssistantApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.rag_server_host = "codex.petra.ac.id"
        self.rag_server_port = 50001
        self.whisper_model = whisper.load_model("small", device="cuda")
    
    def initUI(self):
        self.setWindowTitle("RAG AI Chatbot")
        self.setGeometry(100, 100, 500, 400)
        
        self.label = QLabel("Tekan tombol untuk mulai berbicara:", self)
        self.label.setAlignment(Qt.AlignCenter)
        
        self.recordButton = QPushButton("🎤 Mulai Bicara", self)
        self.recordButton.clicked.connect(self.startListening)
        
        self.responseText = QTextEdit(self)
        self.responseText.setReadOnly(True)
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.recordButton)
        layout.addWidget(self.responseText)
        
        self.setLayout(layout)
    
    def startListening(self):
        self.recordButton.setEnabled(False)
        self.label.setText("🎧 Mendengarkan...")
        self.sr_worker = SpeechRecognitionWorker(self.whisper_model)
        self.sr_worker.text_signal.connect(self.processSpeech)
        self.sr_worker.start()
    
    def processSpeech(self, text):
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
        cleaned_response = bersihkan_teks(response)
        self.responseText.setText(cleaned_response)
        self.tts_worker = TextToSpeechWorker(cleaned_response)
        self.tts_worker.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RAGVoiceAssistantApp()
    window.show()
    sys.exit(app.exec_())