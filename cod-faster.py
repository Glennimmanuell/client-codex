import sys
import json
import socket
from faster_whisper import WhisperModel
import re
import struct
import numpy as np
import sounddevice as sd
import soundfile as sf
import pyaudio as aud
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QTextEdit, QHBoxLayout, QComboBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPalette, QColor
import os
import threading
import wave
import tempfile
import subprocess
import io

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

class PiperTTSModel:
    def __init__(self, model_path="model.onnx"):
        self.model_path = model_path
        self.sample_rate = 22050
        
    def generate_speech(self, text, volume_boost=2.0):
        try:
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav.close()
            output_file = temp_wav.name
            
            cmd = ["piper", "-m", self.model_path, "--output_file", output_file]
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode != 0:
                print(f"Piper TTS error: {stderr}")
                raise Exception(f"Piper TTS failed: {stderr}")
            
            audio_data, sample_rate = sf.read(output_file)
            
            if volume_boost != 1.0:
                audio_data = np.clip(audio_data * volume_boost, -1.0, 1.0)
            
            os.unlink(output_file)
            
            return audio_data, sample_rate
            
        except Exception as e:
            print(f"Error in Piper TTS: {e}")
            raise e

class SpectrumWorker(QThread):
    update_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, mode="mic"):
        super().__init__()
        self.running = True
        self.mode = mode
        self.audio = aud.PyAudio()
        self.stream = self.audio.open(
            format=FMT,
            channels=CHANNELS,
            rate=FREQ,
            input=True,
            frames_per_buffer=BUFFER
        )
        self.tts_data = None
        self.tts_pos = 0
    
    def set_tts_data(self, data, sample_rate):
        if sample_rate != FREQ:
            ratio = FREQ / sample_rate
            n_samples = int(len(data) * ratio)
            self.tts_data = np.interp(
                np.linspace(0, len(data)-1, n_samples),
                np.arange(len(data)),
                data
            )
        else:
            self.tts_data = data
        self.tts_pos = 0
    
    def run(self):
        while self.running:
            if self.mode == "mic":
                data = self.stream.read(BUFFER, exception_on_overflow=False)
                data_int = np.frombuffer(data, dtype=np.int16)
                self.update_signal.emit(data_int)
            elif self.mode == "tts" and self.tts_data is not None:
                end_pos = min(self.tts_pos + BUFFER, len(self.tts_data))
                if self.tts_pos < len(self.tts_data):
                    chunk = self.tts_data[self.tts_pos:end_pos]
                    if len(chunk) < BUFFER:
                        chunk = np.pad(chunk, (0, BUFFER - len(chunk)), 'constant')
                    chunk_int16 = (chunk * 32767).astype(np.int16)
                    self.update_signal.emit(chunk_int16)
                    self.tts_pos += BUFFER
                    duration = BUFFER / FREQ
                    threading.Event().wait(duration / 4)
                else:
                    self.tts_pos = 0
    
    def stop(self):
        self.running = False
        self.quit()
        self.wait()
    
    def switch_to_mic(self):
        self.mode = "mic"
    
    def switch_to_tts(self):
        self.mode = "tts"

class SpeechRecognitionWorker(QThread):
    text_signal = pyqtSignal(str)
    
    def __init__(self, whisper_model):
        super().__init__()
        self.whisper_model = whisper_model
    
    def run(self):
        try:
            duration = 5
            sample_rate = 16000
            
            print("Mulai merekam selama %d detik..." % duration)
            
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
            sd.wait()
            
            temp_audio_file = "temp_recorded.wav"
            sf.write(temp_audio_file, audio_data, sample_rate)
            
            print("Rekaman selesai. Mulai transkripsi dengan faster-whisper...")
            
            segments, info = self.whisper_model.transcribe(
                temp_audio_file, 
                beam_size=5,
                language=None,
                condition_on_previous_text=False
            )
            
            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            
            transcribed_text = ""
            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                transcribed_text += segment.text + " "
            
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
            
            text = transcribed_text.strip()
            self.text_signal.emit(text if text else "Tidak ada suara yang terdeteksi.")
            
        except Exception as e:
            print(f"Error saat speech recognition: {e}")
            self.text_signal.emit(f"Error saat menangkap suara: {e}")

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
            self.response_signal.emit(f"Error: {e}")
        finally:
            client_socket.close()

class TextToSpeechWorker(QThread):
    finished = pyqtSignal()
    audio_ready = pyqtSignal(np.ndarray, int)
    
    def __init__(self, text, voice_model="model.onnx", volume_boost=2.0):
        super().__init__()
        self.text = text
        self.volume_boost = volume_boost
        self.piper_model = PiperTTSModel(model_path=voice_model)
    
    def run(self):
        try:
            audio_data, sample_rate = self.piper_model.generate_speech(self.text, self.volume_boost)
            self.audio_ready.emit(audio_data, sample_rate)
            
            sd.play(audio_data, sample_rate)
            sd.wait()
            
            self.finished.emit()
        except Exception as e:
            print(f"Error saat menghasilkan suara: {e}")
            self.finished.emit()

def create_dark_palette():
    palette = QPalette()
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.Window, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(150, 150, 150))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(150, 150, 150))
    return palette

class RAGVoiceAssistantApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.rag_server_host = "codex.petra.ac.id"
        # self.rag_server_host = "gpu3.petra.ac.id"
        # self.rag_server_host = "spin5.petra.ac.id"
        self.rag_server_port = 50001
        
        print("Loading faster-whisper model...")
        model_size = "small"
        try:
            self.whisper_model = WhisperModel(
                model_size, 
                device="cuda",
                compute_type="int8"
            )
            print(f"Faster-whisper model '{model_size}' loaded successfully!")
        except Exception as e:
            print(f"Error loading faster-whisper model: {e}")
        
        self.spectrum_worker = SpectrumWorker()
        self.spectrum_worker.update_signal.connect(self.updateSpectrum)
        
        self.voice_model_path = "model.onnx"
        self.volume_boost = 2.0
        
        self.voice_models = self.find_voice_models()
        if self.voice_models and len(self.voice_models) > 0:
            self.voice_model_path = self.voice_models[0]
            print(f"Found voice model: {self.voice_model_path}")
        
        try:
            self.piper_model = PiperTTSModel(model_path=self.voice_model_path)
            print("Piper TTS model setup successfully")
        except Exception as e:
            print(f"Error setting up Piper model: {e}")
            print("Please ensure you have a valid Piper voice model (.onnx file)")
    
    def find_voice_models(self):
        """Find all .onnx files that might be piper voice models"""
        models = []
        try:
            for file in os.listdir('.'):
                if file.endswith('.onnx'):
                    models.append(file)
        except Exception as e:
            print(f"Error finding voice models: {e}")
        return models
    
    def initUI(self):
        self.setWindowTitle("Smart Lab Expert")
        self.setGeometry(100, 100, 800, 400)
        
        plt.style.use('dark_background')
        self.figure, self.ax = plt.subplots(facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setVisible(False)
        self.x = np.arange(0, BUFFER, 1)
        self.line, = self.ax.plot(self.x, np.random.rand(BUFFER), '#00ff00')
        self.ax.set_ylim(-60000, 60000)
        self.ax.set_xlim(0, BUFFER)
        self.ax.axis("off")
        
        self.label = QLabel("Tekan tombol untuk mulai berbicara:", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: white; font-size: 12pt;")
        
        self.recordButton = QPushButton("🎤 Mulai Bicara", self)
        self.recordButton.clicked.connect(self.startListening)
        self.recordButton.setStyleSheet("""
            QPushButton {
                background-color: #1a1a1a; 
                color: white; 
                border: 1px solid #5c5c5c;
                border-radius: 4px;
                padding: 6px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #2a2a2a;
            }
            QPushButton:pressed {
                background-color: #444444;
            }
            QPushButton:disabled {
                background-color: #0a0a0a;
                color: #5c5c5c;
            }
        """)
        
        self.responseText = QTextEdit(self)
        self.responseText.setReadOnly(True)
        self.responseText.setStyleSheet("""
            QTextEdit {
                background-color: #121212; 
                color: white; 
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 8px;
                font-size: 11pt;
            }
        """)
        
        self.voice_models = self.find_voice_models()
        self.voiceModelSelector = QComboBox(self)
        if self.voice_models:
            self.voiceModelSelector.addItems(self.voice_models)
            self.voiceModelSelector.currentTextChanged.connect(self.changeVoiceModel)
        else:
            self.voiceModelSelector.addItem("No voice models found")
            self.voiceModelSelector.setEnabled(False)
        self.voiceModelSelector.setStyleSheet("""
            QComboBox {
                background-color: #1a1a1a; 
                color: white; 
                border: 1px solid #5c5c5c;
                border-radius: 4px;
                padding: 6px;
                font-size: 10pt;
            }
            QComboBox:hover {
                background-color: #2a2a2a;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a1a;
                color: white;
                selection-background-color: #2a2a2a;
            }
        """)
        
        self.statusLabel = QLabel("🚀 Menggunakan Faster-Whisper + Piper TTS", self)
        self.statusLabel.setAlignment(Qt.AlignCenter)
        self.statusLabel.setStyleSheet("color: #00ff00; font-size: 10pt; padding: 5px;")
        
        voiceModelLayout = QHBoxLayout()
        voiceModelLayout.addWidget(QLabel("Voice Model:", self))
        voiceModelLayout.addWidget(self.voiceModelSelector)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.label)
        layout.addWidget(self.statusLabel)
        layout.addWidget(self.recordButton)
        layout.addLayout(voiceModelLayout)
        layout.addWidget(self.responseText)
        
        self.setStyleSheet("background-color: black;")
        self.setLayout(layout)
    
    def changeVoiceModel(self, model_name):
        """Change the current voice model to the selected one"""
        if model_name and model_name != "No voice models found":
            self.voice_model_path = model_name
            try:
                self.piper_model = PiperTTSModel(model_path=self.voice_model_path)
                print(f"Changed voice model to: {model_name}")
            except Exception as e:
                print(f"Error changing voice model: {e}")
    
    def updateSpectrum(self, data):
        self.line.set_ydata(data)
        self.canvas.draw()
    
    def startListening(self):
        self.canvas.setVisible(True)
        self.spectrum_worker.switch_to_mic()
        self.spectrum_worker.running = True
        self.spectrum_worker.start()
        self.recordButton.setEnabled(False)
        self.label.setText("🎧 Mendengarkan...")
        self.statusLabel.setText("🚀 Faster-Whisper sedang memproses audio...")
        self.sr_worker = SpeechRecognitionWorker(self.whisper_model)
        self.sr_worker.text_signal.connect(self.processSpeech)
        self.sr_worker.start()
    
    def processSpeech(self, text):
        self.spectrum_worker.stop()
        self.canvas.setVisible(False)
        self.label.setText("Tekan tombol untuk mulai berbicara:")
        self.statusLabel.setText("🚀 Menggunakan Faster-Whisper + Piper TTS")
        self.recordButton.setEnabled(True)
        if text:
            self.responseText.setText(f"📝 Anda berkata: {text}\n\n🔄 Memproses...")
            self.getAIResponse(text)
    
    def getAIResponse(self, text):
        self.rag_worker = RAGServerWorker(self.rag_server_host, self.rag_server_port, text)
        self.rag_worker.response_signal.connect(self.displayResponse)
        self.rag_worker.start()
    
    def displayResponse(self, response):
        # Clean the response text if needed
        clean_response = bersihkan_teks(response)
        self.responseText.setText(clean_response)
        
        # Start text-to-speech with Piper TTS only
        self.tts_worker = TextToSpeechWorker(
            clean_response, 
            voice_model=self.voice_model_path,
            volume_boost=self.volume_boost
        )
        self.tts_worker.audio_ready.connect(self.startTTSVisualization)
        self.tts_worker.finished.connect(self.onTTSFinished)
        self.tts_worker.start()
    
    def startTTSVisualization(self, audio_data, sample_rate):
        self.canvas.setVisible(True)
        model_name = os.path.basename(self.voice_model_path)
        self.label.setText(f"🔊 Codex Berbicara (Piper: {model_name})...")
        
        self.spectrum_worker.set_tts_data(audio_data, sample_rate)
        self.spectrum_worker.switch_to_tts()
        self.spectrum_worker.running = True
        self.spectrum_worker.start()
    
    def onTTSFinished(self):
        self.spectrum_worker.stop()
        self.canvas.setVisible(False)
        self.label.setText("Tekan tombol untuk mulai berbicara:")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(create_dark_palette())
    
    window = RAGVoiceAssistantApp()
    window.show()
    sys.exit(app.exec_())