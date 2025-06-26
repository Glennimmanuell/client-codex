import sys
import json
import socket
import re
import struct
import numpy as np
import sounddevice as sd
import soundfile as sf
import pyaudio as aud
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QTextEdit, QHBoxLayout, QComboBox, QGroupBox, QLineEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPalette, QColor
import os
import threading
import wave
import tempfile
import subprocess
import io
import time
from gtts import gTTS
import pygame

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

class GTTSModel:
    def __init__(self, lang='id', slow=False):
        self.lang = lang
        self.slow = slow
        pygame.mixer.init()
        
    def generate_speech(self, text):
        try:
            temp_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_mp3.close()
            
            tts = gTTS(text=text, lang=self.lang, slow=self.slow)
            tts.save(temp_mp3.name)
            
            # Convert to wav for consistency
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav.close()
            
            # Use pygame to load and convert
            pygame.mixer.music.load(temp_mp3.name)
            
            # Read audio data using soundfile after conversion
            audio_data, sample_rate = sf.read(temp_mp3.name)
            
            os.unlink(temp_mp3.name)
            
            return audio_data, sample_rate
            
        except Exception as e:
            print(f"Error in gTTS: {e}")
            raise e

class SpectrumWorker(QThread):
    update_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, mode="tts"):
        super().__init__()
        self.running = True
        self.mode = mode
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
            if self.mode == "tts" and self.tts_data is not None:
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
    
    def __init__(self, text, voice_model="model.onnx", volume_boost=2.0, use_piper=True):
        super().__init__()
        self.text = text
        self.volume_boost = volume_boost
        self.use_piper = use_piper
        if use_piper:
            self.piper_model = PiperTTSModel(model_path=voice_model)
        else:
            self.gtts_model = GTTSModel()
    
    def run(self):
        try:
            print("=" * 60)
            print("🔊 TTS PROCESSING START")
            print(f"🔧 Using: {'Piper TTS' if self.use_piper else 'Google TTS'}")
            
            tts_start_time = time.time()
            
            if self.use_piper:
                print("🚀 Generating speech with Piper TTS...")
                synthesis_start = time.time()
                audio_data, sample_rate = self.piper_model.generate_speech(self.text, self.volume_boost)
                synthesis_end = time.time()
                print(f"⏱️  Piper synthesis time: {synthesis_end - synthesis_start:.2f} seconds")
            else:
                print("🌐 Generating speech with Google TTS...")
                synthesis_start = time.time()
                audio_data, sample_rate = self.gtts_model.generate_speech(self.text)
                synthesis_end = time.time()
                print(f"⏱️  gTTS synthesis time: {synthesis_end - synthesis_start:.2f} seconds")
            
            self.audio_ready.emit(audio_data, sample_rate)
            
            playback_start = time.time()
            print("🎵 Starting audio playback...")
            sd.play(audio_data, sample_rate)
            sd.wait()
            playback_end = time.time()
            
            tts_end_time = time.time()
            
            print(f"⏱️  Audio playback time: {playback_end - playback_start:.2f} seconds")
            print(f"⏱️  Total TTS time: {tts_end_time - tts_start_time:.2f} seconds")
            print("✅ TTS PROCESSING COMPLETE")
            print("=" * 60)
            
            self.finished.emit()
        except Exception as e:
            print(f"❌ Error during TTS generation: {e}")
            print("=" * 60)
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
        #self.rag_server_host = "codex.petra.ac.id"
        #self.rag_server_host = "gpu3.petra.ac.id"
        self.rag_server_host = "spin5.petra.ac.id"
        self.rag_server_port = 50001
        
        # Initialize TTS models
        print("🚀 Initializing TTS models...")
        self.init_tts_models()
        
        self.spectrum_worker = SpectrumWorker()
        self.spectrum_worker.update_signal.connect(self.updateSpectrum)
        
        self.volume_boost = 2.0
    
    def init_tts_models(self):
        print("🔊 Initializing TTS models...")
        
        # Find voice models for Piper
        self.voice_models = self.find_voice_models()
        if self.voice_models and len(self.voice_models) > 0:
            self.voice_model_path = self.voice_models[0]
            print(f"✅ Found Piper voice model: {self.voice_model_path}")
            try:
                self.piper_model = PiperTTSModel(model_path=self.voice_model_path)
                print("✅ Piper TTS model initialized successfully")
            except Exception as e:
                print(f"❌ Error setting up Piper model: {e}")
                self.piper_model = None
        else:
            print("⚠️  No Piper voice models (.onnx files) found")
            self.piper_model = None
            self.voice_model_path = None
        
        # Initialize gTTS
        try:
            self.gtts_model = GTTSModel()
            print("✅ Google TTS model initialized successfully")
        except Exception as e:
            print(f"❌ Error setting up gTTS model: {e}")
            self.gtts_model = None
    
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
        self.setWindowTitle("Smart Lab Expert - Keyboard Input Assistant")
        self.setGeometry(100, 100, 900, 600)
        
        plt.style.use('dark_background')
        self.figure, self.ax = plt.subplots(facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setVisible(False)
        self.x = np.arange(0, BUFFER, 1)
        self.line, = self.ax.plot(self.x, np.random.rand(BUFFER), '#00ff00')
        self.ax.set_ylim(-60000, 60000)
        self.ax.set_xlim(0, BUFFER)
        self.ax.axis("off")
        
        # Main layout
        layout = QVBoxLayout()
        
        # Status and info
        self.label = QLabel("Ketik pertanyaan Anda di bawah:", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: white; font-size: 12pt;")
        
        self.statusLabel = QLabel("🚀 Ready - Type your question and press Send", self)
        self.statusLabel.setAlignment(Qt.AlignCenter)
        self.statusLabel.setStyleSheet("color: #00ff00; font-size: 10pt; padding: 5px;")
        
        # Input Group
        inputGroup = QGroupBox("Input Pertanyaan")
        inputGroup.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 2px solid #3a3a3a;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        inputLayout = QVBoxLayout()
        
        # Text input field
        self.textInput = QLineEdit(self)
        self.textInput.setPlaceholderText("Ketik pertanyaan Anda di sini...")
        self.textInput.setStyleSheet("""
            QLineEdit {
                background-color: #1a1a1a; 
                color: white; 
                border: 1px solid #5c5c5c;
                border-radius: 4px;
                padding: 8px;
                font-size: 12pt;
            }
            QLineEdit:focus {
                border: 2px solid #00ff00;
            }
        """)
        self.textInput.returnPressed.connect(self.processInput)
        
        inputLayout.addWidget(self.textInput)
        inputGroup.setLayout(inputLayout)
        
        # Settings Group
        settingsGroup = QGroupBox("TTS Settings")
        settingsGroup.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 2px solid #3a3a3a;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        settingsLayout = QVBoxLayout()
        
        # TTS Settings
        ttsLayout = QHBoxLayout()
        ttsLayout.addWidget(QLabel("Text-to-Speech:", self))
        self.ttsSelector = QComboBox(self)
        self.ttsSelector.addItems(["Piper TTS", "Google TTS"])
        self.ttsSelector.setStyleSheet(self.get_combobox_style())
        ttsLayout.addWidget(self.ttsSelector)
        
        # Voice Model Settings (for Piper)
        voiceModelLayout = QHBoxLayout()
        voiceModelLayout.addWidget(QLabel("Piper Voice Model:", self))
        self.voiceModelSelector = QComboBox(self)
        self.voice_models = self.find_voice_models()
        if self.voice_models:
            self.voiceModelSelector.addItems(self.voice_models)
            self.voiceModelSelector.currentTextChanged.connect(self.changeVoiceModel)
        else:
            self.voiceModelSelector.addItem("No voice models found")
            self.voiceModelSelector.setEnabled(False)
        self.voiceModelSelector.setStyleSheet(self.get_combobox_style())
        voiceModelLayout.addWidget(self.voiceModelSelector)
        
        settingsLayout.addLayout(ttsLayout)
        settingsLayout.addLayout(voiceModelLayout)
        settingsGroup.setLayout(settingsLayout)
        
        # Send button
        self.sendButton = QPushButton("📤 Kirim Pertanyaan", self)
        self.sendButton.clicked.connect(self.processInput)
        self.sendButton.setStyleSheet("""
            QPushButton {
                background-color: #1a1a1a; 
                color: white; 
                border: 1px solid #5c5c5c;
                border-radius: 4px;
                padding: 8px;
                font-size: 14pt;
                font-weight: bold;
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
        
        # Response text area
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
        
        # Add all components to layout
        layout.addWidget(self.canvas)
        layout.addWidget(self.label)
        layout.addWidget(self.statusLabel)
        layout.addWidget(inputGroup)
        layout.addWidget(settingsGroup)
        layout.addWidget(self.sendButton)
        layout.addWidget(self.responseText)
        
        self.setStyleSheet("background-color: black;")
        self.setLayout(layout)
    
    def get_combobox_style(self):
        return """
            QComboBox {
                background-color: #1a1a1a; 
                color: white; 
                border: 1px solid #5c5c5c;
                border-radius: 4px;
                padding: 6px;
                font-size: 10pt;
                min-width: 150px;
            }
            QComboBox:hover {
                background-color: #2a2a2a;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a1a;
                color: white;
                selection-background-color: #2a2a2a;
                border: 1px solid #5c5c5c;
            }
        """
    
    def changeVoiceModel(self, model_name):
        """Change the current voice model to the selected one"""
        if model_name and model_name != "No voice models found":
            self.voice_model_path = model_name
            try:
                self.piper_model = PiperTTSModel(model_path=self.voice_model_path)
                print(f"🔄 Changed Piper voice model to: {model_name}")
            except Exception as e:
                print(f"❌ Error changing voice model: {e}")
    
    def updateSpectrum(self, data):
        self.line.set_ydata(data)
        self.canvas.draw()
    
    def processInput(self):
        text = self.textInput.text().strip()
        if not text:
            self.responseText.setText("❌ Silakan masukkan pertanyaan terlebih dahulu.")
            return
        
        # Disable input while processing
        self.textInput.setEnabled(False)
        self.sendButton.setEnabled(False)
        
        tts_method = self.ttsSelector.currentText()
        self.label.setText("🔄 Memproses pertanyaan...")
        self.statusLabel.setText(f"🚀 Using Keyboard Input + {tts_method}")
        
        self.responseText.setText(f"📝 Pertanyaan: {text}\n\n🔄 Memproses...")
        
        # Clear input field
        self.textInput.clear()
        
        # Get AI response
        self.getAIResponse(text)
    
    def getAIResponse(self, text):
        self.rag_worker = RAGServerWorker(self.rag_server_host, self.rag_server_port, text)
        self.rag_worker.response_signal.connect(self.displayResponse)
        self.rag_worker.start()
    
    def displayResponse(self, response):
        # Re-enable input controls
        self.textInput.setEnabled(True)
        self.sendButton.setEnabled(True)
        self.label.setText("Ketik pertanyaan Anda di bawah:")
        
        # Clean the response text if needed
        clean_response = bersihkan_teks(response)
        self.responseText.setText(clean_response)
        
        use_piper = self.ttsSelector.currentText() == "Piper TTS"
        
        # Check if selected TTS model is available
        if use_piper and not self.piper_model:
            self.responseText.append("\n\n❌ Piper TTS model not available. Switching to Google TTS.")
            use_piper = False
        elif not use_piper and not self.gtts_model:
            self.responseText.append("\n\n❌ Google TTS not available. Please check your internet connection.")
            return
        
        # Start text-to-speech
        self.tts_worker = TextToSpeechWorker(
            clean_response, 
            voice_model=self.voice_model_path if use_piper else None,
            volume_boost=self.volume_boost,
            use_piper=use_piper
        )
        self.tts_worker.audio_ready.connect(self.startTTSVisualization)
        self.tts_worker.finished.connect(self.onTTSFinished)
        self.tts_worker.start()
    
    def startTTSVisualization(self, audio_data, sample_rate):
        self.canvas.setVisible(True)
        
        if self.ttsSelector.currentText() == "Piper TTS":
            model_name = os.path.basename(self.voice_model_path) if self.voice_model_path else "Unknown"
            self.label.setText(f"🔊 Codex Berbicara (Piper: {model_name})...")
        else:
            self.label.setText("🔊 Codex Berbicara (Google TTS)...")
        
        self.spectrum_worker.set_tts_data(audio_data, sample_rate)
        self.spectrum_worker.running = True
        self.spectrum_worker.start()
    
    def onTTSFinished(self):
        self.spectrum_worker.stop()
        self.canvas.setVisible(False)
        self.label.setText("Ketik pertanyaan Anda di bawah:")

if __name__ == "__main__":
    print("🚀 Smart Lab Expert - Keyboard Input Assistant")
    print("=" * 60)
    print("Features:")
    print("⌨️  Keyboard Input: Type your questions directly")
    print("🔊 TTS Options: Piper TTS vs Google TTS")
    print("⏱️  Performance tracking with detailed timing")
    print("🎨 Dark theme UI with spectrum visualization")
    print("=" * 60)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(create_dark_palette())
    
    window = RAGVoiceAssistantApp()
    window.show()
    sys.exit(app.exec_())