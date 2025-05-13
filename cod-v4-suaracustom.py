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
from langdetect import detect
from pydub import AudioSegment
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QTextEdit, QHBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPalette, QColor
import os
import threading
import wave
import tempfile
import onnxruntime as ort

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

class OnnxTTSModel:
    def __init__(self, model_path="model.onnx", config_path="model.onnx.json"):
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        
        # Load model configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Get input and output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get configuration values from the Piper model
        self.sample_rate = self.config.get("audio", {}).get("sample_rate", 22050)
        self.phoneme_id_map = self.config.get("phoneme_id_map", {})
        
        # Inference parameters
        self.inference_config = self.config.get("inference", {})
        self.noise_scale = self.inference_config.get("noise_scale", 0.667)
        self.length_scale = self.inference_config.get("length_scale", 1.0)
        self.noise_w = self.inference_config.get("noise_w", 0.8)
        
        print(f"Loaded Piper TTS model with sample rate: {self.sample_rate}Hz")
        
    def text_to_phonemes(self, text):
        """Convert raw text to phoneme IDs using the phoneme_id_map"""
        phoneme_ids = []
        
        # Add start token if needed (usually "^")
        if "^" in self.phoneme_id_map:
            phoneme_ids.extend(self.phoneme_id_map["^"])
        
        # Convert each character to its phoneme ID
        for char in text:
            if char in self.phoneme_id_map:
                phoneme_ids.extend(self.phoneme_id_map[char])
            else:
                # For unknown characters, try lowercase
                if char.lower() in self.phoneme_id_map:
                    phoneme_ids.extend(self.phoneme_id_map[char.lower()])
                else:
                    # Use space as fallback for unknown characters
                    print(f"Unknown character: '{char}', replacing with space")
                    phoneme_ids.extend(self.phoneme_id_map.get(" ", [0]))
        
        # Add end token if needed
        if "_" in self.phoneme_id_map:
            phoneme_ids.extend(self.phoneme_id_map["_"])
            
        return np.array(phoneme_ids, dtype=np.int64)
    
    def preprocess_text(self, text):
        """Prepare text for Piper model inference"""
        # Convert text to phoneme IDs
        phoneme_ids = self.text_to_phonemes(text)
        
        # Create model inputs dictionary
        model_inputs = {
            "phoneme_ids": phoneme_ids,
            "speaker_id": np.array([0], dtype=np.int64),  # Default speaker ID
            "speed": np.array([1.0], dtype=np.float32)  # Default speed
        }
        
        # Add inference parameters if the model supports them
        if "noise_scale" in [input.name for input in self.session.get_inputs()]:
            model_inputs["noise_scale"] = np.array([self.noise_scale], dtype=np.float32)
        
        if "length_scale" in [input.name for input in self.session.get_inputs()]:
            model_inputs["length_scale"] = np.array([self.length_scale], dtype=np.float32)
            
        if "noise_w" in [input.name for input in self.session.get_inputs()]:
            model_inputs["noise_w"] = np.array([self.noise_w], dtype=np.float32)
        
        return model_inputs
    
    def generate_speech(self, text):
        """Generate speech audio from input text using the Piper ONNX model"""
        try:
            print(f"Generating speech for text: '{text}'")
            
            # Preprocess text according to Piper model requirements
            model_inputs = self.preprocess_text(text)
            
            # Get actual input names from the model
            input_names = [input.name for input in self.session.get_inputs()]
            
            # Prepare the input feed dictionary with only valid inputs
            input_feed = {name: model_inputs[name] for name in model_inputs if name in input_names}
            
            # Run inference
            output = self.session.run([self.output_name], input_feed)[0]
            
            # Convert output to audio format
            audio_data = output.squeeze().astype(np.float32)
            
            # Normalize if needed (Piper output is typically already normalized)
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                audio_data = np.clip(audio_data, -1.0, 1.0)
            
            print(f"Generated audio with length: {len(audio_data)} samples")
            return audio_data, self.sample_rate
            
        except Exception as e:
            print(f"Error in speech generation: {e}")
            # Return empty audio on error
            return np.zeros(1000, dtype=np.float32), self.sample_rate

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
            sample_rate = 48000
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
            sd.wait()
            temp_audio_file = "temp_audio.wav"
            sf.write(temp_audio_file, audio_data, sample_rate)
            
            result = self.whisper_model.transcribe(temp_audio_file)
            text = result["text"].strip()
            
            self.text_signal.emit(text if text else "Tidak ada suara yang terdeteksi.")
        except Exception as e:
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
    
    def __init__(self, text, use_onnx=True):
        super().__init__()
        self.text = text
        self.use_onnx = use_onnx
        if use_onnx:
            self.onnx_model = OnnxTTSModel()
    
    def run(self):
        try:
            if self.use_onnx:
                # Generate speech using the ONNX model
                audio_data, sample_rate = self.onnx_model.generate_speech(self.text)
                
                # Emit audio data for visualization
                self.audio_ready.emit(audio_data, sample_rate)
                
                # Play audio
                sd.play(audio_data, sample_rate)
                sd.wait()
            else:
                # Fallback to gTTS if ONNX model fails
                from gtts import gTTS
                lang = 'id' if detect(self.text) == 'id' else 'en'
                tts = gTTS(text=self.text, lang=lang)
                
                temp_file = "temp_tts.mp3"
                tts.save(temp_file)
                
                audio = AudioSegment.from_mp3(temp_file)
                samples = np.array(audio.get_array_of_samples()).astype(np.float32)
                samples_normalized = samples / np.iinfo(audio.array_type).max
                
                self.audio_ready.emit(samples_normalized, audio.frame_rate)
                
                sd.play(samples_normalized, audio.frame_rate)
                sd.wait()
                
                os.remove(temp_file)
            
            self.finished.emit()
        except Exception as e:
            print(f"Error saat menghasilkan suara: {e}")
            print("Mencoba menggunakan gTTS sebagai fallback...")
            # If ONNX model fails, fallback to gTTS
            if self.use_onnx:
                self.use_onnx = False
                self.run()
            else:
                self.finished.emit()

def create_dark_palette():
    palette = QPalette()
    # Text colors
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
    # Background colors
    palette.setColor(QPalette.Window, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    # Highlight colors
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
        self.rag_server_port = 50001
        self.whisper_model = whisper.load_model("small", device="cuda")
        self.spectrum_worker = SpectrumWorker()
        self.spectrum_worker.update_signal.connect(self.updateSpectrum)
        
        # Initialize the ONNX TTS model
        try:
            self.onnx_model = OnnxTTSModel()
            self.use_onnx_tts = True
            print("ONNX TTS model loaded successfully")
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            print("Fallback to gTTS")
            self.use_onnx_tts = False
    
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
        
        # Add a toggle button for TTS model
        self.ttsToggleButton = QPushButton("🔊 Mode TTS: ONNX", self)
        self.ttsToggleButton.clicked.connect(self.toggleTTSMode)
        self.ttsToggleButton.setStyleSheet("""
            QPushButton {
                background-color: #1a1a1a; 
                color: #00ff00; 
                border: 1px solid #5c5c5c;
                border-radius: 4px;
                padding: 6px;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #2a2a2a;
            }
        """)
        
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.recordButton)
        buttonLayout.addWidget(self.ttsToggleButton)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.label)
        layout.addLayout(buttonLayout)
        layout.addWidget(self.responseText)
        
        self.setStyleSheet("background-color: black;")
        self.setLayout(layout)
    
    def toggleTTSMode(self):
        self.use_onnx_tts = not self.use_onnx_tts
        if self.use_onnx_tts:
            self.ttsToggleButton.setText("🔊 Mode TTS: ONNX")
            self.ttsToggleButton.setStyleSheet("""
                QPushButton {
                    background-color: #1a1a1a; 
                    color: #00ff00; 
                    border: 1px solid #5c5c5c;
                    border-radius: 4px;
                    padding: 6px;
                    font-size: 10pt;
                }
                QPushButton:hover {
                    background-color: #2a2a2a;
                }
            """)
        else:
            self.ttsToggleButton.setText("🔊 Mode TTS: gTTS")
            self.ttsToggleButton.setStyleSheet("""
                QPushButton {
                    background-color: #1a1a1a; 
                    color: #ff9900; 
                    border: 1px solid #5c5c5c;
                    border-radius: 4px;
                    padding: 6px;
                    font-size: 10pt;
                }
                QPushButton:hover {
                    background-color: #2a2a2a;
                }
            """)
    
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
        self.sr_worker = SpeechRecognitionWorker(self.whisper_model)
        self.sr_worker.text_signal.connect(self.processSpeech)
        self.sr_worker.start()
    
    def processSpeech(self, text):
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
        self.responseText.setText(response)
        self.tts_worker = TextToSpeechWorker(response, use_onnx=self.use_onnx_tts)
        self.tts_worker.audio_ready.connect(self.startTTSVisualization)
        self.tts_worker.finished.connect(self.onTTSFinished)
        self.tts_worker.start()
    
    def startTTSVisualization(self, audio_data, sample_rate):
        self.canvas.setVisible(True)
        model_type = "ONNX" if self.use_onnx_tts else "gTTS"
        self.label.setText(f"🔊 Codex Berbicara ({model_type})...")
        
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