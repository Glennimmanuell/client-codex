import sys
import json
import whisper
import re
import numpy as np
import sounddevice as sd
import soundfile as sf
import pyaudio as aud
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from gtts import gTTS
from langdetect import detect
from pydub import AudioSegment
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QTextEdit, QHBoxLayout, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPalette, QColor
import os
import threading
import wave
import tempfile
import traceback
import requests

# Import RAG components - these would be your actual RAG implementation
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama  # Use Ollama integration

# Audio constants
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

# Replace the RAGServerWorker with an integrated local RAG processor
class RAGProcessorWorker(QThread):
    response_signal = pyqtSignal(str)
    
    def __init__(self, rag_chain, prompt):
        super().__init__()
        self.rag_chain = rag_chain
        self.prompt = prompt
        
        # Default response in case of failure
        self.default_responses = [
            "Maaf, saya tidak dapat memproses permintaan tersebut. Bisa diulangi?",
            "Terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba lagi.",
            "Saya mengalami kesulitan memahami permintaan Anda. Bisa dijelaskan dengan cara lain?"
        ]
    
    def run(self):
        try:
            # Format prompt with context for Gemma 3
            # Create a system prompt that instructs the model to provide a lab expert response
            system_prompt = """Anda adalah sebuah Lab Expert yang memiliki pengetahuan mendalam tentang topik-topik laboratorium. 
            Berikan jawaban yang jelas, informatif, dan akurat. Jawaban Anda sebaiknya singkat dan mudah dipahami.
            Jika Anda tidak tahu jawaban, katakan dengan jujur."""
            
            # Combine system prompt with user query
            full_prompt = f"{system_prompt}\n\nPertanyaan: {self.prompt}\n\nJawaban:"
            
            # Make direct API call to Ollama for more control
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "gemma3",
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "top_k": 40,
                            "num_predict": 512
                        }
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "").strip()
                else:
                    # Fallback to RAG chain if direct API fails
                    rag_response = self.rag_chain({"query": full_prompt})
                    response_text = rag_response.get("result", "").strip()
            except:
                # Use the RAG chain if direct API call fails
                rag_response = self.rag_chain({"query": full_prompt})
                response_text = rag_response.get("result", "").strip()
            
            # Clean the response
            response_text = bersihkan_teks(response_text)
            
            # Emit the final response
            if not response_text or len(response_text) < 5:
                import random
                self.response_signal.emit(random.choice(self.default_responses))
            else:
                self.response_signal.emit(response_text)
                
        except Exception as e:
            import random
            error_msg = str(e)
            print(f"RAG Error: {error_msg}")
            print(traceback.format_exc())
            self.response_signal.emit(random.choice(self.default_responses))

class TextToSpeechWorker(QThread):
    finished = pyqtSignal()
    audio_ready = pyqtSignal(np.ndarray, int)
    
    def __init__(self, text):
        super().__init__()
        self.text = text
    
    def run(self):
        try:
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

# New class to initialize and hold RAG components
class RAGSystem:
    def __init__(self):
        print("Initializing RAG system...")
        # Load embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Load vector store
        # Assumption: The vector store DB is already created and stored in the ./chroma_db directory
        self.vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )
        
        # Set up retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Load LLM using Ollama
        print("Connecting to Ollama for Gemma 3...")
        self.llm = Ollama(
            model="gemma3",
            temperature=0.7,
            top_p=0.95,
            repeat_penalty=1.15,
            max_tokens=512
        )
        
        # Create the RAG chain
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
        print("RAG system initialized!")
    
    def get_chain(self):
        return self.rag_chain

class RAGVoiceAssistantApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.whisper_model = whisper.load_model("small", device="cuda")
        self.spectrum_worker = SpectrumWorker()
        self.spectrum_worker.update_signal.connect(self.updateSpectrum)
        
        # Initialize RAG system
        self.statusLabel = QLabel("Memuat sistem RAG dengan Ollama Gemma 3...", self)
        self.statusLabel.setAlignment(Qt.AlignCenter)
        self.statusLabel.setStyleSheet("color: #ffaa00; font-size: 10pt;")
        self.layout().addWidget(self.statusLabel)
        
        # Check if Ollama is running
        self.checkOllamaThread = threading.Thread(target=self.checkOllamaService)
        self.checkOllamaThread.start()
    
    def checkOllamaService(self):
        import requests
        import time
        
        try:
            # Check if Ollama is running
            try:
                response = requests.get("http://localhost:11434/api/version", timeout=2)
                if response.status_code == 200:
                    self.statusLabel.setText("Ollama terdeteksi. Memuat sistem RAG...")
                    self.initRagSystemThread = threading.Thread(target=self.initializeRagSystem)
                    self.initRagSystemThread.start()
                else:
                    self.statusLabel.setText("Ollama tidak merespon. Pastikan Ollama sudah berjalan.")
                    self.statusLabel.setStyleSheet("color: #ff0000; font-size: 10pt;")
            except requests.exceptions.ConnectionError:
                self.statusLabel.setText("Ollama tidak ditemukan. Memastikan Ollama berjalan...")
                self.statusLabel.setStyleSheet("color: #ff0000; font-size: 10pt;")
                
                # Try to start Ollama service (this is OS-specific)
                try:
                    import subprocess
                    subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    time.sleep(5)  # Wait for Ollama to start
                    
                    # Check again
                    try:
                        response = requests.get("http://localhost:11434/api/version", timeout=2)
                        if response.status_code == 200:
                            self.statusLabel.setText("Ollama berhasil dimulai. Memuat sistem RAG...")
                            self.statusLabel.setStyleSheet("color: #ffaa00; font-size: 10pt;")
                            self.initRagSystemThread = threading.Thread(target=self.initializeRagSystem)
                            self.initRagSystemThread.start()
                        else:
                            self.statusLabel.setText("Gagal memulai Ollama. Silakan mulai secara manual.")
                            self.statusLabel.setStyleSheet("color: #ff0000; font-size: 10pt;")
                    except:
                        self.statusLabel.setText("Gagal memulai Ollama. Silakan mulai secara manual.")
                        self.statusLabel.setStyleSheet("color: #ff0000; font-size: 10pt;")
                except:
                    self.statusLabel.setText("Gagal memulai Ollama. Silakan mulai secara manual.")
                    self.statusLabel.setStyleSheet("color: #ff0000; font-size: 10pt;")
        except Exception as e:
            self.statusLabel.setText(f"Error: {str(e)}")
            self.statusLabel.setStyleSheet("color: #ff0000; font-size: 10pt;")
    
    def initializeRagSystem(self):
        try:
            # First check if Gemma 3 model is available in Ollama
            import requests
            import json
            import time
            
            try:
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    models = json.loads(response.text).get("models", [])
                    gemma3_available = any(model.get("name") == "gemma3" for model in models)
                    
                    if not gemma3_available:
                        self.statusLabel.setText("Mengunduh model Gemma 3. Ini mungkin memerlukan waktu...")
                        self.statusLabel.setStyleSheet("color: #ffaa00; font-size: 10pt;")
                        
                        # Pull the model
                        requests.post("http://localhost:11434/api/pull", 
                                      json={"name": "gemma3"},
                                      stream=True)
                        
                        # Wait for model to be ready (simple poll)
                        for _ in range(60):  # Wait up to 5 minutes
                            response = requests.get("http://localhost:11434/api/tags")
                            if response.status_code == 200:
                                models = json.loads(response.text).get("models", [])
                                if any(model.get("name") == "gemma3" for model in models):
                                    break
                            time.sleep(5)
            except Exception as e:
                self.statusLabel.setText(f"Error checking model: {str(e)}")
                self.statusLabel.setStyleSheet("color: #ff0000; font-size: 10pt;")
                time.sleep(3)
            
            self.statusLabel.setText("Memuat sistem RAG dengan Ollama Gemma 3...")
            self.rag_system = RAGSystem()
            # Update UI from main thread
            QApplication.instance().processEvents()
            self.statusLabel.setText("Sistem RAG siap!")
            self.statusLabel.setStyleSheet("color: #00ff00; font-size: 10pt;")
            # Hide status after 3 seconds
            threading.Timer(3.0, lambda: self.statusLabel.setVisible(False)).start()
        except Exception as e:
            self.statusLabel.setText(f"Error: {str(e)}")
            self.statusLabel.setStyleSheet("color: #ff0000; font-size: 10pt;")
    
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
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.label)
        layout.addWidget(self.recordButton)
        layout.addWidget(self.responseText)
        
        self.setStyleSheet("background-color: black;")
        self.setLayout(layout)
    
    def updateSpectrum(self, data):
        self.line.set_ydata(data)
        self.canvas.draw()
    
    def startListening(self):
        # Check if RAG system is ready
        if not hasattr(self, 'rag_system') or (hasattr(self, 'initRagSystemThread') and self.initRagSystemThread.is_alive()):
            self.responseText.setText("Sistem RAG sedang dimuat. Mohon tunggu...")
            return
        
        # Check if Ollama check is still running
        if hasattr(self, 'checkOllamaThread') and self.checkOllamaThread.is_alive():
            self.responseText.setText("Menunggu koneksi ke Ollama. Mohon tunggu...")
            return
            
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
        # Use local RAG processor instead of remote server
        self.rag_worker = RAGProcessorWorker(self.rag_system.get_chain(), text)
        self.rag_worker.response_signal.connect(self.displayResponse)
        self.rag_worker.start()
    
    def displayResponse(self, response):
        self.responseText.setText(response)
        self.tts_worker = TextToSpeechWorker(response)
        self.tts_worker.audio_ready.connect(self.startTTSVisualization)
        self.tts_worker.finished.connect(self.onTTSFinished)
        self.tts_worker.start()
    
    def startTTSVisualization(self, audio_data, sample_rate):
        self.canvas.setVisible(True)
        self.label.setText("🔊 Lab Expert Berbicara...")
        
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