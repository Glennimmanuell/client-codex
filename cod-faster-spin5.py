import json
import re
import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import tempfile
import subprocess
import time
from gtts import gTTS
import threading
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama

def bersihkan_teks(text):
    """Clean and format response text"""
    try:
        filtered_response = re.sub(r'\*\s*', '', text)
        filtered_response = re.sub(r'^(\d+)\.\s*', r'\1. ', filtered_response, flags=re.MULTILINE)
        filtered_response = re.sub(r' +', ' ', filtered_response)
        filtered_response = re.sub(r'\n{3,}', '\n\n', filtered_response)
        return filtered_response.strip()
    except Exception as e:
        return "Terjadi kesalahan saat memproses teks."

class PiperTTSModel:
    """Piper TTS model handler"""
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
    """Google TTS model handler"""
    def __init__(self, lang='id', slow=False):
        self.lang = lang
        self.slow = slow
        
    def generate_speech(self, text):
        try:
            temp_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_mp3.close()
            
            tts = gTTS(text=text, lang=self.lang, slow=self.slow)
            tts.save(temp_mp3.name)
            
            # Read audio data using soundfile
            audio_data, sample_rate = sf.read(temp_mp3.name)
            
            os.unlink(temp_mp3.name)
            
            return audio_data, sample_rate
            
        except Exception as e:
            print(f"Error in gTTS: {e}")
            raise e

class RAGSystem:
    """Integrated RAG System"""
    def __init__(self):
        self.PERSIST_DIR = "./vector_db"
        self.DOCS_DIR = "./Docs"
        self.chat_engine = None
        self.init_rag()
    
    def init_rag(self):
        """Initialize RAG system"""
        try:
            print("🚀 Initializing RAG System...")
            
            # Configure LlamaIndex
            embed_model = FastEmbedEmbedding(model_name="intfloat/multilingual-e5-large")
            llm = Ollama(model="gemma2:latest", base_url="http://spin5.petra.ac.id:11434", request_timeout=120.0)
            Settings.llm = llm
            Settings.embed_model = embed_model
            
            # Check if Docs directory exists and has files
            if not os.path.exists(self.DOCS_DIR) or not os.listdir(self.DOCS_DIR):
                print(f"⚠️  Warning: No documents found in {self.DOCS_DIR}")
                print("Creating dummy index for testing...")
                # Create a minimal index for testing
                from llama_index.core import Document
                dummy_doc = Document(text="This is a test document for the RAG system.")
                index = VectorStoreIndex.from_documents([dummy_doc])
            else:
                # Load or create index
                if not os.path.exists(self.PERSIST_DIR):
                    print(f"📄 Loading documents from {self.DOCS_DIR}...")
                    documents = SimpleDirectoryReader(self.DOCS_DIR).load_data()
                    print(f"✅ Loaded {len(documents)} documents.")

                    print("🔨 Building VectorStoreIndex...")
                    index = VectorStoreIndex.from_documents(documents)
                    index.storage_context.persist(persist_dir=self.PERSIST_DIR)
                else:
                    print("📦 Loading existing VectorStoreIndex from storage...")
                    storage_context = StorageContext.from_defaults(persist_dir=self.PERSIST_DIR)
                    index = load_index_from_storage(storage_context)
            
            # Create chat engine
            self.chat_engine = index.as_chat_engine()
            print("✅ RAG System initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            self.chat_engine = None
    
    def convert_numbers_to_text(self, text):
        """Convert numbers in text to Indonesian words"""
        numbers_dict = {
            '0': 'nol', '1': 'satu', '2': 'dua', '3': 'tiga', '4': 'empat',
            '5': 'lima', '6': 'enam', '7': 'tujuh', '8': 'delapan', '9': 'sembilan',
            '10': 'sepuluh', '11': 'sebelas', '12': 'dua belas', '13': 'tiga belas',
            '14': 'empat belas', '15': 'lima belas', '16': 'enam belas',
            '17': 'tujuh belas', '18': 'delapan belas', '19': 'sembilan belas',
            '20': 'dua puluh', '30': 'tiga puluh', '40': 'empat puluh',
            '50': 'lima puluh', '60': 'enam puluh', '70': 'tujuh puluh',
            '80': 'delapan puluh', '90': 'sembilan puluh', '100': 'seratus',
            '1000': 'seribu', '1000000': 'satu juta', '1000000000': 'satu miliar'
        }
        
        def number_to_indonesian(num):
            if num == 0:
                return 'nol'
            
            if str(num) in numbers_dict:
                return numbers_dict[str(num)]
            
            # Handle 21-99
            if 21 <= num <= 99:
                tens = (num // 10) * 10
                ones = num % 10
                if ones == 0:
                    return numbers_dict[str(tens)]
                else:
                    return f"{numbers_dict[str(tens)]} {numbers_dict[str(ones)]}"
            
            # Handle 101-999
            if 101 <= num <= 999:
                hundreds = num // 100
                remainder = num % 100
                result = f"{numbers_dict[str(hundreds)]} ratus"
                if remainder > 0:
                    result += f" {number_to_indonesian(remainder)}"
                return result
            
            # Handle 1001-9999
            if 1001 <= num <= 9999:
                thousands = num // 1000
                remainder = num % 1000
                if thousands == 1:
                    result = "seribu"
                else:
                    result = f"{number_to_indonesian(thousands)} ribu"
                if remainder > 0:
                    result += f" {number_to_indonesian(remainder)}"
                return result
            
            # Handle 10000-999999
            if 10000 <= num <= 999999:
                thousands = num // 1000
                remainder = num % 1000
                result = f"{number_to_indonesian(thousands)} ribu"
                if remainder > 0:
                    result += f" {number_to_indonesian(remainder)}"
                return result
            
            # Handle 1000000+
            if num >= 1000000:
                millions = num // 1000000
                remainder = num % 1000000
                result = f"{number_to_indonesian(millions)} juta"
                if remainder > 0:
                    result += f" {number_to_indonesian(remainder)}"
                return result
            
            return str(num)
        
        def replace_number(match):
            number_str = match.group()
            try:
                clean_number = number_str.replace(',', '').replace('.', '')
                
                if '.' in number_str and number_str.count('.') == 1:
                    parts = number_str.split('.')
                    if len(parts[1]) <= 2:
                        integer_part = int(parts[0].replace(',', ''))
                        decimal_part = parts[1]
                        result = number_to_indonesian(integer_part)
                        if decimal_part != '0' and decimal_part != '00':
                            result += f" koma {' '.join([numbers_dict.get(d, d) for d in decimal_part])}"
                        return result
                
                num = int(clean_number)
                return number_to_indonesian(num)
            except ValueError:
                return number_str
        
        number_pattern = r'\b\d{1,3}(?:[,.]\d{3})*(?:\.\d{1,2})?\b|\b\d+\b'
        converted_text = re.sub(number_pattern, replace_number, text)
        
        return converted_text
    
    def get_response(self, prompt):
        """Get response from RAG system"""
        try:
            if not self.chat_engine:
                return "RAG system tidak tersedia. Silakan periksa konfigurasi."
            
            start_time = time.time()
            print(f"🤖 Processing prompt: {prompt[:50]}...")
            
            response = self.chat_engine.chat(prompt)
            response_text = str(response)
            
            # Convert numbers to text
            response_text = self.convert_numbers_to_text(response_text)
            
            # Remove unwanted parts for display
            filtered_response = re.sub(r"\*\*.*?\*\*|<think>.*?</think>", "", response_text).strip()
            
            end_time = time.time()
            print(f"⏱️  Response generated in {end_time - start_time:.2f} seconds")
            
            return response_text
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "Terjadi kesalahan saat memproses permintaan."

class IntegratedVoiceAssistant:
    def __init__(self):
        self.volume_boost = 2.0
        
        # Initialize RAG system
        print("🚀 Initializing RAG System...")
        self.rag_system = RAGSystem()
        
        # Initialize TTS models
        print("🚀 Initializing TTS models...")
        self.init_tts_models()
        
        # Default settings
        self.use_piper = True
        self.current_voice_model = None
    
    def init_tts_models(self):
        """Initialize available TTS models"""
        print("🔊 Initializing TTS models...")
        
        # Find voice models for Piper
        self.voice_models = self.find_voice_models()
        if self.voice_models and len(self.voice_models) > 0:
            self.current_voice_model = self.voice_models[0]
            print(f"✅ Found Piper voice model: {self.current_voice_model}")
            try:
                self.piper_model = PiperTTSModel(model_path=self.current_voice_model)
                print("✅ Piper TTS model initialized successfully")
            except Exception as e:
                print(f"❌ Error setting up Piper model: {e}")
                self.piper_model = None
        else:
            print("⚠️  No Piper voice models (.onnx files) found")
            self.piper_model = None
            self.current_voice_model = None
        
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
    
    def text_to_speech(self, text):
        """Convert text to speech and play it"""
        try:
            print("=" * 60)
            print("🔊 TTS PROCESSING START")
            print(f"🔧 Using: {'Piper TTS' if self.use_piper else 'Google TTS'}")
            
            tts_start_time = time.time()
            
            if self.use_piper and self.piper_model:
                print("🚀 Generating speech with Piper TTS...")
                synthesis_start = time.time()
                audio_data, sample_rate = self.piper_model.generate_speech(text, self.volume_boost)
                synthesis_end = time.time()
                print(f"⏱️  Piper synthesis time: {synthesis_end - synthesis_start:.2f} seconds")
            elif self.gtts_model:
                print("🌐 Generating speech with Google TTS...")
                synthesis_start = time.time()
                audio_data, sample_rate = self.gtts_model.generate_speech(text)
                synthesis_end = time.time()
                print(f"⏱️  gTTS synthesis time: {synthesis_end - synthesis_start:.2f} seconds")
            else:
                print("❌ No TTS model available")
                return
            
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
            
        except Exception as e:
            print(f"❌ Error during TTS generation: {e}")
            print("=" * 60)
    
    def show_settings(self):
        """Display current settings"""
        print("\n" + "=" * 50)
        print("⚙️  CURRENT SETTINGS")
        print("=" * 50)
        print(f"🔊 TTS Engine: {'Piper TTS' if self.use_piper else 'Google TTS'}")
        if self.use_piper and self.current_voice_model:
            print(f"🎤 Voice Model: {self.current_voice_model}")
        print(f"🔈 Volume Boost: {self.volume_boost}x")
        print(f"🤖 RAG System: {'Active' if self.rag_system.chat_engine else 'Inactive'}")
        print("=" * 50)
    
    def change_settings(self):
        """Interactive settings change"""
        while True:
            print("\n" + "⚙️" * 20)
            print("SETTINGS MENU")
            print("⚙️" * 20)
            print("1. Toggle TTS Engine (Piper/Google)")
            print("2. Change Voice Model (Piper only)")
            print("3. Change Volume Boost")
            print("4. Show Current Settings")
            print("5. Back to Main Menu")
            
            choice = input("\nPilih opsi (1-5): ").strip()
            
            if choice == "1":
                self.use_piper = not self.use_piper
                engine = "Piper TTS" if self.use_piper else "Google TTS"
                print(f"✅ TTS Engine changed to: {engine}")
                
            elif choice == "2":
                if not self.voice_models:
                    print("❌ No Piper voice models available")
                    continue
                    
                print("\nAvailable Voice Models:")
                for i, model in enumerate(self.voice_models, 1):
                    marker = " (current)" if model == self.current_voice_model else ""
                    print(f"{i}. {model}{marker}")
                
                try:
                    model_choice = int(input("Select model number: ")) - 1
                    if 0 <= model_choice < len(self.voice_models):
                        self.current_voice_model = self.voice_models[model_choice]
                        self.piper_model = PiperTTSModel(model_path=self.current_voice_model)
                        print(f"✅ Voice model changed to: {self.current_voice_model}")
                    else:
                        print("❌ Invalid selection")
                except (ValueError, IndexError):
                    print("❌ Invalid input")
                    
            elif choice == "3":
                try:
                    new_boost = float(input(f"Current volume boost: {self.volume_boost}x\nEnter new volume boost (e.g., 2.0): "))
                    if 0.1 <= new_boost <= 5.0:
                        self.volume_boost = new_boost
                        print(f"✅ Volume boost changed to: {self.volume_boost}x")
                    else:
                        print("❌ Volume boost must be between 0.1 and 5.0")
                except ValueError:
                    print("❌ Invalid number format")
                    
            elif choice == "4":
                self.show_settings()
                
            elif choice == "5":
                break
                
            else:
                print("❌ Invalid option")
    
    def run(self):
        """Main application loop"""
        print("🚀 Smart Lab Expert - Integrated Voice Assistant with RAG")
        print("=" * 70)
        print("Features:")
        print("⌨️  Terminal Input: Type your questions directly")
        print("🤖 Integrated RAG: Direct document-based AI responses")
        print("🔊 TTS Options: Piper TTS vs Google TTS")
        print("⏱️  Performance tracking with detailed timing")
        print("⚙️  Interactive settings menu")
        print("=" * 70)
        
        self.show_settings()
        
        while True:
            print("\n" + "🤖" * 20)
            print("MAIN MENU")
            print("🤖" * 20)
            print("1. Ask Question")
            print("2. Settings")
            print("3. Exit")
            
            choice = input("\nPilih opsi (1-3): ").strip()
            
            if choice == "1":
                question = input("\n📝 Masukkan pertanyaan Anda: ").strip()
                
                if not question:
                    print("❌ Pertanyaan tidak boleh kosong!")
                    continue
                
                print(f"\n📤 Memproses pertanyaan: {question}")
                
                # Get AI response from integrated RAG
                response = self.rag_system.get_response(question)
                
                # Clean and display response
                clean_response = bersihkan_teks(response)
                print("\n" + "💬" * 50)
                print("AI RESPONSE:")
                print("💬" * 50)
                print(clean_response)
                print("💬" * 50)
                
                # Ask if user wants TTS
                tts_choice = input("\n🔊 Play audio response? (y/n): ").strip().lower()
                if tts_choice in ['y', 'yes', 'ya']:
                    self.text_to_speech(clean_response)
                
            elif choice == "2":
                self.change_settings()
                
            elif choice == "3":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid option")

if __name__ == "__main__":
    assistant = IntegratedVoiceAssistant()
    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\n\n👋 Program terminated by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")