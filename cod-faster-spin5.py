import json
import socket
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

class TerminalVoiceAssistant:
    def __init__(self):
        self.rag_server_host = "spin5.petra.ac.id"
        self.rag_server_port = 50001
        self.volume_boost = 2.0
        
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
    
    def get_ai_response(self, prompt):
        """Get response from RAG server"""
        try:
            print("🔄 Connecting to AI server...")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.rag_server_host, self.rag_server_port))
            
            request_data = json.dumps({"prompt": prompt}, ensure_ascii=False)
            client_socket.sendall(request_data.encode('utf-8') + b'\n')
            
            data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk

            response_json = json.loads(data.decode('utf-8', errors='ignore'))
            response_text = response_json.get("response", "").strip()
            
            return response_text
        except Exception as e:
            return f"Error connecting to server: {e}"
        finally:
            client_socket.close()
    
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
        print(f"🖥️  Server: {self.rag_server_host}:{self.rag_server_port}")
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
            print("4. Change Server Address")
            print("5. Show Current Settings")
            print("6. Back to Main Menu")
            
            choice = input("\nPilih opsi (1-6): ").strip()
            
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
                new_host = input(f"Current server: {self.rag_server_host}\nEnter new server address: ").strip()
                if new_host:
                    try:
                        new_port = int(input(f"Current port: {self.rag_server_port}\nEnter new port: "))
                        self.rag_server_host = new_host
                        self.rag_server_port = new_port
                        print(f"✅ Server changed to: {self.rag_server_host}:{self.rag_server_port}")
                    except ValueError:
                        print("❌ Invalid port number")
                        
            elif choice == "5":
                self.show_settings()
                
            elif choice == "6":
                break
                
            else:
                print("❌ Invalid option")
    
    def run(self):
        """Main application loop"""
        print("🚀 Smart Lab Expert - Terminal Voice Assistant")
        print("=" * 60)
        print("Features:")
        print("⌨️  Terminal Input: Type your questions directly")
        print("🔊 TTS Options: Piper TTS vs Google TTS")
        print("⏱️  Performance tracking with detailed timing")
        print("⚙️  Interactive settings menu")
        print("=" * 60)
        
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
                
                print(f"\n📤 Mengirim pertanyaan: {question}")
                
                # Get AI response
                response = self.get_ai_response(question)
                
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
    assistant = TerminalVoiceAssistant()
    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\n\n👋 Program terminated by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")