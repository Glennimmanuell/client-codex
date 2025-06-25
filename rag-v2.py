import socket
import json
import os
import re
import threading
import time
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama

# Konfigurasi model LlamaIndex
#embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
embed_model = FastEmbedEmbedding(model_name="intfloat/multilingual-e5-large")
llm = Ollama(model="gemma2:latest", base_url="http://localhost:11434", request_timeout=120.0)
Settings.llm = llm
Settings.embed_model = embed_model

# Direktori penyimpanan index
PERSIST_DIR = "./vector_db"
DOCS_DIR = "./Docs"

# Konfigurasi removed - konversi angka dilakukan di post-processing

# Cek apakah folder Docs/ kosong atau tidak ada
if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
    raise FileNotFoundError(f"Tidak ada dokumen di {DOCS_DIR}. Tambahkan file terlebih dahulu.")

# Load atau buat index
if not os.path.exists(PERSIST_DIR):
    print(f"Loading documents from {DOCS_DIR}...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print(f"Loaded {len(documents)} documents.")

    print("Building VectorStoreIndex...")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    print("Loading existing VectorStoreIndex from storage...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Membuat chat engine
chat_engine = index.as_chat_engine()
print("Chat engine initialized successfully!")


class DeepSeekRAGServer:
    def __init__(self, host='0.0.0.0', port=50001):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def start_server(self):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"DeepSeek RAG Server is running on {self.host}:{self.port}")

        try:
            while True:
                client_socket, client_address = self.server_socket.accept()
                print(f"Accepted connection from {client_address}")
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            print("Server shutting down...")
        finally:
            self.server_socket.close()

    def handle_client(self, client_socket):
        try:
            data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break

            if data:
                request_str = data.decode("utf-8").strip()
                print(f"Received request: {request_str[:50]}...")  # DEBUG

                try:
                    request = json.loads(request_str)
                    prompt = request.get("prompt", "")
                except json.JSONDecodeError:
                    prompt = request_str  # Jika tidak dalam format JSON, anggap sebagai string

                if prompt:
                    response_text = self.process_with_rag(prompt)
                    response_data = {"response": response_text}

                    # Kirim response ke client
                    response_json = json.dumps(response_data, ensure_ascii=False)
                    client_socket.sendall(response_json.encode("utf-8") + b"\n")
                    client_socket.shutdown(socket.SHUT_WR)  # Pastikan pengiriman selesai
                    print(f"Response sent to client: {response_text[:50]}...")  # DEBUG
                else:
                    error_response = json.dumps({"error": "Empty prompt received"}, ensure_ascii=False)
                    client_socket.sendall(error_response.encode("utf-8") + b"\n")
                    client_socket.shutdown(socket.SHUT_WR)
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()

    def convert_numbers_to_text(self, text):
        """Konversi angka dalam teks ke bentuk kata-kata bahasa Indonesia"""
        # Dictionary untuk konversi angka dasar
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
            """Konversi angka ke bahasa Indonesia"""
            if num == 0:
                return 'nol'
            
            if str(num) in numbers_dict:
                return numbers_dict[str(num)]
            
            # Handle angka 21-99
            if 21 <= num <= 99:
                tens = (num // 10) * 10
                ones = num % 10
                if ones == 0:
                    return numbers_dict[str(tens)]
                else:
                    return f"{numbers_dict[str(tens)]} {numbers_dict[str(ones)]}"
            
            # Handle angka 101-999
            if 101 <= num <= 999:
                hundreds = num // 100
                remainder = num % 100
                result = f"{numbers_dict[str(hundreds)]} ratus"
                if remainder > 0:
                    result += f" {number_to_indonesian(remainder)}"
                return result
            
            # Handle angka 1001-9999
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
            
            # Handle angka 10000-999999
            if 10000 <= num <= 999999:
                thousands = num // 1000
                remainder = num % 1000
                result = f"{number_to_indonesian(thousands)} ribu"
                if remainder > 0:
                    result += f" {number_to_indonesian(remainder)}"
                return result
            
            # Handle angka 1000000 ke atas (sederhana)
            if num >= 1000000:
                millions = num // 1000000
                remainder = num % 1000000
                result = f"{number_to_indonesian(millions)} juta"
                if remainder > 0:
                    result += f" {number_to_indonesian(remainder)}"
                return result
            
            return str(num)  # fallback
        
        # Regex untuk menangkap angka (termasuk desimal dan ribuan dengan koma/titik)
        def replace_number(match):
            number_str = match.group()
            try:
                # Handle angka dengan koma sebagai pemisah ribuan
                clean_number = number_str.replace(',', '').replace('.', '')
                
                # Cek apakah mengandung desimal (titik setelah digit)
                if '.' in number_str and number_str.count('.') == 1:
                    parts = number_str.split('.')
                    if len(parts[1]) <= 2:  # kemungkinan desimal
                        integer_part = int(parts[0].replace(',', ''))
                        decimal_part = parts[1]
                        result = number_to_indonesian(integer_part)
                        if decimal_part != '0' and decimal_part != '00':
                            result += f" koma {' '.join([numbers_dict.get(d, d) for d in decimal_part])}"
                        return result
                
                # Handle integer
                num = int(clean_number)
                return number_to_indonesian(num)
            except ValueError:
                return number_str  # kembalikan original jika tidak bisa dikonversi
        
        # Pattern untuk menangkap berbagai format angka
        number_pattern = r'\b\d{1,3}(?:[,.]\d{3})*(?:\.\d{1,2})?\b|\b\d+\b'
        converted_text = re.sub(number_pattern, replace_number, text)
        
        return converted_text

    def process_with_rag(self, prompt):
        try:
            start_time = time.time()
            print(f"Processing prompt: {prompt}")  # DEBUG
            
            response = chat_engine.chat(prompt)
            response_text = str(response)
            
            # Konversi angka ke bentuk teks
            response_text = self.convert_numbers_to_text(response_text)

            # Hapus bagian yang tidak perlu dari output terminal (tetap ada dalam respons ke klien)
            filtered_response = re.sub(r"\*\*.*?\*\*|<think>.*?</think>", "", response_text).strip()

            print(f"Filtered response (hidden special parts): {filtered_response}")  # DEBUG

            end_time = time.time()
            print(f"Response generated in {end_time - start_time:.2f} seconds")

            return response_text  # Kirim respons asli tanpa filter ke klien
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Terjadi kesalahan saat memproses permintaan."

if __name__ == "__main__":
    try:
        rag_server = DeepSeekRAGServer()
        rag_server.start_server()
    except Exception as e:
        print(f"Server startup error: {e}")
