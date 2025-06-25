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

def measure_embedding_performance(embed_model, test_texts=None):
    """
    Mengukur performa embedding model
    """
    if test_texts is None:
        test_texts = [
            "Ini adalah contoh teks untuk pengujian embedding.",
            "Kecepatan embedding sangat penting untuk aplikasi real-time.",
            "Model embedding yang baik harus cepat dan akurat.",
            "Pengujian performa membantu memilih model terbaik.",
            "Benchmark embedding dilakukan dengan berbagai ukuran teks."
        ]
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK EMBEDDING MODEL: {embed_model.__class__.__name__}")
    print(f"{'='*60}")
    
    # Test embedding initialization time
    init_start = time.time()
    try:
        # Test dengan single text
        _ = embed_model.get_text_embedding(test_texts[0])
        init_end = time.time()
        print(f"✓ Initialization time: {init_end - init_start:.3f} seconds")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return None
    
    # Test single embedding speed
    single_start = time.time()
    try:
        embedding = embed_model.get_text_embedding(test_texts[0])
        single_end = time.time()
        embedding_dim = len(embedding) if embedding else 0
        single_time = single_end - single_start
        print(f"✓ Single text embedding: {single_time:.3f} seconds")
        print(f"✓ Embedding dimension: {embedding_dim}")
    except Exception as e:
        print(f"✗ Single embedding failed: {e}")
        return None
    
    # Test batch embedding speed
    batch_start = time.time()
    try:
        batch_embeddings = [embed_model.get_text_embedding(text) for text in test_texts]
        batch_end = time.time()
        batch_time = batch_end - batch_start
        avg_time_per_text = batch_time / len(test_texts)
        print(f"✓ Batch embedding ({len(test_texts)} texts): {batch_time:.3f} seconds")
        print(f"✓ Average per text: {avg_time_per_text:.3f} seconds")
        
        # Calculate throughput
        chars_total = sum(len(text) for text in test_texts)
        throughput_chars = chars_total / batch_time
        throughput_texts = len(test_texts) / batch_time
        
        print(f"✓ Throughput: {throughput_texts:.2f} texts/second")
        print(f"✓ Throughput: {throughput_chars:.0f} characters/second")
        
    except Exception as e:
        print(f"✗ Batch embedding failed: {e}")
        return None
    
    # Test dengan teks yang lebih panjang
    long_text = " ".join(test_texts * 10)  # Gabungkan semua teks jadi satu yang panjang
    long_start = time.time()
    try:
        _ = embed_model.get_text_embedding(long_text)
        long_end = time.time()
        long_time = long_end - long_start
        chars_per_sec = len(long_text) / long_time
        print(f"✓ Long text ({len(long_text)} chars): {long_time:.3f} seconds")
        print(f"✓ Long text throughput: {chars_per_sec:.0f} characters/second")
    except Exception as e:
        print(f"✗ Long text embedding failed: {e}")
    
    performance_stats = {
        'model_name': embed_model.__class__.__name__,
        'init_time': init_end - init_start,
        'single_embedding_time': single_time,
        'batch_time': batch_time,
        'avg_time_per_text': avg_time_per_text,
        'throughput_texts_per_sec': throughput_texts,
        'throughput_chars_per_sec': throughput_chars,
        'embedding_dimension': embedding_dim
    }
    
    print(f"{'='*60}\n")
    return performance_stats

# Konfigurasi model LlamaIndex dengan benchmark
print("Initializing embedding models...")

# Pilih model embedding (uncomment salah satu)
#embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
embed_model = FastEmbedEmbedding(model_name="intfloat/multilingual-e5-large")

# Jalankan benchmark
embedding_stats = measure_embedding_performance(embed_model)

# Simpan hasil benchmark ke file
if embedding_stats:
    benchmark_file = f"embedding_benchmark_{int(time.time())}.json"
    with open(benchmark_file, 'w') as f:
        json.dump(embedding_stats, f, indent=2)
    print(f"Benchmark results saved to: {benchmark_file}")

llm = Ollama(model="gemma2:latest", base_url="http://localhost:11434", request_timeout=120.0)
Settings.llm = llm
Settings.embed_model = embed_model

# Direktori penyimpanan index
PERSIST_DIR = "./vector_db"
DOCS_DIR = "./Docs"

# Cek apakah folder Docs/ kosong atau tidak ada
if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
    raise FileNotFoundError(f"Tidak ada dokumen di {DOCS_DIR}. Tambahkan file terlebih dahulu.")

# Load atau buat index dengan pengukuran waktu
if not os.path.exists(PERSIST_DIR):
    print(f"\n{'='*50}")
    print("BUILDING VECTOR INDEX")
    print(f"{'='*50}")
    
    # Load documents
    doc_start = time.time()
    print(f"Loading documents from {DOCS_DIR}...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    doc_end = time.time()
    print(f"✓ Loaded {len(documents)} documents in {doc_end - doc_start:.2f} seconds")
    
    # Calculate total text length
    total_chars = sum(len(doc.text) for doc in documents)
    print(f"✓ Total text length: {total_chars:,} characters")

    # Build index with timing
    index_start = time.time()
    print("Building VectorStoreIndex...")
    index = VectorStoreIndex.from_documents(documents)
    index_end = time.time()
    
    build_time = index_end - index_start
    chars_per_sec = total_chars / build_time if build_time > 0 else 0
    
    print(f"✓ VectorStoreIndex built in {build_time:.2f} seconds")
    print(f"✓ Processing speed: {chars_per_sec:.0f} characters/second")
    
    # Save index
    save_start = time.time()
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    save_end = time.time()
    print(f"✓ Index saved in {save_end - save_start:.2f} seconds")
    
    # Save indexing stats
    indexing_stats = {
        'embedding_model': embed_model.__class__.__name__,
        'document_count': len(documents),
        'total_characters': total_chars,
        'load_time': doc_end - doc_start,
        'build_time': build_time,
        'save_time': save_end - save_start,
        'total_time': save_end - doc_start,
        'processing_speed_chars_per_sec': chars_per_sec,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    stats_file = f"indexing_stats_{int(time.time())}.json"
    with open(stats_file, 'w') as f:
        json.dump(indexing_stats, f, indent=2)
    print(f"✓ Indexing stats saved to: {stats_file}")
    print(f"{'='*50}\n")
    
else:
    print("Loading existing VectorStoreIndex from storage...")
    load_start = time.time()
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    load_end = time.time()
    print(f"✓ Index loaded in {load_end - load_start:.2f} seconds")

# Membuat chat engine
chat_engine = index.as_chat_engine()
print("Chat engine initialized successfully!")


class DeepSeekRAGServer:
    def __init__(self, host='0.0.0.0', port=50001):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.query_stats = []  # Track query performance

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
            self.save_query_stats()
        finally:
            self.server_socket.close()

    def save_query_stats(self):
        if self.query_stats:
            stats_file = f"query_stats_{int(time.time())}.json"
            with open(stats_file, 'w') as f:
                json.dump(self.query_stats, f, indent=2)
            print(f"Query statistics saved to: {stats_file}")

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
                print(f"Received request: {request_str[:50]}...")

                try:
                    request = json.loads(request_str)
                    prompt = request.get("prompt", "")
                except json.JSONDecodeError:
                    prompt = request_str

                if prompt:
                    response_text, query_time = self.process_with_rag(prompt)
                    response_data = {"response": response_text, "processing_time": query_time}

                    response_json = json.dumps(response_data, ensure_ascii=False)
                    client_socket.sendall(response_json.encode("utf-8") + b"\n")
                    client_socket.shutdown(socket.SHUT_WR)
                    print(f"Response sent in {query_time:.2f}s: {response_text[:50]}...")
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
            
            if 21 <= num <= 99:
                tens = (num // 10) * 10
                ones = num % 10
                if ones == 0:
                    return numbers_dict[str(tens)]
                else:
                    return f"{numbers_dict[str(tens)]} {numbers_dict[str(ones)]}"
            
            if 101 <= num <= 999:
                hundreds = num // 100
                remainder = num % 100
                result = f"{numbers_dict[str(hundreds)]} ratus"
                if remainder > 0:
                    result += f" {number_to_indonesian(remainder)}"
                return result
            
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
            
            if 10000 <= num <= 999999:
                thousands = num // 1000
                remainder = num % 1000
                result = f"{number_to_indonesian(thousands)} ribu"
                if remainder > 0:
                    result += f" {number_to_indonesian(remainder)}"
                return result
            
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

    def process_with_rag(self, prompt):
        try:
            start_time = time.time()
            print(f"Processing prompt: {prompt}")
            
            response = chat_engine.chat(prompt)
            response_text = str(response)
            
            response_text = self.convert_numbers_to_text(response_text)
            filtered_response = re.sub(r"\*\*.*?\*\*|<think>.*?</think>", "", response_text).strip()

            end_time = time.time()
            query_time = end_time - start_time
            
            # Save query statistics
            query_stat = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'prompt_length': len(prompt),
                'response_length': len(response_text),
                'processing_time': query_time,
                'chars_per_second': len(response_text) / query_time if query_time > 0 else 0
            }
            self.query_stats.append(query_stat)
            
            print(f"Response generated in {query_time:.2f} seconds ({len(response_text)} chars)")
            return response_text, query_time
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Terjadi kesalahan saat memproses permintaan.", 0.0

if __name__ == "__main__":
    try:
        rag_server = DeepSeekRAGServer()
        rag_server.start_server()
    except Exception as e:
        print(f"Server startup error: {e}")
