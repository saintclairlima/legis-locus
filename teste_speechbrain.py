import os
import torch
import soundfile as sf
import numpy as np
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

# 1. Otimização de threads
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"

print("Iniciando o download e carregamento do modelo (pode levar alguns instantes)...")

# 2. Instancia o modelo
verification = SpeakerRecognition.from_hparams(
    # source="speechbrain/spkrec-ecapa-voxceleb",
    # savedir="pretrained_models/spkrec-ecapa-voxceleb",
    source="speechbrain/spkrec-resnet-voxceleb",
    savedir="pretrained_models/spkrec-resnet-voxceleb",
    local_strategy=LocalStrategy.COPY
)

print("Modelo carregado com sucesso em CPU!")

print("Carregando arquivos de áudio e extraindo embeddings...")

# 3. Função alternativa para carregar áudio usando soundfile (evita dependência de k2)
def load_audio_alternative(path):
    """Carrega áudio usando soundfile e retorna tensor no formato esperado"""
    signal, sr = sf.read(str(path))
    if signal.ndim > 1:
        signal = signal.mean(axis=1)  # Converte stereo para mono
    return torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

# 4. Descobre todos os arquivos de áudio na pasta e extrai embeddings
audio_dir = "audios"
audio_files = sorted(
    [f for f in os.listdir(audio_dir) if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))]
)

if not audio_files:
    raise FileNotFoundError(f"Nenhum arquivo de áudio encontrado em {audio_dir}")

print(f"Encontrados {len(audio_files)} arquivos de áudio. Extraindo embeddings...")

embeddings = []
for filename in audio_files:
    path = os.path.join(audio_dir, filename)
    audio_tensor = load_audio_alternative(path)
    emb = verification.encode_batch(audio_tensor)
    embeddings.append(emb)
    print(f"  - {filename}")

print("\n--- Matriz de Similaridade ---")
from torch.nn.functional import cosine_similarity

def compute_similarity(emb1, emb2):
    emb1_flat = emb1.view(-1) if emb1.dim() > 1 else emb1
    emb2_flat = emb2.view(-1) if emb2.dim() > 1 else emb2
    return cosine_similarity(emb1_flat.unsqueeze(0), emb2_flat.unsqueeze(0)).item()

n = len(audio_files)
similarity_matrix = np.zeros((n, n), dtype=float)
for i in range(n):
    for j in range(n):
        similarity_matrix[i, j] = compute_similarity(embeddings[i], embeddings[j])

# Imprime a matriz com cabeçalho
header = ["      "] + [f"{name[:12]:12}" for name in audio_files]
print("\t".join(header))
for i, name in enumerate(audio_files):
    row_values = [f"{similarity_matrix[i, j]:.4f}" for j in range(n)]
    print(f"{name[:12]:12}\t" + "\t".join(row_values))

# Saída adicional: exibe as comparações de cada par
print("\n--- Comparações por par ---")
for i in range(n):
    for j in range(i + 1, n):
        print(f"{audio_files[i]} x {audio_files[j]} -> {similarity_matrix[i, j]:.4f}")
