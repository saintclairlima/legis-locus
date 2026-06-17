import json
import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from pydub import AudioSegment
from speechbrain.inference.speaker import SpeakerRecognition

# Otimizações de hardware
load_dotenv()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def inicializar_modelo_speechbrain():
    """Carrega a configuração da GPU e inicializa o modelo ResNet do SpeechBrain."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Dispositivo em uso: {device}")

    print("-> Carregando o modelo de Embeddings do SpeechBrain...")
    # O from_hparams baixa e faz cache dos pesos na pasta especificada
    modelo = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-resnet-voxceleb",
        savedir="pretrained_models/spkrec-resnet-voxceleb",
        run_opts={"device": str(device)}
    )
    
    return modelo, device


def extrair_embeddings_sb(caminho_audio, modelo_inferencia, device):
    """Lê um áudio via PyDub e extrai o vetor de embeddings via SpeechBrain."""
    try:
        # 1. Carregamento e padronização (Mono, 16kHz são exigências do VoxCeleb)
        audio = AudioSegment.from_file(caminho_audio)
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # 2. Conversão para Float32 e formato Tensor PyTorch
        amostras = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        
        # O SpeechBrain espera [batch, time]. 
        # IMPORTANTE: O tensor DEVE ser enviado para o mesmo device do modelo.
        waveform = torch.from_numpy(amostras).unsqueeze(0).to(device)
        
        # 3. Inferência bloqueando o cálculo de gradientes para economizar VRAM
        with torch.no_grad():
            embedding_tensor = modelo_inferencia.encode_batch(waveform)
        
        # 4. Formatação da saída
        # O resultado vem como (1, 1, 256). Squeeze remove as dimensões de tamanho 1.
        # Depois, mandamos de volta para a CPU para converter em lista Python nativa.
        vetor_np = embedding_tensor.squeeze().cpu().numpy()
        
        if vetor_np.size == 0:
            return None
            
        return vetor_np.tolist()
        
    except Exception as e:
        print(f" -> Erro ao processar '{Path(caminho_audio).name}': {e}")
        return None


def processar_diretorio_sb(pasta_origem, arquivo_destino, modelo_inferencia, device):
    """Varre a pasta de áudios, extrai os embeddings e exporta para JSON."""
    pasta_raiz = Path(pasta_origem)
    arquivos_audio = list(pasta_raiz.glob("**/*.mp3")) + list(pasta_raiz.glob("**/*.wav"))
    total = len(arquivos_audio)
    
    if total == 0:
        print(f"\nALERTA: Nenhum áudio encontrado em '{pasta_raiz.absolute()}'")
        return

    print(f"\n-> Iniciando processamento de {total} arquivos...")
    dados_embeddings = {}

    for idx, caminho_arquivo in enumerate(arquivos_audio, 1):
        nome_video = caminho_arquivo.parent.name
        nome_amostra = caminho_arquivo.name
        
        print(f"[{idx:03d}/{total}] Extraindo (SpeechBrain): {nome_video}/{nome_amostra}", end=" ")
        
        vetor = extrair_embeddings_sb(str(caminho_arquivo), modelo_inferencia, device)
        
        if vetor:
            print("-> OK")
            dados_embeddings.setdefault(nome_video, {})[nome_amostra] = vetor
        else:
            print("-> FALHOU")

    Path(arquivo_destino).parent.mkdir(parents=True, exist_ok=True)
    
    with open(arquivo_destino, "w", encoding="utf-8") as arq:
        json.dump(dados_embeddings, arq)
        
    print(f"\n-> Concluído! {len(dados_embeddings)} vídeos mapeados no dicionário.")
    print(f"-> Arquivo salvo em: {Path(arquivo_destino).absolute()}")


# --- Execução Principal ---
if __name__ == "__main__":
    PASTA_AUDIOS = "dados/audios"
    # Salva com um nome diferente para não sobrescrever o do Pyannote
    JSON_SAIDA = "dados/embeddings_videos_speechbrain.json" 
    
    try:
        modelo_sb, hw_device = inicializar_modelo_speechbrain()
        processar_diretorio_sb(PASTA_AUDIOS, JSON_SAIDA, modelo_sb, hw_device)
    except KeyboardInterrupt:
        print("\n-> Processamento interrompido pelo usuário.")
    except Exception as erro_fatal:
        print(f"\n-> Falha fatal na execução: {erro_fatal}")