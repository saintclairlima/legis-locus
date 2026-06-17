import json
import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from pydub import AudioSegment
from pyannote.audio import Model, Inference

# Otimizações de hardware
load_dotenv()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def inicializar_modelo_pyannote():
    """Carrega as variáveis de ambiente, configura a GPU e inicializa o modelo."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Dispositivo em uso: {device}")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("ERRO: Token HF_TOKEN não encontrado no arquivo .env")

    print("-> Carregando o modelo de Embeddings...")
    embedding_model = Model.from_pretrained("pyannote/embedding", token=hf_token)
    embedding_model.to(device)
    
    return Inference(embedding_model, window="whole")


def extrair_embeddings(caminho_audio, modelo_inferencia):
    """Lê um áudio via PyDub e extrai o vetor de embeddings via Pyannote."""
    try:
        # 1. Carregamento e padronização (Mono, 16kHz)
        audio = AudioSegment.from_file(caminho_audio)
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # 2. Conversão para Float32 e formato Tensor PyTorch
        amostras = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        waveform = torch.from_numpy(amostras).unsqueeze(0)
        
        # 3. Inferência
        audio_in_memory = {"waveform": waveform, "sample_rate": 16000}
        embedding_objeto = modelo_inferencia(audio_in_memory)
        
        # 4. Extração segura e padronização da saída
        if hasattr(embedding_objeto, "data"):
            vetor = embedding_objeto.data
        elif hasattr(embedding_objeto, "cpu"):
            vetor = embedding_objeto.cpu().numpy()
        else:
            vetor = embedding_objeto
            
        vetor_np = np.array(vetor)
        
        if vetor_np.size == 0:
            return None
            
        return vetor_np.flatten().tolist()
        
    except Exception as e:
        print(f" -> Erro ao processar '{Path(caminho_audio).name}': {e}")
        return None


def processar_diretorio(pasta_origem, arquivo_destino, modelo_inferencia):
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
        
        # Formatando o output para ficar limpo (ex: [001/514])
        print(f"[{idx:03d}/{total}] Extraindo: {nome_video}/{nome_amostra}", end=" ")
        
        vetor = extrair_embeddings(str(caminho_arquivo), modelo_inferencia)
        
        if vetor:
            print("-> OK")
            # setdefault cria o sub-dicionário se não existir, substituindo o antigo bloco "if"
            dados_embeddings.setdefault(nome_video, {})[nome_amostra] = vetor
        else:
            print("-> FALHOU")

    # Garante que a pasta de destino exista antes de tentar salvar
    Path(arquivo_destino).parent.mkdir(parents=True, exist_ok=True)
    
    with open(arquivo_destino, "w", encoding="utf-8") as arq:
        json.dump(dados_embeddings, arq)
        
    print(f"\n-> Concluído! {len(dados_embeddings)} vídeos mapeados no dicionário.")
    print(f"-> Arquivo salvo em: {Path(arquivo_destino).absolute()}")


# --- Execução Principal ---
if __name__ == "__main__":
    PASTA_AUDIOS = "dados/audios"
    JSON_SAIDA = "dados/embeddings_videos_pyannote.json"
    
    try:
        modelo = inicializar_modelo_pyannote()
        processar_diretorio(PASTA_AUDIOS, JSON_SAIDA, modelo)
    except KeyboardInterrupt:
        print("\n-> Processamento interrompido pelo usuário.")
    except Exception as erro_fatal:
        print(f"\n-> Falha fatal na execução: {erro_fatal}")