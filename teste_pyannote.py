import os
from dotenv import load_dotenv
import numpy as np
import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline

# Carrega as variáveis do arquivo .env
load_dotenv()

# CONFIGURAÇÕES
HF_TOKEN = os.getenv("HF_TOKEN")
CAMINHO_VIDEO = "./dados/videos/vid_4187.mp4"

def carregar_pipeline():
    print("-> Carregando o modelo Pyannote v3.1...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", 
        token=HF_TOKEN
    )
    
    # Verifica se há uma GPU disponível e joga o modelo para ela
    if torch.cuda.is_available():
        print("-> GPU detectada! Rodando com aceleração CUDA.")
        pipeline.to(torch.device("cuda"))
    else:
        print("-> Nenhuma GPU encontrada. Rodando na CPU.")
        
    return pipeline

def extrair_audio_do_video(caminho_arquivo):
    print(f"-> Extraindo áudio de {caminho_arquivo} usando FFmpeg...")
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
        
    # Pydub lê o MP4 de forma externa, sem quebrar o Python com DLLs do torchcodec
    audio = AudioSegment.from_file(caminho_arquivo)
    
    # O Pyannote exige áudio Mono em 16000Hz (16kHz)
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    # Transforma o áudio digitalizado em matriz numérica NumPy
    dados_brutos = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # Normaliza o áudio para a escala decimal [-1.0, 1.0] que o PyTorch trabalha
    dados_normalizados = dados_brutos / (2**15)
    
    # Formata como um Tensor de canais [1, amostras]
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    waveform = torch.FloatTensor(dados_normalizados).unsqueeze(0).to(dispositivo)
    
    return {
        "waveform": waveform,
        "sample_rate": audio.frame_rate
    }

def main():
    if HF_TOKEN == "SEU_TOKEN_AQUI_HF":
        print("[ERRO] Altere a variável HF_TOKEN com o seu token do Hugging Face.")
        return

    # 1. Prepara o áudio na memória
    try:
        dados_audio = extrair_audio_do_video(CAMINHO_VIDEO)
    except Exception as e:
        print(f"[ERRO ao ler vídeo]: {e}")
        return

    # 2. Inicializa o Pyannote
    pipeline = carregar_pipeline()
    if pipeline is None:
        print("[ERRO] Não foi possível baixar o modelo. Verifique o token ou a internet.")
        return

    print("-> Processando a diarização (isso pode levar alguns minutos)...")
    # Executa a inteligência artificial sobre os dados da memória
    resultado = pipeline(file=dados_audio)

    print("\n" + "="*30)
    print("       RESULTADOS DA DIARIZAÇÃO")
    print("="*30)
    
    # Varre as marcações encontradas e exibe na tela
    for segmento, _, speaker in resultado.speaker_diarization.itertracks(yield_label=True):
        inicio = segmento.start
        fim = segmento.end
        print(f"[{speaker}]: {inicio:.2f}s -> {fim:.2f}s")

if __name__ == "__main__":
    main()
