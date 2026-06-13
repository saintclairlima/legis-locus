print ('importando bibliotecas...')

import os
from dotenv import load_dotenv
import json
import datetime
import torch
import numpy as np
from pydub import AudioSegment
from pyannote.audio import Pipeline

# Carrega as variáveis do arquivo .env
load_dotenv()

# CONFIGURAÇÕES
HF_TOKEN = os.getenv("HF_TOKEN")
PASTA_VIDEOS = "./dados/videos/"
ARQUIVO_SAIDA_JSON = "resultado_diarizacao.json"

def carregar_pipeline():
    print("-> Carregando o modelo Pyannote v3.1...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", 
        token=HF_TOKEN
    )
    
    if torch.cuda.is_available():
        print("-> GPU detectada! Rodando com aceleração CUDA.")
        pipeline.to(torch.device("cuda"))
    else:
        print("-> Nenhuma GPU encontrada. Rodando na CPU.")
        
    return pipeline

def extrair_audio_do_video(caminho_arquivo):
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
        
    audio = AudioSegment.from_file(caminho_arquivo)
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    dados_brutos = np.array(audio.get_array_of_samples(), dtype=np.float32)
    dados_normalizados = dados_brutos / (2**15)
    
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    waveform = torch.FloatTensor(dados_normalizados).unsqueeze(0).to(dispositivo)
    
    return {
        "waveform": waveform,
        "sample_rate": audio.frame_rate
    }

def formatar_tempo_hms(segundos):
    """Converte segundos para o formato HH:MM:SS.mmm"""
    td = datetime.timedelta(seconds=segundos)
    # Garante o formato correto extraindo os componentes
    total_segundos = int(td.total_seconds())
    horas = total_segundos // 3600
    minutos = (total_segundos % 3600) // 60
    segs = total_segundos % 60
    milisegundos = int(round((segundos - total_segundos) * 1000))
    
    # Correção caso o arredondamento dos milisegundos passe de 1000
    if milisegundos >= 1000:
        segs += 1
        milisegundos -= 1000
        
    return f"{horas:02d}:{minutos:02d}:{segs:02d}.{milisegundos:03d}"

def agrupar_segmentos(resultado_pipeline):
    """Agrupa falas consecutivas do mesmo orador"""
    segmentos_brutos = []
    
    # Extrai os dados do Pyannote
    for segmento, _, speaker in resultado_pipeline.speaker_diarization.itertracks(yield_label=True):
        segmentos_brutos.append({
            "rotulo": f"[{speaker.upper()}]",
            "inicio": segmento.start,
            "fim": segmento.end
        })
        
    if not segmentos_brutos:
        return []
        
    # Algoritmo de agrupamento contíguo
    segmentos_agrupados = []
    segmento_atual = segmentos_brutos[0]
    
    for proximo_segmento in segmentos_brutos[1:]:
        # Se for o mesmo orador, estende o tempo de fim do bloco atual
        if proximo_segmento["rotulo"] == segmento_atual["rotulo"]:
            segmento_atual["fim"] = proximo_segmento["fim"]
        else:
            # Se mudou de orador, fecha o bloco atual e começa um novo
            segmentos_agrupados.append(segmento_atual)
            segmento_atual = proximo_segmento
            
    # Adiciona o último segmento processado
    segmentos_agrupados.append(segmento_atual)
    return segmentos_agrupados

def main():
    if HF_TOKEN == "xxxxxxxxxxxxxxx" or HF_TOKEN == "SEU_TOKEN_AQUI_HF":
        print("[ERRO] Altere a variável HF_TOKEN com o seu token do Hugging Face.")
        return

    # Lista os arquivos especificados ou todos os mp4 da pasta
    videos_alvo = ["vid_4147.mp4", "vid_4161.mp4", "vid_4166.mp4", "vid_4171.mp4", "vid_4187.mp4", "vid_4188.mp4"]
    
    pipeline = carregar_pipeline()
    if pipeline is None:
        print("[ERRO] Não foi possível baixar o modelo.")
        return

    dados_finais_json = {}

    for nome_arquivo in videos_alvo:
        caminho_completo = os.path.join(PASTA_VIDEOS, nome_arquivo)
        
        if not os.path.exists(caminho_completo):
            print(f"[AVISO] Arquivo pulado (não encontrado): {nome_arquivo}")
            continue
            
        print(f"\n-> Processando arquivo: {nome_arquivo}")
        
        try:
            # 1. Extração do Áudio
            print("   Extraindo e normalizando áudio...")
            dados_audio = extrair_audio_do_video(caminho_completo)
            
            # 2. Execução da IA
            print("   Rodando inteligência artificial de diarização...")
            resultado = pipeline(file=dados_audio)
            
            # 3. Agrupamento Contíguo das Falas
            print("   Agrupando falas contíguas do mesmo orador...")
            segmentos_agrupados = agrupar_segmentos(resultado)
            
            # 4. Formatação conforme especificação do JSON
            lista_formatada = []
            for seg in segmentos_agrupados:
                bloco = {
                    "rotulo": seg["rotulo"],
                    "tempo": {
                        "inicio": formatar_tempo_hms(seg["inicio"]),
                        "fim": formatar_tempo_hms(seg["fim"])
                    },
                    "tempo_segundos": {
                        "inicio": f"{seg['inicio']:.3f}s",
                        "fim": f"{seg['fim']:.3f}s"
                    }
                }
                lista_formatada.append(bloco)
                
            dados_finais_json[nome_arquivo] = lista_formatada
            print(f"   ✓ {nome_arquivo} processado com sucesso!")
            
        except Exception as e:
            print(f"[ERRO ao processar {nome_arquivo}]: {e}")

    # 5. Salvando o arquivo JSON final
    print(f"\n-> Gravando resultados em: {ARQUIVO_SAIDA_JSON}")
    with open(ARQUIVO_SAIDA_JSON, "w", encoding="utf-8") as f:
        json.dump(dados_finais_json, f, indent=4, ensure_ascii=False)
        
    print("-> Processo concluído com sucesso!")

if __name__ == "__main__":
    main()
