import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path


def limpar_rotulo(rotulo):
    """Remove caracteres como '[' e ']' do rótulo para evitar problemas no nome do arquivo."""
    return rotulo.replace("[", "").replace("]", "")


def extrair_segmentos_audio(json_path, pasta_videos, pasta_saida_base):
    # Carrega os dados do JSON
    with open(json_path, "r", encoding="utf-8") as f:
        dados_videos = json.load(f)

    # Garante que a pasta base de saída exista
    Path(pasta_saida_base).mkdir(parents=True, exist_ok=True)

    # Iterar por cada vídeo no JSON
    for nome_video, falas in dados_videos.items():
        caminho_video = os.path.join(pasta_videos, nome_video)

        # Verifica se o vídeo realmente existe na pasta informada
        if not os.path.exists(caminho_video):
            print(f"Aviso: Vídeo {nome_video} não encontrado em {pasta_videos}. Pulando...")
            continue

        # 1. Cria a pasta com o nome do vídeo (removendo a extensão .mp4 do nome da pasta)
        nome_pasta_video = Path(nome_video).stem
        pasta_destino_video = os.path.join(pasta_saida_base, nome_pasta_video)
        Path(pasta_destino_video).mkdir(parents=True, exist_ok=True)

        print(f"\nProcessando: {nome_video} -> {pasta_destino_video}")

        # Dicionário para controlar a numeração sequencial por rótulo nesta pasta
        contador_rotulos = defaultdict(int)

        # 2. Processa cada momento de fala
        for fala in falas:
            rotulo = limpar_rotulo(fala["rotulo"])

            # Captura o tempo inicial e final em segundos (removendo o 's' do final)
            inicio_fala = float(fala["tempo_segundos"]["inicio"].replace("s", ""))
            fim_fala = float(fala["tempo_segundos"]["fim"].replace("s", ""))

            duracao_fala = fim_fala - inicio_fala

            # b. Calcula quantos pedaços inteiros de 30 segundos cabem neste bloco
            # Pedaços menores que 30s no final serão naturalmente descartados pelo range
            num_pedacos = int(duracao_fala // 30)

            for i in range(num_pedacos):
                # Calcula o tempo exato de início e fim do fragmento dentro do vídeo original
                inicio_fragmento = inicio_fala + (i * 30)

                # Incrementa o contador do rótulo atual para gerar o sufixo (01, 02, etc.)
                contador_rotulos[rotulo] += 1
                numero_sequencial = f"{contador_rotulos[rotulo]:02d}"

                # c. Define o nome do arquivo final de saída (.mp3)
                nome_arquivo_saida = f"{rotulo}.{numero_sequencial}.mp3"
                caminho_saida_completo = os.path.join(
                    pasta_destino_video, nome_arquivo_saida
                )

                # Comando FFmpeg:
                # -ss: ponto de partida no vídeo original
                # -t: duração do corte (30 segundos)
                # -vn: desativa o vídeo (extrai apenas áudio)
                # -q:a 2: define qualidade excelente para o MP3 (VBR)
                comando_ffmpeg = [
                    "ffmpeg",
                    "-y",  # Sobrescreve o arquivo se já existir
                    "-ss",
                    str(inicio_fragmento),
                    "-t",
                    "30",
                    "-i",
                    caminho_video,
                    "-vn",
                    "-q:a",
                    "2",
                    caminho_saida_completo,
                ]

                # Executa o comando omitindo o output do terminal para não poluir o console
                subprocess.run(
                    comando_ffmpeg,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        print(f"Finalizado: {nome_video}. Total de fragmentos gerados: {dict(contador_rotulos)}")


# --- Execução do Script ---
if __name__ == "__main__":
    # Configure os caminhos do seu projeto aqui
    CAMINHO_JSON = "resultado_diarizacao.json"  # Substitua pelo nome real do seu arquivo JSON
    PASTA_VIDEOS = "dados/videos"
    PASTA_AUDIOS = "dados/audios"

    extrair_segmentos_audio(CAMINHO_JSON, PASTA_VIDEOS, PASTA_AUDIOS)