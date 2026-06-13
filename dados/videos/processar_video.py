import csv
import subprocess
from pathlib import Path
from datetime import timedelta


VIDEO_FILE = "./dados/videos/vid_4187.mp4"
CSV_FILE = "./dados//videos/vid_4187.csv"
OUTPUT_DIR = "./dados/audios/4187"
SEGMENT_DURATION = 30  # segundos


def parse_time(time_str):
    h, m, s = time_str.split(":")
    return timedelta(
        hours=int(h),
        minutes=int(m),
        seconds=float(s)
    ).total_seconds()


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.3f}"


Path(OUTPUT_DIR).mkdir(exist_ok=True)

with open(CSV_FILE, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        nome = row["nomeOrador"].strip()
        inicio = parse_time(row["tempoInicial"])
        fim = parse_time(row["tempoFinal"])

        duracao_total = fim - inicio

        qtd_segmentos = int(duracao_total // SEGMENT_DURATION)

        for i in range(qtd_segmentos):
            seg_inicio = inicio + i * SEGMENT_DURATION

            nome_arquivo = (
                f"{nome.replace(' ', '_')}_{i+1:03d}.mp3"
            )

            caminho_saida = str(
                Path(OUTPUT_DIR) / nome_arquivo
            )

            cmd = [
                "./ffmpeg/bin/ffmpeg.exe",
                "-y",
                "-ss",
                format_time(seg_inicio),
                "-t",
                str(SEGMENT_DURATION),
                "-i",
                VIDEO_FILE,
                "-vn",
                "-q:a",
                "0",
                "-map",
                "a",
                caminho_saida,
            ]

            print("Executando:", " ".join(cmd))
            subprocess.run(cmd, check=True)

print("Concluído.")