# Processo de seleção dos arquivos

Foram identificados a lista com os 24 parlamentares vigentes.

Foram procurados vídeos no LegisVideos de forma que pudesse haver uma amostra de fala de cada deputada ou deputado. 

Assim, 22 deputados foram identificados nos vídeos de id 4161, 4166, 4171, 4187 e 4188, salvo Dr. Bernardo e Ivanilson Oliveira.

Foi encontrada fala de Dr. Bernardo no vídeo de id 4147, sendo salvo no recorte do vídeo correspondente ao tempo (00:00:39 -> 00:08:58).

Não foi encontrada fala registrada do Deputado Ivanilson Oliveira, que ficou fora do primeiro teste 

# Identificação de falas

Usando o script em `teste_pyannote.py`, foram marcados os trechos de fala de cada parlamentar, e sua rotulação, com minutagem de início e final de falas, salvo em `resultado_diarizacao.json`.

# Coisas a fazer:

- Segmentar os vídeos para gerar amostras de áudio para uso posterior (especialmente na fase de treinamento do classificador com base em embeddings) 

- Testar armazenamento da "assinatura de fala" de embeddings interna, mantida pelo pyannote (verificar tamanho, formato e modelo usado - documentação)

- Testar representação de embeddings usando o modelo do speechbrain

- Comparar a acurácia de predição dos dois modelos de embeddings (tanto por distância do cosseno quanto via classificador)