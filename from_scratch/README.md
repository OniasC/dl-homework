# Trabalho "From Scratch" de RNN, GRU e LSTM

Foram usados os seguintes artigos do Kaggle como referência para este trabalho:


https://www.kaggle.com/code/fareselmenshawii/rnn-from-scratch

https://www.kaggle.com/code/fareselmenshawii/lstm-from-scratch

https://www.kaggle.com/code/fareselmenshawii/gru-from-scratch

Para LSTM, também foi usada a referência https://d2l.ai/, que contém um tópico sobre LSTM from scratch.

## Datasets e Modelos usados

Os artigos apresentam um modelo de geração de texto a partir de um *prompt* inicial. RNN, GRU e LSTM usam textos diferentes que seram detalhados em cada sessão.

Como texto adicional, também foram usados dois livros em português de domínio público de Machado de Assis, adquiridos através desse site: https://machado.mec.gov.br/obra-completa-lista/itemlist/category/23-romance

Além dos livros de Machado de Assis, também foi usado um dataset de futebol de robôs para regressão de dados, ao invés de classificação. Uma descrição desse dataset está nos notebooks com _robot em seus nomes.


## RNN from scratch

#### Classificação
[rnn.ipynb](rnn.ipynb) utiliza [dinos.txt](dinos.txt) para gerar nomes de dinossauro.

#### Regressão

[rnn_robot.ipynb](rnn_robot.ipynb) utiliza o dataset da RoboCin [data/quadrado_opt_X_Y.csv](../data/quadrado_opt_1_1.csv) para tentar estimar dados de posição do robô. Ao não convergir, tentou-se ter o mesmo resultado usando o mesmo modelo, porém pronto de uma biblioteca estabelecida, para averiguar se o problema era o modelo ou sua implementação. Isso pode ser visto em [rnn_robot_pytorch.ipynb](rnn_robot_pytorch.ipynb), que também não convergiu, indicando que o problema está em outro ponto.

## LSTM from scratch

#### Classificação

[lstm.ipynb](lstm.ipynb) utiliza [text.txt](text.txt) para gerar *output* de texto.

Como o treinamento de classificação demorou muito, também usou-se o exemplo da d2l.ia, em [lstm_pytorch.ipynb](lstm_pytorch.ipynb). Ele usa como base de treino [data/timemachine.txt](../data/timemachine.txt). Também foram usados como referência para teste, os textos [data/quincas_borba.txt](../data/quincas_borba.txt) e [data/memorias_postumas.txt](../data/memorias_postumas.txt)

#### Regressão

[lstm_robot.ipynb](lstm_robot.ipynb) utiliza o dataset da RoboCin [data/quadrado_opt_X_Y.csv](../data/quadrado_opt_1_1.csv) para tentar estimar dados de posição do robô.

## GRU from scratch

Procedimento bastante similar ao de LSTM.

#### Classificação

[gru.ipynb](gru.ipynb) utiliza [text.txt](text.txt) para gerar *output* de texto. Também foram usados como referência para teste, os textos [data/quincas_borba.txt](../data/quincas_borba.txt) e [data/memorias_postumas.txt](../data/memorias_postumas.txt)

#### Regressão

[gru_robot.ipynb](gru_robot.ipynb) utiliza o dataset da RoboCin [data/quadrado_opt_X_Y.csv](../data/quadrado_opt_1_1.csv) para tentar estimar dados de posição do robô.
