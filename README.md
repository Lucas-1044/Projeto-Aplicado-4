# PROJETO APLICADO IV - PREVISAO DE SERIE TEMPORAL

## Previsao de passageiros aereos com ARIMA, Prophet e LSTM

---

## Resumo executivo

Este projeto compara tres abordagens para previsao de serie temporal:

- `ARIMA(1,1,1)`
- `Prophet`
- `LSTM`

O estudo utiliza a serie mensal de passageiros aereos entre janeiro de 1949 e dezembro de 1960, com 144 observacoes.

### Resultado principal

No conjunto de teste, o melhor desempenho foi obtido pelo `Prophet`, com menor `RMSE`.

---

## Base de dados

- Nome: Airline Passengers
- Periodo: janeiro de 1949 a dezembro de 1960
- Frequencia: mensal
- Total de registros: 144
- Variaveis: data e numero de passageiros

### Caracteristicas observadas

- Ausencia de valores faltantes
- Ausencia de registros duplicados
- Tendencia de crescimento bem definida
- Sazonalidade anual regular
- Aumento da amplitude ao longo do tempo

---

## Analise descritiva

### Estatisticas da serie

- Media: `280.30`
- Mediana: `265.50`
- Desvio padrao: `119.97`
- Coeficiente de variacao: `42.80%`
- Minimo: `104`
- Maximo: `622`
- Amplitude: `518`

### Interpretacao

A serie apresenta variabilidade elevada e crescimento consistente ao longo do periodo analisado. Esse comportamento justifica a avaliacao de modelos capazes de representar tendencia e sazonalidade.

---

## Qualidade dos dados

- Valores ausentes: `0`
- Registros duplicados: `0`

Conclusao: a base esta limpa e pronta para analise.

---

## Estacionariedade

### Teste ADF na serie original

- Estatistica de teste: `0.815369`
- P-value: `0.991880`

Conclusao: a serie original nao e estacionaria.

### Primeira diferenca

- Estatistica de teste: `-2.829267`
- P-value: `0.054213`

Conclusao: a primeira diferenca ficou muito proxima do limiar de 5%, mas ainda acima dele.

### Segunda diferenca

- Estatistica de teste: `-16.384232`
- P-value: `2.73e-29`

Conclusao: a serie se torna claramente estacionaria na segunda diferenca.

Observacao metodologica: o notebook compara os modelos com a configuracao `ARIMA(1,1,1)`, conforme definido no codigo original, mesmo com o diagnostico sugerindo maior atencao ao parametro de diferenciacao.

---

## ACF e PACF

- ACF: usada para indicar o componente de media movel (`q`)
- PACF: usada para indicar o componente autoregressivo (`p`)

Leitura adotada no projeto:

- `p = 1`
- `q = 1`

---

## Divisao treino e teste

- Treino: `115` observacoes
- Teste: `29` observacoes

---

## Modelos avaliados

### ARIMA(1,1,1)

Justificativa:

- modelo classico e interpretavel
- adequado para series com tendencia e dependencia temporal

Metrica obtida:

- MAE: `85.25`
- RMSE: `97.50`
- MAPE: `21.29%`

### Prophet

Justificativa:

- decomposicao automatica de tendencia e sazonalidade
- nao exige estacionariedade previa
- apropriado para padroes anuais regulares

Metrica obtida:

- MAE: `33.90`
- RMSE: `41.33`
- MAPE: `7.72%`

### LSTM

Justificativa:

- rede recorrente para sequencias
- capturar relacoes nao lineares
- usar janela temporal de 12 meses

Configuracao:

- `look_back = 12`
- `epochs = 50`
- `batch_size = 32`
- `Adam(learning_rate=0.001)`
- `Dropout = 0.2`

Metrica obtida na reproducao local do bloco neural:

- MAE: `72.01`
- RMSE: `80.17`
- MAPE: `17.35%`

---

## Comparacao dos modelos

| Modelo | MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|
| ARIMA(1,1,1) | 85.25 | 97.50 | 21.29 |
| Prophet | 33.90 | 41.33 | 7.72 |
| LSTM | 72.01 | 80.17 | 17.35 |

### Melhor modelo

O `Prophet` apresentou o menor erro nas tres metricas avaliadas e, por isso, foi o modelo mais adequado neste experimento.

---

## Discussao sintetica

- O `ARIMA(1,1,1)` oferece boa interpretabilidade, mas mostrou maior erro na previsao do trecho final da serie.
- O `Prophet` capturou com mais eficiencia a combinacao de tendencia e sazonalidade anual.
- O `LSTM` teve desempenho intermediario, mas ainda inferior ao `Prophet` neste conjunto reduzido de dados.

---

## Resultados esperados em graficos

Ao executar o notebook, sao geradas as seguintes figuras:

1. Serie temporal completa
2. Analise de distribuicao
3. Diferenciacao da serie
4. ACF e PACF
5. Divisao treino e teste
6. Resultado do ARIMA
7. Serie completa com Prophet
8. Componentes do Prophet
9. Prophet versus valores reais
10. Treinamento da LSTM
11. Previsao da LSTM
12. Comparacao entre modelos
13. Comparacao final das previsoes

---

## Conclusao

O projeto atende ao objetivo de comparar tres abordagens de previsao para uma serie temporal classica. Para esta base, o `Prophet` apresentou o melhor desempenho geral, seguido pela `LSTM` e, por ultimo, pelo `ARIMA(1,1,1)` na configuracao usada no notebook.
