# INFNET-ANALISE DE FRAUDE CARTAO CREDITO
## Objetivos do projeto

### Objetivo principal:

**Classificação de transações como autênticas ou fraudulentas**. Para sermos precisos, dados os dados sobre **Time**, **Amount** e recursos transformados **V1** a **V28** para uma determinada transação, nossa meta é classificar corretamente a transação como **autêntica** ou **fraudulenta**. Empregamos diferentes técnicas para criar modelos de classificação e compará-los por meio de várias métricas de avaliação.

### Objetivos secundários:

Responder às seguintes perguntas usando ferramentas e técnicas de aprendizado de máquina e estatísticas.

- Quando uma transação fraudulenta é feita, ela é seguida logo por uma ou mais transações fraudulentas? Em outras palavras, os invasores fazem transações fraudulentas consecutivas em um curto espaço de tempo?


- O valor de uma transação fraudulenta é geralmente maior do que o de uma transação autêntica?


- Há alguma indicação nos dados de que as transações fraudulentas ocorrem em um período de alta transação?


- Os dados mostram que o número de transações é alto em alguns intervalos de tempo e baixo em outros. A ocorrência de fraudes está relacionada a esses intervalos de tempo?


- Há alguns pontos de tempo que exibem um número alto de transações fraudulentas. Isso se deve ao alto número de transações totais ou a algum outro motivo? 

**Nesta parte, classificaremos as transações como autênticas ou fraudulentas com base nas informações disponíveis sobre os recursos independentes (tempo, valor e as variáveis transformadas V1-V28). Um problema com o conjunto de dados é que ele é altamente desequilibrado em termos da variável-alvo Classe. Assim, corremos o risco de treinar os modelos com uma amostra representativa de transações fraudulentas de tamanho extremamente pequeno. Empregamos diferentes abordagens para lidar com esse problema. O desempenho de cada modelo é verificado por meio de várias métricas de avaliação e está resumido em uma tabela.**


Qualquer previsão sobre uma variável de destino categórica binária se enquadra em uma das quatro categorias:
- Verdadeiro positivo: O modelo de classificação prevê corretamente que o resultado é positivo
- True Negative (Verdadeiro negativo): O modelo de classificação prevê corretamente que o resultado será negativo.
- Falso positivo: O modelo de classificação prevê incorretamente que o output é positivo
- Falso negativo: O modelo de classificação prevê incorretamente que o resultado será negativo

| Estado real / Estado previsto $\rightarrow$ | Positivo | Negativo |
| :---: | :---: | :---: |
| Positivo | Verdadeiro positivo | Falso negativo | Negativo
| Negativo | Falso positivo | Verdadeiro negativo |

Deixe que **TP**, **TN**, **FP** e **FN** denotem, respectivamente, o número de **verdadeiros positivos**, **verdadeiros negativos**, **falso positivos** e **falso negativos** entre as previsões feitas por um determinado modelo de classificação. A seguir, apresentamos as definições de algumas métricas de avaliação com base nessas quatro quantidades.

$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Number of total predictions}} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Métrica de recuperação de precisão**

\begin{align*}
&\text{Precisão} = \frac{\text{Número de previsões positivas verdadeiras}}{\text{Número de previsões positivas totais}} = \frac{TP}{TP + FP}\\\\
&\text{Recall} = \frac{\text{Número de previsões positivas verdadeiras}}{\text{Número total de casos positivos}} = \frac{TP}{TP + FN}\\\\
&\text{Fowlkes-Mallows index (FM)} = \text{Geometric mean of Precision and Recall} = \sqrt{\text{Precision} \times \text{Recall}}\\\\
&F_1\text{-Score} = \text{Média harmônica de precisão e recuperação} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\\\\
&F_{\beta}\text{-Score} = \frac{\left(1 + \beta^2\right) \times \text{Precision} \times \text{Recall}}{\left(\beta^2 \times \text{Precision}\right) + \text{Recall}},
\end{align*}

em que $\beta$ é um fator positivo, escolhido de forma que o Recall seja $\beta$ vezes mais importante que a Precisão na análise. As escolhas populares de $\beta$ são $0,5$, $1$ e $2$.


- Métricas de sensibilidade-especificidade**

\begin{align*}
&\text{Sensibilidade} = \frac{\text{Número de previsões positivas verdadeiras}}{\text{Número total de casos positivos}} = \frac{TP}{TP + FN}\\\\
&\text{Especificidade} = \frac{\text{Número de previsões negativas verdadeiras}}{\text{Número total de casos negativos}} = \frac{TN}{TN + FP}\\\\
&\text{G-mean} = \text{Média geométrica de sensibilidade e especificidade} = \sqrt{\text{Sensibilidade} \times \text{Specificity}}
\end{align*}

- **Métricas de área sob a curva (AUC)**

Considere as seguintes quantidades:

\begin{align*}
&\text{True Positive Rate (TPR)} = \frac{\text{Number of true positive predictions}}{\text{Number of total positive cases}} = \frac{TP}{TP + FN}\\\\
&\text{False Positive Rate (FPR)} = \frac{\text{Number of false positive predictions}}{\text{Number of total negative cases}} = \frac{FP}{FP + TN}
\end{align*}

A curva ROC (Receiver Operating Characteristic, Característica de Operação do Receptor) é obtida plotando-se a TPR em relação à FPR para vários valores de probabilidade de limite. A área sob a curva ROC (ROC-AUC) serve como uma métrica de avaliação válida.

Da mesma forma, a curva Precision-Recall (PR) é obtida plotando-se a Precision em relação à Recall para um número de valores de probabilidade de limite. A área sob a curva PR (PR-AUC) também é uma métrica de avaliação válida. Outra métrica amplamente usada nesse sentido é a precisão média (Average Precision, AP), que é uma média ponderada de precisões em cada limite, com os pesos sendo o aumento na recuperação do limite anterior.

- **Outras métricas**

\begin{align*}&\text{Matthews Correlation Coefficient (MCC)} = \frac{\left(TP \times TN\right) - \left(FP \times FN\right)}{\sqrt{\left(TP + FP\right) \times \left(TP + FN\right) \times \left(TN + FP\right) \times \left(TN + FN\right)}}
\end{align*}

Diferentemente das métricas anteriores, **MCC** varia de $-1$ (pior cenário) a $1$ (melhor cenário: previsão perfeita).

Observe que **Recall** e **Sensibilidade** são essencialmente a mesma quantidade.

Entre as métricas discutidas, algumas boas opções para avaliar modelos, em especial para conjuntos de dados desequilibrados, são **MCC** e **$F_1$-Score**, enquanto **Precision** e **Recall** também fornecem informações úteis. Não daremos muita importância à métrica **Accuracy** neste projeto, pois ela produz conclusões errôneas quando as classes não estão equilibradas. No problema em questão, o falso negativo (uma transação fraudulenta classificada como autêntica) é mais perigoso do que o falso positivo (uma transação autêntica classificada como fraudulenta), pois, no primeiro caso, o fraudador pode causar mais danos financeiros, enquanto no segundo caso o banco pode verificar a autenticidade da transação do usuário do cartão depois de tomar as medidas necessárias para proteger o cartão. Considerando esse fato, damos ao **$F_2$-Score** uma importância especial na avaliação dos modelos.


