# Model Card — IsolationForest para detecção de falha em rolamentos

> Inspirado no template **Model Cards for Model Reporting** (Mitchell et al., 2019).
> Referência: <https://arxiv.org/abs/1810.03993>

---

## Detalhes do modelo

| Campo | Valor |
|---|---|
| Nome | `IForestDetector` (modelo principal) |
| Versão | 0.1.0 |
| Algoritmo base | `sklearn.ensemble.IsolationForest` (n_estimators=100, contamination='auto') |
| Modelos comparados | `OCSVMDetector`, `LOFDetector`, `AutoEncoderDetector` (PyTorch) |
| Tipo de aprendizado | Não supervisionado (treinamento exclusivamente em janelas saudáveis) |
| Entradas | Vetor de 11 features extraídas de janela de vibração (1 segundo @ 20 kHz) |
| Saída | Score de anomalia ∈ ℝ; flag binário `score > threshold[bearing]` |
| Limiar | p99 dos scores saudáveis **por rolamento** (≤ 1% de falsos alarmes por bearing) |
| Random state | 42 (todos os modelos) |
| Treinado em | IMS/NASA Run 2, primeiros 40% dos snapshots (período saudável) |
| Avaliado em | Os 60% restantes (split temporal, sem leakage) |
| Linguagem / framework | Python 3.12, scikit-learn 1.5+, PyTorch 2.2+ (apenas para AutoEncoder) |
| Licença | MIT |
| Mantenedor | Renan Miqueloti — `<renanmiqueloti@gmail.com>` |

---

## Uso pretendido

### Casos de uso primários

- **Manutenção preditiva industrial** — detecção precoce de degradação progressiva em rolamentos a partir de sinais de acelerômetro montados próximos ao mancal.
- **Triagem em frota de máquinas** — sinalizar quais ativos merecem inspeção manual prioritária quando rótulos de falha não estão disponíveis.
- **Demonstração técnica e baseline** — ponto de partida razoável antes de investir em modelos mais complexos ou rotulagem manual.

### Usuários pretendidos

Engenheiros de confiabilidade, técnicos de manutenção, cientistas de dados e engenheiros de ML trabalhando com sinais de vibração de máquinas rotativas.

### Casos de uso fora do escopo

- **Diagnóstico clínico ou decisões de segurança crítica sem human-in-the-loop.** O modelo emite alertas, não diagnósticos definitivos. Toda decisão de parar ou trocar um equipamento deve incluir verificação por engenheiro qualificado.
- **Transferência cega para máquinas com características de operação muito diferentes** (rotação, carga, geometria do rolamento) — ver "Limitações" abaixo.
- **Identificação do tipo exato de falha** (BPFO vs. BPFI vs. BSF vs. cage). O modelo detecta *anomalia*, não classifica modo de falha. As feature bands fornecem indícios físicos, mas a atribuição requer análise espectral fina dedicada.
- **Predição de tempo até falha (RUL — Remaining Useful Life).** O dashboard projeta uma estimativa heurística por regressão linear nos últimos snapshots; isso é um indicador grosseiro, não um modelo de RUL validado.

---

## Fatores e desempenho

### Métricas (Bearing 1 — falha documentada de pista externa)

| Métrica | Valor | IC 95% (bootstrap, 1.000 reamostras) |
|---|---|---|
| ROC-AUC (dataset completo, 3.936 linhas) | 0.8705 | reportado em `results/comparison.parquet` |
| Antecedência da 1ª detecção | 47 horas | (relativa ao último snapshot monitorado) |
| Falsos alarmes em janelas saudáveis | ≤ 1% por design (limiar = p99 healthy) | calibrado por bearing |

### AUC por rolamento

| Bearing | AUC | Condição real |
|---|---|---|
| Bearing 1 | 0.8705 | Falha documentada (outer race) |
| Bearing 2 | 0.7676 | Sem falha registrada |
| Bearing 3 | 0.6174 | Sem falha registrada |
| Bearing 4 | 0.8034 | Sem falha registrada |

> Bearings 2-4 não tiveram falha real — o AUC nesses casos compara janelas iniciais vs. tardias do *mesmo* rolamento saudável. Valores moderados ali são consistentes com leve drift de operação ao longo de 7 dias.

### Comparação dos 4 modelos

Os quatro detectores foram treinados nos mesmos dados saudáveis e avaliados no mesmo conjunto de teste temporal. O ranking detalhado com IC bootstrap está em `results/comparison.parquet` e visualizado em `docs/assets/model_comparison.png`. IsolationForest foi escolhido como modelo principal pelo equilíbrio entre desempenho, velocidade de inferência e disponibilidade de explicabilidade exata via `TreeExplainer`.

---

## Dados de treinamento

Ver [`docs/DATASET.md`](DATASET.md) para o datasheet completo do conjunto IMS/NASA Bearing Run 2.

Pontos críticos:

- **Período**: 12-19 de fevereiro de 2004 (≈ 7 dias contínuos).
- **Volume**: 984 snapshots × 4 rolamentos = 3.936 linhas de feature; cada snapshot é 1 segundo @ 20 kHz (20 480 amostras).
- **Rótulos pseudo-supervisionados**: y=0 nos primeiros 40% dos snapshots (período saudável documentado), y=1 no restante. Os rótulos servem **apenas para avaliação** — o modelo é treinado sem usá-los.
- **Split**: temporal por timestamp único (70/30). Embaralhar antes de dividir vazaria o futuro no treino.

---

## Avaliação

- **Conjunto de teste**: últimos 30% dos snapshots (cutoff: 2004-02-16 16:52).
- **Métrica primária**: ROC-AUC com IC bootstrap (95%, 1.000 reamostras).
- **Métricas secundárias**: F1, precisão, recall, taxa de falsos alarmes.
- **Comparação**: IsolationForest, One-Class SVM (kernel RBF), LOF (`novelty=True`), AutoEncoder denso (PyTorch, early stopping em validação).
- **Reprodutibilidade**: `random_state=42` em todos os modelos. Pipeline completo verificável via `make install download features train compare`.

### Drift detection

Após o treino, o pipeline calcula PSI (Population Stability Index) entre o conjunto de referência (treino) e novas janelas. Relatório em `results/drift_report.json`.

---

## Considerações éticas

Manutenção preditiva é um caso de uso de baixo risco direto a pessoas — o modelo não toma decisões sobre indivíduos. Ainda assim, alguns pontos merecem atenção:

- **Custo de falso negativo**: deixar passar uma falha incipiente pode gerar parada não programada, prejuízo financeiro ou risco de segurança em equipamentos pesados. O design "≤ 1% de falsos alarmes por bearing" prioriza precisão; aplicações com tolerância menor a falso negativo devem **reduzir o limiar** e aceitar mais falsos positivos.
- **Custo de falso positivo**: parada para inspeção sem necessidade. Em frotas grandes, o agregado de falsos positivos drena recursos de manutenção.
- **Viés de transferência**: o modelo aprendeu a "normalidade" de rolamentos específicos do IMS Run 2. Aplicar diretamente em outras máquinas sem recalibração resultará em alertas espúrios ou silêncio falso. **Sempre re-treine ou ao menos re-calibre o limiar com dados saudáveis da máquina alvo.**
- **Mão de obra**: alertas sem verificação humana podem desencadear paradas operacionais. O modelo é uma ferramenta de triagem, não substitui a avaliação do engenheiro.

---

## Limitações conhecidas

1. **Domínio restrito**: validado apenas em rolamentos de esfera de classe industrial em regime aproximadamente estacionário (rotação ≈ constante). Variação significativa de carga/rotação degrada o desempenho — features como RMS misturam efeito de carga com efeito de defeito.
2. **Modo de falha único validado**: o Bearing 1 do Run 2 falhou na pista externa. Falhas de pista interna, esfera, gaiola, lubrificação ou desalinhamento podem apresentar assinatura espectral diferente — o modelo provavelmente detecta degradação genérica, mas a taxa de detecção varia por modo.
3. **Sensibilidade a posição do sensor**: features dependem da função de transferência mecânica entre o defeito e o acelerômetro. Mudar de posição do sensor sem recalibrar invalida limiares.
4. **Dataset pequeno** (3.936 linhas, 4 rolamentos, 1 falha real). IC bootstrap mostra incerteza considerável em F1 e recall. Resultados são **indicadores**, não garantias.
5. **Assume 20 kHz de taxa de amostragem.** Para outras taxas, os limites de banda (`band_2000_5000`, etc.) precisam ser revistos relativamente à frequência de Nyquist e às frequências características do rolamento alvo.
6. **AutoEncoder pode overfitar**: com poucas amostras saudáveis (≈ 1.500 vetores de feature), redes com mais capacidade do que a configurada começam a memorizar o ruído. Early stopping mitiga, mas não elimina.

---

## Recomendações de uso

- Antes de aplicar em uma máquina nova: colete ≥ 1 semana de dados saudáveis, recalcule features com a mesma `fs`, re-fit o modelo, recalibre o limiar p99 com dados da máquina alvo.
- Mantenha um pipeline de **drift monitoring** — quando PSI > 0.2 nas features dominantes, re-treine.
- Combine com inspeção manual periódica nos primeiros 30 dias após deploy para validar a taxa de falsos alarmes em campo.
- Para falha catastrófica iminente (deterioração rápida em horas), considere baixar o limiar para p95 ou p90 — aceita mais falsos positivos em troca de antecedência maior.
- Não use o score como entrada direta para decisão automática de parada de equipamento sem human-in-the-loop.

---

## Como reproduzir

```bash
git clone https://github.com/RenanMiqueloti/industrial-anomaly-detection.git
cd industrial-anomaly-detection
make install download features train compare
```

Ou, sem Kaggle:

```bash
make install demo features train
```

---

## Citação

```
@software{miqueloti2024iad,
  author = {Miqueloti, Renan},
  title = {industrial-anomaly-detection: unsupervised bearing fault detection},
  year = {2026},
  url = {https://github.com/RenanMiqueloti/industrial-anomaly-detection}
}
```
