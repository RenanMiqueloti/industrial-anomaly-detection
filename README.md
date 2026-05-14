# industrial-anomaly-detection

![CI](https://github.com/RenanMiqueloti/industrial-anomaly-detection/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)

**Detecção preditiva de falhas em rolamentos industriais com aprendizado não supervisionado.**

Modelos treinados **exclusivamente em dados saudáveis** — sem nenhum rótulo de falha — capazes de detectar degradação de rolamentos horas antes do colapso, com limiar calibrado por rolamento (≤ 1% de falsos alarmes por design).

> **Resultado principal — Bearing 1 (IMS/NASA Run 2):**
> AUC = **0.8735** (dataset completo) · TP = **69.0%** no período rotulado como degradado · FP = **1.0%** no período saudável de calibração · limiar = p99 dos scores saudáveis de cada rolamento.
>
> Os outros 3 rolamentos não têm falha documentada pelo paper. O dashboard os classifica em três tiers — **falha em progressão**, **anomalias recorrentes** (sinal acima do limite sem correspondência no ground truth) e **tendência estável** — em vez de carimbar "falha" em qualquer score que cruze o limiar.

---

## Sinal — saudável vs. falha

Antes de qualquer modelagem: a diferença entre um rolamento saudável e um com falha na pista externa é visível tanto no domínio do tempo (amplitude e impactos periódicos) quanto no espectro (elevação na faixa 2–5 kHz, região BPFO/BPFI).

![Comparação de sinal: Bearing 1 saudável vs. degradado — forma de onda e PSD](docs/assets/signal_overview.png)

As 11 features extraídas (7 domínio do tempo + 4 bandas espectrais) capturam exatamente essas diferenças — sem arquitetura profunda, apenas engenharia de features precisa aplicada ao domínio do problema.

---

## Score de anomalia ao longo do tempo

No test split (30% finais dos snapshots, cobrindo as últimas ~47h do run), o score do Bearing 1 fica acima do limiar de forma sustentada — todos os snapshots do test cruzam, dando 47h de lead time até a falha real. A regressão linear nos últimos 25% dos snapshots mostra a tendência crescente.

![Timeline do Bearing 1 — score de anomalia com 1ª detecção anotada](docs/assets/b1_timeline.png)

---

## Separabilidade dos scores por rolamento

Os rótulos do dataset seguem o ground truth do paper IMS Run 2: **somente o Bearing 1 falha (pista externa) no fim do período**. Os outros três permanecem saudáveis até o final — e portanto não têm classe degradada para calcular AUC.

![Distribuição de scores por rolamento — AUC por bearing](docs/assets/score_separability.png)

| Rolamento | AUC¹ | Estado no dashboard² | Condição real (paper) |
|---|---|---|---|
| **Bearing 1** | **0.8735** | 🔴 Falha em progressão | Falha documentada — pista externa (outer race) |
| Bearing 2 | N/A | 🟠 Anomalias recorrentes | Sem falha registrada |
| Bearing 3 | N/A | ✅ Tendência estável | Sem falha registrada |
| Bearing 4 | N/A | 🟠 Anomalias recorrentes | Sem falha registrada |

> ¹ AUC calculado nos 3.936 rows completos. Indefinido para B2/B3/B4 porque o paper não documenta classe degradada — recall/precision/F1 também caem com isso, e o dashboard troca os KPIs por taxa bruta de alerta nesses casos.
>
> ² Estado derivado da fração de snapshots recentes acima do limite p99: ≥ 60% → falha · ≥ 10% → recorrente · < 10% → estável. B2/B4 ficam acima do limite com frequência maior do que o nível de calibração (~1%) — provável drift operacional ou acoplamento via eixo com o B1, mas o paper não atribui falha a eles.

---

## Comparação de modelos

Três detectores de anomalia não supervisionados, treinados nos mesmos dados saudáveis e avaliados com intervalos de confiança bootstrap (1.000 reamostras):

| Modelo | ROC-AUC (IC 95%) | F1 (IC 95%) | Train time |
|---|---|---|---|
| IsolationForest | 0.930 [0.916, 0.943] | 0.667 [0.630, 0.701] | 0.14s |
| OC-SVM | 0.943 [0.929, 0.954] | 0.667 [0.629, 0.702] | 0.03s |
| AutoEncoder | 0.944 [0.931, 0.955] | 0.666 [0.628, 0.701] | 31.29s |

![Comparação de modelos — ROC-AUC e F1 com IC bootstrap](docs/assets/model_comparison.png)

> Os três modelos são estatisticamente equivalentes (CIs sobrepostos) com os rótulos do paper. IsolationForest é mantido como default pelo custo/benefício — score interpretável, SHAP exato via TreeExplainer, ~200× mais rápido que o AutoEncoder. A figura ainda mostra o LOF de uma rodada anterior (removido do benchmark) e será regenerada na próxima `make compare`.

---

## Pipeline

![Pipeline — do arquivo bruto ao dashboard em 6 passos](docs/assets/pipeline.png)

| Etapa | Comando | Saída |
|---|---|---|
| 1 · Download (Kaggle) | `make download` | `data/raw/2nd_test/` (984 arquivos, ~680 MB) |
| 2–3 · Features | `make features` | `data/features/features.parquet` (3.936 × 11) |
| 4 · Treino + limiares | `make train` | `results/iforest_model.joblib` + `threshold.json` |
| 5 · Benchmark 3 modelos | `make compare` | `results/comparison.parquet` + gráfico |
| 6 · Dashboard | `make dashboard` | Streamlit em `http://localhost:8502` |

---

## Tutorial — do zero aos resultados

### 1. Pré-requisito: Kaggle CLI

```bash
pip install kaggle
# Coloque ~/.kaggle/kaggle.json com suas credenciais
# Guia: https://www.kaggle.com/docs/api
```

### 2. Instalar

```bash
git clone https://github.com/RenanMiqueloti/industrial-anomaly-detection.git
cd industrial-anomaly-detection
make install       # pip install -e ".[dev]"
```

### 3. Baixar o dataset

```bash
make download
```

Baixa o [IMS/NASA Bearing Dataset](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset) (Run 2) via Kaggle CLI.

> **Sem Kaggle?** Rode `make demo` no lugar de `make download` — gera um conjunto sintético compacto (60 snapshots, 4 rolamentos) com a mesma estrutura do IMS Run 2, suficiente pra validar o pipeline ponta a ponta. Não substitui o dataset real para análise de resultados.

```
Downloading bearing-dataset.zip to data/raw
100%|████████████████| 680M/680M [02:14<00:00, 5.11MB/s]
IMS Run 2 dir: data/raw/2nd_test/2nd_test (984 arquivos)
```

### 4. Extrair features

```bash
make features
```

Cada snapshot (1 segundo a 20 kHz, 4 rolamentos simultâneos) gera um vetor de 11 features.
Resultado: parquet com 3.936 linhas × 11 features + metadados (`timestamp`, `bearing_id`, `y`).

```
INFO build_ims_features: 984 snapshots × 4 bearings = 3936 rows
INFO Feature matrix saved → data/features/features.parquet
```

### 5. Treinar e calibrar limiares por rolamento

```bash
make train
```

O IsolationForest é ajustado **exclusivamente nos snapshots saudáveis** (primeiros 40% do B1 + todos os snapshots de B2/B3/B4, que o paper documenta como saudáveis). O limiar de cada rolamento é o p99 dos seus próprios scores saudáveis. `make compare` também grava `ocsvm_bN` e `ae_bN` no `threshold.json` — o dashboard usa o limite específico do modelo selecionado.

```
INFO Temporal split: 2752 train rows, 1184 test rows (cutoff: 2004-02-17 05:12)
INFO Threshold bearing 1 (p99 healthy): 0.5422
INFO Threshold bearing 2 (p99 healthy): 0.5292
INFO Threshold bearing 3 (p99 healthy): 0.6503
INFO Threshold bearing 4 (p99 healthy): 0.5681
```

### 6. Benchmark dos 3 modelos

```bash
make compare
```

Treina, avalia e salva os 3 detectores. IC bootstrap com 1.000 reamostras.

### 7. Dashboard interativo

```bash
make dashboard
```

Abre em `http://localhost:8502`. Funcionalidades principais:

- **Hero banner** — estado em três tiers (`falha`, `recorrente`, `estável`) derivado da fração de snapshots recentes acima do limite. Para o B1: lead time, recall e taxa de falsos alarmes. Para B2/B4: nota explícita de "sem falha documentada pelo paper". Para B3: "sem anomalias significativas"
- **KPIs adaptativos** — quando o rolamento tem falha documentada (B1), mostra recall/F1/false alarms. Quando não tem (B2/B3/B4), mostra taxa bruta de flagged + status
- **Auto-diagnóstico** — parágrafo gerado automaticamente. Linguagem suaviza para B2/B4: relata fração flagged e feature dominante sem alegar "1ª detecção" ou "antecedência"
- **Separabilidade de scores** — histograma healthy vs. degraded por rolamento com AUC no título (N/A explícito para rolamentos sem classe degradada)
- **Timeline detalhada** — 7 dias de score. Projeção linear "🔮 Falha prevista em Xh" só dispara se o estado já é `falha` — evita projeção fantasma em rolamento saudável
- **Multi-bearing** — 4 rolamentos em paralelo com limiares individuais por rolamento E por modelo (cada um dos 3 modelos tem seu próprio p99 calibrado)
- **Inspeção de snapshot** — barra de z-score por feature + histograma de posição percentil. Badge "ACIMA DO LIMITE" (laranja) em vez de "ANOMALIA DETECTADA" (vermelho) quando o rolamento não tem falha documentada
- **Explicabilidade SHAP** — waterfall por snapshot sob demanda (TreeExplainer para IsolationForest, KernelExplainer para OC-SVM/AutoEncoder)

---

## Arquitetura técnica

```
industrial-anomaly-detection/
├── src/
│   ├── ingest.py          # parse de timestamps IMS, validação de layout Kaggle
│   ├── dataset.py         # build_ims_features → parquet com metadados
│   ├── features.py        # time-domain (7) + spectral bands (4) @ 20 kHz
│   ├── evaluate.py        # bootstrap_ci, plot_roc, plot_comparison
│   ├── compare.py         # benchmark 3 modelos + salva joblibs
│   ├── explain.py         # SHAP (TreeExplainer / KernelExplainer)
│   ├── cli.py             # download | features | train | eval | compare | explain
│   ├── dashboard.py       # Streamlit dashboard v2
│   └── models/
│       ├── iforest.py     # IForestDetector
│       ├── ocsvm.py       # OCSVMDetector
│       └── autoencoder.py # AutoEncoderDetector (PyTorch, early stopping)
├── tests/                 # 75+ testes (pytest + fixtures sintéticas)
├── docs/assets/           # figuras versionadas no repo
├── data/                  # gitignored — gerado pelo pipeline
├── results/               # gitignored — gerado pelo pipeline
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

---

## Features de vibração

Implementadas em [`src/features.py`](src/features.py).

**Domínio do tempo** (7 features):

| Feature | O que captura |
|---|---|
| `rms` | Energia total de vibração — degradação disseminada |
| `peak` | Amplitude máxima — impactos severos |
| `crest_factor` | Relação pico/RMS — impactos localizados incipientes |
| `kurtosis` | Impulsividade — fadiga de superfície localizada |
| `skewness` | Assimetria — dano direcional preferencial |
| `std` | Variabilidade — instabilidade mecânica |
| `p2p` | Amplitude pico-a-pico — folgas mecânicas |

**Energia espectral por banda** (4 features — Nyquist = 10 kHz @ 20 kHz):

| Feature | Frequência | O que indica |
|---|---|---|
| `band_0_500` | 0–500 Hz | Desbalanceamento, ressonâncias estruturais |
| `band_500_2000` | 500–2k Hz | Harmônicos fundamentais de defeito de rolamento |
| `band_2000_5000` | **2–5 kHz** | **Frequência característica de defeito de pista (BPFO/BPFI)** (dominante no B1, z=+129σ no pico) |
| `band_5000_10000` | 5–10 kHz | Dano avançado, impactos de esfera |

```python
from src.features import extract_all

feats = extract_all(window, fs=20_000)  # → dict[str, float], 11 chaves
```

---

## Dataset — IMS/NASA Bearing Run 2

| Campo | Detalhe |
|---|---|
| Origem | University of Cincinnati — NASA Prognostics Center of Excellence |
| Período | 12 a 19 de fevereiro de 2004 (≈ 7 dias de monitoramento contínuo) |
| Snapshots | 984 arquivos × 4 rolamentos = **3.936 linhas** no parquet |
| Taxa de amostragem | 20.000 Hz (Nyquist = 10 kHz) |
| Rolamento com falha | **Bearing 1** — pista externa (outer race fault) ao final do período |
| Timestamps | Reais — nome do arquivo `YYYY.MM.DD.HH.MM.SS` |
| Intervalo entre snapshots | ~10 minutos |
| Rótulos | Ground truth do paper: **B1** → y=1 nos últimos 60% (degradação documentada) · **B2/B3/B4** → y=0 sempre (saudáveis até o fim) |
| Split treino/teste | Temporal por timestamp único (70/30) — sem data leakage |

---

## Modelos

| Modelo | Ponto forte | Limitação |
|---|---|---|
| **IsolationForest** | Robusto em alta dimensão, rápido, TreeExplainer exato | Cortes axis-aligned perdem interações |
| **One-Class SVM** | Fronteiras não-lineares (kernel RBF) | Sensível a hiperparâmetros; quadrático em n |
| **AutoEncoder** | Erro de reconstrução codifica normalidade complexa | Pode overfitar com poucos dados saudáveis |

Todos treinados **sem rótulos de falha** e avaliados no mesmo conjunto de teste temporal.

---

## Decisões de design

**Features handcrafted, não waveform bruta.**
Em vibração de rolamentos com datasets na ordem de 10³–10⁴ janelas, features de domínio (RMS, curtose, energia espectral por banda) superam arquiteturas end-to-end. A interpretabilidade é um requisito, não uma concessão — engenheiros precisam entender por que o modelo alertou.

**Aprendizado não supervisionado por necessidade real.**
Em manutenção preditiva industrial, dados rotulados de falha são raros e caros. Treinar apenas em dados saudáveis é o único protocolo que escala para frotas de máquinas sem histórico rotulado.

**Limiar calibrado por rolamento, não global.**
Cada rolamento tem um nível basal de vibração diferente. Um limiar único subalerta rolamentos naturalmente mais ruidosos e sobre-alerta os silenciosos. O p99 dos scores saudáveis de cada bearing garante ≤ 1% de falsos alarmes **por rolamento**, por design.

**Split temporal, não aleatório.**
Embaralhar antes de dividir cria data leakage temporal (o modelo vê o futuro no treino). O split é feito por timestamp único — todos os bearings aparecem em treino e teste, e a ordem cronológica é preservada.

**IC bootstrap em todas as métricas.**
Métricas únicas sem intervalos de confiança são ruído em datasets pequenos. Cada número reportado inclui IC de 95% com 1.000 reamostras bootstrap.

---

## Reprodutibilidade

```bash
make install download features train compare
make dashboard
```

Todas as sementes aleatórias são fixas (`random_state=42`). Os resultados foram gerados a partir de um clone limpo sem nenhuma etapa manual além da configuração do `kaggle.json`.

---

## Documentação adicional

- **[Model Card](docs/MODEL_CARD.md)** — uso pretendido, métricas com IC, limitações conhecidas, considerações éticas.
- **[Dataset Datasheet](docs/DATASET.md)** — composição do IMS/NASA Run 2, processo de coleta, splits recomendados, licença.
