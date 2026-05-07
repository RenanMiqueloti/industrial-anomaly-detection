# Datasheet — IMS/NASA Bearing Dataset (Run 2)

> Inspirado no template **Datasheets for Datasets** (Gebru et al., 2021).
> Referência: <https://arxiv.org/abs/1803.09010>

> Este documento descreve apenas o subconjunto **Run 2** usado neste projeto.
> Run 1 e Run 3 também estão disponíveis no dataset original e seguem padrão similar — ver "Composição".

---

## Motivação

### Para que o dataset foi criado?

O dataset foi gerado pelo **Center for Intelligent Maintenance Systems (IMS) da University of Cincinnati** com o objetivo de estudar prognóstico de degradação em rolamentos de esfera sob condições controladas até a falha real (run-to-failure). Os dados foram disponibilizados publicamente via **NASA Prognostics Center of Excellence Data Repository** para promover pesquisa em manutenção preditiva e benchmarking de algoritmos de detecção de anomalia.

### Quem criou e quem financiou?

- Pesquisadores do IMS Center / University of Cincinnati (Lee, Qiu, Yu, Lin et al.).
- Disponibilização pelo NASA Ames Prognostics Data Repository.

### Citação canônica

```
Lee, J., Qiu, H., Yu, G., & Lin, J. (2007). Bearing Data Set.
IMS, University of Cincinnati, NASA Ames Prognostics Data Repository.
```

---

## Composição

### O que cada instância representa?

Uma **instância** é um *snapshot* de 1 segundo de vibração amostrada simultaneamente em todos os rolamentos do banco de testes. No Run 2, cada snapshot tem:

- **20 480 amostras** (1 segundo @ 20 kHz)
- **4 colunas** (uma por rolamento)
- Formato: arquivo de texto, separado por tabulação, sem cabeçalho, sem timestamps internos
- Nome do arquivo: `YYYY.MM.DD.HH.MM.SS` (timestamp da coleta — fonte canônica do tempo)

### Quantas instâncias existem?

| Run | Período | Snapshots | Colunas | Falha real |
|---|---|---|---|---|
| Run 1 | 22/Out/2003 – 25/Nov/2003 (≈ 34 dias) | 2 156 | 8 (4 rolamentos × 2 sensores) | Bearing 3 (IR) + Bearing 4 (RE) |
| **Run 2** ← este projeto | 12/Fev/2004 – 19/Fev/2004 (≈ 7 dias) | **984** | **4 (1 sensor / rolamento)** | **Bearing 1 — outer race** |
| Run 3 | 04/Mar/2004 – 04/Abr/2004 (≈ 31 dias) | 6 324 | 4 | Bearing 3 (OR) |

Após extração de features pelo pipeline deste projeto:

- **3.936 linhas** de feature (984 snapshots × 4 bearings)
- **11 features** por linha (7 domínio do tempo + 4 bandas espectrais)
- **5 colunas de metadados**: `timestamp`, `bearing_id`, `y` (rótulo pseudo-supervisionado), `_meta_y`, `_meta_timestamp`

### O dataset contém todas as instâncias possíveis ou é uma amostra?

Cobre o experimento completo do Run 2 — desde a primeira aquisição até a parada do banco no fim da semana. Não é uma amostra — é o experimento na íntegra.

### Os dados estão associados a rótulos?

**Não no arquivo bruto.** O dataset original informa o modo de falha de cada rolamento (em PDFs e README), mas não fornece rótulos por janela. Este projeto adota convenção pseudo-supervisionada documentada na literatura:

- `y = 0`: primeiros **40%** dos snapshots → considerados saudáveis (período antes do início da degradação detectável)
- `y = 1`: restantes **60%** → considerados anômalos (período de degradação progressiva)

Esses rótulos são usados **apenas para avaliação do modelo**. O treinamento é não supervisionado — o modelo nunca vê `y`.

### Existe informação ausente?

- O dataset não documenta condições ambientais (temperatura, umidade) por snapshot.
- Não há registro fino do *exato* momento em que a falha começou — a literatura coloca em torno de 80% do experimento para o Run 2.
- A carga aplicada e a rotação são fixas (≈ 6.000 lbs / 2.000 RPM), declaradas globalmente, não por snapshot.

### Existem relações explícitas entre instâncias?

Sim — todas as instâncias são temporalmente ordenadas. Snapshots são tirados a cada ≈ 10 minutos durante 7 dias. Bearings 1-4 são fisicamente acoplados ao mesmo eixo, sob a mesma carga e rotação, então mostram correlação de carga embora cada um degrade de forma independente.

### Existem splits recomendados?

Sim — **temporal**, não aleatório. Embaralhar snapshots criaria leakage (o modelo veria amostras do futuro durante o treino). Este projeto usa split 70/30 por timestamp único, corte em **2004-02-16 16:52**.

### Erros, ruído, redundâncias?

- **Ruído elétrico** está presente nos sinais — esperado em ambiente industrial e parte do desafio.
- O dataset Kaggle tem aninhamento extra de pasta (`bearing-dataset/2nd_test/2nd_test/`) — `src/cli.py:_find_ims_run_dir()` lida com esse caso.
- Não há valores ausentes nos arquivos de Run 2.

---

## Coleta dos dados

### Como os dados foram adquiridos?

Sensores PCB 353B33 (acelerômetros piezoelétricos) montados em cada mancal. Aquisição via NI DAQCard-6062E a 20 kHz simultaneamente em todos os canais. O banco de testes foi um eixo com 4 rolamentos em série, acionado por motor AC a 2.000 RPM, com carga radial constante de 6.000 lbs aplicada hidraulicamente.

### Quem participou da coleta? Foram pagos?

Equipe de pesquisa do IMS Center. Pesquisa institucional — sem participantes humanos.

### Período da coleta

Run 2: 12 a 19 de fevereiro de 2004.

### Existe revisão ética / IRB?

Não aplicável — sem dados humanos.

---

## Pré-processamento, limpeza, rotulagem

### Os dados foram pré-processados antes da publicação?

Não pelos autores. Os arquivos disponibilizados são as séries cruas de aceleração.

### Pipeline aplicado por **este projeto**

1. **Ingest** (`src/ingest.py`): parse de timestamp do nome do arquivo, leitura tab-separated, validação de layout (20 480 × 4).
2. **Feature extraction** (`src/features.py`): 11 features por janela (RMS, peak, crest factor, kurtosis, skewness, std, p2p, energia em 4 bandas espectrais via Welch PSD).
3. **Dataset build** (`src/dataset.py`): empilha features de todos os bearings e snapshots, adiciona metadados, aplica rótulos pseudo-supervisionados pelo critério dos 40% iniciais.
4. **Output**: parquet em `data/features/features.parquet` (3.936 × 16 colunas).

### Os dados pré-processados estão disponíveis?

Não no repositório (gitignored — gerados pelo pipeline). Para reproduzir: `make install download features`.

---

## Uso

### Para que o dataset já foi usado?

Centenas de publicações em manutenção preditiva, prognóstico de RUL, fault classification e detecção de anomalia em rolamentos. Tornou-se um benchmark de fato no campo.

### Existe repositório de papers que usam este dataset?

A NASA mantém um pôster de citações no Prognostics Data Repository. Google Scholar > 1.000 citações para o dataset em conjunto.

### Existem tarefas para as quais o dataset não deve ser usado?

- **Benchmark de transfer learning entre máquinas reais.** O banco de testes do IMS é específico — generalização para frotas industriais reais requer validação adicional.
- **Diagnóstico de modos de falha não cobertos.** Run 2 só tem falha em pista externa; o dataset não diz nada sobre falha de gaiola, esferas ou lubrificação inadequada.
- **Validação de RUL (remaining useful life)**: o ponto exato de início da degradação não é conhecido com precisão; modelos de RUL treinados aqui herdam essa imprecisão.

---

## Distribuição

### Como o dataset é distribuído?

- **Canônico**: <https://data.nasa.gov/dataset/IMS-University-of-Cincinnati-Bearing-Dataset/3yud-nd96> (NASA PCoE Data Repository).
- **Mirror prático no Kaggle**: <https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset> (usado por este projeto via `kaggle datasets download`).

### Tamanho

- Run 2 zip: ≈ 680 MB compactado.
- Descompactado: ≈ 1.4 GB.

### Licença

Dados de origem governamental (NASA) — domínio público nos EUA. O mirror Kaggle herda essa condição. Sem restrição comercial conhecida; cite os autores ao publicar trabalhos baseados no dataset.

### DOI / identificador persistente

NASA Ames Prognostics Data Repository — sem DOI, mas o link `data.nasa.gov` é considerado persistente.

---

## Manutenção

### Quem mantém o dataset?

NASA Ames Prognostics Center of Excellence. O dataset foi publicado em 2007 e não recebe atualizações — é um snapshot histórico do experimento.

### Existem erratas ou versões?

Não. Este projeto trata os arquivos exatamente como distribuídos.

### O dataset pode ser estendido / aumentado?

Não pelos autores originais. Outros pesquisadores podem complementar com seus próprios runs, mas a comparação direta exige cuidado com banco de testes, sensor, carga e rotação.
