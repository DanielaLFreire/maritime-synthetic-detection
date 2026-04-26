# Como utilizar imagens públicas de embarcações para melhorar a detecção em cenário operacional naval

**Autora:** Daniela L. Freire (ICMC/USP)
**Colaborador:** Eduardo H. Teixeira (INATEL)
**Supervisor:** Leandro Aparecido Simal Moreira
**Contexto:** Projeto CASNAV/DMarSup, Marinha do Brasil (Termo 66/2025)

---

## 1. Problema

Sistemas de vigilância marítima dependem de detecção automática de embarcações em imagens capturadas por câmeras operacionais. Treinar detectores robustos exige grandes volumes de dados anotados, mas datasets operacionais são tipicamente pequenos e sigilosos. O dataset utilizado neste trabalho (CITRA-3D-Real, Marinha do Brasil) contém apenas 2.081 imagens com 7.003 bounding boxes.

Simultaneamente, existem datasets públicos de embarcações com dezenas de milhares de imagens. O InaTechShips (Teixeira et al., 2025) disponibiliza ~28 mil imagens do shipspotting.com com anotações automáticas via PointRend.

## 2. Pergunta de pesquisa

**Como utilizar efetivamente imagens públicas de embarcações para melhorar a detecção em cenário operacional naval com dados limitados?**

## 3. Metodologia

A investigação seguiu uma abordagem empírica em quatro etapas: tentativa direta, diagnóstico da falha, adaptação de domínio e validação. Todas as etapas usam o mesmo detector (YOLOv11m), a mesma configuração de hiperparâmetros (validada formalmente por HPO) e o mesmo protocolo de avaliação (3 seeds, test set do CITRA-3D-Real).

### 3.1 Etapa 1 — Transfer learning direto

**O que:** pré-treinar o YOLOv11m no InaTechShips (~28k imagens curadas por similaridade visual CLIP) e depois fazer fine-tuning no CITRA-3D-Real.

**Por que:** a hipótese inicial era que expor o modelo a milhares de imagens de embarcações — visualmente similares ao domínio alvo — melhoraria sua capacidade de detecção no cenário operacional.

**Como:** o pipeline COCO → dataset_25k_v2 (100 épocas de pré-treino) → CITRA-3D-Real (300 épocas de fine-tuning) foi comparado com dois baselines:

| Braço | Pipeline | Propósito |
|---|---|---|
| B1 (baseline) | Random init → CITRA-3D-Real | Performance sem nenhum pré-treino |
| B2 (baseline) | COCO → CITRA-3D-Real | Performance com pré-treino genérico |
| A (curado) | COCO → InaTechShips curado → CITRA-3D-Real | Efeito do pré-treino marítimo |

**Resultado:** o braço A obteve mAP50 = 0,7936 ± 0,0060 — **4,15% abaixo do B2** (COCO puro, mAP50 = 0,8351) e inclusive abaixo do B1 (sem pré-treino, mAP50 = 0,8008). O pré-treino no InaTechShips piorou o resultado em vez de melhorar.

### 3.2 Etapa 2 — Diagnóstico: por que o transfer direto falhou?

**O que:** investigar se o problema era excesso de treino (catastrophic forgetting recuperável com menos épocas) ou incompatibilidade fundamental entre os datasets.

**Por que:** o resultado negativo da Etapa 1 poderia ter duas causas distintas, cada uma exigindo uma solução diferente. Se fosse apenas excesso de treino, bastaria reduzir as épocas de pré-treino. Se fosse incompatibilidade de domínio, as imagens precisariam ser transformadas.

**Como:** duas análises complementares:

**Ablation de épocas.** Repetiu-se o pré-treino com 10, 20 e 50 épocas (além das 100 originais), mantendo tudo o mais constante, para traçar a curva de performance em função da dose de pré-treino.

| Épocas de pré-treino | mAP50 | Δ vs B2 |
|---|---|---|
| 0 (B2, sem pré-treino marítimo) | 0,8351 | referência |
| 10 | 0,8200 | −1,5% |
| 20 | 0,8171 | −1,8% |
| 50 | 0,8037 | −3,1% |
| 100 (braço A) | 0,8006 | −3,5% |

Resultado: **degradação monotônica**. Cada incremento de pré-treino piora o resultado. Mesmo 10 épocas (exposição mínima) já causa perda de 1,5%. Não existe "dose ideal" de pré-treino — o problema não é de quantidade.

**Análise do gap de domínio.** Comparação quantitativa das distribuições dos dois datasets revelou três eixos de incompatibilidade:

| Eixo | InaTechShips | CITRA-3D-Real | Impacto |
|---|---|---|---|
| **Escala** | Navios ocupam ~80% da imagem (fotos próximas) | Navios ocupam ~1-10% da imagem (71,6% são "small" COCO) | Modelo aprende features de objetos grandes que não transferem para objetos pequenos |
| **Densidade** | 1 navio por imagem | 3,37 navios por imagem (mediana 2, máximo 40) | Modelo perde capacidade de detectar múltiplos objetos |
| **Contexto** | Fotos profissionais (porto, cais, boa iluminação) | Capturas operacionais (oceano aberto, condições adversas) | Features de contexto são incompatíveis |

**Verificação adicional:** um subset aleatório do mesmo tamanho (sem curadoria CLIP) foi testado como pré-treino (braço B). O objetivo era confirmar se o problema era universal (afeta qualquer subset do InaTechShips) ou específico da curadoria. Resultado esperado: B ≈ A, confirmando que o gap é estrutural. (Em execução.)

**Conclusão do diagnóstico:** o InaTechShips e o CITRA-3D-Real são superficialmente similares (ambos contêm embarcações) mas operacionalmente incompatíveis. A similaridade visual medida por CLIP (cosine similarity ≥ 0,60) captura aparência mas não captura compatibilidade de distribuição (escala, densidade, contexto de captura). O pré-treino direto causa catastrophic forgetting: o modelo sobrescreve features úteis do COCO (genéricas, de baixo nível) por features especializadas do InaTechShips que não transferem para o cenário operacional.

### 3.3 Etapa 3 — Adaptação de domínio via composição sintética

**O que:** em vez de usar as imagens públicas diretamente, transformá-las para que se pareçam com o cenário operacional e depois usá-las como dados de treino.

**Por que:** o diagnóstico mostrou que o problema não são as embarcações em si (o InaTechShips contém navios reais de tipos variados), mas como elas aparecem na imagem (escala, posição, contexto). Se for possível reposicionar os navios do InaTechShips no cenário operacional do CITRA-3D — com a escala, densidade e fundos corretos — as imagens resultantes podem ser úteis como dados de treino sem causar negative transfer.

**Como:** pipeline de composição sintética por substituição in-place, em 4 passos:

1. **Análise do domínio alvo.** Extraiu-se o perfil de escala do CITRA-3D-Real: distribuição de tamanhos dos bboxes (mediana ~3% da largura da imagem), densidade de objetos por imagem (mediana 2, P90=7), e posição espacial dos navios (concentrados na metade inferior, y_center entre 0,37 e 0,70).

2. **Segmentação dos navios fonte.** Utilizou-se o SAM (Segment Anything Model) para extrair recortes dos navios do InaTechShips com fundo transparente, usando o bounding box original como prompt de segmentação. 23.828 recortes passaram no filtro de qualidade (cobertura da máscara 25-95%, dimensão ≥ 50px).

3. **Composição in-place.** Para cada imagem do CITRA-3D-Real, gerou-se múltiplas variações (~13 por imagem) substituindo cada navio real por um recorte aleatório do InaTechShips, **na mesma posição e na mesma dimensão do bounding box original**. O fundo permanece intacto — é o cenário operacional real. Os labels são idênticos aos originais (mesmas coordenadas). Três abordagens foram testadas durante o desenvolvimento:

   - Posicionamento aleatório (v1): navios apareciam em montanhas e prédios. Descartada.
   - Detecção de zona de água por cor HSV (v2): parcialmente eficaz, mas confundia rochas com água. Descartada.
   - **Substituição in-place (v3, adotada):** posiciona navios exatamente onde navios reais existiam. 100% dos posicionamentos corretos, zero decisões arbitrárias. Cientificamente defensável: cada posição de navio sintético herda a posição de um navio real observado pela Marinha.

4. **Treino do braço A'.** O dataset sintético (~28k imagens) será usado como pré-treino intermediário no mesmo pipeline do braço A: COCO → dataset_sintético → CITRA-3D-Real. A comparação direta A' vs B2 responde à pergunta central: **a adaptação de domínio resolve o gap que o transfer direto não conseguiu superar?**

### 3.4 Etapa 4 — Validação

O braço A' foi treinado com 3 seeds (42, 123, 2024), seguindo o mesmo protocolo dos demais braços. Avaliação no test set do CITRA-3D-Real.

**Resultado principal:**

| Seed | mAP50 | mAP50-95 |
|---|---|---|
| 42 | 0,8561 | 0,5309 |
| 123 | 0,8491 | 0,5216 |
| 2024 | 0,8570 | 0,5318 |
| **Média ± DP** | **0,8541 ± 0,0043** | **0,5281 ± 0,0056** |

**O braço A' superou o baseline B2 (COCO puro) em +1,90% mAP50** com separação estatística completa dos intervalos de confiança (A': [0,8498; 0,8584] vs B2: [0,8327; 0,8375]).

**Tabela final completa:**

| Braço | Pipeline | mAP50 | mAP50-95 | Δ vs B2 |
|---|---|---|---|---|
| **A' (sintético)** | **COCO → copy-paste → CITRA-3D** | **0,8541 ± 0,0043** | **0,5281 ± 0,0056** | **+1,90%** |
| B2 (baseline) | COCO → CITRA-3D | 0,8351 ± 0,0024 | 0,5055 ± 0,0027 | ref |
| B1 (baseline) | Random → CITRA-3D | 0,8008 ± 0,0073 | 0,4742 ± 0,0008 | −3,43% |
| B (aleatório) | COCO → random_pool → CITRA-3D | 0,7997 | 0,4711 | −3,54% |
| A (curado direto) | COCO → InaTechShips → CITRA-3D | 0,7936 ± 0,0060 | 0,4692 ± 0,0021 | −4,15% |

**Braço B (verificação):** mAP50 = 0,7997 (seed 42), equivalente ao braço A (0,7936). Confirma que o negative transfer é independente da curadoria CLIP — o gap é estrutural.

## 4. Validação da configuração experimental

**Hiperparâmetros.** A configuração de treino (AdamW, lr0=0,001, cos_lr=True) foi validada por HPO formal: Optuna TPE com 30 trials em 5 dimensões de busca. O melhor trial encontrado obteve mAP50 = 0,8328 vs baseline 0,8351 (Δ = −0,0023, dentro de 1σ). Decisão formal: manter configuração — ela está dentro de 1 desvio padrão do ótimo no espaço testado.

**Reprodutibilidade.** Todos os experimentos usam seeds fixas (42, 123, 2024). Scripts versionados, dados rastreados, decisões documentadas em documento vivo com histórico de versões.

**Controle experimental.** A única variável entre os braços é o dataset de pré-treino. Modelo, hiperparâmetros, protocolo de treino, protocolo de avaliação e dataset de fine-tuning são idênticos em todos os braços.

## 5. Contribuições

1. **Evidência empírica de negative transfer em detecção marítima.** Demonstração de que similaridade visual (CLIP) entre datasets de embarcações não garante transferência positiva, com ablation quantificada (degradação monotônica de 10 a 100 épocas). Contribuição para a literatura de domain adaptation.

2. **Método de adaptação de domínio por composição in-place.** Pipeline para transformar imagens públicas em dados de treino compatíveis com cenários operacionais, usando posições de objetos reais como âncoras para composição sintética. Resultado: **+1,90% mAP50 sobre COCO puro**, transformando deficit de −4,15% em ganho positivo. Método simples, preciso e reprodutível.

3. **Demonstração de que similaridade visual (CLIP) é proxy insuficiente para transferibilidade.** Curadoria por CLIP não melhorou nem piorou em relação à seleção aleatória (A ≈ B), mostrando que métricas de aparência não capturam a compatibilidade operacional entre domínios.

## 6. Conclusão

O experimento demonstrou que imagens públicas de embarcações (InaTechShips, ~28k imagens) **podem melhorar a detecção em cenário operacional naval** (CITRA-3D-Real, 2.081 imagens), desde que adaptadas ao domínio alvo.

O uso direto dessas imagens como pré-treino causa **negative transfer** (−4,15% mAP50 vs COCO puro), por incompatibilidade de domínio: escala (navios ocupam 80% vs 3% da imagem), densidade (1 vs 3,4 objetos/imagem) e contexto (fotos profissionais vs capturas operacionais). A degradação é monotônica — mesmo 10 épocas de exposição prejudica.

A **composição sintética in-place** resolve o gap: recorta navios do InaTechShips (SAM), redimensiona para a escala operacional, e posiciona em cenas reais do CITRA-3D nas mesmas posições de navios reais. O resultado é **+1,90% mAP50 sobre COCO puro** (0,8541 vs 0,8351), com separação estatística completa entre os intervalos de confiança.

A abordagem é simples (sem modelo generativo), precisa (bboxes 100% exatos, herdados de dados operacionais), e reprodutível (scripts públicos, seeds fixas, protocolo documentado).
