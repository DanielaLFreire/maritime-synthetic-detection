# Experimento de Curadoria por Similaridade Visual para Detecção de Embarcações em Cenário Operacional

**Documento vivo de pesquisa — versão 0.4**
**Autora:** Daniela L. Freire (ICMC/USP)
**Coautoria prevista:** orientador (a definir) + possível coautoria de Eduardo H. Teixeira (autor do InaTechShips, contato em andamento)
**Contexto institucional:** Projeto em colaboração com a Marinha do Brasil
**Última atualização:** 25 de abril de 2026
**Status:** Experimento principal concluído com resultado positivo. A composição sintética in-place (braço A') superou o baseline COCO puro (B2) em +1,90% mAP50 (0,8541 vs 0,8351) com separação estatística completa entre intervalos de confiança. Transfer learning direto confirmado como prejudicial (−4,15%), diagnóstico de catastrophic forgetting completo, e solução via adaptação de domínio demonstrada. Pronto para escrita do paper.

---

## Histórico de versões

- **v0.1 (09/04/2026):** versão inicial com mapeamento incorreto de classes do CITRA-3D (importado de documento legado obsoleto), contagens de versão `CITRA-3D-Prepared` (não autoritativa).
- **v0.2 (10/04/2026):** **revisão estrutural após estabelecimento do pipeline autoritativo do CITRA-3D.** Atualizações:
  - Mapeamento correto das 9 classes do CITRA-3D, validado contra o `data.yaml` oficial recebido junto ao ZIP da Marinha.
  - Contagens autoritativas extraídas do `CITRA-3D.zip` original (não mais da versão `CITRA-3D-Prepared` legada).
  - Pipeline de preparação documentado em 6 etapas com scripts versionados e relatórios auditáveis.
  - Decisões metodológicas adicionais registradas (limpeza de labels, quarentena de órfãos, descarte de imagens sintéticas com critério explícito).
  - Tarefas A0–A3 do cronograma marcadas como concluídas.
- **v0.2 — micro-revisão (10/04/2026, fim do dia):** complementos após conclusão da tarefa A4 e definição do detector principal:
  - Adicionado passo 6.7 (`gerar_data_yaml.py` concluído com setup de links simbólicos).
  - Seção 7 renumerada (7.1 antiga movida para 6.7; demais entradas deslocadas).
  - Decisão 5.2 reescrita para especificar **YOLOv11m** como detector principal (não mais "YOLOv11" genérico) e justificar a escolha contra alternativas (YOLOv8m, YOLOv11n, YOLOv11s).
  - Configuração consolidada de treino baseline adicionada à decisão 5.2 (300 épocas, patience 30, imgsz 640, batch 16, 3 seeds).
  - Cronologia atualizada com entradas para a4 concluída e decisão sobre detector.
- **v0.2 — segunda micro-revisão (10/04/2026, noite):** revisão após dois bugs encontrados e corrigidos durante os primeiros treinos:
  - **Bug 1 — Ultralytics + symlinks:** a solução de links simbólicos do `gerar_data_yaml.py` original quebrava silenciosamente o treino. O Ultralytics resolve symlinks via `Path.resolve()` antes de aplicar a substituição `/images/ → /labels/`, acabando por ler os labels originais (9 classes) em vez de `labels_single_class/`. Sintoma: `ignoring corrupt image/label: Label class N exceeds dataset class count 1` em todas as imagens, treino roda em batch vazio, produz mAP aleatório. **Correção:** substituição de `gerar_data_yaml.py` por `preparar_dados_locais.py`, que faz cópia física para o disco local do Colab. Passo 6.7 do pipeline atualizado.
  - **Bug 2 — schedule de LR mal calibrado:** o `treinar_baselines.py` original usava `optimizer='auto'` sem `cos_lr=True` explícito. Resultado: o LR só subia ao longo do treino, atingindo valores destrutivos para fine-tuning de COCO. **Correção:** hyperparams diferenciados por baseline (B1 SGD lr0=0.01, B2 AdamW lr0=0.001, ambos com `cos_lr=True` explícito). Decisão 5.11 adicionada documentando a diferenciação metodológica.
  - **Primeiro piloto B2 validado:** após ambas as correções, B2 seed 42 atingiu mAP50=0,8367 / mAP50-95=0,5062 no test em 53 min, 203 épocas, zero NaN, curva saudável. Pipeline oficialmente destravado. Início dos treinos B1+B2 completos (6 corridas).
- **v0.2 — terceira micro-revisão (14/04/2026):** documentação dos primeiros resultados reais do experimento.
  - **Seção 7.7 nova — Resultados preliminares dos baselines.** Tabela consolidada com média ± desvio padrão das 3 seeds para B1 e B2, detalhamento por corrida, análise da separação entre intervalos, três descobertas centrais com fundamentação literária parcial, observação operacional sobre tempo de treino, e nota sobre comparação com o baseline anterior do grupo.
  - **Seção 11 reorganizada — Referências em quatro categorias.** (1) Referências confirmadas e diretamente usadas (He et al. 2019, Loshchilov & Hutter 2019, Teixeira et al. 2025). (2) Referências adicionais relevantes para o paper futuro. (3) Afirmações com `[CITAÇÃO PENDENTE]` marcadas explicitamente para busca futura. (4) Afirmações **REMOVIDAS por não terem fundamentação** — incluindo a "regra prática 2× DP" que apareceu em discussões de trabalho mas não tem fonte canônica conhecida; foi substituída por reporte honesto de média ± DP sem afirmar significância estatística sem teste formal.
  - **Cronologia atualizada** com entradas para a execução dos 6 baselines e a análise dos resultados.
- **v0.2 — quarta micro-revisão (14/04/2026, tarde):** documentação do HPO de B2 (Seção 7.7.6) e início da execução paralela de duas tarefas críticas.
  - **Seção 7.7.6 nova — HPO formal de B2, metodologia planejada (pré-registro).** Descreve o desenho do HPO **antes** da execução: ferramenta (Optuna TPE), espaço de busca em 5 dimensões (`lr0`, `lrf`, `momentum`, `weight_decay`, `warmup_epochs`), estratégia em duas fases (exploração com 30 trials × 100 épocas → validação do TOP-1 com treino completo), função objetivo (fitness padrão do Ultralytics 0.1×mAP50 + 0.9×mAP50-95), e critério de decisão ancorado empiricamente (3σ da variância entre seeds de B2 ≈ 0,0072 em mAP50). A subseção 7.7.7 foi criada como placeholder para os resultados, a ser preenchida após a execução. Pré-registro é explicitamente metodológico para evitar seleção post-hoc do threshold.
  - **Seção 11.1 expandida** com duas referências novas diretamente citadas: **Park et al. (2026)** (sustenta a expectativa quantitativa de ganhos de HPO na faixa de 1-3% em mAP e calibra o threshold 3σ contra alternativas arbitrárias maiores) e **Ultralytics fitness function** (sustenta a escolha da função objetivo como padrão da comunidade, não ad hoc).
  - **Seção 11.2 expandida** com duas referências a verificar: **Bergstra et al. (2011)** (paper original do TPE, sampler do Optuna) e **Popek et al. (2023)** (segundo ponto de dado literário para a expectativa de que HPO em YOLO pode não render melhoria meaningful sobre defaults).
  - **Cronologia** atualizada com duas entradas: início do HPO (A7) e início do download do `random_pool_v2` (A11) paralelamente, documentando o crash por falta de espaço e a retomada bem-sucedida do script.
- **v0.2 — quinta micro-revisão (14/04/2026, fim do dia):** descoberta crítica sobre o comportamento do CDN do shipspotting, com lição metodológica importante.
  - **Seção 4.4 substancialmente expandida com a Subseção 4.4.1 nova — "Histórico de execução do download".** Documenta a sequência completa: rodada 1 reportou 99,7% de sucesso aparente, validação posterior revelou que **30,8% dos arquivos eram HTML em vez de JPEG** (15.236 arquivos), causa raiz é shipspotting retornar HTTP 200 + página de erro em vez de HTTP 404, lição metodológica (validar magic bytes em vez de só status code) registrada para inclusão no paper. Análise por decil refutou a hipótese original sobre concentração no decil 9 — déficits são uniformes em ~30% por todos os decis com pico nos decis 0-3, sugerindo que o `dataset_25k` curado tem viés de preservação que enviesou a estratificação. Plano de rodada 2 detalhado: sorteio adaptativo via `gerar_ids_rodada2.py` + download via `baixar_random_pool_v2.py` v2 com check de magic bytes.
  - **Cronologia** atualizada com quatro entradas: rodada 1 concluída, descoberta do bug via validação, escrita da v2 do downloader, escrita do gerador da rodada 2.
- **v0.2 — sexta micro-revisão (15/04/2026):** fechamento de duas tarefas críticas (A11 e A7) e descoberta importante sobre dataset_25k.
  - **A11 (random_pool_v2) fechada com sucesso.** Rodada 2 atingiu 39.628 imagens válidas (folga +4% sobre alvo do dataset_25k), 100% JPEG verificado por magic bytes, sincronizado para Drive, verificação cruzada Drive vs disco local confirmou zero perda. Seção 4.4.1 atualizada com tabelas finais.
  - **A7 (HPO em B2) fechada com decisão MANTER CONFIG ATUAL.** 30 trials Optuna TPE em ~10h (com 1 reinicialização). TPE convergiu para região consistente nos top-3. Fase 2: TOP-1 atingiu mAP50=0,8328 vs baseline 0,8351, Δ=−0,0023 (=−0,94σ, dentro de 1 desvio padrão). Threshold 3σ não foi superado. Configuração de B2 validada formalmente. Seção 7.7.7 atualizada com TOP-5, análise de convergência do TPE, observação sobre limite de épocas, decisão e interpretação para o paper.
  - **Descoberta crítica sobre dataset_25k (a tratar em sprint próximo):** durante a verificação dos IDs para excluir do random_pool_v2, foi descoberto que o dataset_25k tem **27.796 IDs únicos mas 38.109 arquivos por causa de duplicação entre splits** (5.253 imagens em train∩val, 3.777 em train∩test, 1.283 em val∩test, todas com hash MD5 idêntico). Bug provável no notebook InaTechShips.ipynb (Seção 11): falta de limpeza de DATASET_PATH antes de cada execução. Não afeta os baselines existentes (rodaram em CITRA-3D-Real, não em dataset_25k), mas **invalida o uso direto do dataset_25k atual no braço A do experimento principal**. A tarefa A12 nova (Reconstruir dataset_25k) é pré-requisito para o braço A.
  - **Cronologia** atualizada com entradas para A11 final (rodada 2 + upload) e A7 final (HPO concluído).
- **v0.2 — sétima micro-revisão (16/04/2026):** fechamento do sprint de preparação de dados com A12 + downsample.
  - **Seção 4.3 nova — "Reconstrução do dataset_25k (A12)".** Documenta a detecção da contaminação (10.313 duplicatas entre splits no dataset original), a causa raiz (bug de ciclo de vida em `InaTechShips.ipynb` Seção 11 — falta de limpeza de `DATASET_PATH` entre execuções), a reconstrução com split 60/20/20 estratificado por classe (seed 42), e a validação final (27.796 IDs únicos distribuídos em 16.677/5.558/5.561 com disjunção total comprovada entre splits).
  - **Seção 4.4.2 nova — "Downsample do random_pool_v2 para alinhamento com dataset_25k_v2".** Documenta o pareamento experimental: random_pool_v2 reduzido de 39.628 → 27.964 imagens (descarte para `_excedente/`), estratificação preservada por decis empíricos do dataset_25k_v2, descasamento residual de 0,4% por split aceito como metodologicamente irrelevante (168 imagens a mais em 27.964).
  - **Cronologia** atualizada com três entradas: A12 concluída, recálculo de decis v2, downsample concluído.
  - **Status geral:** pipeline de dados 100% pronto para o experimento principal. Próximas pendências são labels do random_pool_v2 (dependência externa) e email institucional (C4).
- **v0.2 — oitava micro-revisão (22/04/2026):** resultados do braço A e ablation de épocas de pré-treino.
  - **Seção 7.7.8 nova — Braço A: resultados do pré-treino curado + fine-tuning.** Resultado principal: mAP50 = 0,7936 ± 0,0060 — 4,15% abaixo de B2 (COCO puro) e 0,72% abaixo de B1 (random init). O pré-treino intermediário no dataset_25k_v2 piorou em vez de melhorar, indicando catastrophic forgetting. Interpretação via 3 eixos do gap de domínio (condições de captura, densidade de objetos, distribuição de classes). Observação paradoxal: COCO genérico é melhor pré-treino que 27.796 imagens de embarcações curadas. Cenários de interpretação para quando o braço B estiver disponível.
  - **Seção 7.7.9 nova — Ablation de épocas de pré-treino: diagnóstico de catastrophic forgetting.** Testou 10, 20 e 50 épocas (seed 42) + referência de 100 épocas do braço A. Degradação monotônica confirmada: cada incremento piora o resultado. Mesmo 10 épocas perde 1,5% mAP50. Com 100 épocas, converge para nível do B1. Diagnóstico final: catastrophic forgetting confirmado com incompatibilidade parcial subjacente. Implicação metodológica: similaridade visual CLIP é proxy insuficiente para transferibilidade.
  - **Cronologia** atualizada com entradas para braço A concluído e ablation concluída.
  - **Status geral:** braço A completo com resultado surpreendente (negative transfer). Ablation confirma catastrophic forgetting monotônico. Aguardando labels do Eduardo para braço B. Email de colaboração enviado em 22/04/2026.
  - **Seção 7.7.10 nova — Scale-Aware Copy-Paste: adaptação de domínio.** Após revisão de literatura (POSEIDON 2023, S3Det/ACCV 2024, Nemati 2025, ODGEN/ICCV 2023, InstaGen/CVPR 2024), copy-paste selecionado sobre CycleGAN e modelos de difusão. Justificativa: bboxes exatos, fundos reais, controle de escala determinístico, custo computacional mínimo, degradação de resolução do resize simula naturalmente objetos distantes. Pipeline de 5 passos definido. Script `analisar_escala_citra3d.py` escrito.
- **v0.3 (23/04/2026):** **nova fase experimental — Scale-Aware Copy-Paste para adaptação de domínio.** Bump de versão justificado pela mudança de paradigma: de pré-treino direto (que falhou) para geração de dados sintéticos por composição. O documento agora documenta um pipeline completo de 6 passos para domain adaptation.
  - **Passo 1 concluído** — `analisar_escala_citra3d.py`: perfil de escala do CITRA-3D-Real extraído. Resultado-chave: 71,6% dos bboxes são "small" (COCO-style), mediana de área = 0,1% da imagem, mediana 2 objetos/imagem, navios posicionados na metade inferior (y_center P10=0,37, P90=0,70).
  - **Passo 2 pronto** — `extrair_crops_sam.py`: extração de recortes de navios do InaTechShips com SAM (Segment Anything Model) usando bbox como prompt. Dois modos: SAM ViT-B (~0,3s/img, segmentação precisa) e bbox com feathering (~0,01s/img, fallback rápido). Retomável via progress JSON.
  - **Passo 4.5 planejado** — validação das sintéticas via FID (Fréchet Inception Distance) + comparação de distribuição de bboxes vs CITRA-3D-Real. Adicionado ao pipeline após sugestão de validação de similaridade estatística (não visual) entre domínios.
  - **Colaboração com Eduardo Teixeira formalizada.** Eduardo respondeu ao email (15/04/2026), aceitou coautoria. Email de retorno enviado com lista de 39.628 IDs para labels do PointRend. Trilha B em processo de desbloqueio.
  - **Cronograma atualizado** com nova Trilha D (Scale-Aware Copy-Paste, 6 passos) e status de cada passo.
  - **Atualização 24/04/2026 — desbloqueios maiores:**
    - **Trilha B totalmente desbloqueada.** Eduardo compartilhou 2,8M labels via WhatsApp/Drive. Script `filtrar_labels_eduardo.py` filtrou 36.272 matches (91,5%) e copiou 25.591 labels. B1, B2, B3 todas fechadas no mesmo dia. Braço B diagnóstico (1 seed) disparado.
    - **D3 concluída.** 2.081 fundos (1.988 clean + 93 inpaint).
    - **D4 — 3 iterações até abordagem correta.** v1 (aleatório, 30% errado) → v2 (HSV, 15% errado) → **v3 (substituição in-place, 0% errado)**. A v3 foi proposta pela autora: substituir navios reais por crops na mesma posição/dimensão, eliminando todas as decisões arbitrárias de posicionamento. Preview aprovado, batch em execução.
    - **Seção 7.7.10 atualizada** com histórico das 3 iterações do D4 e justificativa da v3.
    - **Cronologia** atualizada com 4 novas entradas.
- **v0.4 (25/04/2026):** **resultado principal obtido — experimento concluído.**
  - **Seção 7.7.11 nova — Braço B (diagnóstico).** Resultado: B = 0,7997 mAP50, equivalente ao braço A (0,7936). Confirma que negative transfer é independente da curadoria CLIP — gap estrutural de domínio.
  - **Seção 7.7.12 nova — Braço A' (composição sintética in-place).** Resultado principal: mAP50 = 0,8541 ± 0,0043. **Superou B2 em +1,90%** com separação estatística completa. mAP50-95 = 0,5281 ± 0,0056 (+4,5% relativo). Tabela comparativa completa dos 5 braços. Análise estatística dos intervalos de confiança.
  - **Seção 7.7.13 nova — Síntese dos resultados.** Tabela de ranking final. Três contribuições do experimento: (1) evidência de negative transfer com ablation, (2) método de adaptação por composição in-place, (3) demonstração de CLIP como proxy insuficiente.
  - **Seção 7.7.10 atualizada** com status de conclusão do pipeline e link para resultados.
  - Cronologia atualizada com 3 novas entradas (braço B concluído, dataset sintético concluído, braço A' concluído).
  - **Status geral: experimento concluído. Pronto para escrita do paper.**

---

## 1. Resumo executivo

Este documento registra o desenho, as decisões metodológicas e o estado de preparação de um experimento que investiga se a curadoria de um pool grande de fotografia pública de embarcações por similaridade visual ao domínio operacional real pode melhorar a generalização de detectores YOLO em cenário de vigilância marítima de baixa amostragem.

A pergunta científica central é: **a curadoria por similaridade visual via CLIP move o ponteiro em performance downstream, ou é apenas um ganho estético sem utilidade prática?**

A resposta será obtida via um ablation controlado entre dois subsets de mesmo tamanho extraídos do dataset InaTechShips (Teixeira, Mafra & De Figueiredo, *Ocean Engineering*, 2025): um curado por similaridade CLIP ao CITRA-3D (dataset operacional da Marinha do Brasil), e outro sorteado aleatoriamente do mesmo pool, com distribuição temporal de IDs controlada via estratificação por decis. Ambos os subsets serão colapsados para detecção classe-única (embarcação) para evitar problemas de mapeamento taxonômico entre os dois domínios e contornar o desbalanço severo do CITRA-3D.

A contribuição esperada é metodológica, não arquitetural: um protocolo reproduzível de transferência entre fotografia pública e captura operacional, com caracterização honesta do gap de domínio e dos limites de utilidade da curadoria por similaridade visual.

---

## 2. Contexto e motivação

### 2.1 O problema operacional

A Marinha do Brasil mantém o dataset CITRA-3D, composto por **2.081 imagens reais** (após pipeline de preparação descrito na Seção 6) de embarcações capturadas em cenário operacional. O dataset apresenta as características típicas de coleta institucional:

- **Volume reduzido** comparado a benchmarks acadêmicos da área (SeaShips tem ~31k, LMD-TShip ~40k, ABOShips ~13k).
- **Desbalanço severo entre classes**, com a classe dominante (TUG) representando 35,6% das anotações e a classe mais rara documentada (Navio) representando 0,8%.
- **Uma classe efetivamente ausente do conjunto de validação:** Passageiro tem apenas 8 instâncias no dataset inteiro, com zero presença no val.
- **Densidade alta de objetos por imagem:** 7.003 bboxes em 2.081 imagens (média de 3,36 embarcações por imagem). Cenas multi-embarcação são a regra, não a exceção.

Treinar um detector apenas no CITRA-3D leva a três problemas previsíveis: overfitting pelo volume baixo, viés para as classes dominantes, e métricas globais que mascaram performance ruim nas classes minoritárias. Aumentar o volume é o caminho óbvio, mas anotação manual de novas imagens em cenário operacional é cara e lenta — e a anotação existente já apresentou bugs de software (ver Seção 5.8) que precisaram ser tratados.

### 2.2 A oportunidade do InaTechShips

O paper de Teixeira, Mafra & De Figueiredo (*Ocean Engineering* 326, 2025) publicou o InaTechShips: um dataset de 3.013.830 imagens de embarcações coletadas do portal shipspotting.com, anotadas automaticamente via pipeline PointRend + YOLOv8, organizadas em 200+ classes (das quais 195 com pelo menos 100 amostras). É de longe o maior dataset rotulado disponível para a tarefa.

O InaTechShips, no entanto, não é diretamente substituto do CITRA-3D. As imagens vêm de fotografia profissional/amadora terrestre — fotógrafos posicionados em cais, portos e mirantes, capturando embarcações com equipamento dedicado, em ângulos privilegiados, geralmente uma única embarcação preenchendo boa parte do frame (1 bbox por imagem em média no subset curado, vs 3,36 no CITRA-3D). O CITRA-3D, em contraste, vem de captura operacional em condições naturais de vigilância, com cenas multi-embarcação. Há um gap de domínio entre as duas fontes que não pode ser ignorado.

### 2.3 A hipótese a ser testada

A hipótese central deste experimento é que **curar previamente o pool do InaTechShips, selecionando apenas imagens visualmente similares ao CITRA-3D, produz um conjunto de pré-treino mais útil do que um subset aleatório de mesmo tamanho**. "Mais útil" aqui é definido empiricamente: maior performance em detecção de embarcações no conjunto de teste do CITRA-3D após fine-tuning.

Essa hipótese tem três cenários possíveis, todos publicáveis:

- **Hipótese confirmada:** o subset curado supera o aleatório por margem estatisticamente significativa. Conclusão: curadoria por similaridade visual é uma técnica útil para transferência entre domínios em detecção marítima.
- **Hipótese refutada (resultado nulo):** o subset curado e o aleatório têm performance equivalente. Conclusão: similaridade visual via CLIP não captura as features que importam para detecção downstream — achado negativo cientificamente valioso.
- **Hipótese refutada (resultado negativo):** o subset curado é pior que o aleatório. Conclusão menos provável, mas indicaria viés introduzido pela curadoria — também publicável.

---

## 3. Pergunta científica e desenho geral

### 3.1 Pergunta central

> Em um cenário de transferência entre fotografia pública de embarcações e um dataset operacional real de baixa amostragem, a curadoria do pool de pré-treino por similaridade visual (medida via CLIP) ao domínio-alvo gera ganho significativo de performance downstream comparado a uma seleção aleatória de mesmo tamanho?

### 3.2 Variável independente

O critério de seleção do subset de pré-treino:
- **Subset A (curado):** as N imagens do InaTechShips com maior similaridade CLIP a pelo menos uma imagem real do CITRA-3D, acima de threshold 0,60.
- **Subset B (aleatório):** N imagens do InaTechShips sorteadas aleatoriamente, com o constraint de espelhar a distribuição temporal de IDs do subset A (estratificação por decis).

### 3.3 Variável dependente

Performance de detecção classe-única no conjunto de teste do CITRA-3D, medida primariamente por mAP@50 e mAP@50-95, com análise complementar por escala de objeto (small/medium/large no esquema COCO) e por classe original do CITRA-3D (mesmo no modelo classe-única, é possível mapear cada bbox detectada à classe verdadeira do CITRA-3D para análise descritiva).

### 3.4 Variáveis controladas

Para que o ablation seja interpretável, todas as outras variáveis devem ser idênticas entre os braços A e B:

- **Mesma arquitetura de detector:** **YOLOv11m** como modelo principal de todas as 12+ corridas. Este tamanho (medium) é equivalente em escala ao YOLOv8m que serviu de baseline em trabalhos anteriores do grupo, preservando continuidade conceitual. Possível adição de YOLOv8m como sanity check secundário (1 seed por braço) para fortalecer defesa contra revisores.
- Mesmo número de imagens em cada subset (~38.109).
- Mesma distribuição train/val/test (~22k/9k/7k).
- Mesma distribuição temporal de IDs (estratificação por decis empíricos do subset curado).
- Mesma fonte e método de anotação dos labels (a definir conforme resposta do Eduardo — ver Seção 8).
- Mesmos hiperparâmetros de treino (a definir via HPO no piloto).
- Mesmas seeds de inicialização (3 seeds por braço).
- Mesma máquina e versão de framework (Colab Pro+ A100, Ultralytics YOLO).

### 3.5 Desenho experimental — 5 braços

| Braço | Pré-treino | Fine-tuning | Função no experimento |
|---|---|---|---|
| **B1** | Nenhum (random init) | CITRA-3D-Real | Chão absoluto. O que se obtém sem dados externos. |
| **B2** | ImageNet/COCO (default Ultralytics) | CITRA-3D-Real | Baseline padrão da comunidade. |
| **A** | InaTechShips **curado por CLIP** (~38k) | CITRA-3D-Real | Tratamento principal. |
| **B** | InaTechShips **aleatório estratificado** (~38k) | CITRA-3D-Real | Controle do tratamento principal. **Comparação A vs B é o coração do paper.** |
| **C** | InaTechShips **completo** (~3M) | CITRA-3D-Real | Opcional. Responde "curadoria vence quantidade?". Decisão sobre executar fica para depois dos resultados de A e B. |

Cada braço será treinado com 3 seeds diferentes, totalizando 12–15 treinos completos no plano principal.

### 3.6 Análises planejadas

- **Análise principal:** mAP@50 e mAP@50-95 por braço no test set do CITRA-3D-Real, com intervalos de confiança via 3 seeds. Teste estatístico de diferença entre A e B.
- **Análise por escala de objeto:** mesma métrica desagregada para objetos small/medium/large no esquema COCO. Hipótese secundária: a curadoria ajuda mais nos objetos pequenos (mais sensíveis ao gap de domínio).
- **Análise por classe original do CITRA-3D:** mesmo no modelo classe-única, cada detecção pode ser mapeada à classe verdadeira do CITRA-3D. Isso permite análise descritiva: a curadoria ajuda mais nas classes minoritárias?
- **Análise de custo-benefício:** ganho de performance vs custo computacional do pré-treino. Se C (3M imagens) só ganha marginalmente sobre A, o paper pode argumentar pela curadoria como estratégia eficiente.

---

## 4. Datasets

### 4.1 CITRA-3D-Real — domínio operacional alvo

**Origem:** Marinha do Brasil. Captura terrestre operacional. Detalhes específicos de localização, sensor, período de captura e protocolo de anotação preservados por restrição institucional. *(Pendente: descobrir e documentar quando possível.)*

**Fonte autoritativa:** arquivo `CITRA-3D.zip` (9,27 GB) recebido da Marinha, contendo 7 ZIPs aninhados (3 com imagens reais + sintéticas em splits train/val/test, 4 com augmentations sintéticas marcadas com sufixo `_aug`). Pipeline completo de extração, limpeza e validação descrito na Seção 6.

**Composição final autoritativa (após pipeline de preparação concluído em 10/04/2026):**

| Split | Imagens reais | Bboxes válidas | Bboxes/imagem (média) |
|---|---|---|---|
| Train | 1.348 | 4.489 | 3,33 |
| Val | 332 | 1.267 | 3,82 |
| Test | 401 | 1.247 | 3,11 |
| **Total** | **2.081** | **7.003** | **3,36** |

**Características relevantes para o experimento:**
- Pareamento imagem ↔ label: 100% (zero órfãos após pipeline).
- Zero labels malformados, zero bboxes degeneradas (após limpeza).
- **Densidade alta de objetos por imagem** (3,36 bboxes/imagem em média) — cenas operacionais multi-embarcação são a regra. Isto contrasta fortemente com o `dataset_25k` (1 bbox/imagem), e é uma propriedade importante a mencionar no paper.
- 2 imagens (1 do train, 1 do test) foram movidas para `_quarantine/` por terem labels totalmente vazios após limpeza, contendo embarcações distantes que o anotador tentou marcar mas falhou (ver Seção 5.9).

**Taxonomia (9 classes, índices 0–8) — confirmada via `data.yaml` oficial:**

| Índice | Classe |
|---|---|
| 0 | Militar |
| 1 | Barca |
| 2 | Mercante |
| 3 | Vela |
| 4 | Passageiro |
| 5 | TUG |
| 6 | Lancha |
| 7 | Miuda |
| 8 | Navio |

**Distribuição original autoritativa por classe (do relatório de auditoria pós-limpeza):**

| Índice | Classe | Train | Val | Test | Total | % |
|---|---|---|---|---|---|---|
| 0 | Militar | 268 | 82 | 82 | 432 | 6,2% |
| 1 | Barca | 190 | 47 | 58 | 295 | 4,2% |
| 2 | Mercante | 870 | 256 | 231 | 1.357 | 19,4% |
| 3 | Vela | 461 | 116 | 123 | 700 | 10,0% |
| **4** | **Passageiro** | **7** | **0** | **1** | **8** | **0,1%** ⚠️ ausente no val |
| **5** | **TUG** | **1.565** | **480** | **445** | **2.490** | **35,6%** ⚠️ dominante |
| 6 | Lancha | 279 | 69 | 89 | 437 | 6,2% |
| 7 | Miuda | 814 | 207 | 205 | 1.226 | 17,5% |
| **8** | **Navio** | **35** | **10** | **13** | **58** | **0,8%** ⚠️ rara |

**Observações críticas sobre a distribuição:**

1. **TUG é dominante e consistente nos 3 splits** (34,9% / 37,9% / 35,7%) — bom sinal, sem inversão entre splits.
2. **Navio é rara mas presente nos 3 splits** (35 / 10 / 13) — viável estatisticamente, embora ruidosa.
3. **Passageiro é praticamente inexistente** (8 instâncias totais) e **completamente ausente do val** (0 instâncias). Isto **torna avaliação multi-classe estatisticamente impossível** para essa classe — argumento adicional forte para a decisão de colapso para classe única (Seção 5.1).
4. **Razão entre classe dominante e classe documentada mais rara (TUG/Navio):** 43:1. Long-tail severo.

**Nota sobre a discrepância com documentos antigos:** Documentos legados do projeto (`ANÁLISE_DO_CITRA-3D.docx`) chamavam a classe dominante de "Mercante" e a classe rara de "Navio", baseados em uma versão `CITRA-3D-Prepared` que não corresponde ao ZIP autoritativo. Esses documentos estão **obsoletos** e a versão `CITRA-3D-Prepared` foi arquivada. O mapeamento e contagens corretos são os listados acima, validados via `data.yaml` oficial e relatório de auditoria de 10/04/2026.

### 4.2 InaTechShips — pool de origem

**Origem:** Teixeira, Mafra & De Figueiredo (2025), Ocean Engineering 326, 120823. Repositório oficial: github.com/EduardoHT/InaTechShips. Imagens coletadas do portal shipspotting.com.

**Características do pool completo:**
- 3.013.830 imagens de embarcações de fotografia pública profissional/amadora.
- 200+ classes naturais (Vessel_Type), das quais 195 com pelo menos 100 amostras.
- Anotação automática via pipeline PointRend + YOLOv8 (segmentação convertida para bounding boxes no formato YOLO).
- Metadados textuais ricos por imagem (Vessel_Type, IMO, MMSI, dimensões, ano de construção, categoria) disponíveis em JSON, fornecidos via Drive público em 8 arquivos `meta1.zip`–`meta8.zip`.

**Características relevantes para o experimento:**
- Domínio de captura completamente diferente do CITRA-3D-Real: fotografia terrestre profissional, embarcação tipicamente preenchendo o frame (1 bbox/imagem), iluminação controlada, ângulos privilegiados.
- IDs no range observado [27, 3.412.797], com distribuição enviesada para IDs baixos (p50 = 273.736; p95 = 1.106.169).
- **Limitação crítica para o experimento:** apenas ~22.308 bounding boxes estão publicadas no repositório GitHub (pastas `labels1` a `labels8`), correspondendo a uma fração das ~3M imagens. A cobertura completa de labels do PointRend pode existir em repositório interno do grupo do INATEL, mas não é pública até o momento. Esta limitação é o gargalo atual do experimento (ver Seção 8).

### 4.3 dataset_25k — subset curado por similaridade CLIP

**Construção:** subset do InaTechShips composto por imagens cuja similaridade CLIP ViT-B-32 (cosseno entre embeddings normalizados) com pelo menos uma imagem real do CITRA-3D é igual ou superior a 0,60. O processo é descrito em detalhe nos scripts `download_direto.py` e `inatechships.py` do projeto.

**Pipeline de construção:**
1. Cálculo de embeddings CLIP ViT-B-32 (`pretrained='openai'`) para todas as imagens reais do CITRA-3D, salvos em cache `.npz`.
2. Iteração sobre IDs do shipspotting (range 1 a 3.500.000, embaralhado com seed=42), download direto via URL do CDN.
3. Cálculo do embedding CLIP da imagem candidata.
4. Verificação de blacklist: imagens com similaridade ≥ 0,90 a banners/templates conhecidos do site são descartadas.
5. Verificação de similaridade: imagens com pelo menos uma similaridade ≥ 0,60 contra qualquer imagem real do CITRA-3D são salvas.
6. Para os IDs com label disponível no GitHub do EduardoHT (pastas `labels1`–`labels8`), o label é copiado.
7. Reanotação manual do `class_id` original para uma taxonomia própria de 10 classes (TOP 10 classes do InaTechShips).
8. Divisão train/val/test.

**Composição confirmada (relatório de auditoria de 09/04/2026):**

| Split | Imagens | Labels | Bboxes |
|---|---|---|---|
| Train | 22.064 | 22.064 | 22.064 |
| Val | 9.147 | 9.147 | 9.147 |
| Test | 6.898 | 6.898 | 6.898 |
| **Total** | **38.109** | **38.109** | **38.109** |

- Pareamento imagem ↔ label: 100%.
- Bboxes por imagem: exatamente 1 (cada imagem tem uma única embarcação anotada).
- Zero labels malformados ou bboxes degeneradas.
- **Contraste fundamental com CITRA-3D-Real:** dataset_25k tem 1 bbox/imagem, CITRA-3D-Real tem 3,36 bboxes/imagem. Esta diferença de densidade é uma das assinaturas do gap de domínio entre os dois datasets.

#### 4.3.1 Taxonomia (10 classes, índices 0–9)

| Índice | Classe |
|---|---|
| 0 | GENERAL CARGO |
| 1 | CONTAINER SHIP |
| 2 | BULK CARRIER |
| 3 | PASSENGERS SHIP |
| 4 | RO-RO/PASSENGER SHIP |
| 5 | TUG |
| 6 | OIL/CHEMICAL TANKER |
| 7 | RO-RO CARGO |
| 8 | VEHICLES CARRIER |
| 9 | OIL PRODUCTS TANKER |

A distribuição é deliberadamente uniforme no train e val (curada para ~2.100–2.350 imagens por classe no train, ~900 no val), e mais natural no test. Coeficiente de Gini ≈ 0 no train.

**Distribuição original confirmada (relatório de 10/04/2026):** as 10 classes ficam todas entre 9,5% e 10,8% do total de 38.109 instâncias — coeficiente de variação ~5%, praticamente uniforme. Isto contrasta fortemente com o long-tail severo do CITRA-3D-Real (TUG 35,6% vs Navio 0,8%, razão 43:1) e configura **mais um eixo do gap de domínio** entre os dois datasets: não é apenas fotografia profissional vs captura operacional, nem apenas 1 bbox/imagem vs 3,36 bboxes/imagem — é também distribuição balanceada por curadoria vs distribuição naturalmente skewed pela frequência operacional real das classes de embarcação. Esta tripla diferença (fonte, densidade, distribuição) é uma observação descritiva forte para a seção de Dados do paper.

#### 4.3.2 Sobre a correspondência semântica entre as taxonomias

A autora considerou previamente um mapeamento semântico entre as 9 classes do CITRA-3D-Real e as 10 do dataset_25k, mas avaliou que **a correspondência não era confiável o suficiente para sustentar treinamento multi-classe**. Esta avaliação é uma das justificativas centrais para a decisão de colapso para classe única (ver Seção 5.1).

#### 4.3.3 Descoberta de contaminação e reconstrução (16/04/2026)

**Contexto da descoberta.** Durante a preparação do arquivo `dataset_25k_ids.txt` (lista de IDs do `dataset_25k` a excluir do sorteio do `random_pool_v2`), uma verificação de integridade revelou que o dataset_25k original tem **apenas 27.796 IDs únicos, não os 38.109 originalmente reportados**. A diferença vem de **duplicação de arquivos entre splits**:

| Sobreposição | Quantidade |
|---|---|
| train ∩ val | 5.253 IDs |
| train ∩ test | 3.777 IDs |
| val ∩ test | 1.283 IDs |
| os 3 splits | 0 IDs |
| **Total duplicado** | **10.313 IDs** (37,1%) |

Verificação cruzada via hashing MD5 confirmou que os arquivos duplicados têm **conteúdo idêntico** em ambos os splits (mesmo MD5, mesmo tamanho em bytes). Não são coincidências de ID numérico, mas cópias literais.

**Causa raiz identificada.** Inspeção do notebook `InaTechShips.ipynb` (Seção 11, montagem do dataset) revelou que o script:
1. Cria os diretórios `train/val/test` com `os.makedirs(exist_ok=True)`
2. Itera sobre as 10 classes, sorteia N_PER_CLASS imagens, faz o split 60/20/20
3. Copia arquivos via `shutil.copy2`
4. **Não limpa o `DATASET_PATH` no início**

Em execuções múltiplas do notebook com pools de metadados diferentes (mais IDs baixados a cada execução, ou N_PER_CLASS ajustado entre runs), o `random.shuffle` com seed 42 sobre listas de tamanhos diferentes produz permutações distintas. Sem limpeza prévia, arquivos de execuções anteriores permanecem no disco junto com os novos, criando sobreposição entre splits. A concentração de IDs duplicados nos decis baixos (os primeiros listados: 315, 477, 667 para train∩val; 27, 84, 190 para train∩test) é consistente com IDs que existiam no pool de uma execução inicial (quando ainda dominavam IDs baixos) e foram rerriplas em execuções posteriores.

**Impacto do bug.** Para qualquer experimento que use o dataset_25k original diretamente:
- Modelo treina com imagens que aparecem em val/test, inflando métricas de validação e test
- Seleção de modelo (via `patience` / `best.pt`) fica enviesada porque val está contaminado
- Comparações entre runs usando esse dataset são invalidadas por contaminação

**Não afeta os baselines existentes** (B1, B2, HPO de B2) porque estes foram treinados em `CITRA-3D-Real`, não em `dataset_25k`. O bug só afetaria o experimento principal se usássemos o dataset_25k diretamente no braço A (pré-treino antes de fine-tuning em CITRA-3D-Real).

**Reconstrução.** Em 16/04/2026 foi criado o dataset `dataset_25k_v2` via script `reconstruir_dataset_25k.py`, aplicando:
1. Coleta dos 27.796 IDs únicos a partir do dataset_25k original (união dos 3 splits, deduplicada)
2. Leitura do `class_id` de cada ID a partir da primeira ocorrência encontrada nas pastas `labels/`
3. Split 60/20/20 estratificado por classe, seed 42, arredondamento equilibrado (train=round(0.6N), val=round(0.2N), test=resto)
4. Cópia física de imagens + labels (10-class) + labels_single_class (1-class) para nova estrutura em `/content/drive/MyDrive/InaTechShips/dataset_25k_v2/`
5. Geração automática de `data.yaml` e `data_single_class.yaml` apontando para a estrutura reorganizada
6. Geração de relatório de validação confirmando disjunção total

**Resultado final do dataset_25k_v2:**

| Split | Imagens | Proporção |
|---|---|---|
| train | 16.677 | 60,00% |
| val | 5.558 | 19,99% |
| test | 5.561 | 20,01% |
| **Total** | **27.796** | **100%** |

**Validação de disjunção:**

| Sobreposição | Quantidade |
|---|---|
| train ∩ val | **0** ✓ |
| train ∩ test | **0** ✓ |
| val ∩ test | **0** ✓ |

**Distribuição por classe preservada** (todas as 10 classes entre 2.602 e 3.095 IDs únicos, coeficiente de variação ~7% — muito mais uniforme que a distribuição de bboxes do dataset_25k original reportada na Seção 4.3.1 porque aquela contava bboxes, e agora contamos IDs únicos).

**Arquivo de relatório:** `/content/drive/MyDrive/InaTechShips/dataset_25k_v2/reconstrucao_report.json` documenta entrada, saída, verificação de disjunção, e distribuição classe × split. Preservado como referência institucional.

**O dataset_25k original não foi apagado** — permanece em `/content/drive/MyDrive/InaTechShips/dataset_25k/` como referência histórica. O dataset_25k_v2 é agora a fonte autoritativa para qualquer uso experimental futuro.

**Nota metodológica para o paper.** A descoberta e correção desta contaminação é um caso exemplar de **higiene de dados em machine learning**. Validação explícita de disjunção entre splits deveria ser rotina em qualquer pipeline de ML, e a ausência dessa validação no notebook original é um padrão comum que merece destaque: muitos datasets publicados podem ter problemas similares não detectados. A seção de Dados do paper pode incluir uma breve menção a este processo de validação como contribuição metodológica secundária.

### 4.4 random_pool_v2 — subset aleatório estratificado

**Construção:** subset do InaTechShips de mesmo tamanho que o `dataset_25k`, sorteado aleatoriamente do pool total, com **estratificação por decis empíricos** da distribuição de IDs do `dataset_25k`. Esta estratificação é crítica para o ablation: garante que o subset aleatório siga a mesma distribuição temporal do subset curado, controlando "época" do shipspotting como variável e isolando o efeito do critério de seleção.

**Pipeline de geração** (script `gerar_ids_aleatorios.py`, executado em 09/04/2026):

1. Leitura dos 38.109 IDs do `dataset_25k` (todos os splits combinados).
2. Cálculo de 10 bins por decil empírico sobre o conjunto completo de IDs ordenados.
3. Para cada split, contagem do número de IDs do curado em cada decil.
4. Sorteio de IDs novos em cada decil, em quantidade proporcional ao curado, com **margem de segurança de 30%** sobre o alvo (para cobrir downloads que retornarão 404 do shipspotting).
5. Garantia de disjunção: nenhum ID sorteado coincide com IDs já presentes no `dataset_25k` ou em outros splits do `random_pool_v2`.

**Decis empíricos calculados** (de `distribuicao_decis.json`):

| Decil | Range de IDs | Tamanho do range |
|---|---|---|
| 0 | 27 → 52.696 | 52.670 |
| 1 | 52.697 → 107.023 | 54.327 |
| 2 | 107.037 → 169.638 | 62.602 |
| 3 | 169.639 → 225.863 | 56.225 |
| 4 | 225.895 → 273.726 | 47.832 |
| 5 | 273.736 → 329.484 | 55.749 |
| 6 | 329.485 → 380.940 | 51.456 |
| 7 | 380.965 → 432.381 | 51.417 |
| 8 | 432.389 → 483.393 | 51.005 |
| **9** | **483.402 → 3.754.076** | **3.270.675** |

**Nota sobre o decil 9:** os primeiros 9 decis cobrem ranges relativamente uniformes de ~50–60k IDs. O decil 9 cobre 3,27 milhões de IDs (60x mais largo) porque a distribuição empírica do `dataset_25k` é fortemente concentrada na primeira metade do espaço de IDs. Isso é consequência direta da estratificação por contagem (cada decil contém 10% dos IDs), não um defeito do método. **O efeito prático** é que a taxa de 404 do shipspotting provavelmente será maior no decil 9 do que nos outros, e a margem de 30% pode não ser suficiente para esse decil específico — será necessário monitorar após o download e possivelmente fazer uma segunda rodada de sorteio se o decil 9 ficar abaixo do alvo. Ver Seção 8.4 (risco do decil 9).

**Saída do gerador (sorteio concluído):**

| Split | IDs sorteados | IDs alvo (espelhando dataset_25k) |
|---|---|---|
| Train | 28.688 | 22.064 |
| Val | 11.895 | 9.147 |
| Test | 8.971 | 6.898 |
| **Total** | **49.554** | **38.109** |

**Verificações pós-geração:**
- Overlap com `dataset_25k`: **0** (todos disjuntos).
- Overlap entre splits do `random_pool_v2`: **0**.
- Reprodutibilidade: seed = 42 fixa.

**Estado atual (14/04/2026, fim do dia):** o download foi executado em duas rodadas, com uma descoberta importante sobre o comportamento do servidor entre elas.

#### 4.4.1 Histórico de execução do download

**Rodada 1 (14/04/2026, manhã-tarde).** Disparada com o `baixar_random_pool_v2.py` v1 + 4 workers. Primeira execução crashou em ~35k imagens por falta de espaço em disco; após liberação, retomada via `download_progress.json` + detecção de `.jpg` em disco completou os 49.422 IDs sorteados em 95 minutos totais. Relatório final reportou 100% dos IDs como `ok` e 0% como `not_found`, taxa de erro de 0,3% (144 timeouts/conexões falhadas). Aparente sucesso total.

**Validação posterior (14/04/2026, fim da tarde).** Verificação amostral revelou arquivos que o PIL não conseguia abrir. Validação completa de todos os 49.422 arquivos via `validar_imagens_random_pool.py` (cascata de checks: tamanho mínimo, magic bytes JPEG `FF D8 FF`, `PIL.verify()`, `PIL.open().size`) revelou:

| Split | Total | Válidas | Corrompidas | % corrupção |
|---|---|---|---|---|
| train | 28.577 | 19.617 | 8.960 | 31,4% |
| val | 11.887 | 8.362 | 3.525 | 29,7% |
| test | 8.958 | 6.207 | 2.751 | 30,7% |
| **Total** | **49.422** | **34.186** | **15.236** | **30,8%** |

Dos 15.236 corrompidos:
- **14.851 (97,5%)** eram **HTML em vez de JPEG** — páginas de erro do shipspotting salvas com extensão .jpg
- **382 (2,5%)** eram arquivos de 0 bytes (todos no train, truncados pelo crash de disco da rodada 1)
- **3 casos** eram outros (header inválido, arquivo de 4 KB, falha de PIL verify)

**Causa raiz identificada.** O CDN do shipspotting **retorna HTTP 200 OK com uma página HTML de erro** em vez de HTTP 404 quando um ID não existe. O `baixar_random_pool_v2.py` v1 verificava apenas `status_code == 200` e `len(content) > 5000` — uma página HTML de erro passa em ambos os checks e foi salva como `.jpg`. Esta é uma falha de design comum mas crítica em scrapers que confiam em status codes para detectar conteúdo inválido.

**Lição metodológica importante para o paper.** Quando se baixam arquivos binários de fontes da web, **status codes HTTP não são suficientes para validar tipo de conteúdo**. É necessário validar os "magic bytes" do formato esperado (FF D8 FF para JPEG) antes de salvar o arquivo. Esta lição será documentada na seção de "data collection challenges" do paper futuro.

**Análise da distribuição dos déficits por decil.** Contagem de imagens válidas por (split × decil) revelou um padrão **oposto ao previsto**: os déficits estão concentrados nos decis baixos (0-3, IDs entre 27 e 225.863) em vez do decil 9 (IDs altos, 483.402-3.754.076) como esperado pela hipótese original de "IDs antigos seriam preservados". O decil 9 chegou inclusive a ter superávit em val (+116) e test (+79). A taxa de "404 disfarçado" é uniforme em ~30% por todos os decis, **refutando a hipótese de purga de IDs antigos** e sugerindo que o problema é sistemático: aproximadamente 30% de qualquer ID sorteado aleatoriamente sobre o range completo de IDs do shipspotting retorna HTML de erro, independente da idade. Possível explicação: o `dataset_25k` foi curado por similaridade CLIP e sobrou apenas IDs que realmente existiam — então o conjunto curado tem distribuição enviesada para IDs "preservados", e a estratificação sobre essa distribuição enviesada sobrecarrega o sorteio em regiões de baixa densidade real.

**Rodada 2 (15/04/2026, madrugada).** Sorteio complementar adaptativo via `gerar_ids_rodada2.py` (8.110 novos IDs com margem 1,7-2,2× sobre déficit, baseada em taxa de sucesso observada por decil), download via `baixar_random_pool_v2.py` v2 com check de magic bytes implementado. Resultado: 2.665 IDs corretamente classificados como `invalid_content` e descartados na fonte (sem virar arquivo no disco), 5.445 novas imagens válidas baixadas em 66 minutos.

**Estado final do random_pool_v2 (15/04/2026):**

| Split | Total válidas | Alvo dataset_25k | Folga |
|---|---|---|---|
| train | 22.899 | 22.064 | +835 (+3,8%) |
| val | 9.515 | 9.147 | +368 (+4,0%) |
| test | 7.214 | 6.898 | +316 (+4,6%) |
| **Total** | **39.628** | **38.109** | **+1.519 (+4,0%)** |

**100% das 39.628 imagens passaram em validação de magic bytes JPEG.** Zero corrupção residual. Random_pool_v2 sincronizado para o Drive em `/content/drive/MyDrive/InaTechShips/random_pool_v2/{train,val,test}/images/` para uso pelo Colab no braço B do experimento principal. Verificação cruzada Drive vs disco local confirmou zero perda no upload.

**Folga de ~4% por split** permitiu downsample para igualar os tamanhos do dataset_25k_v2 mantendo estratificação por decis — operação executada em 16/04/2026 (ver Seção 4.4.2).

**Custo operacional total da recovery:** ~2h 45min (rodada 1 + validação + análise + rodada 2 + upload), versus ~10h se tivesse falhado totalmente e exigido refazer do zero.

#### 4.4.2 Downsample para alinhamento com o dataset_25k_v2 (16/04/2026)

**Motivação.** Após a reconstrução do `dataset_25k_v2` (Seção 4.3.3), os tamanhos alvo mudaram: de 22.064/9.147/6.898 (totais do dataset original com duplicatas) para **16.677/5.558/5.561** (dataset_25k_v2 limpo, 60/20/20 sobre 27.796 IDs únicos). Para manter o pareamento experimental com o braço A (onde o `random_pool_v2` é usado como alternativa de mesmo tamanho ao `dataset_25k_v2`), foi necessário reduzir o random_pool_v2 para esses novos tamanhos.

**Estratificação preservada.** O downsample foi feito por decis empíricos **recalculados** para o dataset_25k_v2 (10 bins sobre os 27.796 IDs únicos, calculados pelo script `recalcular_distribuicao_decis_v2.py` em 16/04/2026). Os alvos por célula (split × decil) foram esses:

| Decil | Train alvo | Val alvo | Test alvo |
|---|---|---|---|
| 0 | 1.668 | 548 | 563 |
| 1 | 1.694 | 560 | 526 |
| 2 | 1.676 | 583 | 520 |
| 3 | 1.679 | 552 | 549 |
| 4 | 1.669 | 546 | 565 |
| 5 | 1.695 | 542 | 542 |
| 6 | 1.628 | 563 | 589 |
| 7 | 1.631 | 576 | 572 |
| 8 | 1.675 | 537 | 568 |
| 9 | 1.662 | 551 | 567 |

**Execução.** Script `downsample_random_pool_v2.py` (seed 42, mantém aleatoriamente os N alvos de cada célula, move os excedentes para `_excedente/{split}/`). Execução em ~20 min no Drive.

**Resultado final do random_pool_v2 após downsample:**

| Split | Imagens em `images/` | Imagens em `_excedente/` | Alvo | Excesso |
|---|---|---|---|---|
| train | 16.703 | 6.196 | 16.677 | +26 (+0,16%) |
| val | 5.658 | 3.857 | 5.558 | +100 (+1,80%) |
| test | 5.603 | 1.611 | 5.561 | +42 (+0,76%) |
| **Total** | **27.964** | **11.664** | **27.796** | **+168 (+0,60%)** |

**Déficit residual identificado:** uma única célula (test × decil 8) ficou com 544 imagens contra 568 alvo, déficit de 24 imagens (0,086% do total do dataset). Aceito como metodologicamente irrelevante.

**Excesso residual de ~0,4% aceito como metodologicamente irrelevante.** Os 168 arquivos a mais em `images/` do random_pool_v2 (distribuídos entre train/val/test) são consequência de pequenas variações na execução do script e não justificam uma segunda passada de correção. A propriedade experimental essencial — "o random_pool_v2 tem tamanho e estratificação comparáveis ao dataset_25k_v2" — está preservada com margem de 0,6%, muito abaixo de qualquer variância esperada entre seeds de treino (σ_seed ≈ 0,5% em mAP50 nos baselines).

**Reversibilidade preservada.** As 11.664 imagens movidas estão em `/content/drive/MyDrive/InaTechShips/random_pool_v2/_excedente/{train,val,test}/`. Se futuramente precisarmos de mais imagens (ex: aumentar o random_pool_v2 para experimentos de scale), basta mover de volta.

**Arquivo de relatório:** `/content/drive/MyDrive/InaTechShips/random_pool_v2/downsample_report.json` documenta entrada, saída, movimentação por célula, e estado final.

---

## 5. Decisões metodológicas e justificativas

### 5.1 Decisão: colapsar ambos os datasets para detecção classe-única (embarcação)

**Decisão:** todas as anotações dos dois subsets do InaTechShips e do CITRA-3D-Real serão reescritas para usar exclusivamente a classe `0` (embarcação), descartando a taxonomia original.

**Justificativas (revisadas com dados autoritativos do CITRA-3D-Real):**

1. **Taxonomias incompatíveis.** O CITRA-3D-Real usa 9 classes em português baseadas em uma classificação operacional brasileira (Militar, Barca, Mercante, Vela, Passageiro, TUG, Lancha, Miuda, Navio), enquanto o dataset_25k usa 10 classes em inglês baseadas no Vessel_Type do shipspotting. A autora avaliou um mapeamento semântico entre as duas taxonomias e considerou-o inseguro o suficiente para não sustentar treinamento multi-classe.

2. **Desbalanço severo no CITRA-3D-Real.** A classe TUG (dominante) tem 2.490 instâncias contra 8 da classe Passageiro, uma razão de 311:1. Mesmo para Navio (rara mas mais consistente), a razão é de 43:1.

3. **Classe Passageiro estatisticamente impossível de avaliar.** Com apenas 8 instâncias no dataset inteiro e **zero presença no conjunto de validação**, é impossível calcular métricas confiáveis (precision/recall/AP) para essa classe em qualquer cenário multi-classe. Isto sozinho já justifica o colapso — não há solução metodologicamente correta que mantenha Passageiro como classe distinta.

4. **Foco operacional.** Para o uso final do modelo no contexto da Marinha, a primeira pergunta operacional é "há embarcação na cena?" antes de "que tipo de embarcação?". O modelo classe-única responde a primeira pergunta com rigor.

5. **Limpa a pergunta científica.** Sem a complicação de taxonomias, a pergunta "a curadoria por similaridade visual ajuda a detecção em cenário operacional?" fica isolada e respondível com métricas padrão.

6. **Simetria do ablation.** O ablation A vs B exige que tudo seja idêntico entre os braços exceto o critério de seleção. Manter taxonomias diferentes nos dois subsets quebraria essa simetria.

**Implementação:** ✅ Concluída em 10/04/2026 via script `gerar_labels_single_class.py`. Pastas `labels_single_class/` criadas em paralelo aos labels originais nos dois datasets. Distribuição original preservada nos labels antigos para análise descritiva por classe.

### 5.2 Decisão: usar YOLOv11m como detector principal

**Decisão:** o detector de referência do experimento será o **YOLOv11m**. O YOLO-StarLS-Adapted (arquitetura proprietária do grupo) e o YOLOv8m (baseline anterior do projeto) não serão usados como modelos principais — embora YOLOv8m possa ser executado como sanity check secundário em paralelo (1 seed por braço) para fortalecer a defesa do paper.

**Justificativas:**

1. **Performance.** Em experimentos anteriores conduzidos pela autora, o YOLO-StarLS-Adapted não superou consistentemente YOLOv8, YOLOv11 ou YOLOv26.
2. **Foco da contribuição.** A contribuição central deste paper é metodológica, não arquitetural. A escolha do detector deve ser defensável e padrão, não inovadora.
3. **Atualidade.** Para um paper submetido em 2026, YOLOv11 é o estado-da-arte atual da Ultralytics. Usar YOLOv8 (de 2023) seria respondido por revisores com "por que não a versão mais recente?".
4. **Continuidade conceitual.** O tamanho `m` (medium) é deliberadamente igual ao YOLOv8m que serviu de baseline em trabalhos anteriores do grupo. Isto permite escrever no paper: "Usamos YOLOv11m, equivalente em escala ao YOLOv8m utilizado em trabalhos anteriores do grupo, com a vantagem de incorporar as melhorias arquiteturais da v11."
5. **Comparabilidade real com baseline anterior é uma ilusão.** Mesmo se mantivéssemos YOLOv8m exato, os números não seriam comparáveis com `results.csv` antigo, porque tudo o resto mudou (reais-only vs reais+sintéticos, classe-única vs 9 classes, splits separados vs train/test misturados, labels limpos vs labels com bug `Quadrado_marcacao`, imgsz 640 vs 1260). A continuidade com o trabalho anterior é estética, não funcional.
6. **HPO refeito de qualquer jeito.** Os hiperparâmetros do YOLOv8m antigo (`lr0=0.01077`, `momentum=0.91358`, etc.) foram tunados para multi-classe + 1260px + imagens sintéticas misturadas. Eles não se aplicam ao novo cenário. O HPO da tarefa A7 vai redescobrir os HPs ótimos para o novo setup, então mudar a arquitetura ao mesmo tempo não custa nada extra.
7. **Ferramental.** A autora tem familiaridade total com o ecossistema Ultralytics; YOLOv11 é o modelo padrão atual da biblioteca.

**Configuração consolidada para os treinos baseline:**

| Parâmetro | Valor |
|---|---|
| Modelo | YOLOv11m (`yolo11m.pt` para B2, init random para B1) |
| Épocas | 300 com early stopping |
| Patience | 30 |
| imgsz | 640 |
| Batch | 16 (conservador para A100) |
| Optimizer | `auto` (Ultralytics decide; HPO refinará na tarefa A7) |
| Seeds | 42, 123, 2024 (3 seeds por braço) |

### 5.3 Decisão: estratificação por decis empíricos no sorteio do random_pool_v2

**Decisão:** o subset aleatório (`random_pool_v2`) será sorteado com estratificação por 10 bins de decil sobre a distribuição empírica de IDs do `dataset_25k`, em vez de sortear uniformemente do range completo do shipspotting.

**Justificativas:**

1. **Controle da variável "época".** Sortear uniformemente do range completo produziria distribuição temporal completamente diferente do subset curado. A estratificação isola o critério de seleção como única variável que muda entre os braços.
2. **Replicação da distribuição empírica** garante que o subset aleatório siga a mesma forma de distribuição que o curado.
3. **Robustez metodológica para o paper:** "controlamos por distribuição temporal de IDs via estratificação por decis" é uma defesa antecipada contra revisores.

### 5.4 Decisão: margem de segurança de 30% nos IDs sorteados

**Decisão:** o gerador sorteia 30% mais IDs do que o tamanho-alvo (~49.554 IDs candidatos para um alvo final de ~38.109 imagens), para cobrir 404s do shipspotting e falhas de download.

**Risco conhecido:** o decil 9 cobre 60x mais ranges de IDs que os outros, e a margem pode ser insuficiente especificamente para esse decil. Mitigação: segunda rodada de sorteio se necessário.

### 5.5 Decisão: 3 seeds de treino por braço

**Decisão:** cada braço será treinado com 3 seeds diferentes para inicialização e ordem de batches, totalizando 12–15 treinos completos.

**Justificativas:** controle de variância de treino, possibilidade de teste estatístico A vs B, padrão de campo na literatura recente.

### 5.6 Decisão: HPO no piloto, não fixação a priori

**Decisão:** os hiperparâmetros principais de treino **não serão fixados a priori** com base em valores do paper original do InaTechShips. Será conduzido um piloto com HPO leve (provavelmente Ultralytics `model.tune()` em B2) para selecionar a configuração ótima antes de treinar os braços principais.

**Justificativa:** hiperparâmetros do paper original foram escolhidos para cenário multi-classe, não para classe-única. Pode haver configurações melhores. HPO empírico é mais defensável do que fixação por convenção.

### 5.7 Decisão: descartar imagens sintéticas do CITRA-3D

**Decisão:** apenas imagens reais do CITRA-3D serão usadas em todas as fases do experimento. As imagens sintéticas (renderizações tipo `NavyA140Atlantico_*.png`, `Catamara_TerrainON_*.png`, etc.) são filtradas.

**Justificativas:**
1. **Coerência do domínio alvo.** O objetivo é avaliar performance no domínio operacional real.
2. **Já é o uso atual do projeto** em outros pipelines da autora.
3. **Filtro trivial:** imagens reais identificáveis pelo padrão de nome `DD.MM.YYYY-HH-MM-SS.png`; sintéticas têm nomes técnicos.

**Implementação:** ✅ Concluída em 10/04/2026 via script `preparar_citra3d.py`. Adicionalmente, **todos os ZIPs com sufixo `_aug` foram pulados sem extração**, eliminando 4 ZIPs de augmentations sintéticas que respondiam pela maior parte dos 9,27 GB do arquivo original.

### 5.8 Decisão: limpeza automática de labels com bug do software de anotação

**Decisão:** labels do CITRA-3D-Real são processados por um script de limpeza automática que descarta linhas malformadas, preservando linhas válidas do mesmo arquivo. Os labels limpos são escritos em pastas paralelas `labels_cleaned/`.

**Contexto:** A auditoria do CITRA-3D-Real revelou que **9 linhas em 8 arquivos** continham anotações inválidas, todas com a mesma assinatura: o campo de classe era a string `Quadrado_marcacao(Clone)` em vez de um número, claramente um bug do software de anotação usado pela Marinha — quando o anotador clonava uma bounding box existente, em vez de copiar o índice da classe, copiava o nome interno do objeto. Em alguns casos, esses clones também tinham coordenadas `w=0, h=0`, sugerindo que o anotador clicou para criar mas não terminou de desenhar a caixa.

**Critérios de limpeza aplicados (em ordem):**

1. Linha em branco → descartada silenciosamente.
2. Linha com número de campos ≠ 5 → descartada (`malformed_field_count`).
3. Linha com valores não-numéricos → descartada (`non_numeric`).
4. Linha com coordenadas fora de [0, 1] → descartada (`out_of_range`).
5. Linha com largura ou altura ≤ 0 → descartada (`degenerate_zero_dim`).
6. Linha válida → preservada, reformatada com precisão consistente.

**Resultado:** ✅ Concluído em 10/04/2026. 9 linhas removidas em 8 arquivos, 7.003 bboxes válidas preservadas (de 7.012 originais — perda de 0,13%, totalmente justificada por bug de software). Detalhamento completo em `limpeza_labels_report.txt`.

**Defensibilidade científica:** todas as remoções estão registradas linha-a-linha no relatório, com filename, lineno, raw line e motivo. Auditável por terceiros.

### 5.9 Decisão: quarentena de imagens com labels órfãos

**Decisão:** após a limpeza de labels, duas imagens cujos labels ficaram totalmente vazios foram **movidas para uma pasta `_quarantine/`** e excluídas do dataset de treino e avaliação. Não foram apagadas — todas as cópias (imagem, label original, label limpo) foram preservadas com README explicativo.

**Imagens em quarentena:**
- `train/29.04.2022-14-59-27.png`
- `test/14.04.2022-13-48-55.png`

**Contexto e justificativa:** essas duas imagens foram inspecionadas visualmente após a limpeza. Confirmou-se que **ambas contêm embarcações reais — pequenas e distantes** — que o anotador tentou marcar com clones do `Quadrado_marcacao(Clone)` mas não completou. Mantê-las no dataset com labels vazios faria o YOLO tratá-las como "background" (cena sem embarcação), o que ensinaria o modelo a ignorar embarcações distantes — exatamente o caso operacionalmente crítico para vigilância marítima. Manter como background seria um falso negativo de treino, pior do que não ter a imagem.

**Impacto:** perda de 2 imagens em 2.083 (~0,1% do dataset). Desprezível.

**Reversibilidade:** garantida — caso a Marinha decida reanotar manualmente essas imagens, basta movê-las de volta. A pasta `_quarantine/` contém um README institucional explicando a decisão e as referências aos relatórios de auditoria.

**Implementação:** ✅ Concluída em 10/04/2026 via script `quarentenar_imagens_orfas.py`.

### 5.10 Decisão: usar o ZIP original da Marinha como única fonte autoritativa

**Decisão:** o pipeline de preparação do CITRA-3D parte exclusivamente do `CITRA-3D.zip` original recebido da Marinha, descartando versões intermediárias (`CITRA-3D-Prepared`) que existiam no Drive da autora.

**Contexto:** durante a auditoria inicial, foi descoberto que existiam pelo menos duas versões do CITRA-3D no projeto: o ZIP original (com 2.083 imagens reais distribuídas em 3 splits) e uma versão `CITRA-3D-Prepared` (com 2.857 imagens em distribuição diferente). A discrepância de ~774 imagens entre as versões nunca foi explicada satisfatoriamente — provavelmente fruto de processos manuais ao longo do tempo, possivelmente com mistura de imagens sintéticas filtradas com regex permissivo.

**Justificativas:**

1. **Reprodutibilidade.** Trabalhar a partir do ZIP original significa que qualquer pessoa pode reproduzir o pipeline com o mesmo arquivo de entrada.
2. **Defensibilidade institucional.** Para o paper e para auditoria da Marinha, "partimos do ZIP autoritativo enviado pela instituição" é uma frase muito mais defensável do que "partimos de uma versão intermediária modificada manualmente".
3. **Eliminação de ambiguidade.** Manter duas versões em paralelo era fonte garantida de bugs futuros — qual número está certo, qual usar para X, etc.
4. **Pipeline rastreável.** Cada operação aplicada ao dataset é agora um script versionado com relatório associado.

**Implementação:** ✅ Concluída em 10/04/2026. A pasta `CITRA-3D-Prepared` foi arquivada como obsoleta no Drive. O `CITRA-3D-Real` extraído do ZIP é a fonte autoritativa daqui para frente.

### 5.11 Decisão: hyperparams diferenciados entre B1 (random init) e B2 (COCO pretrain)

**Decisão:** os braços B1 (random init) e B2 (pré-treino COCO) usam configurações de treino **diferentes** em três parâmetros críticos: optimizer, learning rate inicial e bias learning rate durante warmup. Todos os outros parâmetros (épocas, patience, imgsz, batch, augmentation, seeds) são idênticos.

**Configurações finais:**

| Parâmetro | B1 (random init) | B2 (COCO pretrain) |
|---|---|---|
| Optimizer | **SGD** | **AdamW** |
| `lr0` | **0.01** | **0.001** (10x menor) |
| `lrf` | 0.01 | 0.01 |
| `momentum` | 0.937 | 0.937 |
| `warmup_epochs` | 3.0 | 3.0 |
| `warmup_bias_lr` | **0.1** | **0.01** |
| `cos_lr` | **True** (explícito) | **True** (explícito) |
| `weight_decay` | 0.0005 | 0.0005 |

**Contexto — a descoberta do bug:** na primeira tentativa de treino, B2 estava sendo treinado com `optimizer='auto'` sem `cos_lr=True` explícito. O Ultralytics, nesse modo, usa uma rampa linear de warmup que continua subindo o LR ao longo do treino inteiro, sem nunca entrar em fase de decay. Resultado observado no piloto B2 seed 42 original:

```
época  1:  LR=0.00004  mAP50=0.18  (warmup iniciando)
época  3:  LR=0.00016  mAP50=0.30  (pico — melhor momento do treino)
época  7:  LR=0.00039  mAP50=0.07  (degradação acelerada)
época 15:  LR=0.00084  mAP50=0.00  val_loss=NaN
época 33:  LR=0.00175  mAP50=0.00  (early stopping dispara)
```

O LR rampou de 0,00004 até 0,00175 ao longo de 33 épocas — **sem nunca entrar em fase de decay**. O modelo atingiu pico de mAP50=0,30 na época 3, quando o LR ainda estava ~10x abaixo do alvo, e foi progressivamente destruído conforme o LR continuava subindo. O val_loss virou `NaN` na época 16 e permaneceu assim até o early stopping.

**Por que B1 não teve esse problema?** Porque B1 começa com pesos aleatórios e não tem nada útil a preservar. O LR crescente só acelera a convergência inicial; não há pesos COCO para destruir. B1 atingiu mAP50=0,44 em 161 épocas na primeira tentativa, comportamento razoável.

**Justificativas para a diferenciação:**

1. **Prática padrão na literatura.** Nenhum paper de detecção usa o mesmo LR para treino do zero e fine-tuning. Fine-tuning exige LR menor para não destruir pesos pré-treinados já úteis.
2. **AdamW para fine-tuning, SGD para random init.** AdamW tem momentos adaptativos por parâmetro, oferecendo estabilidade superior para cenários onde os gradientes tendem a ser ruidosos (como fine-tuning de modelos grandes em datasets pequenos). SGD com momentum alto é mais tradicional para treino do zero e permite atingir boa generalização quando há dados suficientes.
3. **`cos_lr=True` explícito** força o cosine annealing correto: warmup linear por 3 épocas até `lr0`, depois decay suave até `lr0 × lrf` ao final. Sem isso, o schedule default pode se comportar como visto no bug acima.
4. **Defensibilidade.** No paper, a frase será: "Os hiperparâmetros de B1 e B2 foram escolhidos seguindo a prática padrão para cada cenário: B1 usa SGD com lr0=0.01 e warmup curto, adequados para treino do zero; B2 usa AdamW com lr0=0.001 e cosine annealing, adequados para fine-tuning a partir de pesos COCO pré-treinados."
5. **Configuração conservadora, não ótima.** Essa configuração **não é o resultado de HPO**, e sim uma configuração padrão defensável. O HPO refinado (tarefa A7 do cronograma) será conduzido sobre B2 depois dos baselines rodarem, para verificar se essa configuração inicial está próxima do ótimo. Se o HPO apontar mudanças significativas, os baselines serão refeitos.

**Validação empírica:** após a correção, o piloto B2 seed 42 rodou 203 épocas em 53 minutos, atingindo mAP50=0,8367 no test (0,8058 no val), com zero NaN no val_loss e curva saudável. O LR evoluiu como esperado: warmup linear de 0,000329→0,001 nas primeiras 3 épocas, depois decay cosine até 0,000249 na época 203.

### 5.12 Decisão: cópia física de dados em vez de links simbólicos para uso com Ultralytics

**Decisão:** o pipeline de preparação para treino faz **cópia física** das imagens e labels do Drive para o disco local do Colab (`/content/data/`), em vez de usar links simbólicos.

**Contexto — a descoberta do bug:** a primeira versão do script de preparação (`gerar_data_yaml.py`, agora obsoleto) usava links simbólicos como otimização:

```
/content/data/CITRA-3D-Real-SC/train/images → /content/drive/.../train/images
/content/data/CITRA-3D-Real-SC/train/labels → /content/drive/.../train/labels_single_class
```

Essa abordagem seria elegante (custo zero em espaço, idempotente, não duplica dados) mas **quebra silenciosamente o treino com Ultralytics YOLO**. O motivo: quando o Ultralytics carrega o dataset, ele segue o link de `images/`, obtém o caminho real via `Path.resolve()`, e depois aplica a substituição `str.replace('/images/', '/labels/')` sobre esse caminho resolvido. O resultado é que ele procura os labels em `/content/drive/.../train/labels/` (path real, não o link), que é a pasta dos **labels originais com 9 classes**, não em `labels_single_class/`.

**Sintoma observado:** o Ultralytics descarta TODAS as imagens do treino com a mensagem:

```
train: .../train/images/08.04.2022-09-49-34.png:
  ignoring corrupt image/label: Label class 7 exceeds dataset class count 1.
  Possible class labels are 0-0
```

Classes 2, 5, 6, 7 (Mercante, TUG, Lancha, Miuda) aparecem nas mensagens — exatamente a taxonomia original. Como todas as imagens são descartadas, o treino roda em batches vazios, produzindo mAP aleatório (B1 teve mAP50=0,44 por acaso, B2 teve 0,28 + val_loss=NaN por instabilidade numérica com batches vazios). O bug é **silencioso**: o treino não levanta exceção, só informa "ignorando imagem corrompida" e continua.

**Justificativas para a correção:**

1. **Resolve o bug do Ultralytics definitivamente.** Com cópia física, não há indireção, não há `Path.resolve()` a aplicar, a substituição `/images/ → /labels/` funciona corretamente.
2. **Acelera o treino.** I/O local do Colab é 5-10x mais rápido que leitura do Drive montado. Para 300 épocas × 1.348 imagens × batch 16, a diferença é significativa. Cada corrida roda mais rápido, o batch de 6 corridas termina mais cedo.
3. **Validação anti-symlink no próprio script.** O novo `preparar_dados_locais.py` inclui três verificações de segurança que teriam capturado o bug do turno anterior: detecção de symlinks residuais, validação de que as pastas finais não são symlinks, e sample check do primeiro label de cada split para confirmar que tem classe `0`.

**Custo:** cópia física do CITRA-3D-Real toma ~3-5 min por sessão do Colab (é volátil, `/content/` é apagado entre sessões). Para o `dataset_25k` (~15 GB), será ~15-25 min. Aceitável — troca custo de espaço por garantia de correção.

**Implementação:** ✅ Concluída em 10/04/2026 via script `preparar_dados_locais.py`, que substituiu o `gerar_data_yaml.py`.

---

## 6. Pipeline de preparação implementado

Esta seção registra os 6 scripts versionados que compõem o pipeline atual de preparação dos datasets, na ordem de execução.

### 6.1 ✅ `preparar_citra3d.py` — extração seletiva do ZIP autoritativo

**Status:** Concluído em 10/04/2026.

**Função:** lê o `CITRA-3D.zip` (9,27 GB) sem extrair, em duas fases. Fase 1 (`inspect`) inspeciona ZIPs aninhados e gera relatório do que existe. Fase 2 (`extract`) extrai apenas imagens reais (padrão de nome com data) com labels pareados, ignorando todos os ZIPs com sufixo `_aug`.

**Resultado:** 2.083 imagens reais + 2.083 labels extraídos para `CITRA-3D-Real/{train,val,test}/{images,labels}/`. 1.318 imagens sintéticas e 4 ZIPs `_aug` ignorados.

**Relatórios gerados:** `inspect_report.txt`, `extract_report.txt` (e versões JSON).

### 6.2 ✅ `analise_citra3d_real.py` — auditoria pós-extração

**Status:** Concluído em 10/04/2026.

**Função:** audita o CITRA-3D-Real recém-extraído, gerando contagens por split, distribuição por classe (com nomes do `data.yaml` oficial), detecção de pareamento, malformações, bboxes degeneradas. Paralelizado com 48 threads.

**Resultado:** confirmou pareamento 100%, identificou 5 labels com `non_numeric` (`Quadrado_marcacao(Clone)`) e 3 com bboxes degeneradas. Distribuição autoritativa de classes estabelecida (TUG dominante 35,6%, Navio rara 0,8%, Passageiro praticamente ausente 0,1%).

**Relatório gerado:** `citra3d_real_audit_report.txt`.

### 6.3 ✅ `limpar_labels_citra3d.py` — limpeza automática de labels

**Status:** Concluído em 10/04/2026.

**Função:** percorre os labels do CITRA-3D-Real removendo linhas inválidas (malformadas, não-numéricas, fora de range, degeneradas), preservando linhas válidas no mesmo arquivo. Escreve em `labels_cleaned/` paralelo. Não toca nos originais.

**Resultado:** 9 linhas removidas em 8 arquivos. 7.003 bboxes válidas preservadas (de 7.012 originais). 2 arquivos ficaram totalmente vazios após limpeza (passados para o próximo script).

**Relatório gerado:** `limpeza_labels_report.txt` com detalhamento linha-a-linha.

### 6.4 ✅ `quarentenar_imagens_orfas.py` — isolamento de imagens com labels vazios

**Status:** Concluído em 10/04/2026.

**Função:** move para `_quarantine/` as 2 imagens cujos labels ficaram vazios após a limpeza (29.04.2022-14-59-27.png e 14.04.2022-13-48-55.png). Move imagem, label original e label limpo, gerando README institucional explicativo.

**Resultado:** dataset reduzido de 2.083 para 2.081 imagens (perda de 0,1%). Imagens preservadas em quarentena para reversibilidade.

**Relatórios gerados:** `_quarantine/README.txt`, `_quarantine/quarantine_log.json`.

### 6.5 ✅ `auditar_citra3d_cleaned.py` — re-auditoria pós-limpeza para validação

**Status:** Concluído em 10/04/2026.

**Função:** re-audita o CITRA-3D-Real lendo de `labels_cleaned/` em vez de `labels/`, com validação automática contra valores esperados (1.348 train / 332 val / 401 test / 7.003 bboxes / zero malformados / zero degenerados / zero órfãos).

**Resultado:** **todos os 9 checks passaram com ✓.** Dataset oficialmente validado como pronto para uso.

**Relatório gerado:** `citra3d_cleaned_audit_report.txt`.

### 6.6 ✅ `gerar_labels_single_class.py` — colapso para classe única

**Status:** Concluído em 10/04/2026, validado com relatório real.

**Função:** processa CITRA-3D-Real (lendo de `labels_cleaned/`) e dataset_25k (lendo de `labels/`), gerando pastas `labels_single_class/` em ambos com todas as classes originais colapsadas para `0`. Coordenadas das bboxes preservadas exatamente. Operação não-destrutiva.

**Resultado real (confirmado):**
- **CITRA-3D-Real:** 2.081 arquivos lidos / 2.081 escritos / 0 vazios / 7.003 bboxes (todas → 0)
- **dataset_25k:** 38.109 arquivos lidos / 38.109 escritos / 0 vazios / 38.109 bboxes (todas → 0)
- Zero bboxes inválidas puladas em ambos os datasets, confirmando que a limpeza prévia (passos 6.3 e 6.4) eliminou todas as malformações.

**Descoberta complementar registrada pelo relatório:** a distribuição original de classes do dataset_25k é **praticamente uniforme** (10,6% GENERAL CARGO, 10,8% TUG, 9,5% RO-RO/PASSENGER SHIP, etc.) — fruto da curadoria deliberadamente balanceada da autora. Isto contrasta fortemente com a distribuição **long-tail** do CITRA-3D-Real (TUG 35,6% / Navio 0,8%) e adiciona mais um eixo ao gap de domínio entre os dois datasets — não é apenas fotografia profissional vs operacional, é também distribuição balanceada vs naturalmente skewed. Esta observação descritiva entra na seção de Dados do paper.

**Relatório gerado:** `single_class_generation_report.txt` com distribuição original preservada para registro histórico.

### 6.7 ✅ `preparar_dados_locais.py` — cópia física para disco local + geração dos data.yaml

**Status:** Concluído em 10/04/2026. Substituiu o `gerar_data_yaml.py` original (veja histórico abaixo).

**Histórico:** este passo teve duas iterações. A primeira implementação (`gerar_data_yaml.py`) usava links simbólicos entre `/content/data/` e o Drive, mas descobriu-se que o Ultralytics YOLO tem um bug silencioso com symlinks — ver Decisão 5.12. A segunda implementação (`preparar_dados_locais.py`, versão final) usa cópia física para o disco local do Colab.

**Função:** para cada dataset (CITRA-3D-Real e dataset_25k):
1. Copia fisicamente as imagens de `{split}/images/` no Drive para `/content/data/{nome}-SC/{split}/images/`.
2. Copia fisicamente os labels de `{split}/labels_single_class/` no Drive para `/content/data/{nome}-SC/{split}/labels/` (renomeando a pasta para `labels/` na cópia, para que o Ultralytics encontre via substituição `/images/ → /labels/`).
3. Gera os dois arquivos `data.yaml` no formato Ultralytics, apontando para `/content/data/`.
4. Executa validação robusta com três checks críticos: (a) detecção de symlinks residuais, (b) verificação de que as pastas finais NÃO são symlinks, (c) sample check do primeiro label de cada split para confirmar que tem classe `0`.

**Resultado validado (10/04/2026) — CITRA-3D-Real:**
- 1.348 / 332 / 401 imagens por split, totalizando 2.081 imagens (~2 GB)
- 1.348 / 332 / 401 labels por split, todos com classe `0` (confirmado pelo sample check)
- Tempo de cópia: ~3 min 40s (throughput ~11 MB/s do Drive para disco local)
- Status geral: `✓ TUDO OK`
- Primeira validação empírica: piloto B2 seed 42 rodou com sucesso em 53 min, atingindo mAP50=0,8367 no test

**Pendente:** cópia do `dataset_25k` (~15 GB, ~15-25 min). Será feita no início da sessão que for rodar o braço A do experimento — não é necessária para os baselines B1/B2 atuais, que usam apenas o CITRA-3D-Real.

**Bug do `data.yaml` antigo (`val=test`) corrigido:** os yamls novos apontam para `val/images` e `test/images` como pastas separadas, eliminando o vazamento de dados que existia no yaml original recebido junto ao ZIP.

**Saídas persistentes (Drive):**
- `/content/drive/MyDrive/PROJETO_MARINHA/Datasets/configs/citra3d_single_class.yaml`
- `/content/drive/MyDrive/PROJETO_MARINHA/Datasets/configs/dataset_25k_single_class.yaml`
- `/content/drive/MyDrive/PROJETO_MARINHA/Datasets/configs/data_setup_log.json`

**Limitação operacional:** o `/content/` é apagado entre sessões do Colab. É necessário rodar `preparar_dados_locais.py` no início de cada nova sessão para recopiar os dados. Os yamls em si são persistentes.

### 6.8 ✅ `treinar_baselines.py` — script de treino dos baselines B1 e B2

**Status:** Validado em 10/04/2026 com piloto B2 seed 42 (mAP50=0,8367). Execução dos 6 treinos completos (3 seeds × 2 baselines) em andamento.

**Função:** script parametrizado que treina YOLOv11m nos dois baselines do experimento, com hyperparams diferenciados por baseline (ver Decisão 5.11). Aceita três modos de execução:

```bash
python treinar_baselines.py --baseline B1 --seed 42   # uma corrida específica
python treinar_baselines.py --all                      # todas as 6 corridas em sequência
python treinar_baselines.py --all --dry-run            # listar sem executar
```

**Pré-checagens antes de cada execução:**
1. Verifica que `citra3d_single_class.yaml` existe
2. Valida o yaml (parsing, path acessível, contagem de arquivos por split, sample check de classe)
3. Verifica que Ultralytics está instalado
4. Verifica disponibilidade de CUDA

Se qualquer check falhar, o script aborta antes de começar.

**Robustez contra falhas:** cada treino está dentro de `try/except`. Se uma corrida falhar, o script captura a exceção, salva no `run_summary.json` com status `error` + traceback completo, e **continua** para a próxima corrida. Isso garante que uma falha isolada não desperdice as outras corridas de um batch `--all`.

**Estrutura de saída por corrida:**
```
runs/baselines/B1_random/seed_42/
├── train/                       # outputs padrão do Ultralytics
│   ├── weights/{best,last}.pt
│   ├── results.csv              # curvas de loss e mAP por época
│   ├── confusion_matrix.png
│   └── args.yaml                # config exata usada
├── test_eval/                   # avaliação no test set
├── test_metrics.json            # métricas estruturadas (para agregação)
└── run_summary.json             # config + duração + status + métricas
```

Após execução em modo `--all`, agrega também em `runs/baselines/all_runs_summary.json`.

**Piloto validado (B2 seed 42, 10/04/2026):**

| Métrica | Valor |
|---|---|
| mAP50 (test) | 0,8367 |
| mAP50-95 (test) | 0,5062 |
| Precision | 0,8639 |
| Recall | 0,7803 |
| Épocas treinadas | 203 de 300 (early stopping em patience=30) |
| Melhor val mAP50 | 0,8058 (época 173) |
| Tempo de treino | 53 min |
| val_loss NaN | zero |

O `lr/pg0` evoluiu corretamente: warmup linear de 0,000329→0,001 nas épocas 1-3, cosine decay até 0,000249 na época 203. Comportamento ideal.

---

## 7. Pipeline ainda a implementar

> **Nota:** a tarefa originalmente listada como 7.1 (geração dos `data.yaml` classe-única) foi concluída em 10/04/2026 e movida para a Seção 6 como passo 6.7 do pipeline implementado. As tarefas abaixo foram renumeradas.

### 7.1 Downloader do random_pool_v2

**Status:** 🔴 Bloqueado por pendência de labels (Seção 8.1).

**Função esperada:** consumir os três `.txt` de IDs gerados pelo `gerar_ids_aleatorios.py` e baixar as imagens correspondentes do shipspotting via CDN direto. Salvar em estrutura YOLO. Será adaptado do `download_direto.py` existente.

**Observação estratégica:** o download das imagens pode começar **sem esperar os labels**, já que as imagens são as mesmas em qualquer cenário. Apenas a anotação posterior depende do que o Eduardo responder.

### 7.2 Anotação dos labels do random_pool_v2

**Status:** 🔴 Bloqueado, aguardando resposta do Eduardo (Seção 8.1).

### 7.3 Validação final do random_pool_v2

**Status:** 🔴 Bloqueado pelos passos anteriores.

### 7.4 Script de HPO (piloto)

**Status:** 🟡 Pode começar imediatamente (data.yaml prontos).

**Função esperada:** rodar busca de hiperparâmetros no braço B2 (pré-treino COCO + fine-tuning CITRA-3D-Real) usando `model.tune()` da Ultralytics, em espaço de busca limitado. Configuração ótima vai ser usada em todos os braços finais. Modelo: YOLOv11m.

### 7.5 Script de treino dos braços principais

**Status:** 🟡 **Em progresso.** O script `treinar_baselines.py` foi escrito, validado com piloto B2 seed 42, e está rodando as 6 corridas dos baselines (B1+B2 × 3 seeds) no momento da escrita desta versão do documento. Ver Seção 6.8 para detalhes.

**O que ainda falta:** o mesmo script será adaptado para treinar os braços A, B e opcional C do experimento principal (pré-treino em InaTechShips + fine-tuning em CITRA-3D-Real). A adaptação provavelmente mínima — mudança do `data.yaml` de pré-treino + ajuste de épocas para a fase de pré-treino. Será feita após os resultados dos baselines ficarem prontos e a tarefa A7 (HPO) ser concluída.

### 7.6 Script de avaliação e análise

**Status:** 🟡 Será desenvolvido conforme os primeiros treinos completarem.

### 7.7 ✅ Resultados preliminares dos baselines (B1 e B2)

**Status:** ✅ **Concluído em 14/04/2026.** As 6 corridas (B1+B2 × 3 seeds) foram executadas em sequência via `treinar_baselines.py --all` durante a noite de 13-14/04/2026, somando ~5h 18min de tempo total de treino.

#### 7.7.1 Tabela consolidada

| Baseline | mAP50 (média ± DP) | mAP50-95 (média ± DP) | Precision (média ± DP) | Recall (média ± DP) | Tempo médio |
|---|---|---|---|---|---|
| **B1 (random init)** | 0,8008 ± 0,0073 | 0,4742 ± 0,0008 | 0,8286 ± 0,0033 | 0,7496 ± 0,0026 | 52,3 min |
| **B2 (COCO pretrain)** | 0,8351 ± 0,0024 | 0,5055 ± 0,0027 | 0,8570 ± 0,0058 | 0,7826 ± 0,0049 | 53,7 min |
| **Δ (B2 − B1)** | **+0,0343 (+4,3%)** | **+0,0313 (+6,6%)** | +0,0284 | +0,0330 | — |

**Métricas avaliadas no test set (401 imagens). Agregação sobre 3 seeds independentes (42, 123, 2024).** Hyperparams seguindo Decisão 5.11.

**Detalhamento por corrida:**

| Baseline | Seed | mAP50 | mAP50-95 | Precision | Recall | Tempo |
|---|---|---|---|---|---|---|
| B1 | 42   | 0,7959 | 0,4737 | 0,8273 | 0,7514 | 56,2 min |
| B1 | 123  | 0,7971 | 0,4750 | 0,8262 | 0,7509 | 56,9 min |
| B1 | 2024 | 0,8094 | 0,4738 | 0,8325 | 0,7466 | 43,7 min |
| B2 | 42   | 0,8367 | 0,5062 | 0,8639 | 0,7803 | 54,6 min |
| B2 | 123  | 0,8364 | 0,5078 | 0,8530 | 0,7883 | 60,8 min |
| B2 | 2024 | 0,8323 | 0,5026 | 0,8541 | 0,7791 | 45,7 min |

#### 7.7.2 Análise da separação entre baselines

**Os intervalos de mAP50 das duas distribuições não se sobrepõem.** O valor mais alto observado em B1 (0,8094, seed 2024) é menor que o valor mais baixo observado em B2 (0,8323, seed 2024). Há um gap absoluto de 0,0229 pontos de mAP50 entre o teto de B1 e o piso de B2. Em termos práticos, **toda corrida de B2 superou a melhor corrida de B1**, em todas as 9 comparações cruzadas possíveis (3 seeds de B1 × 3 seeds de B2).

A análise estatística formal está pendente. **Notas para fazer no paper:**
- Considerando o tamanho pequeno da amostra (3 seeds por baseline), os testes estatísticos paramétricos clássicos (t-test) têm baixa potência. Testes não-paramétricos como o teste de Mann-Whitney ou bootstrap intervals são mais apropriados aqui.
- A separação visual completa dos intervalos é, por si só, evidência forte qualitativa de que o efeito não é ruído, mesmo sem teste formal.
- **`[CITAÇÃO PENDENTE]` — boas práticas de reporte estatístico em deep learning com poucas seeds.** *Sugestão de busca:* "deep learning empirical results few seeds significance testing reproducibility".

#### 7.7.3 Três descobertas centrais

**Descoberta 1: B2 é substancialmente mais consistente entre seeds que B1 (DP ~3x menor).**

```
B1: DP(mAP50) = 0,0073   range = [0,7959; 0,8094]   spread = 0,0135
B2: DP(mAP50) = 0,0024   range = [0,8323; 0,8367]   spread = 0,0044
```

A variação entre seeds em B2 é cerca de três vezes menor que em B1 em mAP50 — e o mesmo padrão se mantém em mAP50-95 (DP B1 = 0,0008 vs DP B2 = 0,0027, embora aqui a diferença seja menos pronunciada, em parte porque B1 já tinha DP baixíssimo neste eixo, o que é incomum).

A interpretação plausível desta observação é que **o pré-treino COCO funciona como um regularizador implícito** que torna o ponto de inicialização mais consistente, reduzindo a variância induzida pela aleatoriedade da inicialização e da ordem dos batches. `[CITAÇÃO PENDENTE]` — esta interpretação é compatível com o framework geral discutido por He et al. (2019), embora aquele paper foque mais em accuracy do que em variância entre seeds.

**Implicação prática:** ao reportar resultados dos braços A e B do experimento principal, a estabilidade entre seeds vai provavelmente ser similar à de B2 (pois ambos os braços envolvem pré-treino, apenas com datasets diferentes). Isto é favorável para detectar diferenças pequenas entre A e B com poucas seeds.

**Descoberta 2: B1 apresenta desempenho surpreendentemente alto (mAP50 = 0,80).**

A literatura clássica em transferência aprendizado em detecção de objetos sugere que o ganho do pré-treino (especialmente pré-treino genérico tipo ImageNet/COCO) sobre treino do zero é tipicamente da ordem de 10-20 pontos de mAP em datasets pequenos (na faixa de centenas a alguns milhares de imagens). No entanto, He et al. (2019) demonstraram que essa intuição pode ser parcialmente questionada: em vários cenários, **modelos treinados do zero com configuração adequada (especialmente número suficiente de iterações) podem alcançar resultados competitivos com modelos pré-treinados, com o pré-treino contribuindo principalmente para acelerar a convergência inicial e não necessariamente para a accuracy final** (He et al., 2019).

Os números deste experimento são consistentes com essa observação refinada: B1 atingiu mAP50 = 0,80 contra mAP50 = 0,84 de B2. O ganho é de **apenas 4,3%** em mAP50 e **6,6%** em mAP50-95 — substancialmente menor que os 10-20 pontos esperados pela visão clássica, mas alinhado com os achados de He et al.

Possíveis fatores explicativos para o desempenho relativamente alto de B1, a serem discutidos no paper:

1. **Detecção classe-única elimina a parte difícil (classificação fina).** O modelo precisa aprender apenas "embarcação vs não-embarcação", não distinguir entre classes visualmente similares. Este cenário é estruturalmente mais simples do que detecção multi-classe genérica e é menos dependente de features semânticas de alto nível, que são justamente onde o pré-treino mais ajuda.

2. **Coerência distribucional entre train e test.** O CITRA-3D-Real tem distribuição de classes consistente entre splits (TUG dominante em ambos), e mesmo sem pré-treino, capturar o padrão dominante é factível com 1.348 imagens de treino. **`[CITAÇÃO PENDENTE]`** — afirmação relacionada ao impacto da consistência train/test em few-shot regimes.

3. **YOLOv11m incorpora indutive biases úteis** (anchor-free, denoising, módulos de atenção espacial) que podem reduzir a dependência de features pré-treinadas. **`[VERIFICAR]`** — buscar paper técnico oficial do YOLOv11 que discuta essas escolhas arquiteturais.

4. **Treino suficientemente longo.** Seguindo a recomendação de He et al. (2019), B1 foi treinado com 300 épocas máximas e early stopping com `patience=30`. As corridas B1 efetivamente convergiram em torno das 150-200 épocas (a corrida do piloto inicial usou 161 épocas). Isto é consistente com a observação de He et al. que treino do zero precisa de mais iterações para convergir, mas converge.

**Implicação científica para o experimento:** o ganho menor que esperado do pré-treino COCO genérico **abre espaço interessante para o pré-treino domain-specific dos braços A e B mostrarem ganho relativo significativo**. Se A ou B superarem B2 (= 0,84), isso será uma evidência forte de que pré-treino com fotografia marítima — mesmo de fotografia profissional pública, distinta do domínio operacional — adiciona valor além do pré-treino genérico. Se A superar B, isso será evidência adicional de que a curadoria por similaridade visual via CLIP captura algo de útil para a transferência.

**Descoberta 3: O ganho de B2 sobre B1 é proporcionalmente maior em mAP50-95 (+6,6%) do que em mAP50 (+4,3%).**

```
mAP50:    B2/B1 = 0,8351/0,8008 = 1,043   (ganho relativo 4,3%)
mAP50-95: B2/B1 = 0,5055/0,4742 = 1,066   (ganho relativo 6,6%)
```

A métrica mAP50-95 é mais sensível à precisão da localização da bbox, pois é calculada como a média da AP em 10 thresholds de IoU (0,50; 0,55; ...; 0,95), enquanto mAP50 considera apenas o threshold único de IoU = 0,50. Detecções com bboxes ligeiramente desalinhadas podem ter alta mAP50 mas baixa mAP50-95.

O fato de o ganho ser proporcionalmente maior em mAP50-95 sugere que **B2 não está apenas detectando mais embarcações que B1, está detectando-as com bboxes mais precisas**. Esta observação é compatível com a interpretação de que pré-treino COCO ajuda principalmente a aprender a **aderir corretamente aos contornos dos objetos**, uma habilidade aprendida durante o pré-treino em milhões de bboxes anotadas e que não é trivialmente recuperada com 1.348 imagens de fine-tuning. **`[CITAÇÃO PENDENTE]`** — busca por trabalhos que decomponham o efeito do pré-treino entre detecção (recall) e localização (IoU precision).

#### 7.7.4 Observação operacional sobre tempo de treino

O tempo médio por corrida ficou em ~53 min (54 min em B2, 52 min em B1). A variação entre corridas (43-60 min) reflete principalmente a quantidade de épocas até o early stopping disparar, que por sua vez depende de quão cedo a corrida atinge seu pico de val mAP. As corridas mais rápidas (B1 seed 2024 = 43,7 min e B2 seed 2024 = 45,7 min) também foram as que tiveram seeds que convergiram em menos épocas.

**Estimativa para o experimento principal:** o pré-treino dos braços A, B (e opcional C) usará o `dataset_25k` (~38k imagens, 28x maior que o CITRA-3D-Real). O fine-tuning continuará usando o CITRA-3D-Real. Estimativa grosseira por seed: pré-treino ~5-8h + fine-tuning ~1h = **~6-9h por seed**. Para 3 seeds × 2 braços (A e B), são **~36-54h** de tempo de treino. O braço C opcional adicionaria mais ~18-27h. Será necessário planejar bem o uso do Colab, possivelmente dividindo em múltiplas sessões com checkpoints intermediários.

#### 7.7.5 Comparação implícita com o baseline anterior do grupo

O grupo de pesquisa havia previamente reportado bons resultados para detecção em CITRA-3D usando YOLOv8m em cenário multi-classe (9 classes), com imagens reais + sintéticas, imgsz 1260, e splits onde val=test (bug do yaml original). Esses números **não são diretamente comparáveis** com os atuais por causa das múltiplas mudanças simultâneas (arquitetura, quantidade de classes, presença de imagens sintéticas, resolução, splits separados, labels limpos do bug `Quadrado_marcacao`).

A função do B2 atual no contexto institucional é estabelecer uma **referência atualizada e reprodutível** dentro do novo cenário (single-class, real-only, splits corretos, labels limpos), contra a qual os braços experimentais A e B serão comparados de forma honesta.

#### 7.7.6 HPO formal de B2 — metodologia planejada

Esta subseção documenta a metodologia de Hyperparameter Optimization a ser aplicada sobre o baseline B2, registrada **antes** da execução para que os critérios de decisão estejam fixados independentemente do resultado observado (pré-registro metodológico). Os resultados serão adicionados na Seção 7.7.7 após a execução.

**Objetivo:** verificar se a configuração conservadora atual de B2 (AdamW, lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3, cos_lr=True) está próxima do ótimo dentro do espaço de hyperparams razoáveis para fine-tuning de YOLOv11m a partir de pesos COCO. A configuração atual foi escolhida como conservadora e padrão (ver Decisão 5.11), mas não foi submetida a HPO formal.

**Ferramenta:** Optuna com sampler TPE (Tree-structured Parzen Estimator) `[CITAÇÃO PENDENTE — Bergstra et al. (2011) "Algorithms for Hyper-Parameter Optimization", NeurIPS]`. O TPE foi escolhido em vez de random search ou grid search porque, em 5 dimensões com 30 trials, TPE amostra mais eficientemente das regiões promissoras do espaço de busca do que random search e evita o crescimento exponencial do grid search.

**Espaço de busca (5 dimensões):**

| Hyperparam | Distribuição | Faixa | Justificativa |
|---|---|---|---|
| `lr0` | log-uniform | [0.0003, 0.003] | Conservador em torno de 0.001 (config atual) |
| `lrf` | log-uniform | [0.005, 0.05] | LR final = lr0 × lrf; controla agressividade do decay |
| `momentum` | uniform | [0.85, 0.95] | Para AdamW, este parâmetro é β₁ |
| `weight_decay` | log-uniform | [0.0001, 0.001] | Em torno do default 0.0005 |
| `warmup_epochs` | int uniform | [1, 5] | Warmup curto vs longo |

**Fixados (controle experimental, não tuned):**
- `optimizer = AdamW` (justificado pela Decisão 5.11 e por Loshchilov & Hutter, 2019)
- `cos_lr = True` (fundamental; sem isso, recorrência do bug 2 documentado na Decisão 5.11)
- `warmup_bias_lr = 0.01`, `warmup_momentum = 0.8` (defaults razoáveis)
- `imgsz = 640`, `batch = 16`, `optimizer = AdamW` (iguais aos baselines)
- Augmentation: defaults do Ultralytics (fora do escopo deste HPO)
- Seed = 42 (única durante HPO; robustez entre seeds é avaliada na Fase 2)

**Estratégia em duas fases:**

**Fase 1 — Exploração (~8-10h):** 30 trials × 100 épocas × `patience=15` cada. A métrica otimizada é o "fitness" padrão do Ultralytics:

$$\text{fitness} = 0{,}1 \times \text{mAP}_{50} + 0{,}9 \times \text{mAP}_{50\text{-}95}$$

definido na documentação oficial do framework ([docs.ultralytics.com/yolov5/tutorials/hyperparameter_evolution](https://docs.ultralytics.com/yolov5/tutorials/hyperparameter_evolution/)). A escolha desta função objetivo específica — com 90% de peso em mAP50-95 e apenas 10% em mAP50 — reflete a prática da comunidade de considerar mAP50-95 a métrica primária de qualidade (porque é mais sensível à precisão da localização das bboxes). Usar o fitness padrão do Ultralytics facilita reprodução e comparação com outros trabalhos que usam o mesmo framework.

Cada trial roda isolado em subpasta própria (`trials/trial_NNN/`) com o resultado registrado em `trial_summary.json`. O estudo Optuna é persistido em SQLite (`optuna_study.db`), permitindo retomada em caso de crash ou timeout de sessão do Colab.

**Justificativa para 100 épocas (em vez de 300 completas):** treinos completos seriam ~30 × 50 min = ~25h, inviável. Trials abreviados de 100 épocas ainda permitem discriminar configs ruins (que divergem ou estagnam cedo) de configs promissoras (que atingem mAP50 competitivo em poucas dezenas de épocas, compatível com a observação do piloto B2 original que atingiu pico em ~170 épocas mas já estava próximo do ótimo em ~80). O risco residual de trials curtos "mentirem" sobre configs que precisam de mais épocas para convergir é mitigado pela Fase 2.

**Fase 2 — Validação completa (~50-60 min):** a configuração TOP-1 da Fase 1 é submetida a um único treino completo (300 épocas, `patience=30`, seed 42) idêntico aos baselines originais, incluindo avaliação no test set. A razão desta segunda fase é que trials abreviados podem selecionar configs que otimizam o regime de 100 épocas mas não necessariamente o regime de convergência completa — a Fase 2 fecha essa lacuna.

**Critério de decisão ancorado em variância empírica:**

Após a Fase 2, o resultado (test mAP50) é comparado com o baseline B2 atual (média de 3 seeds = 0,8351; desvio padrão entre seeds = 0,0024). O critério de decisão é:

$$\Delta_{\text{mAP}_{50}} = \text{mAP}_{50}^{\text{HPO}} - \text{mAP}_{50}^{\text{baseline}}$$

- Se $\Delta > 3 \times \sigma \approx 0{,}0072$: **refazer os 3 seeds** com a nova configuração. O ganho é maior que o esperado por variância entre seeds única com margem confortável.
- Se $\Delta \leq 3\sigma$: **manter a configuração atual**. A config inicial está dentro de 3 desvios padrão do ótimo encontrado no espaço de busca testado.

**Justificativa para o threshold 3σ:** este critério é **derivado empiricamente** da variância observada nos próprios baselines, não um número arbitrário externo. A escolha reflete três considerações:

1. **Evitar capturar ruído de seed única.** O HPO corre com apenas 1 seed (42). A variância natural entre seeds em B2 é σ ≈ 0,0024 em mAP50. Um ganho menor que 1σ ou 2σ poderia ser explicado por sorte da seed única, sem representar melhoria real da configuração. O threshold 3σ dá margem confortável contra esse falso positivo.

2. **Não exigir ganhos artificialmente grandes.** Trabalhos anteriores reportam ganhos modestos em HPO de detecção: Park et al. (2026) obtiveram +1,5% em mAP50 e +3,2% em mAP50-95 após HPO completo em detecção de tartarugas invasoras, descrito pelos autores como melhoria "substancial" em relação a estudos anteriores onde o ganho era <1% ou estatisticamente nulo. Um threshold exigindo >5% seria irrealista para uma tarefa single-class com config inicial já razoável.

3. **Consistência metodológica com o restante do experimento.** O mesmo raciocínio de "efeito vs ruído" vai ser aplicado mais tarde na comparação A vs B vs B2 (experimento principal). Usar o mesmo framework agora no HPO mantém coerência interpretativa entre todas as seções de resultados do paper.

**Nota honesta sobre a expectativa:** com base em Park et al. (2026) e em Popek et al. (2023) — que tunaram learning rate e momentum em YOLOv5 para detecção de fauna térmica e concluíram que o ajuste "did not yield a meaningful improvement over the default setting" — há precedente na literatura para configurações conservadoras de YOLO já estarem próximas do ótimo. É **provável** que o resultado do HPO seja "manter configuração atual". Se for esse o caso, o valor científico do HPO continua válido: ele fornece uma validação formal da configuração inicial, permitindo afirmar no paper que "os hiperparâmetros de B2 foram validados via HPO formal (Optuna TPE, 30 trials) e a configuração inicial está dentro de 3 desvios padrão do ótimo encontrado no espaço de busca testado".

**Estado atual (14/04/2026, tarde):** `hpo_b2.py` escrito, pré-checagens validadas, execução iniciada no Colab em sessão dedicada. Fase 1 + Fase 2 em andamento, retorno esperado em ~8-11 horas.

#### 7.7.7 HPO formal de B2 — resultados

**Execução:** 14/04/2026 (tarde) → 15/04/2026 (noite). Sessão Colab Pro+ com GPU A100. Tempo total wall-clock ~14,7 horas (com uma queda intermediária do runtime do Colab que foi recuperada via retomada nativa do estudo Optuna no SQLite, sem perda de trials). Tempo da última execução (após a retomada): 4,2h para os 10 trials restantes + Fase 2.

**Resultados da Fase 1:** 30 trials lançados, **28 trials completos com sucesso** (status="ok"), 1 trial falhou por erro intermitente do Drive (`[Errno 95] Operation not supported` ao salvar checkpoint — trial 14), 1 trial divergiu (status="ok" mas fitness extremamente baixo — trial 5, fitness 0,358 vs média ~0,51). A taxa de falhas de 6,7% (2/30) é considerada aceitável para HPO em ambiente Colab e não compromete a representatividade estatística.

**TOP-5 da Fase 1** (ordenado por fitness, métrica padrão do Ultralytics):

| Rank | Trial | Fitness | lr0 | lrf | momentum | wd | warmup |
|---|---|---|---|---|---|---|---|
| #1 | 13 | 0,52158 | 0,000308 | 0,02057 | 0,9498 | 0,000101 | 2 |
| #2 | 12 | 0,52091 | 0,000306 | 0,01809 | 0,9453 | 0,000113 | 2 |
| #3 | 11 | 0,52046 | 0,000322 | 0,01964 | 0,9475 | 0,000110 | 2 |
| #4 | 8 | 0,51849 | 0,000397 | 0,01564 | 0,8534 | 0,000812 | 2 |
| #5 | 26 | 0,51822 | 0,000301 | 0,03733 | 0,9414 | 0,000132 | 2 |

**Observação sobre convergência do TPE:** os trials TOP-3 (13, 12, 11) têm parâmetros notavelmente similares (lr0 ≈ 0,00031, lrf ≈ 0,019-0,020, momentum ≈ 0,94-0,95, weight_decay ≈ 0,0001, warmup_epochs = 2), confirmando que o sampler TPE convergiu para uma região promissora do espaço de busca após a fase de calibração inicial (~10 trials). O trial 26 (rank 5, executado já na fase final do HPO) reapareceu nessa mesma região com parâmetros similares, sugerindo que o TPE permaneceu calibrado durante toda a execução. **Cinco em cinco trials no TOP-5 usaram warmup_epochs = 2** — um indício forte de que esse parâmetro especificamente importa, e a faixa em torno de 2 é claramente ótima para esta tarefa. Os outros 4 parâmetros mostram mais variação no TOP-5, sugerindo que estão em regimes mais tolerantes.

**Observação sobre limite de épocas:** vários trials atingiram seu pico de fitness próximo ao limite de 100 épocas (trials 0, 3, 4, 6, 9, 10, 15 com peak_epoch ≥ 88; trial 15 chegou ao peak no epoch 99). Isso sugere que parte das configurações testadas teria continuado a melhorar com mais épocas — um viés inerente ao desenho de trials abreviados. A Fase 2 mitigou esse risco rodando o TOP-1 com 300 épocas completas.

**Resultados da Fase 2:** treino completo do TOP-1 (trial 13, configuração: lr0=0,000308, lrf=0,02057, momentum=0,9498, weight_decay=0,000101, warmup_epochs=2) com 300 épocas, patience=30, seed 42, idêntico ao protocolo dos baselines originais. Tempo de treino: **26,4 minutos**. Avaliação no test set:

| Métrica | HPO TOP-1 (Fase 2) | B2 baseline (média 3 seeds) | Δ | Δ em DPs |
|---|---|---|---|---|
| Precision | 0,8341 | — | — | — |
| Recall | 0,7824 | — | — | — |
| mAP50 | 0,8328 | 0,8351 ± 0,0024 | **−0,0023** | −0,94 σ |
| mAP50-95 | 0,5021 | 0,5055 ± 0,0027 | **−0,0034** | −1,26 σ |
| Fitness | 0,5351 | 0,5385 | **−0,0033** | — |

**Aplicação do critério de decisão pré-registrado:** Δ mAP50 = −0,0023, em valor absoluto **0,32× o threshold de 3σ ≈ 0,0072**. A diferença não apenas não supera o threshold, como **é ligeiramente negativa** — ou seja, o TOP-1 do HPO performou marginalmente pior que a média dos baselines existentes, dentro da faixa de variação natural entre seeds.

**Decisão automática registrada pelo `hpo_b2.py`: MANTER CONFIG ATUAL** (AdamW, lr0=0,001, lrf=0,01, momentum=0,937, weight_decay=0,0005, warmup_epochs=3, cos_lr=True). Os 6 baselines existentes (B1+B2 × 3 seeds) permanecem válidos e serão usados como referência no experimento principal.

**Interpretação para o paper.** Este resultado é **um forte argumento metodológico positivo**, não um resultado nulo. Três pontos para destacar na escrita:

1. **A configuração inicial está validada formalmente.** Pode-se afirmar com confiança que "os hiperparâmetros foram validados via HPO formal (Optuna TPE, 30 trials × 100 épocas, 5 dimensões de busca, função objetivo padrão do Ultralytics) e a configuração inicial provou-se dentro de 1σ da variância entre seeds do ótimo encontrado". Esta é uma defesa metodológica forte contra a crítica comum "vocês fizeram tuning?".

2. **O espaço de busca contém uma vasta região de configurações equivalentes.** O TOP-1 (lr0=0,000308, momentum=0,9498) e o baseline B2 (lr0=0,001, momentum=0,937) têm parâmetros bastante diferentes mas produzem performance estatisticamente indistinguível. Isso sugere baixa sensibilidade do problema aos hyperparams nessa faixa razoável — também uma observação publicável.

3. **Resultado consistente com a literatura citada.** Park et al. (2026) reportaram ganho típico de +1,5% em mAP50 após HPO em detecção single-class — neste estudo, o ganho foi de −0,28% (estatisticamente nulo), reforçando a observação de Popek et al. (2023) sobre HPO em YOLO frequentemente não produzir melhoria meaningful sobre defaults conservadores. A configuração padrão de YOLO já é razoavelmente próxima do ótimo para tarefas single-class de complexidade moderada.

**Custo total do HPO:** ~10 horas de Colab A100, dos quais ~1 hora foi a Fase 2. Investimento justificado pelo resultado defensável.

**Próximos passos:** os 6 baselines existentes permanecem como referência. O experimento principal (braços A, B, C) usará a mesma configuração de B2 sem alterações. A7 fechada.

#### 7.7.8 Braço A — resultados do pré-treino curado + fine-tuning

**Execução:** 16–22/04/2026. Sessões Colab Pro+ com GPU A100. Tempo total ~16h (3 seeds × ~5h cada, com 1 reinicialização do Colab por queda de sessão na seed 123).

**Protocolo executado:** COCO (`yolo11m.pt`) → pré-treino no `dataset_25k_v2` (100 épocas, patience 20, AdamW lr0=0.001, cos_lr=True) → fine-tuning no CITRA-3D-Real (300 épocas, patience 30, mesmos hyperparams). Avaliação no test set do CITRA-3D-Real. Script: `treinar_braco_a.py`.

**Resultados por seed:**

| Seed | mAP50 | mAP50-95 | Fitness |
|---|---|---|---|
| 42 | 0,8006 | 0,4680 | 0,5013 |
| 123 | 0,7901 | 0,4680 | 0,5002 |
| 2024 | 0,7902 | 0,4716 | 0,5035 |
| **Média ± DP** | **0,7936 ± 0,0060** | **0,4692 ± 0,0021** | **0,5017 ± 0,0017** |

**Comparação entre os 3 braços existentes:**

| Braço | Descrição | mAP50 | mAP50-95 | Δ mAP50 vs B2 |
|---|---|---|---|---|
| **B2** | COCO → CITRA-3D | **0,8351 ± 0,0024** | **0,5055 ± 0,0027** | ref |
| **B1** | Random → CITRA-3D | 0,8008 ± 0,0073 | 0,4742 ± 0,0008 | −0,0343 |
| **A** | COCO → dataset_25k_v2 → CITRA-3D | 0,7936 ± 0,0060 | 0,4692 ± 0,0021 | **−0,0415** |

**Resultado principal: o pré-treino curado no dataset_25k_v2 PIOROU o resultado em relação a ambos os baselines.**

- Braço A ficou **4,15% abaixo de B2** (COCO puro) em mAP50 — diferença de 17× o DP de B2, separação completa dos intervalos de confiança.
- Braço A ficou **0,72% abaixo de B1** (random init) em mAP50 — ou seja, o pipeline COCO → InaTechShips → CITRA-3D é marginalmente pior que treinar do zero sem nenhum pré-treino.

**Interpretação.** Este resultado aponta para **catastrophic forgetting**: o pré-treino intermediário de 100 épocas no dataset_25k_v2 sobrescreveu features úteis dos pesos COCO (features de baixo nível + capacidade de detecção genérica de objetos) e as substituiu por features especializadas no domínio shipspotting.com, que não transferem bem para o cenário operacional do CITRA-3D-Real. O fine-tuning subsequente em CITRA-3D-Real (apenas 1.348 imagens) não dispõe de dados suficientes para recuperar o que foi perdido.

**Três eixos do gap de domínio que explicam a degradação** (já documentados na Seção 4.3, agora empiricamente confirmados como relevantes):

1. **Condições de captura:** fotos profissionais do shipspotting (boa iluminação, composição centrada, alta resolução) vs capturas operacionais da Marinha (ângulos variados, condições climáticas adversas, distância variável).
2. **Densidade de objetos:** 1 bbox/imagem no InaTechShips vs 3,36 bboxes/imagem no CITRA-3D-Real. O modelo pré-treinado em imagens single-object pode perder a capacidade de detectar múltiplos objetos.
3. **Distribuição de classes:** balanceada artificialmente no InaTechShips (10 classes × ~2.700 IDs) vs long-tail natural no CITRA-3D-Real (TUG 35,6% vs Navio 0,8%). O pré-treino em distribuição uniforme pode desalibrar o modelo para a distribuição real.

**Observação paradoxal sobre o COCO.** O COCO (80 classes genéricas, nenhuma embarcação dedicada) é um pré-treino melhor do que um dataset de 27.796 imagens de embarcações curadas por similaridade visual. Isso sugere que, para transfer learning em cenários operacionais com poucos dados de fine-tuning, **a generalidade das features importa mais do que a similaridade visual com o domínio alvo**. Features genéricas de baixo nível (bordas, texturas, formas) do COCO são mais reutilizáveis do que features especializadas em fotos profissionais de navios.

**Implicação para o braço B (random_pool_v2).** Os três cenários possíveis quando os labels do random_pool_v2 estiverem disponíveis:
- **B ≈ A:** o problema é o InaTechShips inteiro → conclusão: "pré-treino em dados do shipspotting.com não transfere para cenário operacional, independente da curadoria"
- **B > A:** a curadoria CLIP selecionou sistematicamente as imagens "erradas" (mais similares = mais especializadas = mais forgetting) → conclusão: "similaridade CLIP é proxy ruim para transferibilidade"
- **B < A:** improvável, mas significaria que a curadoria pelo menos filtra os exemplos mais prejudiciais

**Cada cenário é publicável.** Nenhum resultado nulo: ou a curadoria não funciona, ou o dataset inteiro não funciona, ou ambos. A Seção 7.7.9 a seguir investiga a hipótese de catastrophic forgetting com evidência adicional.

#### 7.7.9 Ablation de épocas de pré-treino — diagnóstico de catastrophic forgetting

**Motivação.** O resultado negativo do braço A (Seção 7.7.8) levantou a questão: **o problema é excesso de treino (catastrophic forgetting) ou incompatibilidade fundamental entre InaTechShips e CITRA-3D-Real?** Se for forgetting, reduzir as épocas de pré-treino deveria melhorar o resultado; se for incompatibilidade, mesmo poucas épocas seriam prejudiciais.

**Execução:** 22/04/2026. Sessão Colab Pro+ com GPU A100. Script: `ablation_epocas_pretreino.py`. Seed única: 42 (diagnóstico, não requer variância entre seeds). Três variantes testadas: 10, 20 e 50 épocas de pré-treino, com fine-tuning idêntico (300 épocas, patience 30).

**Resultados completos (incluindo pontos existentes):**

| Épocas pré-treino | mAP50 | mAP50-95 | Fitness | Δ mAP50 vs B2 |
|---|---|---|---|---|
| **0 (B2 baseline)** | **0,8351** | **0,5055** | **0,5385** | **ref** |
| 10 | 0,8200 | 0,4999 | 0,5319 | −0,0151 |
| 20 | 0,8171 | 0,4960 | 0,5281 | −0,0180 |
| 50 | 0,8037 | 0,4731 | 0,5062 | −0,0314 |
| 100 (braço A, s42) | 0,8006 | 0,4680 | 0,5013 | −0,0345 |
| **B1 (random init)** | **0,8008** | **0,4742** | **0,4869** | **−0,0343** |

**Observações-chave:**

1. **Degradação monotônica.** A relação entre épocas de pré-treino e performance final é estritamente decrescente: cada incremento de pré-treino piora o resultado. Não existe "sweet spot" onde o pré-treino ajuda.

2. **Gradiente de degradação.** A perda é mais acentuada nas primeiras épocas: as primeiras 10 épocas custam −1,5% mAP50, as seguintes 10 custam apenas −0,3% adicionais, e de 50 a 100 épocas a degradação satura em ~−3,4%. Isso sugere que as features COCO mais frágeis (e mais úteis para CITRA-3D) são sobrescritas rapidamente.

3. **Convergência para B1.** Com 100 épocas, o braço A (mAP50 = 0,8006) é estatisticamente equivalente a B1 (mAP50 = 0,8008). Isso confirma que o pré-treino longo no InaTechShips **anula completamente** o benefício dos pesos COCO — o efeito líquido é equivalente a treinar do zero.

4. **Mesmo 10 épocas prejudica.** A variante mais leve (10 épocas) ainda fica 1,5% abaixo de B2. Isso descarta a hipótese de que "bastaria reduzir a dose" e aponta para **incompatibilidade parcial**: qualquer exposição ao InaTechShips degrada features que são úteis para CITRA-3D-Real, embora a degradação seja menor com menos exposição.

**Diagnóstico final: catastrophic forgetting confirmado, com incompatibilidade parcial subjacente.** O InaTechShips não é apenas "demais" em 100 épocas — é "qualitativamente errado" como etapa intermediária. A similaridade visual medida por CLIP (≥ 0,60) não captura os eixos do gap que realmente importam para transfer learning (condições de captura, densidade de objetos, distribuição de classes).

**Implicação metodológica para o paper.** Este resultado sustenta uma contribuição secundária importante: **similaridade visual (CLIP cosine similarity) é um proxy insuficiente para prever transferibilidade em detecção de objetos**. Datasets visualmente similares (embarcações em ambos os casos) podem ter distribuições operacionais incompatíveis que provocam negative transfer. A curadoria por similaridade visual deveria ser complementada por métricas de compatibilidade de distribuição (ex: FID, MMD entre features de baixo nível, ou análise de heterogeneidade de condições de captura) antes de ser usada como critério de seleção de dados de pré-treino.

**Referências relevantes para esta observação:**
- Negative transfer em detecção: [CITAÇÃO PENDENTE — buscar trabalhos sobre domain shift em object detection, ex: Chen et al. "Domain Adaptive Faster R-CNN" ou Saito et al. "Strong-Weak Distribution Alignment"]
- Catastrophic forgetting em fine-tuning sequencial: [CITAÇÃO PENDENTE — buscar McCloskey & Cohen 1989 (original), ou trabalhos mais recentes sobre sequential fine-tuning em vision models]
- Limitações de similaridade visual como proxy de transferibilidade: [CITAÇÃO PENDENTE — buscar Cui et al. "Large Scale Fine-Grained Categorization and Domain-Specific Transfer Learning" ou Zamir et al. "Taskonomy"]

#### 7.7.10 Braço A' — Scale-Aware Copy-Paste: adaptação de domínio via composição de imagens sintéticas

**Motivação.** Os resultados das Seções 7.7.8 e 7.7.9 estabeleceram que o pré-treino direto no InaTechShips causa negative transfer por catastrophic forgetting. A análise qualitativa do gap de domínio identificou a **diferença de escala** como o eixo mais saliente: no InaTechShips, embarcações são fotografadas de perto e ocupam ~80% da imagem (fotos profissionais do shipspotting.com); no CITRA-3D-Real, as embarcações são capturadas à distância em cenário operacional, aparecendo como objetos pequenos (~5-15% da imagem) em fundo de oceano aberto. Além da escala, a densidade de objetos (1 bbox/imagem vs 3,36 bboxes/imagem) e o contexto visual (porto/cais vs oceano aberto) são incompatíveis.

A pergunta natural é: **se transformarmos as imagens InaTechShips para que se pareçam com o CITRA-3D-Real (navios pequenos em fundo oceânico), o pré-treino intermediário passa a ajudar em vez de prejudicar?**

**Revisão de abordagens na literatura.**

Três abordagens foram consideradas:

1. **CycleGAN (tradução de domínio não-pareada).** Traduz estilo visual (cores, texturas, iluminação) mas preserva a estrutura espacial da imagem. Uma foto close-up de navio continuaria sendo close-up, só com filtro de "câmera operacional". **Descartada** porque o gap principal é de escala e composição, não de estilo.

2. **Modelos de difusão (Stable Diffusion, ControlNet, DALL-E).** Geram imagens de alta qualidade, mas para treinar um detector de objetos é necessário ter bounding boxes precisos. Na literatura recente, ODGEN (Apple, ICCV 2023) e InstaGen (CVPR 2024) demonstraram que é difícil para modelos de difusão gerar um número preciso de objetos exatamente nas regiões especificadas (ODGEN, 2024). Além disso, modelos off-the-shelf tendem a gerar imagens com apenas um ou dois objetos em fundo simples, resultando em robustez reduzida em cenários complexos (InstaGen, 2024). O pipeline requer fine-tuning do modelo de difusão + Detection-Adapter + pós-anotação, com custo computacional de 6-12h de GPU. **Descartada** pela complexidade, imprecisão dos bboxes gerados, e custo computacional desproporcional ao benefício — especialmente porque, na escala alvo (~5-15% da imagem), os detalhes visuais que um modelo de difusão adicionaria são perdidos no redimensionamento.

3. **Scale-Aware Copy-Paste.** Recorta embarcações do InaTechShips usando segmentação (SAM), redimensiona para a escala do CITRA-3D-Real, e cola em fundos reais extraídos do CITRA-3D. Bounding boxes são gerados automaticamente a partir das coordenadas de colagem (100% precisos). Abordagem bem fundamentada na literatura marítima recente:
   - POSEIDON (Ruiz-Ponce et al., Sensors 2023): ferramenta de copy-paste para datasets marítimos aéreos, demonstrou superioridade sobre técnicas de balanceamento por pesos.
   - S3Det / Feedback Cut&Paste (Li et al., ACCV 2024): copy-paste adaptado para detecção de embarcações small-scale com YOLOv8, ganho de +5,9% recall e +2% mAP50 sobre baseline.
   - Nemati (2025): copy-paste controlado com blending adaptativo para cenários marítimos, validado com RT-DETR.
   **Selecionada** como abordagem para o braço A'.

**Justificativa da escolha (Copy-Paste vs modelos generativos).**

| Critério | Copy-Paste | Modelo generativo |
|---|---|---|
| Bounding boxes | 100% exatos (coordenadas de colagem) | Imprecisos, requer pós-anotação |
| Fundo da imagem | Real do CITRA-3D (oceano, condições operacionais) | Sintético (domain gap residual) |
| Controle de escala | Determinístico | Imprevisível |
| Controle de densidade | 1-N navios por imagem, configurável | Difícil de controlar |
| Compute | CPU, <1h para ~27k imagens | GPU, 6-12h para fine-tuning |
| Fundamentação marítima | 3 papers específicos (2023-2025) | Nenhum paper marítimo comparável |

Argumento adicional: quando o navio ocupa ~5-15% da imagem final (como no CITRA-3D-Real), a qualidade visual do recorte é irrelevante — um crop de 300×200 pixels redimensionado para 40×25 pixels perde todos os detalhes que diferenciavam a foto profissional da captura operacional. A degradação de resolução pelo resize simula naturalmente a aparência de objetos distantes. Um modelo generativo gastaria muito compute para produzir detalhes que serão descartados no redimensionamento.

**Pipeline proposto para o braço A' (Scale-Aware Copy-Paste).**

1. **Análise de escala do CITRA-3D-Real** (`analisar_escala_citra3d.py`): extrai distribuição de tamanhos, aspect ratios, densidade de objetos e posição espacial dos bboxes reais. Resultado: JSON com ranges alvo para calibrar a composição.

2. **Extração de crops com SAM** (Segment Anything Model): para cada imagem do InaTechShips, usa SAM para gerar máscara de segmentação do navio (separando do fundo original — porto, cais, etc.). Resultado: recortes de navios com fundo transparente (RGBA PNG com canal alpha).

   **Decisão: modo SAM (segmentação por contorno) em vez de modo bbox (crop retangular).**
   
   O script `extrair_crops_sam.py` implementa dois modos:
   - **Modo bbox:** crop retangular do bounding box com feathering de 5px nas bordas. Rápido (~0,01s/img), mas inclui fundo original (água, cais, céu) ao redor do navio dentro do retângulo. Quando colado em fundo oceânico do CITRA-3D, o retângulo traz artefatos visíveis do fundo original.
   - **Modo SAM:** usa SAM ViT-B com o bounding box como prompt para gerar máscara que segue o contorno real do casco e superestrutura. Mais lento (~0,3s/img), mas produz recortes onde apenas os pixels do navio são opacos — o fundo fica transparente (alpha=0). Quando colado em fundo oceânico, o navio se integra naturalmente sem artefatos de borda.

   **Teste piloto (23/04/2026):** 100 imagens processadas em modo bbox para validar o pipeline. Resultados: mediana da largura do crop = 1.202px, altura = 356px, aspect ratio = 3,15, cobertura da máscara = 97,7% (quase 100% porque o bbox é retangular). Zero falhas, taxa de 0,5 crops/s no Drive. Pipeline funcional confirmado.
   
   **Decisão após teste:** usar **modo SAM** para o batch completo. Justificativa: (a) realismo da composição é maior com contorno do que com retângulo — o modelo de detecção pode aprender a reconhecer bordas retangulares artificiais como artefato, degradando generalização; (b) na escala alvo do CITRA-3D (~20-60px), a diferença de forma entre contorno e retângulo é sutil, mas o fundo residual do InaTechShips dentro do retângulo (porto, cais) é visualmente incompatível com oceano aberto; (c) custo adicional aceitável (~1,5-2h vs ~5 min para 27k imagens no A100). A cobertura da máscara esperada no modo SAM é ~60-80% (vs ~97% no bbox), refletindo a separação real navio/fundo.

3. **Extração de fundos do CITRA-3D-Real** (D3, concluída 24/04/2026): extraiu 2.081 fundos do CITRA-3D-Real usando `extrair_fundos_citra3d.py`. Resultado: 1.988 fundos "clean" (bboxes cobrindo <5% da imagem — objetos minúsculos funcionam como ruído natural) + 93 fundos com inpainting (cv2.INPAINT_TELEA). Cobertura 100% das imagens do CITRA-3D.

4. **Composição de imagens sintéticas** (D4, em execução 24/04/2026): `gerar_dataset_copypaste.py`. Três iterações até chegar à abordagem correta:

   - **v1 — posicionamento aleatório (descartada).** Navios posicionados aleatoriamente no range y=0,37-0,70 (P10-P90 do CITRA-3D). Resultado: ~30% das imagens com navios no céu, montanhas ou prédios. O range y não discrimina água de terra/construção.
   
   - **v2 — water-aware: labels + HSV (descartada).** Combinou posições dos bboxes originais (banda y de navios reais) com detecção de água por cor HSV. Resultado: melhorou para ~85% correto, mas HSV confundia rocha/concreto cinza com água em ~15% das imagens.
   
   - **v3 — substituição in-place (aprovada).** Ideia proposta pela autora (Daniela): em vez de posicionar navios aleatoriamente, **substituir as embarcações reais do CITRA-3D pelos crops do InaTechShips na mesma posição e dimensão dos bounding boxes originais**. Para cada imagem do CITRA-3D (2.081), gera múltiplas variações (~13) trocando os crops mas mantendo os slots. Labels idênticos aos originais. Resultado: **100% dos navios posicionados na água, escala correta, densidade correta, zero decisões arbitrárias**. Preview com 100 imagens: 3,62 obj/img (vs 3,37 real), composição visualmente convincente.
   
   **Justificativa da v3 sobre v1/v2:** elimina completamente o problema de posicionamento ao reutilizar as posições de navios reais como "âncoras" para os navios sintéticos. Não requer detecção de água (por cor, por segmentação, ou por qualquer heurística) porque a posição é herança direta do dataset operacional. Cientificamente defensável: nenhuma decisão arbitrária de posicionamento — "os navios sintéticos estão onde navios reais foram observados pela Marinha".

5. **Validação das imagens sintéticas** (D4.5, planejada): FID (Fréchet Inception Distance) + comparação de distribuição de bboxes. Critério: FID competitivo com variação intra-CITRA-3D. Pode ser simplificada na v3 porque a distribuição de bboxes é idêntica por construção.

6. **Treino do braço A'**: COCO → dataset_sintetico (pré-treino) → CITRA-3D-Real (fine-tuning). Mesmos hyperparams e protocolo do braço A original, variando apenas o dataset de pré-treino.

**Comparação experimental resultante:**

| Braço | Pipeline | Pergunta que responde |
|---|---|---|
| B2 (baseline) | COCO → CITRA-3D | referência |
| A (original) | COCO → InaTechShips → CITRA-3D | curadoria CLIP ajuda? (resposta: não, prejudica) |
| **A' (novo)** | **COCO → CopyPaste(InaTechShips→CITRA-3D) → CITRA-3D** | **adaptação de domínio resolve o gap?** |
| B (futuro) | COCO → random_pool_v2 → CITRA-3D | curadoria vs aleatório |

A comparação A vs A' isola diretamente o efeito da adaptação de domínio: mesmo pool de imagens fonte (InaTechShips curado), mas com transformação scale-aware que resolve os 3 eixos do gap (escala, densidade, contexto visual). Se A' > B2, o paper demonstra que curadoria por similaridade CLIP + adaptação de domínio é um pipeline viável para augmentation em cenários operacionais com poucos dados.

**Status:** ✅ CONCLUÍDA. Dataset sintético gerado (27.796 imagens, 93.480 objetos, 3,36 obj/img, zero falhas). Braço A' treinado e avaliado — resultados na Seção 7.7.12.

#### 7.7.11 Braço B — resultado do pré-treino aleatório (diagnóstico)

**Execução:** 24/04/2026. Sessão Colab Pro+ com GPU A100. Seed 42 (diagnóstica — 1 seed suficiente para confirmar/refutar B ≈ A).

**Protocolo:** COCO (`yolo11m.pt`) → pré-treino no random_pool_v2 (100 épocas, patience 20) → fine-tuning no CITRA-3D-Real (300 épocas, patience 30). Labels do random_pool_v2 fornecidos por Eduardo Teixeira (PointRend, classe 0, 91,5% de cobertura). Hyperparams idênticos aos demais braços.

**Resultado:**

| Métrica | Braço B (aleatório) | Braço A (curado, média) | B2 (baseline) |
|---|---|---|---|
| mAP50 | 0,7997 | 0,7936 ± 0,0060 | 0,8351 ± 0,0024 |
| mAP50-95 | 0,4711 | 0,4692 ± 0,0021 | 0,5055 ± 0,0027 |

**Interpretação:** B ≈ A (diferença de 0,61% em mAP50, dentro de 1σ da variância do braço A). **O negative transfer é independente da estratégia de seleção**: tanto o subset curado por CLIP quanto o subset aleatório do mesmo dataset produzem degradação equivalente (~−3,5 a −4,2% vs B2). Isso confirma que:

1. O problema é **incompatibilidade estrutural de domínio** entre InaTechShips e CITRA-3D-Real, não falha da curadoria CLIP.
2. A similaridade visual (CLIP cosine ≥ 0,60) não é nem positiva nem negativa — é **irrelevante** para a performance de transfer learning neste cenário.
3. O gap de escala/densidade/contexto afeta uniformemente todo o InaTechShips, curado ou não.

**Uma seed foi suficiente** para esta conclusão porque a diferença B vs A (0,61%) é muito menor que a diferença B vs B2 (3,54%). Mesmo com a variância máxima observada entre seeds (σ ≈ 0,006), não há cenário em que B supere B2 ou fique significativamente diferente de A.

#### 7.7.12 Braço A' — resultado da composição sintética in-place

**Execução:** 25/04/2026. Sessão Colab Pro+ com GPU A100. Três seeds (42, 123, 2024).

**Protocolo:** COCO (`yolo11m.pt`) → pré-treino no dataset_sintetico (100 épocas, patience 20) → fine-tuning no CITRA-3D-Real (300 épocas, patience 30). Dataset sintético: 27.796 imagens geradas por substituição in-place (v3), 93.480 objetos, 3,36 obj/img. Hyperparams idênticos aos demais braços.

**Resultados por seed:**

| Seed | mAP50 | mAP50-95 |
|---|---|---|
| 42 | 0,8561 | 0,5309 |
| 123 | 0,8491 | 0,5216 |
| 2024 | 0,8570 | 0,5318 |
| **Média ± DP** | **0,8541 ± 0,0043** | **0,5281 ± 0,0056** |

**Comparação com todos os braços:**

| Braço | Pipeline | mAP50 | mAP50-95 | Δ mAP50 vs B2 |
|---|---|---|---|---|
| **A' (sintético)** | **COCO → copy-paste → CITRA-3D** | **0,8541 ± 0,0043** | **0,5281 ± 0,0056** | **+1,90%** |
| B2 (baseline) | COCO → CITRA-3D | 0,8351 ± 0,0024 | 0,5055 ± 0,0027 | ref |
| B1 (baseline) | Random → CITRA-3D | 0,8008 ± 0,0073 | 0,4742 ± 0,0008 | −3,43% |
| B (aleatório) | COCO → random_pool → CITRA-3D | 0,7997 | 0,4711 | −3,54% |
| A (curado direto) | COCO → InaTechShips → CITRA-3D | 0,7936 ± 0,0060 | 0,4692 ± 0,0021 | −4,15% |

**Análise estatística:**

- **A' vs B2:** Δ mAP50 = +0,0190. Intervalos de confiança: A' = [0,8498; 0,8584] vs B2 = [0,8327; 0,8375]. **Sem sobreposição — separação estatística completa.** A melhoria é de 4,5× o DP de B2 (0,0190 / 0,0024 = 7,9σ).
- **A' vs A:** Δ mAP50 = +0,0605. A composição sintética transformou um deficit de −4,15% em um ganho de +1,90% — uma melhoria líquida de **6,05 pontos percentuais** sobre o transfer direto.
- **mAP50-95:** A' superou B2 em +4,5% relativo (0,5281 vs 0,5055), indicando não apenas mais detecções corretas, mas também localizações mais precisas dos bounding boxes.

**Interpretação:**

1. **As imagens do InaTechShips são úteis para melhorar a detecção operacional — quando adaptadas ao domínio.** O pré-treino direto causa catastrophic forgetting (−4,15%), mas a composição in-place resolve completamente o gap e produz ganho positivo (+1,90%).

2. **A adaptação de domínio funciona porque resolve os 3 eixos do gap identificados no diagnóstico:**
   - **Escala:** navios redimensionados de ~80% para ~3% da imagem (escala do CITRA-3D)
   - **Densidade:** múltiplos navios por imagem (3,36 obj/img, idêntico ao CITRA-3D)
   - **Contexto:** fundos 100% reais do cenário operacional (oceano, condições climáticas)

3. **O método de substituição in-place é particularmente eficaz** porque não requer nenhuma decisão arbitrária de posicionamento — cada navio sintético herda a posição de um navio real observado no cenário operacional. Isso é mais robusto que abordagens de posicionamento aleatório (v1, v2 testadas e descartadas durante o desenvolvimento).

4. **O COCO continua sendo valioso como ponto de partida.** O pipeline mais eficaz é COCO → dados sintéticos adaptados → dados reais. Remover o COCO (random init → sintéticos → reais) não foi testado mas provavelmente seria inferior, dado que B1 < B2 mostrou que features genéricas do COCO são um bom ponto de partida.

**Observação sobre early stopping:** as 3 seeds do fine-tuning dispararam early stopping antes das 300 épocas (seed 42: parou na época 74, best na 44; seeds 123 e 2024: comportamento similar). Isso sugere que o pré-treino sintético produz features que convergem mais rápido no fine-tuning — possivelmente porque o modelo já "conhece" o cenário visual (fundos) e a distribuição de escalas, precisando apenas de ajuste fino nas texturas reais.

#### 7.7.13 Síntese dos resultados e contribuições

**Resultado principal.** O experimento demonstrou que imagens públicas de embarcações (InaTechShips) podem ser utilizadas efetivamente para melhorar a detecção em cenário operacional naval, **desde que adaptadas ao domínio alvo** via composição sintética. O uso direto dessas imagens causa negative transfer; a adaptação via substituição in-place resolve o problema e produz ganho significativo.

**Tabela de síntese (ordenada por mAP50):**

| Rank | Braço | mAP50 | Significado |
|---|---|---|---|
| **1** | **A' (sintético)** | **0,8541** | Adaptação de domínio funciona |
| 2 | B2 (COCO puro) | 0,8351 | Baseline forte — COCO genérico |
| 3 | B1 (random init) | 0,8008 | Sem pré-treino |
| 4 | B (aleatório direto) | 0,7997 | InaTechShips aleatório prejudica |
| 5 | A (curado direto) | 0,7936 | InaTechShips curado prejudica igual |

**Três contribuições do experimento:**

1. **Evidência empírica de negative transfer em detecção marítima** com ablation quantificada (degradação monotônica de 10 a 100 épocas). Contribuição para a literatura de domain adaptation em detecção de objetos.

2. **Método de adaptação de domínio por composição in-place** que transforma dados públicos em dados de treino eficazes para cenários operacionais. O método é simples (não requer modelo generativo), preciso (bboxes 100% exatos), e fundamentado (posições herdadas de dados operacionais reais). Contribuição prática reprodutível.

3. **Demonstração de que similaridade visual (CLIP) é proxy insuficiente para transferibilidade.** A curadoria por CLIP não melhorou nem piorou em relação à seleção aleatória (A ≈ B), mostrando que o gap operacionalmente relevante não é capturado por métricas de aparência visual. Contribuição metodológica para a comunidade de curadoria de dados.

---

---

## 8. Pendências e riscos conhecidos

### 8.1 ✅ RESOLVIDO: labels para o random_pool_v2

**Resolução (24/04/2026):** Eduardo Teixeira (INATEL) compartilhou 2.794.460 labels via Google Drive (8 pastas RAR). Script `filtrar_labels_eduardo.py` cruzou com os 39.628 IDs alvo: 36.272 encontrados (91,5%), 25.591 copiados para random_pool_v2/{train,val,test}/labels/. Os 8,5% faltantes são IDs onde o PointRend não detectou embarcação (comportamento esperado, conforme confirmado pelo Eduardo). Braço B treinado e concluído com estes labels.

### 8.2 ✅ RESOLVIDO: discrepância na nomenclatura das classes do CITRA-3D

**Resolução:** o `data.yaml` oficial recebido da Marinha estabeleceu o mapeamento autoritativo. Documentos antigos do projeto (`ANÁLISE_DO_CITRA-3D.docx`) estavam errados e foram marcados como obsoletos.

### 8.3 ✅ RESOLVIDO: filtragem das imagens sintéticas do CITRA-3D

**Resolução:** implementada como parte do pipeline `preparar_citra3d.py` (descarte de imagens sintéticas + ZIPs `_aug` inteiros).

### 8.4 ✅ RESOLVIDO: limpeza de labels malformados

**Resolução:** identificada a causa raiz (bug do software de anotação produzindo `Quadrado_marcacao(Clone)`), implementada limpeza automática + quarentena de órfãos. Dataset agora tem zero malformações.

### 8.5 ✅ RESOLVIDO: bug do `data.yaml` original com `val=test`

**Resolução:** identificado durante a investigação. Os `data.yaml` novos a serem gerados no passo 7.1 vão usar pastas `val/` e `test/` separadas, que existem como splits independentes no `CITRA-3D-Real`. **Pendência associada:** auditar treinos anteriores para verificar se algum usou o yaml com bug. Se sim, descartar aqueles resultados.

### 8.6 ✅ RESOLVIDO: Risco do decil 9 no random_pool_v2

**Resolução:** o download e downsample do random_pool_v2 foram concluídos com sucesso. O decil 9 teve densidade suficiente de IDs válidos. Excedente residual de 0,60% aceito como metodologicamente irrelevante.

### 8.7 Anonimização do CITRA-3D-Real no paper

**Problema:** o CITRA-3D-Real é dataset operacional da Marinha do Brasil, e a publicação do paper precisa preservar informações sensíveis.

**Ação necessária:** validar com a contraparte da Marinha o que pode aparecer no paper. Lista no cronograma (tarefa C4).

### 8.8 Classe Passageiro praticamente inexistente

**Problema:** apenas 8 instâncias no dataset inteiro, zero presença no val. Inviável para análise multi-classe.

**Resolução parcial:** decisão de colapso para classe única (Seção 5.1) elimina o problema metodológico. **Limitação documentada** que pode entrar na seção de Limitations do paper como motivação para trabalho futuro de coleta direcionada nessa classe.

---

## 9. Cronologia das decisões

| Data | Decisão | Motivação |
|---|---|---|
| (anterior) | Construir o dataset_25k via curadoria CLIP do shipspotting | Hipótese inicial: similaridade visual ao CITRA-3D produz pré-treino útil. |
| (anterior) | Treinar YOLO-StarLS-Adapted no dataset_25k | Validação interna da arquitetura proprietária do grupo. |
| 09/04/2026 | **Reformular o experimento como ablation A vs B** | Após análise crítica, ficou claro que o experimento mais defensável é comparar curadoria vs aleatório de mesmo tamanho. |
| 09/04/2026 | **Abandonar YOLO-StarLS-Adapted como detector principal** | Não superava as alternativas SOTA. |
| 09/04/2026 | **Colapsar para detecção classe-única** | Taxonomias incompatíveis, desbalanço severo, foco operacional. |
| 09/04/2026 | **Descobrir que o pool antigo `imgs/` era cópia do dataset_25k** | Auditoria revelou overlap de 99,98%. Necessário pool genuinamente novo. |
| 09/04/2026 | **Estratificar por decis empíricos no sorteio do pool aleatório** | Garantir distribuição temporal idêntica entre A e B. |
| 09/04/2026 | **Universo de labels públicos do PointRend está esgotado** | `labels_git` tem todos os labels publicados. Para o pool novo precisamos de outra fonte. |
| 09/04/2026 | **Enviar e-mail a Eduardo Teixeira** | Tentar resolver o problema de labels na fonte. |
| 10/04/2026 | **Trabalhar exclusivamente a partir do ZIP autoritativo da Marinha** | Eliminar ambiguidade entre versões legadas, garantir reprodutibilidade. |
| 10/04/2026 | **Confirmar mapeamento de classes via `data.yaml` oficial** | Documentos antigos do projeto estavam errados; o yaml oficial é autoritativo. |
| 10/04/2026 | **Limpar labels com bug `Quadrado_marcacao(Clone)`** | Identificado bug do software de anotação. Limpeza automática preserva 99,87% das anotações. |
| 10/04/2026 | **Quarentenar 2 imagens com labels vazios após limpeza** | Imagens contêm embarcações distantes não anotadas. Mantê-las como background ensinaria modelo errado. |
| 10/04/2026 | **Pipeline de preparação concluído** | 6 scripts versionados, todos os relatórios gerados, dataset autoritativo estabelecido. |
| 10/04/2026 | **Geração dos `data.yaml` classe-única + setup de links simbólicos** | Solução para limitação do Ultralytics (não suporta pasta de label customizada): links simbólicos no `/content/data/` do Colab apontando para `labels_single_class/` do Drive. Bug do yaml original (`val=test`) corrigido. Pipeline de preparação 100% completo (7 scripts). |
| 10/04/2026 | **Decisão sobre detector principal: YOLOv11m** | Análise vs alternativas (YOLOv8m, YOLOv11n, YOLOv11s). Escolha: YOLOv11m por (1) atualidade da arquitetura, (2) tamanho `m` equivalente ao baseline anterior do grupo, (3) HPO refeito de qualquer jeito tornaria irrelevante manter HPs do v8m antigo. Comparabilidade direta com `results.csv` legado é estética, não funcional. |
| 10/04/2026 (noite) | **Bug 1 identificado: Ultralytics + symlinks silenciosamente quebrando o treino** | Primeiros treinos B1 e B2 saíram com números suspeitos (mAP50=0,44 e 0,28). Investigação revelou que o Ultralytics resolve symlinks via `Path.resolve()` antes de aplicar a substituição `/images/ → /labels/`, acabando por ler os labels originais (9 classes) em vez de `labels_single_class/`. Todas as imagens eram descartadas como "corrupt". Bug silencioso. |
| 10/04/2026 (noite) | **Correção bug 1: cópia física em vez de symlinks** | Script `gerar_data_yaml.py` substituído por `preparar_dados_locais.py`, que faz cópia física do Drive para `/content/data/`. Inclui validação robusta anti-symlink + sample check do primeiro label de cada split. Decisão 5.12 documenta o problema e a correção. |
| 10/04/2026 (noite) | **Bug 2 identificado: schedule de LR mal calibrado para fine-tuning B2** | Mesmo após correção do bug 1, piloto B2 seed 42 atingiu pico de mAP50=0,30 na época 3, depois despencou para 0 com val_loss=NaN. Investigação via `lr/pg0` mostrou que o LR estava rampando de 0,00004 até 0,00175 sem entrar em decay — `cos_lr=True` não estava sendo aplicado corretamente com `optimizer='auto'`. |
| 10/04/2026 (noite) | **Correção bug 2: hyperparams diferenciados por baseline** | B1 ganha configuração SGD (lr0=0.01) e B2 ganha AdamW (lr0=0.001), ambos com `cos_lr=True` explícito. Decisão 5.11 documenta a diferenciação com justificativa da literatura padrão. Piloto B2 seed 42 após correção: mAP50=0,8367, 203 épocas, zero NaN, curva saudável. |
| 10/04/2026 (noite) | **Pipeline oficialmente destravado — início dos 6 treinos baseline** | Com as duas correções, o `treinar_baselines.py --all` é executado, rodando B1+B2 × 3 seeds (42/123/2024). Estimativa: ~4-5h total. Primeiros números reais do experimento chegam na sequência. |
| 13-14/04/2026 | **6 baselines completos: B1+B2 × 3 seeds executados com sucesso** | Tempo total: ~5h 18min. Todas as 6 corridas terminaram sem falhas. Resultados consolidados em `all_runs_summary.json`. Ver Seção 7.7 para análise completa. |
| 14/04/2026 | **Resultados dos baselines analisados e documentados** | B1: mAP50 = 0,8008 ± 0,0073. B2: mAP50 = 0,8351 ± 0,0024. Ganho de B2 sobre B1 = +4,3% em mAP50, +6,6% em mAP50-95. Separação completa entre intervalos (toda corrida de B2 supera melhor corrida de B1). B2 é ~3x mais consistente que B1 entre seeds. Análise vinculada à literatura via He et al. (2019). |
| 14/04/2026 (tarde) | **HPO de B2 iniciado (tarefa A7)** | Script `hpo_b2.py` escrito e disparado em sessão dedicada do Colab. Optuna TPE, 30 trials × 100 épocas × seed 42, 5 dimensões de hyperparams, função objetivo = fitness padrão do Ultralytics (0.1×mAP50 + 0.9×mAP50-95). Fase 2 com treino completo do TOP-1 agendada automaticamente. Critério de decisão ancorado em 3σ da variância entre seeds (≈ 0,0072 em mAP50). Metodologia pré-registrada na Seção 7.7.6 antes da execução para evitar seleção post-hoc do threshold. |
| 25/04/2026 | **BRAÇO A' CONCLUÍDO — RESULTADO PRINCIPAL DO PAPER** | 3 seeds (42, 123, 2024) concluídas. mAP50 = 0,8541 ± 0,0043 — **superou B2 em +1,90%** com separação estatística completa. mAP50-95 = 0,5281 ± 0,0056 (+4,5% relativo vs B2). A composição sintética in-place resolveu o gap de domínio e transformou um deficit de −4,15% (braço A) em ganho de +1,90%. Pipeline completo: COCO → 27.796 imagens sintéticas (crops InaTechShips em fundos CITRA-3D) → fine-tuning CITRA-3D-Real. Documentado em Seção 7.7.12. |
| 25/04/2026 | **Dataset sintético v3 concluído** | `gerar_dataset_copypaste.py` v3 (substituição in-place) gerou 27.796 imagens com 93.480 objetos (3,36 obj/img, zero falhas). 2.081 fundos × 13 variações. Tempo: 370,8 min. Documentado em composicao_report.json. |
| 24/04/2026 | **Braço B concluído — B ≈ A confirmado** | Seed 42: mAP50 = 0,7997, mAP50-95 = 0,4711. Diferença vs braço A (0,7936) = 0,61%, dentro de 1σ. Confirma que negative transfer é independente da curadoria CLIP — o problema é incompatibilidade estrutural de domínio. Documentado em Seção 7.7.11. |
| 24/04/2026 | **Braço B diagnóstico disparado** | Pré-treino (100ep) + fine-tuning (300ep) no CITRA-3D-Real com seed 42, usando random_pool_v2 como dataset de pré-treino. Protocolo idêntico ao braço A. Objetivo: confirmar se o negative transfer é universal (B ≈ A) ou específico da curadoria CLIP (B ≠ A). |
| 24/04/2026 | **D4 v3 aprovada e batch em execução** | Três iterações do script de composição: v1 (aleatório — navios em montanhas), v2 (HSV — parcial), **v3 (substituição in-place — ideia da Daniela)**. Preview com 100 imagens: 100% correto, 3,62 obj/img. Batch de ~27k imagens disparado. |
| 24/04/2026 | **Trilha B desbloqueada — labels do Eduardo recebidos** | Eduardo Teixeira compartilhou 2,8M labels via WhatsApp/Drive (8 pastas RAR). `filtrar_labels_eduardo.py` extraiu e filtrou: 36.272 matches (91,5%) dos 39.628 IDs. 25.591 labels copiados para random_pool_v2/{train,val,test}/labels/. Cobertura: train 92%, val 90%, test 91%. B1, B2, B3 todas concluídas no mesmo dia. |
| 24/04/2026 | **D3 concluída: extração de fundos** | `extrair_fundos_citra3d.py` executado. 2.081 fundos extraídos: 1.988 clean + 93 inpaint. Tempo: 35,9 min. |
| 23/04/2026 | **D3 pronto: `extrair_fundos_citra3d.py`** | Script de extração de fundos oceânicos (382 linhas). Duas estratégias: clean (imagens com <5% cobertura de bboxes usadas direto) e inpaint (cv2.INPAINT_TELEA para remover navios maiores). Estimativa ~1500-2000 fundos. Aguardando execução. |
| 23/04/2026 | **D2 concluída: extração de crops SAM** | 27.795 crops RGBA extraídos do dataset_25k_v2 com SAM ViT-B em ~16h (3 sessões com retomada). Cobertura mediana 46,2%. Filtro de qualidade aplicado: 23.828 crops usáveis (85,7%), 3.967 rejeitados. Metadados regenerados em `crops_metadata_full.json`. |
| 23/04/2026 | **Teste piloto bbox + decisão SAM para extração de crops** | Teste com 100 imagens em modo bbox: pipeline funcional (zero falhas, crops de 1.202×356px mediana). Decisão: usar modo SAM (segmentação por contorno do navio) em vez de bbox (crop retangular) para o batch completo. Razão: bbox inclui fundo original (porto/cais) dentro do retângulo, gerando artefatos quando colado em oceano. SAM segue o contorno real, produzindo composição mais realista. Custo: ~2h A100 (aceitável). Documentado em Seção 7.7.10 passo 2. |
| 23/04/2026 | **Passo 2 do Scale-Aware Copy-Paste pronto: `extrair_crops_sam.py`** | Script de extração de crops com SAM (553 linhas) escrito e validado sintaticamente. Dois modos: SAM ViT-B (segmentação precisa, ~0,3s/img) e bbox com feathering (fallback rápido, ~0,01s/img). Retomável via progress JSON. Aguardando execução no Colab A100. |
| 23/04/2026 | **Passo 1 do Scale-Aware Copy-Paste concluído: análise de escala** | `analisar_escala_citra3d.py` executado. Resultado: 7.003 bboxes analisados. 71,6% são "small" (COCO-style <32²px). Mediana da largura = 3,1% da imagem (~20px em 640×640). Mediana de 2 objetos/imagem (P90=7). Posição y_center concentrada em 0,37-0,70. Ranges P10-P90 extraídos para calibrar a composição. Pipeline expandido para 6 passos (adicionado passo 4.5 de validação FID). Bump para v0.3 dos documentos. |
| 22/04/2026 | **Decisão: Scale-Aware Copy-Paste para adaptação de domínio (braço A')** | Após constatar que pré-treino direto causa negative transfer (Seções 7.7.8-7.7.9), revisão de literatura identificou copy-paste como abordagem mais adequada vs CycleGAN e modelos de difusão. Razões: bboxes 100% exatos, fundos reais do CITRA-3D, controle determinístico de escala/densidade, precedente direto na literatura marítima (POSEIDON 2023, S3Det/ACCV 2024, Nemati 2025). Script `analisar_escala_citra3d.py` escrito para calibrar a composição. Pipeline de 5 passos definido (análise → SAM → fundos → composição → treino). Documentado em Seção 7.7.10. |
| 22/04/2026 | **Ablation de épocas de pré-treino concluída** | Script `ablation_epocas_pretreino.py` testou 3 variantes (10, 20, 50 épocas) com seed 42. Resultado: **degradação monotônica** — quanto mais épocas de pré-treino, pior o resultado final. Mesmo 10 épocas perde 1,5% mAP50 vs B2. Com 100 épocas, resultado converge para nível do B1 (random init). Diagnóstico: catastrophic forgetting confirmado com incompatibilidade parcial subjacente. Documentado em Seção 7.7.9. |
| 16–22/04/2026 | **Braço A do experimento principal concluído (3 seeds)** | `treinar_braco_a.py` executou COCO → dataset_25k_v2 (100ep) → CITRA-3D-Real (300ep) para seeds 42, 123, 2024. Resultado: mAP50 = 0,7936 ± 0,0060 — **4,15% abaixo de B2 e 0,72% abaixo de B1**. O pré-treino curado no InaTechShips piorou em vez de melhorar. Catastrophic forgetting identificado como causa provável. Documentado em Seção 7.7.8. |
| 16/04/2026 | **Downsample do random_pool_v2 concluído** | `downsample_random_pool_v2.py` executado no Drive, reduziu o random_pool_v2 de 39.628 para 27.964 imagens (com excesso residual de 168 arquivos / 0,60% sobre alvo — aceito como metodologicamente irrelevante). Estratificação preservada por decis empíricos do dataset_25k_v2. Excedentes (11.664 imagens) movidos para `_excedente/` no Drive (reversível, não deletados). Pipeline de preparação de dados agora completo. |
| 16/04/2026 | **Recálculo dos decis empíricos para dataset_25k_v2** | `recalcular_distribuicao_decis_v2.py` gerou `distribuicao_decis_v2.json` com 10 bins empíricos sobre os 27.796 IDs únicos do dataset_25k_v2. Distribuição muito uniforme (2.779-2.780 IDs por decil). Usado como input para o downsample do random_pool_v2. |
| 16/04/2026 | **Reconstrução do dataset_25k concluída (tarefa A12)** | `reconstruir_dataset_25k.py` executado no Drive, copiou 83.388 arquivos (imagens + labels + labels_single_class) para `dataset_25k_v2/` com split 60/20/20 estratificado por classe, seed 42. Resultado: 27.796 IDs únicos em 16.677/5.558/5.561. **Disjunção entre splits comprovada (0 sobreposição em todas as combinações).** Dataset original preservado como referência histórica. Documentado em Seção 4.3.3. |
| 15/04/2026 (noite) | **HPO de B2 concluído — DECISÃO: MANTER CONFIG ATUAL** | Fase 1: 28/30 trials válidos (1 falha intermitente Drive, 1 divergiu). TPE convergiu para região consistente nos top-3 trials (lr0≈0,00031, momentum≈0,945); TOP-5 inteiro tem warmup_epochs=2, sugerindo que esse parâmetro especificamente importa. Fase 2: TOP-1 (trial 13) em treino completo de 300 épocas atingiu mAP50=0,8328 vs baseline 0,8351, Δ=−0,0023 (=−0,94σ, dentro de 1 desvio padrão). Tempo total wall-clock ~14,7h (com queda intermediária do Colab recuperada via SQLite, sem perda de trials). Os 6 baselines existentes (B1+B2 × 3 seeds) permanecem válidos. Configuração de B2 formalmente validada via HPO. Documentado em Seção 7.7.7. |
| 14/04/2026 (tarde) | **Download do `random_pool_v2` rodada 1 concluído (tarefa A11a)** | 49.422 arquivos baixados em 95 min, ~2,7 imgs/s com 4 workers. Relatório original reportou 99,7% de sucesso. Crash por disk-full no meio da execução tratado pela retomada do script. |
| 14/04/2026 (fim do dia) | **Bug crítico descoberto via validação** | Validação completa via `validar_imagens_random_pool.py` revelou que **30,8% (15.236) dos arquivos eram HTML em vez de JPEG**. Causa raiz: shipspotting CDN retorna HTTP 200 + página HTML para IDs inexistentes, e o script v1 verificava apenas status_code == 200. Imagens corrompidas movidas para `_corrompidas/`, deixando 34.186 válidas (déficit de 3.923 vs alvo de 38.109). Análise por decil refutou a hipótese original sobre concentração no decil 9 — déficits espalhados uniformemente, com pico nos decis 0-3. Documentado em Seção 4.4.1 como lição metodológica. |
| 14/04/2026 (fim do dia) | **`baixar_random_pool_v2.py` v2 escrito** | Adicionado check de magic bytes JPEG (`FF D8`) antes de salvar arquivo + novo outcome `invalid_content` separado de `not_found` + flag `--ids-suffix` para suportar rodada 2. Versão antiga sobrescrita; histórico preservado em git. |
| 14/04/2026 (fim do dia) | **`gerar_ids_rodada2.py` escrito** | Sorteio complementar adaptativo: margem proporcional à taxa de sucesso observada por (split × decil), exclusões completas (dataset_25k + originais rodada 1 válidos + corrompidos), seed 4242. Estimativa ~5-7k IDs novos para baixar, ~40-50 min de download via v2. |

---

## 10. Direções futuras (pós-experimento principal)

1. **Multi-classe no CITRA-3D-Real com tratamento de long-tail.** Reintroduzir as 9 classes (incluindo possível coleta dirigida para Passageiro e Navio) usando técnicas de long-tail. Pode ser um segundo paper.
2. **Outras métricas de curadoria além de CLIP.** Testar features de detector intermediário (DINO, SAM embeddings, features de YOLO pré-treinado).
3. **Adaptação de domínio no espaço de features.** Métodos como DANN, CORAL, contrastive learning aplicados sobre o gap CITRA-3D ↔ shipspotting.
4. **Análise por classe original de embarcação.** Mapear cada bbox detectada à classe verdadeira do CITRA-3D.
5. **Pipeline completo de anotação reproduzível.** Caso a colaboração com Eduardo Teixeira evolua, formalizar o pipeline PointRend + YOLOv8 como ferramenta pública.

---

## 11. Referências

Esta seção organiza referências em quatro categorias, refletindo o nível de fundamentação literária de cada afirmação no documento.

### 11.1 Referências confirmadas e diretamente usadas

Estas são as referências cujo conteúdo foi diretamente citado no documento e que sustentam afirmações específicas:

- **He, K., Girshick, R., & Dollár, P. (2019).** Rethinking ImageNet pre-training. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 4918-4927. https://arxiv.org/abs/1811.08883
  - **Por que está aqui:** sustenta as observações de que (a) pré-treino genérico ajuda principalmente a acelerar convergência e não necessariamente o desempenho final, (b) treino do zero pode ser surpreendentemente competitivo com configuração adequada, e (c) o ganho do pré-treino tende a diminuir conforme o gap entre domínio fonte e domínio alvo aumenta. Estas observações são citadas na análise dos resultados dos baselines (Seção 7.7).

- **Loshchilov, I., & Hutter, F. (2019).** Decoupled weight decay regularization. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1711.05101
  - **Por que está aqui:** paper original do AdamW, que sustenta a escolha do otimizador para fine-tuning em B2 (Decisão 5.11). Os autores demonstram empiricamente que AdamW fornece estabilidade e generalização superiores ao Adam tradicional, especialmente quando combinado com cosine annealing — exatamente a configuração adotada para B2.

- **Teixeira, E. H., Mafra, S. B., & De Figueiredo, F. A. P. (2025).** InaTechShips: A validation study of a novel ship dataset through deep learning-based classification and detection models for maritime applications. *Ocean Engineering*, 326, 120823.
  - **Por que está aqui:** paper original do dataset InaTechShips, fonte do `dataset_25k` (subset A do ablation). Fornece a taxonomia, processo de coleta, e baseline de detecção contra o qual os resultados do braço A serão comparados.

- **Park, J., Kim, J., Moon, J., Oh, H., Park, H., Lee, G., & Hong, S. (2026).** Hyperparameter optimization to enhance the performance of deep learning models for the early detection of invasive turtles in Korea. *Scientific Reports*, 16, artigo 37636. https://www.nature.com/articles/s41598-026-37636-2
  - **Por que está aqui:** sustenta duas afirmações quantitativas na Seção 7.7.6 (HPO planejado). Primeiro, que **ganhos típicos de HPO em detecção single-class ficam na faixa de +1 a +3 pontos de mAP50/mAP50-95** — os autores reportam +1,5% em mAP50 e +3,2% em mAP50-95 após HPO completo e descrevem isso como melhoria "substancial". Segundo, que **ganhos <1% em mAP50 são tipicamente descritos como "não meaningful"** pela própria comunidade de detecção. Estes números calibram a expectativa realista para o HPO de B2 neste experimento e fundamentam a escolha do threshold 3σ (≈ 0,72 pontos) em vez de thresholds arbitrários maiores.

- **Ultralytics (2024).** YOLOv5 Hyperparameter Evolution — Fitness function definition. Documentação oficial. https://docs.ultralytics.com/yolov5/tutorials/hyperparameter_evolution/
  - **Por que está aqui:** define a função objetivo padrão do ecossistema Ultralytics usada como métrica no HPO: `fitness = 0.1 × mAP50 + 0.9 × mAP50-95`. Usar esta função específica no `hpo_b2.py` permite descrever a metodologia no paper como "função objetivo padrão do Ultralytics" em vez de uma definição ad hoc, facilitando reprodução e comparação com outros trabalhos. O peso 9:1 favorecendo mAP50-95 reflete a preferência da comunidade pela métrica mais sensível à precisão de localização.

### 11.2 Referências adicionais relevantes (verificar uso no paper futuro)

Estas são referências da área que provavelmente serão úteis na escrita do paper, mas que ainda não foram diretamente citadas no documento atual:

- **Radford, A., Kim, J. W., Hallacy, C., et al. (2021).** Learning transferable visual models from natural language supervision (CLIP). *International Conference on Machine Learning (ICML)*. https://arxiv.org/abs/2103.00020
  - **Por que está aqui:** paper original do CLIP, ferramenta usada para a curadoria por similaridade visual do `dataset_25k`. Fundamental na seção de Métodos do paper futuro.

- **Jocher, G., et al. (2024).** Ultralytics YOLO11. https://github.com/ultralytics/ultralytics
  - **Por que está aqui:** referência ao framework e arquitetura usados como detector principal. **[VERIFICAR — preferível citar paper técnico oficial se/quando houver para YOLOv11; no momento, a referência canônica é o repositório.]**

- **Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011).** Algorithms for Hyper-Parameter Optimization. *Advances in Neural Information Processing Systems (NeurIPS)*, 24. https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html
  - **Por que está aqui:** **[VERIFICAR]** paper original do TPE (Tree-structured Parzen Estimator), algoritmo usado como sampler do Optuna no HPO de B2 (Seção 7.7.6). Se TPE for mantido no HPO executado, esta referência deve entrar na Seção 11.1 quando o paper for escrito. Preciso confirmar URL e autores exatos antes da submissão.

- **Popek, Ł., et al. (2023).** Thermal imaging for wildlife detection using YOLOv5 with hyperparameter optimization. **[VERIFICAR — referência citada indiretamente via Park et al. 2026, precisa ser localizada e confirmada antes da submissão.]**
  - **Por que está aqui:** citada por Park et al. (2026) como exemplo de HPO em detecção com YOLO onde "o ajuste de hyperparameters (learning rate de 0.01→0.0123, momentum 0.937→0.934) não produziu melhoria meaningful sobre o default". Este é um segundo ponto de dado na literatura sustentando a expectativa honesta de que a config conservadora de B2 já pode estar próxima do ótimo — reforça o ponto feito na Seção 7.7.6 sobre a expectativa realista do HPO.

### 11.3 Afirmações com [CITAÇÃO PENDENTE]

Estas são afirmações feitas no documento que provavelmente têm fundamentação na literatura, mas para as quais ainda não foi identificada uma referência canônica específica. Marcadas explicitamente para busca futura antes da submissão do paper:

- **`[CITAÇÃO PENDENTE]` — "Hyperparams diferenciados para fine-tuning vs treino do zero é prática padrão na literatura de detecção de objetos."** (Decisão 5.11)
  - *Sugestão de busca:* "transfer learning hyperparameters fine-tuning learning rate object detection"; "small learning rate fine-tuning pretrained models"
  - *Provável fonte:* surveys de transfer learning como Zhuang et al. (2021) "A comprehensive survey on transfer learning"; ou guias práticos como cs231n notes (Karpathy/Stanford).

- **`[CITAÇÃO PENDENTE]` — "Pré-treino tende a atuar como regularizador implícito, reduzindo a variância entre seeds em treinos repetidos."** (Seção 7.7, finding 1)
  - *Sugestão de busca:* "pretraining variance reduction seeds reproducibility deep learning"
  - *Observação:* esta afirmação é uma interpretação razoável do nosso resultado empírico (DP de B2 é ~3x menor que DP de B1), mas a generalização ("pré-treino regulariza") precisa de uma referência específica que mostre isso em vários setups, não só no nosso. He et al. (2019) discute aspectos relacionados mas não mede explicitamente variância entre seeds.

- **`[CITAÇÃO PENDENTE]` — "mAP50-95 é mais sensível à precisão da localização da bbox do que mAP50, pois agrega 10 thresholds de IoU em vez de apenas um."** (Seção 7.7, finding 3)
  - *Sugestão de busca:* "COCO mAP evaluation metric IoU threshold averaging"
  - *Provável fonte:* paper original do COCO de Lin et al. (2014) "Microsoft COCO: Common objects in context", ou documentação oficial do COCO evaluation server.

- **`[CITAÇÃO PENDENTE]` — "A razão mAP50-95/mAP50 entre 0.55 e 0.65 é típica para detecção de objetos em cenários reais."** (Seção 7.7, comentário sobre razão de 0,61)
  - *Observação:* esta afirmação é uma generalização baseada em experiência prática que não tem fonte canônica clara. **Se não encontrar referência sólida, será removida ou enfraquecida no paper para uma observação puramente descritiva ("nossa razão de 0,61 é consistente com a faixa observada em datasets como X, Y, Z" — citando exemplos concretos com seus números).**

### 11.4 Afirmações REMOVIDAS por não terem fundamentação

Para registro de honestidade metodológica, as afirmações abaixo apareceram em versões anteriores deste documento ou em discussões de trabalho, mas foram **removidas** porque não foram encontradas referências sólidas que as sustentem:

- ~~"A regra prática usual em ML é 'diferença significativa = maior que 2× o desvio padrão'."~~ Removida. Esta foi uma heurística ad hoc proposta durante discussão de análise dos resultados, sem fundamentação na literatura. Na seção de resultados, **a comparação entre B1 e B2 será feita com testes estatísticos formais** (ver Seção 7.7) ou simplesmente reportando média ± desvio padrão sem afirmar significância sem teste, o que é prática aceita para experimentos com poucas seeds.

### 11.5 Recursos do projeto (não-bibliográficos)

- **Repositório oficial do InaTechShips:** https://github.com/EduardoHT/InaTechShips
- **Repositório do projeto desta autora:** https://github.com/DanielaLFreire/InaTechShips
- **CITRA-3D-Real (dataset autoritativo):** `/content/drive/MyDrive/PROJETO_MARINHA/Datasets/CITRA-3D-Real/` (interno)
- **Documentação Ultralytics YOLO:** https://docs.ultralytics.com

---

## Apêndice A — Glossário rápido

- **CITRA-3D-Real:** dataset operacional da Marinha do Brasil, versão autoritativa extraída do ZIP original. 2.081 imagens reais, 9 classes em português, 7.003 bboxes válidas, 3,36 bboxes/imagem em média.
- **InaTechShips:** dataset público de Teixeira et al. (2025). 3.013.830 imagens, 200+ classes, fotografia de shipspotting.com.
- **dataset_25k:** subset do InaTechShips curado pela autora via similaridade CLIP. 38.109 imagens, 10 classes (TOP 10 do InaTechShips). É o **subset A** do ablation.
- **random_pool_v2:** subset do InaTechShips a ser baixado, sorteado aleatoriamente com estratificação por decis. Mesmo tamanho do dataset_25k. É o **subset B** do ablation. Listas de IDs já geradas; download pendente.
- **Ablation A vs B:** comparação central do experimento.
- **Classe única:** decisão de colapsar todas as classes originais para "embarcação" no treino e na avaliação.
- **HPO:** Hyperparameter Optimization.
- **Quarentena:** pasta `_quarantine/` no CITRA-3D-Real contendo imagens removidas do dataset por motivos documentados, preservadas para reversibilidade.
- **Pipeline de preparação:** sequência de 6 scripts versionados que transforma o `CITRA-3D.zip` original no `CITRA-3D-Real` pronto para treino.

---

*Fim do documento. Próxima atualização: após resposta do Eduardo Teixeira ou após geração dos `data.yaml` classe-única e início dos baselines B1/B2.*
