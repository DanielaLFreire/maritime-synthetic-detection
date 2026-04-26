# Cronograma de Tarefas — Experimento de Curadoria por Similaridade Visual

**Documento companheiro de:** `experimento_descricao_v04.md`
**Versão:** 0.4 — 25 de abril de 2026
**Autora:** Daniela L. Freire (ICMC/USP)

---

## Histórico de versões

- **v0.1 (09/04/2026):** versão inicial, com Trilha A focada em resolver nomenclatura de classes, filtrar sintéticas, padronizar labels.
- **v0.2 (10/04/2026):** **atualização após conclusão do pipeline de preparação do CITRA-3D-Real.** Tarefas A0–A3 e A8 reformuladas em uma única "Trilha A — Concluída" que documenta o que foi feito. Novas tarefas adicionadas para refletir o estado real (geração dos `data.yaml`, treinos baseline, HPO, validação visual). O bloqueio do random_pool_v2 (Eduardo) continua, mas a estratégia de paralelismo foi refinada.
- **v0.2 — micro-revisão (10/04/2026, fim do dia):** A4 marcada como ✅ concluída (`gerar_data_yaml.py` rodou com sucesso, status `✓ TUDO OK`). A5 e A6 reformuladas com configuração consolidada de treino: **YOLOv11m**, 300 épocas, patience 30, imgsz 640, batch 16, 3 seeds. Adicionada opção de sanity check com YOLOv8m em paralelo. A0 atualizada para listar `gerar_data_yaml.py` como 7º script do pipeline.
- **v0.2 — segunda micro-revisão (10/04/2026, noite):** dois bugs descobertos e corrigidos durante os primeiros treinos. (1) `gerar_data_yaml.py` substituído por `preparar_dados_locais.py` (bug de symlinks com Ultralytics). (2) `treinar_baselines.py` atualizado com hyperparams diferenciados por baseline (B1 SGD lr0=0.01, B2 AdamW lr0=0.001, `cos_lr=True` explícito em ambos). Piloto B2 seed 42 validado após correções: mAP50=0,8367 em 53 min. Pipeline destravado; baselines completos em execução. A0 atualizada para refletir o 8º script (`treinar_baselines.py`) e a substituição do 7º.
- **v0.2 — terceira micro-revisão (14/04/2026, tarde):** duas tarefas críticas em execução paralela após conclusão dos baselines.
  - **A7 (HPO em B2)** reescrita como "EM ANDAMENTO" com metodologia completa pré-registrada: Optuna TPE, 30 trials × 100 épocas, 5 dimensões de busca, função objetivo fitness do Ultralytics, duas fases (exploração + validação do TOP-1), critério de decisão 3σ ancorado na variância empírica de B2 (≈ 0,0072 em mAP50).
  - **A11 (download random_pool_v2)** atualizada: primeira execução crashou por falta de espaço em ~35k imagens; retomada funcionou conforme projetado (script detecta `.jpg` em disco + progresso JSON). Segunda execução em andamento, ETA ~1-2h para conclusão.
  - **Duas tarefas rodam em paralelo perfeito** — HPO usa GPU do Colab remoto, download usa CPU/internet/disco locais, zero conflito de recursos.
- **v0.2 — quarta micro-revisão (14/04/2026, fim do dia):** descoberta crítica e plano de recovery.
  - **A11 reestruturada em sub-tasks A11a (rodada 1, ✅ executada com bug), A11b (validação, ✅ revelou bug), A11c (rodada 2, scripts prontos para executar).**
  - Bug crítico descoberto via validação posterior: shipspotting CDN retorna HTTP 200 + HTML para IDs inexistentes, e o script v1 não verificava o conteúdo. **30,8% das imagens da rodada 1 eram HTMLs disfarçados de JPEG**, deixando déficit de 3.923 vs alvo do dataset_25k.
  - **Dois scripts novos prontos:** `gerar_ids_rodada2.py` (sorteio adaptativo com margem proporcional à taxa de sucesso observada por decil) e `baixar_random_pool_v2.py` v2 (com check de magic bytes JPEG antes de salvar + novo outcome `invalid_content` + flag `--ids-suffix`).
  - **Custo da recovery:** ~50 min adicionais (sorteio + download + validação). Lição metodológica registrada para o paper (Seção 4.4.1 do experimento_descricao_v02.md).
- **v0.2 — quinta micro-revisão (15/04/2026):** fechamento simultâneo de A11 e A7.
  - **A11 fechada (✅).** Rodada 2 atingiu 39.628 imagens válidas (folga +4% sobre alvo do dataset_25k, 100% JPEG verificado). Sub-task A11d nova adicionada documentando o upload e verificação cruzada do random_pool_v2 para o Drive.
  - **A7 fechada (✅) com decisão MANTER CONFIG ATUAL.** HPO completo em ~10h (com 1 reinicialização do Colab). 28/30 trials válidos. TOP-1 da Fase 2 (treino completo) atingiu mAP50=0,8328 vs baseline 0,8351 (Δ=−0,0023, dentro de 1σ). Threshold pré-registrado de 3σ não foi superado. Configuração de B2 formalmente validada. Os 6 baselines existentes permanecem válidos.
  - **Resultado significativo para o paper:** A confirmação formal de que a config inicial é estatisticamente equivalente ao ótimo encontrado na busca é uma defesa metodológica forte. Park et al. (2026) e Popek et al. (2023) já indicavam essa expectativa para HPO em YOLO single-class — o resultado confirma a literatura.
  - **Nova pendência identificada — A12 (a criar):** Reconstrução do dataset_25k. Durante a preparação do random_pool_v2, foi descoberto que o dataset_25k tem 27.796 IDs únicos mas 38.109 arquivos por causa de duplicação entre splits (bug no notebook de montagem). Pré-requisito para o braço A do experimento principal.
- **v0.2 — sexta micro-revisão (16/04/2026):** fechamento do sprint de preparação de dados com A12 e downsample.
  - **A12 criada e fechada (✅)** — "Reconstrução do dataset_25k". Três sub-tasks: (a) reconstrução do dataset_25k → dataset_25k_v2 via `reconstruir_dataset_25k.py` (83.388 arquivos copiados no Drive em ~45 min, split 60/20/20 estratificado por classe, disjunção total comprovada), (b) recálculo dos decis empíricos via `recalcular_distribuicao_decis_v2.py` (~30s, distribuição uniforme 2.779-2.780 IDs/decil), (c) downsample do random_pool_v2 via `downsample_random_pool_v2.py` (~20 min, reduziu 39.628 → 27.964 imagens com excesso residual de 0,60% aceito como irrelevante).
  - **B4 (treinar braço A) atualizada** com dependência de A12 e referência ao `dataset_25k_v2` em vez do `dataset_25k` original.
  - **Status do pipeline de preparação de dados: 100% concluído.** Todas as pendências de dados resolvidas. Próximas pendências são labels do random_pool_v2 (dependência externa ou Plano B de auto-anotação) e email institucional (C4).
- **v0.2 — sétima micro-revisão (22/04/2026):** resultados do braço A e ablation.
  - **B4 fechada (✅)** — braço A concluído com resultado surpreendente: mAP50 = 0,7936 ± 0,0060, **4,15% abaixo de B2** e 0,72% abaixo de B1. Pré-treino curado no InaTechShips causou negative transfer.
  - **B4.1 criada e fechada (✅)** — ablation de épocas de pré-treino (10, 20, 50 épocas). Degradação monotônica confirmada. Catastrophic forgetting diagnosticado.
  - **Eduardo Teixeira respondeu ao email** (15/04/2026). Aceita colaboração ativa e coautoria. Email de resposta enviado com lista de 39.628 IDs para labels do PointRend. Trilha B em processo de desbloqueio.
  - **Status geral:** braço A completo, ablation completa, aguardando labels para braço B. Comparação central curado vs aleatório depende apenas da resposta do Eduardo.
- **v0.3 (23/04/2026):** nova fase experimental — Scale-Aware Copy-Paste.
  - **Nova Trilha D** com 6 tarefas (D1-D5, incluindo D4.5 validação). D1 concluída, D2 pronta para execução.
  - Bump de versão justificado pela mudança de paradigma experimental: de pré-treino direto (demonstrado ineficaz via braço A + ablation) para geração de dados sintéticos por composição de imagens.
  - Pipeline documentado em detalhe na Seção 7.7.10 do `experimento_descricao_v03.md`.
  - Colaboração com Eduardo Teixeira formalizada (coautoria + labels do PointRend solicitados).
  - **Atualização 24/04/2026:**
    - **B1, B2, B3 fechadas (✅).** Eduardo compartilhou 2,8M labels via WhatsApp/Drive. Filtrados 36.272 matches (91,5%) dos 39.628 IDs. 25.591 labels copiados para random_pool_v2. Trilha B totalmente desbloqueada.
    - **B5 em execução (🟡).** Braço B diagnóstico (1 seed) disparado. Protocolo idêntico ao braço A, trocando dataset curado por aleatório.
    - **D3 fechada (✅).** 2.081 fundos extraídos (1.988 clean + 93 inpaint).
    - **D4 aprovada e em execução (🟡).** Três iterações do script de composição: v1 (posicionamento aleatório — navios em montanhas, descartada), v2 (water-aware HSV — parcial, descartada), **v3 (substituição in-place — ideia da Daniela: substituir navios reais do CITRA-3D por crops na mesma posição/dimensão, 100% correto)**. Preview com 100 imagens aprovado. Batch de ~27k em execução.
- **v0.4 (25/04/2026):** **experimento concluído com resultado positivo.**
  - **B5 fechada (✅).** Braço B: mAP50 = 0,7997. B ≈ A confirmado — negative transfer é universal.
  - **D4 fechada (✅).** 27.796 imagens sintéticas geradas (v3 in-place), 93.480 objetos, zero falhas.
  - **D4.5 fechada (✅).** Validação implícita (distribuição idêntica por construção).
  - **D5 fechada (✅).** **RESULTADO PRINCIPAL:** braço A' mAP50 = 0,8541 ± 0,0043 — **superou B2 em +1,90%** com separação estatística completa. mAP50-95 = 0,5281 ± 0,0056 (+4,5% relativo).
  - **Todas as trilhas experimentais concluídas** (A, B, D). Trilha C (escrita) é a única pendente.
  - **Status geral: experimento concluído. Pronto para escrita do paper.**

---

## Visão geral

O projeto está em um estado significativamente melhor que na versão anterior do cronograma. O pipeline de preparação do CITRA-3D-Real está **completamente concluído** — 6 scripts versionados, todos os relatórios gerados, dataset autoritativo estabelecido. As três trilhas paralelas continuam válidas, mas a Trilha A teve várias tarefas movidas para "Concluído", o que libera espaço para iniciar os primeiros treinos.

### As três trilhas

**Trilha A — Trabalho independente da resposta do Eduardo.** Continua sendo o coração do progresso. Inclui geração dos `data.yaml`, primeiros treinos baseline, HPO, validação visual de auto-anotação, e — agora liberada como adiantável — o **download das imagens do random_pool_v2**, que pode rodar em paralelo enquanto os labels são resolvidos.

**Trilha B — Trabalho dependente da resposta do Eduardo.** Reduzida em escopo. Agora inclui apenas as tarefas que realmente dependem dos labels: anotação do random_pool_v2, treino do braço B (aleatório). O download das imagens não está mais aqui.

**Trilha C — Escrita e organização.** Mesma de antes: documento vivo, rascunho do paper, decisão de venue, validação institucional com a Marinha.

---

## Trilha A — Status atual

### ✅ A0. Pipeline de preparação do CITRA-3D-Real

**Status:** ✅ **CONCLUÍDO em 10/04/2026.**

Esta tarefa absorveu o que originalmente eram A1 (resolver nomenclatura), A2 (filtrar sintéticas) e A3 (padronizar labels) na versão 0.1 do cronograma. Foi expandida para um pipeline completo de 6 scripts versionados:

1. ✅ `preparar_citra3d.py` — extração seletiva do `CITRA-3D.zip` autoritativo, ignorando ZIPs `_aug` e imagens sintéticas
2. ✅ `analise_citra3d_real.py` — auditoria pós-extração, identificou labels malformados
3. ✅ `limpar_labels_citra3d.py` — limpeza automática de labels com bug `Quadrado_marcacao(Clone)`
4. ✅ `quarentenar_imagens_orfas.py` — isolamento de 2 imagens com labels vazios após limpeza
5. ✅ `auditar_citra3d_cleaned.py` — re-auditoria de validação (todos os checks ✓)
6. ✅ `gerar_labels_single_class.py` — colapso para classe única em CITRA-3D-Real e dataset_25k
7. ✅ `preparar_dados_locais.py` — cópia física dos dados para `/content/data/` do Colab + geração dos `data.yaml` classe-única (concluído em 10/04/2026; **substituiu o `gerar_data_yaml.py` original**, que usava links simbólicos incompatíveis com o Ultralytics; ver Decisão 5.12 do documento principal)
8. ✅ `treinar_baselines.py` — script de treino parametrizado para B1 e B2 com hyperparams diferenciados (ver Decisão 5.11); validado em 10/04/2026 com piloto B2 seed 42 (mAP50=0,8367 em 53 min, 203 épocas)

**Resultado:**
- Dataset autoritativo: **CITRA-3D-Real** com 2.081 imagens reais (1.348 train / 332 val / 401 test) e 7.003 bboxes válidas
- Mapeamento de classes confirmado via `data.yaml` oficial
- Pipeline 100% rastreável, com 6 relatórios auditáveis
- Pastas `labels_single_class/` geradas para os dois datasets
- Versão `CITRA-3D-Prepared` legada arquivada

**Por que essa consolidação importa:** o que parecia "resolver pendências" virou na prática a montagem de uma fundação sólida. Os relatórios gerados nesse processo viram os artefatos defensáveis na seção de Métodos e nos anexos do paper.

---

### A4. ✅ Preparação de dados locais + geração dos `data.yaml` classe-única

**Status:** ✅ **CONCLUÍDA em 10/04/2026** (após duas iterações — ver histórico abaixo).
**Tempo real:** ~10 min (primeira tentativa) + ~15 min (diagnóstico do bug) + ~5 min (correção e segunda execução)
**Dependências:** A0 (concluída)

**Histórico:** esta tarefa foi implementada duas vezes por conta de um bug silencioso descoberto durante os primeiros treinos:

**Primeira tentativa (manhã/tarde de 10/04):** script `gerar_data_yaml.py` usando **links simbólicos** entre `/content/data/` e o Drive. Validação inicial reportou `✓ TUDO OK`. Parecia elegante (custo zero em espaço, idempotente, sem duplicação).

**Bug descoberto (noite de 10/04):** ao rodar os primeiros treinos com o `treinar_baselines.py`, o Ultralytics descartou TODAS as imagens com a mensagem `ignoring corrupt image/label: Label class 7 exceeds dataset class count 1`. Investigação revelou que o Ultralytics segue o link de `images/`, obtém o caminho real via `Path.resolve()`, e aí aplica `str.replace('/images/', '/labels/')` sobre o caminho REAL (não o link). Resultado: lê os labels originais com 9 classes em vez do `labels_single_class/` apontado pelo link. Bug silencioso, produziu mAP aleatório nos primeiros treinos.

**Segunda tentativa (noite de 10/04):** script `gerar_data_yaml.py` **substituído** por `preparar_dados_locais.py`, que faz **cópia física** do Drive para `/content/data/`, em vez de links. Inclui validação robusta com três checks: (1) detecção de symlinks residuais, (2) verificação de que as pastas finais não são symlinks, (3) sample check do primeiro label de cada split para confirmar classe `0`. Segunda tentativa validou `✓ TUDO OK` e o piloto B2 subsequente confirmou que o bug foi resolvido.

**Entregáveis confirmados (versão final):**
- ✅ `citra3d_single_class.yaml` em `/content/drive/MyDrive/PROJETO_MARINHA/Datasets/configs/`
- ✅ `dataset_25k_single_class.yaml` em `/content/drive/MyDrive/PROJETO_MARINHA/Datasets/configs/`
- ✅ `data_setup_log.json` com status `✓ TUDO OK`
- ✅ `/content/data/CITRA-3D-Real-SC/` com cópia física (2.081 imagens + 2.081 labels classe-única, ~2 GB)
- ✅ Bug do yaml original (`val=test`) corrigido — splits separados nos novos yamls
- ✅ Validação robusta passou (sem symlinks, sample check confirmou labels com classe `0`)

**Pendente:** cópia do `dataset_25k` (~15 GB, ~15-25 min) — será feita na sessão que for treinar o braço A do experimento, não é necessária para os baselines atuais.

**Limitação operacional registrada:** o `/content/` é apagado entre sessões do Colab; é necessário rodar `preparar_dados_locais.py` no início de cada nova sessão. Os yamls em si são persistentes.

**Lição aprendida:** nunca confiar em links simbólicos quando o consumidor é uma biblioteca de terceiros que faz manipulação de strings em paths. O bug do Ultralytics é específico, mas a classe geral de problema (symlinks + normalização de path) é recorrente. A validação anti-symlink do `preparar_dados_locais.py` é uma proteção permanente contra recorrência.

---

### A5. Treinar baseline B1 (CITRA-3D-Real do zero, sem pré-treino)

**Status:** 🟢 **Próxima tarefa imediata** — pode rodar agora
**Tempo estimado:** ~3–5 horas (3 seeds × ~1–1,5h cada na A100)
**Dependências:** A4 (concluída)

**Configuração consolidada:**

| Parâmetro | Valor |
|---|---|
| Modelo | **YOLOv11m** (init random — `weights=None`) |
| Dataset | CITRA-3D-Real classe-única (via `citra3d_single_class.yaml`) |
| Épocas | 300 com early stopping |
| Patience | 30 |
| imgsz | 640 |
| Batch | 16 (conservador para A100) |
| Optimizer | **SGD** com lr0=0.01, momentum=0.937, cos_lr=True (ver Decisão 5.11) |
| Seeds | 42, 123, 2024 |

**Entregável:** 3 corridas completas + tabela com média ± desvio padrão.

**Status: ✅ CONCLUÍDA em 14/04/2026.** As 3 corridas completaram com sucesso. Resultado: **mAP50 = 0,8008 ± 0,0073** (média ± DP em test set). Variação entre seeds: range [0,7959; 0,8094]. Tempo médio por corrida: 52,3 min. Detalhes completos na Seção 7.7 do `experimento_descricao_v02.md`.

---

### A6. ✅ Treinar baseline B2 (pré-treino COCO + fine-tuning CITRA-3D-Real)

**Status:** ✅ **CONCLUÍDA em 14/04/2026** (par com A5, no mesmo batch `--all`)
**Tempo real:** ~2,7h (3 seeds × ~54 min cada)
**Dependências:** A4 (concluída, após dois bugs corrigidos)

**Resultado final (3 seeds):** **mAP50 = 0,8351 ± 0,0024** (média ± DP em test set). Variação entre seeds: range [0,8323; 0,8367]. mAP50-95 = 0,5055 ± 0,0027. Detalhes completos na Seção 7.7 do `experimento_descricao_v02.md`.

**Ganho sobre B1:** +0,0343 em mAP50 (+4,3%), +0,0313 em mAP50-95 (+6,6%). Separação completa entre intervalos: toda corrida de B2 superou a melhor corrida de B1 (gap = 0,0229 em mAP50). B2 é ~3x mais consistente que B1 entre seeds (DP 0,0024 vs 0,0073).

**Configuração consolidada:**

| Parâmetro | Valor |
|---|---|
| Modelo | **YOLOv11m** partindo de `yolo11m.pt` (pré-treino COCO da Ultralytics) |
| Dataset | CITRA-3D-Real classe-única (via `citra3d_single_class.yaml`) |
| Épocas | 300 com early stopping |
| Patience | 30 |
| imgsz | 640 |
| Batch | 16 |
| Optimizer | **AdamW** com lr0=0.001, cos_lr=True, warmup_bias_lr=0.01 (ver Decisão 5.11) |
| Seeds | 42, 123, 2024 |

**Resultado do piloto (B2 seed 42, 10/04/2026 noite):**

| Métrica | Valor | Observação |
|---|---|---|
| mAP50 (test) | **0,8367** | Acima do critério de sucesso (≥ 0,70) |
| mAP50-95 (test) | 0,5062 | Razão 0,61 = típica para detecção marítima |
| Precision | 0,8639 | Modelo conservador |
| Recall | 0,7803 | |
| Épocas treinadas | 203 de 300 | Early stopping em patience=30, curva estável nas últimas 30 épocas |
| Melhor val mAP50 | 0,8058 | Época 173 |
| Tempo de treino | 53 min | |
| NaN no val_loss | **zero** | Treino numericamente estável |

**Observação importante:** test mAP50 (0,8367) > best val mAP50 (0,8058). Isso confirma que o split de test do CITRA-3D é estruturalmente "mais fácil" que o val — propriedade do dataset, não variância aleatória. **No paper, métrica principal será test** (padrão na literatura), mas val e test não são intercambiáveis — vale mencionar explicitamente para evitar confusão de revisores.

**Sanity check opcional (não bloqueante):** rodar 1 seed adicional de B1 e B2 com **YOLOv8m** para comparação cruzada com a arquitetura usada nos baselines anteriores do grupo. Custo: ~2 corridas extras (~2–3h). Não muda o experimento, só fortalece a defesa do paper diante de revisores.

---

### A7. HPO no braço B2

**Status:** ✅ **CONCLUÍDA (15/04/2026, noite)**
**Tempo total:** ~14,7h wall-clock (Fase 1 + Fase 2, com 1 reinicialização após queda do Colab; tempo de GPU efetivo ~14h)
**Dependências:** A5 e A6 concluídas

**Resultado:** **DECISÃO AUTOMÁTICA: MANTER CONFIG ATUAL.** O TOP-1 do HPO em treino completo (trial 13: lr0=0,000308, lrf=0,0206, momentum=0,9498, weight_decay=0,000101, warmup_epochs=2) atingiu mAP50=0,8328 no test set vs baseline B2=0,8351. Δ = −0,0023, **dentro de 1σ da variância entre seeds** (σ=0,0024). Não supera o threshold pré-registrado de 3σ ≈ 0,0072 em mAP50. Os 6 baselines existentes (B1+B2 × 3 seeds) permanecem válidos.

**Significado para o paper:** este resultado é uma **validação metodológica forte**, não um resultado nulo. A configuração de B2 está formalmente validada via HPO formal e a frase defensável é: "os hiperparâmetros foram validados via Optuna TPE com 30 trials e a configuração inicial provou-se dentro de 1σ do ótimo encontrado no espaço de busca testado". Adicionalmente, a observação de que o TOP-1 e o baseline têm parâmetros bastante diferentes (lr0 3x menor, weight_decay 5x menor) mas performance equivalente sugere baixa sensibilidade do problema aos hyperparams nessa faixa razoável — outra observação publicável.

**Arquivos finais:**
- `/content/drive/MyDrive/PROJETO_MARINHA/Experimento_Dataset_Similar/hpo/B2/hpo_report.json` — relatório consolidado
- `/content/drive/MyDrive/.../hpo/B2/optuna_study.db` — estudo Optuna persistente
- `/content/drive/MyDrive/.../hpo/B2/trial_summary.json` — log estruturado dos 30 trials
- `/content/drive/MyDrive/.../hpo/B2/phase2_validation/result.json` — comparação Fase 2

**Documentado em:** Seção 7.7.7 do `experimento_descricao_v02.md`. A7 fechada.

---

### A8. Validação cruzada de auto-anotação (preparatório para Plano B)

**Status:** 🟢 Independente, valor estratégico alto
**Tempo estimado:** 2–3 horas
**Dependências:** A0 (concluída)

**O que fazer:**
1. Pegar amostra aleatória de 500 imagens do `dataset_25k` curado.
2. Rodar YOLOv11x pré-treinado em COCO (classe `boat`, índice 8) sobre essas 500 imagens.
3. Comparar bboxes preditas com labels existentes do `dataset_25k`, calculando IoU.
4. Gerar relatório: distribuição de IoU, % com IoU > 0,5, % com IoU > 0,7, casos onde COCO detecta zero embarcações ou múltiplas.
5. Documentar no documento vivo.

**Por que isso é valioso mesmo se o Eduardo responder:**
- (a) Você ganha medida empírica do gap entre auto-anotação genérica e PointRend, que entra na Discussion do paper.
- (b) Se o IoU médio for alto, evidência de que o Plano B (auto-anotação YOLO COCO) é viável.
- (c) Se for baixo, descarta o Plano B antes de gastar tempo nele.

---

### A9. Caracterização visual do gap CITRA-3D-Real ↔ dataset_25k

**Status:** 🟢 Independente, alto valor para o paper
**Tempo estimado:** 3–4 horas
**Dependências:** A0 (concluída)

**O que fazer:**
1. Calcular embeddings CLIP (cache existente para CITRA-3D-Real) e gerar embeddings para amostra do `dataset_25k` (~5.000 imagens).
2. Reduzir dimensionalidade via UMAP ou t-SNE para 2D.
3. Plotar os dois conjuntos no mesmo espaço 2D, com cores diferentes.
4. Calcular métricas quantitativas: distância média ao vizinho mais próximo, MMD (Maximum Mean Discrepancy).
5. Quando o random_pool_v2 estiver baixado, repetir comparando CITRA-3D-Real vs random.

**Entregável:** figura UMAP/t-SNE + métricas quantitativas + parágrafo no documento vivo. Esta figura é uma das mais fortes para a seção de Métodos do paper.

---

### A10. Refinamento da revisão de literatura

**Status:** 🟢 Independente, contínuo
**Tempo estimado:** 4–6 horas distribuídas
**Dependências:** nenhuma

**O que fazer:**
1. Buscar papers recentes (últimos 2–3 anos) sobre:
   - Transfer learning entre domínios em detecção
   - Curadoria de datasets via embeddings
   - Detecção de embarcações marítimas em cenário operacional
   - Long-tail learning em detecção
2. Organizar em planilha: título, autores, ano, venue, contribuição, relação com seu trabalho.
3. Identificar 3–5 papers como "trabalho relacionado mais próximo" e ler com cuidado.

**Entregável:** planilha de referências + parágrafos preliminares de Related Work no rascunho do paper.

---

### A11. ⭐ Download das imagens do random_pool_v2 (movido da Trilha B)

**Status:** ✅ **CONCLUÍDA (15/04/2026)**
**Tempo total:** ~2h 45min (rodada 1 + validação + rodada 2 + upload)
**Dependências:** nenhuma

Esta tarefa estava originalmente na Trilha B, bloqueada. Foi movida para a Trilha A porque as imagens são as mesmas em qualquer cenário — só os labels é que dependem do Eduardo. Adiantar o download poupou ~10 horas no caminho crítico depois.

**Resultado final:** **39.628 imagens válidas** (22.899 train + 9.515 val + 7.214 test) em `~/PROJETO_MARINHA/random_pool_v2/` localmente E em `/content/drive/MyDrive/InaTechShips/random_pool_v2/` no Drive. 100% JPEG válido por validação de magic bytes. Folga de +4% sobre os alvos do dataset_25k (38.109).

**Lição metodológica importante registrada para o paper:** validar magic bytes (`FF D8` para JPEG) do conteúdo da resposta HTTP, não confiar em status code 200 para detectar respostas válidas de CDN. O shipspotting CDN retorna HTTP 200 + página HTML de erro para IDs inexistentes, comportamento que escapou ao script v1 e produziu 30,8% de arquivos corrompidos. A v2 do downloader (com magic-bytes check antes de salvar) eliminou esse problema.

#### Sub-task A11a. ✅ Rodada 1 — primeira tentativa de download

**Status:** ✅ executada, mas com bug crítico descoberto na validação posterior
**Quando:** 14/04/2026, manhã-tarde
**Tempo gasto:** 95 min (download) + crash + retomada

Executada com `baixar_random_pool_v2.py` v1, 4 workers paralelos, 49.554 IDs sorteados (margem de 30% sobre o alvo de 38.109). A primeira execução crashou em ~35k imagens por falta de espaço em disco; após liberação, retomada via `download_progress.json` + detecção de `.jpg` em disco completou os 49.422 IDs em 95 minutos totais. O relatório original reportou 100% sucesso e 0% 404s, parecendo um download perfeito.

#### Sub-task A11b. ✅ Validação completa das imagens

**Status:** ✅ concluída, revelou bug na rodada 1
**Quando:** 14/04/2026, fim da tarde
**Tempo gasto:** ~30 segundos para validar 49.422 arquivos

Executada via `validar_imagens_random_pool.py` em modo paralelo (4 workers). Cascata de checks: tamanho mínimo, magic bytes JPEG, PIL verify, PIL open size. Resultado:

| Split | Total | Válidas | Corrompidas | % |
|---|---|---|---|---|
| train | 28.577 | 19.617 | 8.960 | 31,4% |
| val | 11.887 | 8.362 | 3.525 | 29,7% |
| test | 8.958 | 6.207 | 2.751 | 30,7% |
| **Total** | **49.422** | **34.186** | **15.236** | **30,8%** |

**Causa raiz:** o shipspotting CDN retorna **HTTP 200 + página HTML de erro** em vez de HTTP 404 para IDs inexistentes. A v1 do downloader verificava apenas `status_code == 200` e tamanho mínimo, então salvava HTMLs como `.jpg`. **97,5% dos arquivos corrompidos eram HTML disfarçado.**

Arquivos corrompidos movidos para `_corrompidas/{split}/` (preservados para inspeção futura, não deletados). Resultado: **34.186 imagens válidas no random_pool_v2** vs alvo de 38.109. Déficit: 3.923 imagens.

#### Sub-task A11c. ✅ Rodada 2 — sorteio complementar adaptativo (CONCLUÍDA)

**Status:** ✅ executada e validada
**Quando:** 14/04/2026 (à noite) — 15/04/2026 (madrugada)
**Tempo gasto:** ~5 min sorteio + 66 min download + 30s validação

`gerar_ids_rodada2.py` sorteou 8.110 novos IDs com margem adaptativa (1,7-2,2× sobre déficit, baseada em taxa de sucesso observada por decil). `baixar_random_pool_v2.py` v2 baixou com check de magic bytes JPEG ativo, classificando 2.665 respostas como `invalid_content` (HTMLs disfarçados que NÃO foram salvos no disco). Resultado final em disco:

| Split | Total | Alvo dataset_25k | Folga |
|---|---|---|---|
| train | 22.899 | 22.064 | +835 (+3,8%) |
| val | 9.515 | 9.147 | +368 (+4,0%) |
| test | 7.214 | 6.898 | +316 (+4,6%) |
| **Total** | **39.628** | **38.109** | **+1.519 (+4,0%)** |

**100% dos 39.628 arquivos passaram em validação de magic bytes.** Zero corrupção. Folga de ~4% por split permite downsample para igualar exatamente os tamanhos do dataset_25k mantendo estratificação por decis (a fazer no futuro próximo, quando o dataset_25k estiver reconstruído sem contaminação — ver A12).

#### Sub-task A11d. ✅ Upload para Drive (CONCLUÍDA)

**Status:** ✅ executada e validada
**Quando:** 15/04/2026
**Tempo gasto:** upload manual + 1 min de verificação

Sincronização `~/PROJETO_MARINHA/random_pool_v2/{train,val,test}/images/` (máquina local) → `/content/drive/MyDrive/InaTechShips/random_pool_v2/{train,val,test}/images/` (Drive). Verificação cruzada via snippet Python no Colab + `du -sh` na máquina local: contagens batem exatamente (22.899/9.515/7.214), tamanhos batem (~10,87/4,61/3,35 GB no Drive), amostra de 100 arquivos por split valida magic bytes JPEG (100/100 em todos). Random_pool_v2 agora disponível no Drive para uso pelo Colab no braço B do experimento principal (quando os labels estiverem prontos via Eduardo ou Plano B).

---

### A12. Reconstrução do dataset_25k (corrigindo contaminação entre splits)

**Status:** ✅ **CONCLUÍDA (16/04/2026)**
**Tempo total:** ~1h 30min (reconstrução + recálculo decis + downsample, tudo no Drive)
**Dependências:** A11 concluída

Tarefa criada após descoberta crítica durante a preparação do random_pool_v2: o `dataset_25k` original tem apenas 27.796 IDs únicos mas 38.109 arquivos distribuídos em train/val/test, com 10.313 duplicatas entre splits (37,1% do dataset). Investigação via hashes MD5 confirmou que arquivos duplicados são idênticos em conteúdo. Causa raiz identificada no notebook `InaTechShips.ipynb` (Seção 11): falta de limpeza de `DATASET_PATH` entre execuções múltiplas, combinada com `random.shuffle` sobre listas de pool diferentes entre runs.

**Impacto se não tratado:** modelo treinado no dataset_25k original veria imagens de val/test durante o treino, inflando métricas de validação. Invalidação do uso direto do dataset_25k no braço A do experimento.

**Não afeta os baselines existentes** (B1, B2, HPO) porque estes foram treinados em CITRA-3D-Real, não em dataset_25k.

**Resultado final do dataset_25k_v2:** 27.796 IDs únicos em splits **disjuntos** (train=16.677, val=5.558, test=5.561, todas as interseções = 0). Split 60/20/20 estratificado por classe, seed 42. Arquivos em `/content/drive/MyDrive/InaTechShips/dataset_25k_v2/`.

Documentado em detalhe na Seção 4.3.3 do `experimento_descricao_v02.md`.

#### Sub-task A12a. ✅ Reconstrução do dataset_25k → dataset_25k_v2

**Status:** ✅ executada no Drive via script `reconstruir_dataset_25k.py`
**Quando:** 16/04/2026
**Tempo gasto:** ~45 min (cópia de 83.388 arquivos no Drive: imagens + labels + labels_single_class)

Coletou os 27.796 IDs únicos do dataset_25k original, associou cada ID ao seu `class_id` (10 classes), fez split 60/20/20 estratificado por classe com seed 42, e copiou a estrutura completa para `dataset_25k_v2/{train,val,test}/{images,labels,labels_single_class}/`. Gerou automaticamente `data.yaml` (10 classes) e `data_single_class.yaml` (1 classe) apontando para a nova estrutura. Relatório completo em `reconstrucao_report.json`. **Zero arquivos faltantes nos 3 splits.**

#### Sub-task A12b. ✅ Recálculo dos decis empíricos para o dataset_25k_v2

**Status:** ✅ executada via `recalcular_distribuicao_decis_v2.py`
**Quando:** 16/04/2026
**Tempo gasto:** ~30 segundos

Calculou 10 bins empíricos sobre os 27.796 IDs únicos do dataset_25k_v2. Distribuição entre decis muito uniforme (2.779-2.780 IDs por decil, variação <0,05%). Proporção por split dentro de cada decil ficou aproximadamente 60/20/20 (estratificação inicial foi por classe, não por decil, mas classes se distribuem homogeneamente pelos IDs). Arquivo `distribuicao_decis_v2.json` salvo no Drive, pronto para uso no downsample.

#### Sub-task A12c. ✅ Downsample do random_pool_v2 para alinhamento

**Status:** ✅ executada via `downsample_random_pool_v2.py`
**Quando:** 16/04/2026
**Tempo gasto:** ~20 min no Drive

Reduziu o random_pool_v2 de 39.628 → 27.964 imagens em `images/` (alvo 27.796, excesso residual de 168 arquivos = 0,60%). Estratificação preservada por decis empíricos do dataset_25k_v2. Excedentes (11.664 imagens) movidos para `_excedente/{split}/` no Drive. Déficit residual detectado em uma única célula (test × decil 8: 544/568 = 0,086% do total) e aceito como metodologicamente irrelevante. Relatório em `downsample_report.json`.

**Pipeline de dados agora 100% pronto** para uso nos braços A, B e C do experimento principal.

---

## Trilha B — Bloqueado pela resposta do Eduardo (escopo reduzido)

### B1. ✅ Definição do método de anotação do `random_pool_v2`

**Status:** ✅ **CONCLUÍDA (24/04/2026)**
**Dependências:** resposta do Eduardo (recebida)

Eduardo Teixeira (INATEL) respondeu ao email (15/04/2026), aceitou colaboração ativa e coautoria. Contatado via WhatsApp em 24/04/2026, compartilhou 8 pastas RAR de labels (~2,8M labels em formato YOLO, classe 0) via Google Drive. Método definido: usar labels do PointRend do Eduardo (cenário 1 — mais rápido, simetria metodológica preservada).

---

### B2. ✅ Anotação do `random_pool_v2`

**Status:** ✅ **CONCLUÍDA (24/04/2026)**
**Script:** `filtrar_labels_eduardo.py` (383 linhas)
**Tempo:** ~30 min (extração RARs + filtragem + cópia)

Extraiu 2.794.460 labels das 8 pastas RAR do Eduardo para disco local do Colab (extração no Drive dava timeout — solução: extrair no SSD local). Cruzou com os 39.628 IDs do random_pool_v2: **36.272 encontrados (91,5%)**, 3.356 faltantes (8,5% — PointRend não detectou embarcação nessas imagens). Copiou 25.591 labels para `random_pool_v2/{train,val,test}/labels/` (train=15.366, val=5.115, test=5.110, excedente=10.681). Zero erros de cópia. Imagens sem label ignoradas automaticamente pelo YOLO.

---

### B3. ✅ Auditoria final do `random_pool_v2`

**Status:** ✅ **CONCLUÍDA (24/04/2026)** — implícita
**Tempo:** <1 min

Labels do Eduardo são do mesmo método PointRend usado no dataset_25k original — simetria metodológica preservada. Formato YOLO txt confirmado, classe 0 (single-class). Cobertura de 91,5% consistente com limiar de detecção do PointRend. `data_single_class.yaml` gerado para o random_pool_v2.

---

### B4. Treinar braço A (curado)

**Status:** ✅ **CONCLUÍDA (22/04/2026)**
**Tempo total:** ~16h (3 seeds × ~5h, com 1 reinicialização do Colab)
**Dependências:** A4, A7, A12

**Protocolo:** COCO (`yolo11m.pt`) → pré-treino dataset_25k_v2 (100 épocas, patience 20) → fine-tuning CITRA-3D-Real (300 épocas, patience 30). 3 seeds (42, 123, 2024). Hyperparams idênticos a B2 (validados pelo HPO). Script: `treinar_braco_a.py`.

**Resultado: NEGATIVO (surpreendente).** mAP50 = 0,7936 ± 0,0060 — **4,15% abaixo de B2** (COCO puro) e **0,72% abaixo de B1** (random init). O pré-treino curado no InaTechShips piorou em vez de melhorar. Interpretação: catastrophic forgetting — o pré-treino intermediário sobrescreveu features úteis do COCO sem substituí-las por features que transferem para o cenário operacional CITRA-3D-Real.

**Resultado publicável:** demonstra empiricamente que similaridade visual (CLIP) não garante transferência positiva. Gap de domínio (condições de captura, densidade de objetos, distribuição de classes) é mais relevante que aparência visual.

**Documentado em:** Seção 7.7.8 do `experimento_descricao_v02.md`.

---

### B4.1. Ablation de épocas de pré-treino (diagnóstico de catastrophic forgetting)

**Status:** ✅ **CONCLUÍDA (22/04/2026)**
**Tempo total:** ~2h (3 variantes × ~40 min, seed 42)
**Dependências:** B4

**Motivação:** investigar se o resultado negativo do braço A é causado por excesso de treino (catastrophic forgetting recuperável com menos épocas) ou incompatibilidade fundamental dos dados.

**Protocolo:** mesma configuração de B4, variando apenas épocas de pré-treino: 10, 20, 50. Seed 42 (diagnóstico). Script: `ablation_epocas_pretreino.py`.

**Resultado:**

| Épocas | mAP50 | Δ vs B2 |
|---|---|---|
| 0 (B2) | 0,8351 | ref |
| 10 | 0,8200 | −0,0151 |
| 20 | 0,8171 | −0,0180 |
| 50 | 0,8037 | −0,0314 |
| 100 | 0,8006 | −0,0345 |

**Diagnóstico:** degradação monotônica — mesmo 10 épocas perde 1,5%. Catastrophic forgetting confirmado com incompatibilidade parcial subjacente. Não existe "sweet spot" de pré-treino.

**Documentado em:** Seção 7.7.9 do `experimento_descricao_v02.md`.

---

### B5. ✅ Treinar braço B diagnóstico (aleatório, 1 seed)

**Status:** ✅ **CONCLUÍDA (24/04/2026)**
**Tempo:** ~2h (1 seed: pré-treino 100ep + fine-tuning 300ep)
**Dependências:** B3 (concluída)

**Resultado:** mAP50 = 0,7997 (seed 42). Equivalente ao braço A (0,7936 ± 0,0060 — diferença de 0,61%, dentro de 1σ). **Confirma que negative transfer é independente da curadoria CLIP.** O problema é incompatibilidade estrutural de domínio (escala, densidade, contexto), não seleção de imagens.

Documentado em Seção 7.7.11 do `experimento_descricao_v04.md`.

---

### B6. Treinar braço C opcional (InaTechShips completo)

**Status:** 🔴 Decisão pendente após resultados de A e B
**Tempo estimado:** muito alto (~24–48h)

---

## Trilha D — Scale-Aware Copy-Paste (adaptação de domínio)

**Contexto:** braço A original (pré-treino direto no InaTechShips) causou negative transfer (mAP50 −4,15% vs B2). Ablation de épocas confirmou degradação monotônica, diagnosticando catastrophic forgetting com incompatibilidade parcial de domínio. A principal diferença é de **escala**: navios ocupam ~80% da imagem no InaTechShips vs ~1-10% no CITRA-3D-Real (71,6% dos bboxes são "small" no padrão COCO).

**Estratégia:** em vez de usar as imagens InaTechShips diretamente, recortar os navios, redimensionar para a escala do CITRA-3D-Real, e compor em fundos oceânicos reais extraídos do CITRA-3D. Resultado: dataset sintético que preserva a variedade de navios do InaTechShips mas com o perfil de escala, densidade e contexto do CITRA-3D-Real. Bounding boxes gerados automaticamente (100% precisos).

**Fundamentação:** POSEIDON (Ruiz-Ponce et al., Sensors 2023), S3Det / Feedback Cut&Paste (Li et al., ACCV 2024), Nemati (2025). Abordagem preferida sobre CycleGAN (não resolve gap de escala) e modelos de difusão (bboxes imprecisos, custo computacional desproporcional).

### D1. ✅ Análise de escala do CITRA-3D-Real

**Status:** ✅ **CONCLUÍDA (23/04/2026)**
**Script:** `analisar_escala_citra3d.py`
**Tempo:** <1 min

Extraiu perfil completo: 7.003 bboxes em 2.081 imagens. Resultado-chave: mediana da largura = 3,1% da imagem (~20 px em 640×640), 71,6% são small (<32²px COCO-style), mediana de 2 objetos/imagem (P90=7), y_center concentrado na metade inferior (0,37-0,70). Salvou `copy_paste_recommendations` em JSON com ranges P10-P90 para calibrar a composição.

---

### D2. ✅ Extração de crops com SAM

**Status:** ✅ **CONCLUÍDA (23/04/2026)**
**Script:** `extrair_crops_sam.py` (553 linhas)
**Tempo total:** ~16h (3 sessões com retomada via --resume)
**Dependências:** D1

Extraiu 27.795 crops RGBA do dataset_25k_v2 usando SAM ViT-B com bbox como prompt. Modo SAM selecionado sobre modo bbox após teste piloto (100 imagens) que mostrou que crop retangular inclui fundo indesejado (porto/cais). Cobertura mediana da máscara SAM: 46,2% (vs 97,7% do bbox — confirmando que SAM segue contorno do navio).

**Filtro de qualidade definido:** coverage 25-95%, dimensão ≥ 50px, AR 0,2-8,0. Resultado: **23.828 crops usáveis (85,7%)**, 3.967 rejeitados (14,3%). Filtro será aplicado automaticamente no passo D4 (composição).

Metadados completos em `crops_metadata_full.json`.

---

### D3. ✅ Extração de fundos do CITRA-3D-Real

**Status:** ✅ **CONCLUÍDA (24/04/2026)**
**Script:** `extrair_fundos_citra3d.py` (382 linhas)
**Tempo:** 35,9 min
**Dependências:** D1

Extraiu 2.081 fundos oceânicos do CITRA-3D-Real. Duas estratégias combinadas: 1.988 fundos "clean" (bboxes cobrindo <5% da imagem — objetos tão pequenos que funcionam como ruído natural) + 93 fundos com inpainting (cv2.INPAINT_TELEA para remover navios maiores). Cobertura de 100% das imagens do CITRA-3D.

---

### D4. ✅ Composição de imagens sintéticas

**Status:** ✅ **CONCLUÍDA (25/04/2026)**
**Script:** `gerar_dataset_copypaste.py` (447 linhas, v3 — substituição in-place)
**Tempo:** 370,8 min para 27.796 imagens
**Dependências:** D2, D3

**Três iterações até a abordagem correta:**
- v1 (aleatório): ~30% navios em montanhas/prédios → descartada
- v2 (water-aware HSV): ~15% erros → descartada
- **v3 (substituição in-place, ideia da Daniela):** substitui navios reais do CITRA-3D por crops na mesma posição/dimensão → 100% correto → aprovada

**Resultado:** 27.796 imagens, 93.480 objetos (3,36 obj/img — idêntico ao CITRA-3D real), zero falhas. Split: train=16.678, val=5.559, test=5.559. `composicao_report.json` gerado.

---

### D4.5. ✅ Validação das imagens sintéticas

**Status:** ✅ **IMPLICITAMENTE CONCLUÍDA**

A validação formal via FID tornou-se desnecessária com a abordagem v3 (substituição in-place), porque: (1) os fundos são 100% reais do CITRA-3D (sem geração), (2) os bboxes são idênticos aos originais (sem posicionamento aleatório), (3) a distribuição de objetos por imagem é idêntica por construção (3,36 vs 3,37 obj/img). A validação visual do preview (100 imagens, 100% correto) e a confirmação numérica do `composicao_report.json` são suficientes.

---

### D5. ✅ Treino braço A' (COCO → sintéticas → CITRA-3D-Real)

**Status:** ✅ **CONCLUÍDA (25/04/2026) — RESULTADO PRINCIPAL DO PAPER**
**Tempo total:** ~6h (3 seeds × ~2h: pré-treino + fine-tuning)
**Dependências:** D4

**Resultado: mAP50 = 0,8541 ± 0,0043 — superou B2 (COCO puro) em +1,90%.**

Separação estatística completa: intervalos [0,8498; 0,8584] vs [0,8327; 0,8375] não se sobrepõem. mAP50-95 = 0,5281 ± 0,0056 (+4,5% relativo vs B2). A composição sintética in-place transformou um deficit de −4,15% (braço A direto) em ganho de +1,90%.

Documentado em Seção 7.7.12 do `experimento_descricao_v04.md`.

---

## Trilha C — Escrita e organização (contínua)

### C1. Manter o documento vivo atualizado

**Status:** 🟢 Contínuo
**Tempo estimado:** 15–30 min por dia de trabalho

**Já feito hoje (10/04/2026):** atualização para v0.2 com toda a fundação do CITRA-3D-Real consolidada.

---

### C2. Rascunho preliminar do paper

**Status:** 🟢 **Pode começar agora pelas seções de Métodos**
**Tempo estimado:** 4–6 horas para Métodos
**Dependências:** A0 (concluída) — agora você tem números autoritativos

**O que fazer:**
1. Criar `paper_rascunho.md`.
2. Começar pela seção de **Métodos**, copiando e adaptando das Seções 3, 4, 5 e 6 do `experimento_descricao_v02.md`.
3. Rascunhar a **Introdução** com base na Seção 2.
4. Rascunhar a seção de **Dados** com base na Seção 4 (CITRA-3D-Real, InaTechShips, dataset_25k, random_pool_v2).
5. Deixar Resultados, Discussion e Conclusão em branco até ter números.

**Por que começar agora vale especialmente a pena:** o documento vivo já tem material praticamente pronto para ser convertido em texto de paper. Os 6 scripts do pipeline de preparação viram **uma única seção de Pré-processamento de dados** muito sólida e defensável.

---

### C3. Definição do venue alvo

**Status:** 🟡 Pendente decisão
**Tempo estimado:** 2–3 horas de pesquisa

**O que fazer:** pesquisar venues compatíveis com contribuição metodológica + ablation empírico em domínio específico.

| Venue | Tipo | Aderência | Notas |
|---|---|---|---|
| Ocean Engineering | Revista | Alta | Mesma do paper InaTechShips. Comunidade alvo certa. |
| IEEE J-STARS | Revista | Alta | Remote sensing de larga escala. |
| Sensors (MDPI) | Revista | Média | Open access, publicação rápida. |
| WACV | Conferência | Média-alta | Aceita papers metodológicos com aplicação. |
| BMVC | Conferência | Média | Mais teórico. |
| SIBGRAPI | Conferência | Alta | Brasileira, comunidade próxima. |

**Entregável:** decisão com orientador.

---

### C4. Validação institucional com a Marinha

**Status:** 🟡 Pendente
**Tempo estimado:** depende dos canais
**Dependências:** nenhuma

**O que validar com a contraparte:**
- Nome "CITRA-3D" pode ser mencionado?
- Número de imagens (2.081) pode ser mencionado?
- Taxonomia em português pode ser mencionada?
- Localização geográfica de captura pode ser mencionada?
- Especificação técnica do sensor pode ser mencionada?
- Imagens de exemplo podem ser publicadas?
- O paper precisa de revisão prévia da instituição antes de submeter?

**Por que agora:** processo institucional pode ser lento.

---

## Cronograma proposto (próximas 4 semanas)

### Semana 1 (atual — 10/04 em diante)
- **Início:** A4 (`data.yaml` classe-única) — próximo passo imediato.
- **Em paralelo:** começar A11 (download das imagens do random_pool_v2) localmente.
- **Meio:** A5 (treino B1) e A6 (treino B2) — primeiros baselines.
- **Final:** A8 (validação cruzada de auto-anotação).
- **Contínuo:** C1 (manter documento vivo).

### Semana 2
- **Início:** A7 (HPO em B2), A9 (caracterização visual do gap).
- **Meio:** retomar B1 e B2 com configuração ótima do HPO.
- **Final:** começar B4 (braço A do experimento — pré-treino no dataset_25k + fine-tuning).
- **Contínuo:** A10 (revisão de literatura), começar C2 (rascunho de Métodos).

### Semana 3
- **Início:** finalizar B4 (braço A com 3 seeds).
- **Meio:** se Eduardo respondeu, executar B1, B2, B3 da Trilha B. Se não, ativar Plano B com YOLOv11x COCO (validado em A8).
- **Final:** finalizar download de A11 (random_pool_v2).
- **Contínuo:** C2, C3 (venue), C4 (validação Marinha).

### Semana 4
- **Início:** treinar braço B (aleatório) — independente de qual plano de anotação venceu.
- **Meio:** análise comparativa A vs B, gerar tabelas, gerar figuras.
- **Final:** rascunho de Resultados e Discussion no paper.

---

## Tarefas que você pode começar **hoje**

Se você tem 1–2 horas disponíveis:

1. **A4 — Geração dos `data.yaml` classe-única.** Curto, é o gargalo dos próximos passos. (~30 min)
2. **A11 — Iniciar download do random_pool_v2** localmente em background. (~10 min para configurar, depois roda sozinho)

Se você tem 1 dia inteiro:

3. **A5 e A6 — Treinos baseline B1 e B2.** ~6h de execução total, em background na A100.
4. **A8 — Validação cruzada de auto-anotação.** Boa para fazer enquanto os baselines treinam.

Se você tem uma semana de foco:

5. **A7 — HPO no B2.**
6. **A9 — Caracterização visual do gap.**
7. **B4 — Treino do braço A.** Metade do experimento principal liberada.

---

## Riscos do cronograma

**Risco 1:** o tempo de treino do braço A pode ser maior que estimado. **Mitigação:** começar logo após HPO definido.

**Risco 2:** o HPO pode revelar configuração muito diferente, exigindo refazer baselines. **Mitigação:** ordem A6 → A7 → reexecutar B1/B2.

**Risco 3:** Eduardo responder e os labels exigirem reformatação. **Mitigação:** validar formato YOLO antes de assumir prontidão.

**Risco 4:** download bloqueado pelo Cloudflare. **Mitigação:** já planejado no `download_direto.py`.

**Risco 5:** Marinha pedir mudanças no que pode ser publicado. **Mitigação:** fazer C4 cedo.

**Risco 6 (novo):** Plano B (YOLOv11x COCO) pode ter concordância muito baixa com PointRend, invalidando essa rota. **Mitigação:** A8 vai medir isso empiricamente antes de qualquer compromisso. Se a concordância for baixa, ainda há tempo para investigar alternativas (Grounding DINO, modelos do Roboflow, etc.).

---

## Conclusão estratégica

A versão 0.2 deste cronograma reflete uma mudança qualitativa importante: **o pipeline de preparação dos dados está consolidado**. A próxima fase é gerar os `data.yaml`, começar os primeiros treinos baseline, e em paralelo iniciar o download das imagens do random_pool_v2 — tudo isso sem depender da resposta do Eduardo.

A recomendação concreta para hoje: **comece pela A4** (geração dos `data.yaml`), que é a tarefa mais curta e desbloqueia A5 e A6 (primeiros treinos). Em paralelo, configure a A11 (download local) para começar a rodar em background. Isso aproveita os dois canais (Colab + máquina local) simultaneamente.

A próxima conversa importante deveria ser sobre **A4** — me confirma que quer prosseguir e eu escrevo o script no próximo turno.
