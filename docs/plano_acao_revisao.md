# Plano de Ação — Revisão Pré-Submissão Ocean Engineering

**Paper:** Visual Similarity Is Not Enough: Domain-Adapted Synthetic Data for Maritime Vessel Detection
**Data:** 26/04/2026
**Baseado em:** Revisão simulada com 20 pontos

---

## Visão geral

| Prioridade | Itens | Tempo estimado | Tipo |
|---|---|---|---|
| P1 — Indispensável | 7 itens | ~3 dias | Experimentos + texto |
| P2 — Fortemente recomendada | 7 itens | ~2 dias | Experimentos + texto |
| P3 — Refinamento | 6 itens | ~1 dia | Texto |

---

## P1 — INDISPENSÁVEL (bloqueia submissão)

### P1.1 ✦ Corrigir data leakage na geração sintética
**Revisão §5 | Esforço: ~8h (geração) + ~6h (retreino)**

O script v3 gerou sintéticas de TODAS as 2,081 imagens e depois splitou 60/20/20.
Fundos do test set vazaram para o train sintético.

- [ ] Rodar `gerar_dataset_copypaste_v4.py` (splits isolados) — ~6h
- [ ] Retreinar braço A' com 3 seeds (42, 123, 2024) — ~6h
- [ ] Comparar resultados v3 vs v4
- [ ] Adicionar frase explícita na Seção 4.2.3:
  > "To prevent data leakage, synthetic images for each partition were
  > generated exclusively from the corresponding CITRA-3D-Real partition.
  > No background, annotation or object position from the validation or
  > test sets was used during pre-training or model selection."

**Status:** Script v4 pronto. Aguardando execução.

---

### P1.2 ✦ Reenquadrar como aplicação civil marítima
**Revisão §1 | Esforço: ~2h (texto)**

A Ocean Engineering exclui submissões focadas em aplicações militares.
Trocar "naval" por "maritime surveillance" em todo o texto.

- [ ] Abstract: trocar "naval scenarios" → "maritime surveillance scenarios"
- [ ] Introduction: enfatizar busca e salvamento, segurança portuária,
      monitoramento de tráfego, vigilância costeira, sistemas autônomos
- [ ] Manter "Brazilian Navy" apenas como fonte institucional do dataset
- [ ] Revisar todas as 20+ ocorrências de "naval" e "operational naval"
- [ ] Argumento central: "o problema é geral para sistemas reais de
      monitoramento marítimo, não exclusivo de defesa"

**Buscar/substituir sugerido:**
```
"operational naval scenarios" → "operational maritime surveillance"
"naval surveillance" → "maritime surveillance"
"naval scenarios" → "maritime monitoring scenarios"
```

---

### P1.3 ✦ Braço B (random) com 3 seeds
**Revisão §7.1 | Esforço: ~6h (Colab)**

Uma seed é insuficiente para afirmar A ≈ B. Rodar seeds 123 e 2024.

- [ ] Pré-treino seed 123 no random_pool_v2 (100 ep) + fine-tuning (300 ep)
- [ ] Pré-treino seed 2024 no random_pool_v2 (100 ep) + fine-tuning (300 ep)
- [ ] Atualizar Tabela 3 com média ± DP de 3 seeds
- [ ] Recalcular comparação A vs B com estatísticas adequadas

**Pode rodar em paralelo com P1.1.**

---

### P1.4 ✦ Suavizar afirmações estatísticas
**Revisão §7.2, §18 | Esforço: ~1h (texto)**

- [ ] Remover "complete statistical separation" → 
      "non-overlapping confidence intervals across the tested seeds"
- [ ] Remover "7.9σ" → apresentar intervalos de confiança 95%
- [ ] Remover "This paper demonstrates that this intuition is wrong" →
      "This paper shows that this intuition does not hold in the studied setting"
- [ ] Remover "perfectly annotated data by construction" →
      "inherits target-domain annotations by construction"
- [ ] Remover "eliminating arbitrary placement decisions" →
      "reducing the need for arbitrary placement decisions"
- [ ] Adicionar tabela com resultados por seed individual
- [ ] Considerar teste pareado (paired t-test ou Wilcoxon) entre A' e B2

---

### P1.5 ✦ Corrigir highlights (limite 85 caracteres)
**Revisão §4 | Esforço: ~15min**

Atuais: 127-141 caracteres. Máximo Elsevier: 85 caracteres com espaços.

Sugestão corrigida (contagem verificada):
```
1. Public ship pre-training causes negative transfer in maritime detection   [73]
2. In-place synthetic composition improves detection over COCO baseline      [72]
3. CLIP similarity fails to predict transfer for vessel detection            [65]
4. Scale, density and context gaps explain transfer failure                  [60]
5. SAM-based ship crops bridge public and operational maritime domains       [68]
```

---

### P1.6 ✦ Verificar permissão de imagens
**Revisão §14 | Esforço: ~1h**

A Figura 1(a) contém marca d'água do shipspotting.com (© Dieter Pots).

- [ ] Verificar licença da imagem no shipspotting.com
- [ ] Opção A: obter permissão escrita do fotógrafo
- [ ] Opção B: substituir por imagem com licença aberta (e.g., Wikimedia Commons)
- [ ] Opção C: usar imagem própria do InaTechShips sem marca d'água
- [ ] Verificar se imagens CITRA-3D-Real podem ser publicadas
      (confirmar com Leandro/CASNAV)

---

### P1.7 ✦ Declarações obrigatórias Elsevier
**Revisão §16 | Esforço: ~30min (texto)**

- [ ] CRediT authorship contribution statement (já tem no \author, 
      mas precisa de seção separada)
- [ ] Declaration of competing interest:
      "The authors declare that they have no known competing financial 
      interests or personal relationships that could have appeared to 
      influence the work reported in this paper."
- [ ] Funding statement:
      "This work was supported by the Brazilian Navy under the 
      CASNAV/DMarSup project (Termo 66/2025)."
- [ ] Renomear seção AI:
      "Declaration of generative AI and AI-assisted technologies in 
      the writing process"

---

## P2 — FORTEMENTE RECOMENDADA (fortalece muito o paper)

### P2.1 Métricas adicionais: AP small, Precision, Recall, curvas PR
**Revisão §9 | Esforço: ~2h (Colab + texto)**

O artigo enfatiza 71.6% small objects mas não reporta AP_small.

- [ ] Extrair do results.csv ou re-rodar val com verbose=True:
      Precision, Recall, AP_small, AP_medium, AP_large
- [ ] Para os braços principais (B2, A, A'): gerar curvas PR
- [ ] Adicionar tabela suplementar com métricas expandidas
- [ ] Reportar tempo de inferência (FPS) — já nos logs do Ultralytics

```python
# Extrair métricas detalhadas
model = YOLO("best.pt")
metrics = model.val(data="...", split="test", verbose=True)
print(f"P={metrics.box.p:.4f} R={metrics.box.r:.4f}")
# AP por tamanho: metrics.box.maps contém AP por classe
```

---

### P2.2 Baseline de copy-paste convencional
**Revisão §8.1 | Esforço: ~4h (geração + treino)**

Demonstra que o ganho vem do posicionamento in-place, não apenas de
ter mais dados sintéticos.

- [ ] Gerar dataset com crops posicionados ALEATORIAMENTE na zona de água
      (reusar v1 ou v2 do script de composição)
- [ ] Treinar com 1 seed (diagnóstico, como braço B)
- [ ] Comparar: A' (in-place) vs A'' (random placement) vs B2 (COCO)
- [ ] Se A' > A'' → confirma que posicionamento in-place é a contribuição

---

### P2.3 Ablação do número de variações
**Revisão §8.5 | Esforço: ~8h (treino)**

O número 13 variações parece arbitrário.

- [ ] Gerar subsets com 1, 3, 5, 10, 13 variações (usando v4)
- [ ] Treinar cada com 1 seed
- [ ] Plotar curva: n_variações vs mAP50
- [ ] Identificar ponto de saturação

---

### P2.4 Descrição detalhada do CITRA-3D-Real
**Revisão §12 | Esforço: ~1h (texto)**

- [ ] Resolução das imagens (e.g., 1920×1080)
- [ ] Tipo de câmera (se não for sensível): fixa, PTZ, etc.
- [ ] Ambiente: baía, costa, porto, mar aberto
- [ ] Período de captura (datas aproximadas)
- [ ] Condições climáticas representadas
- [ ] Critérios e ferramenta de anotação
- [ ] Número de anotadores e controle de qualidade
- [ ] Classes originais antes da fusão (listar as 9)
- [ ] Declarar restrições institucionais sobre detalhes não informados

---

### P2.5 Tabela metodológica de protocolo por braço
**Revisão §13 | Esforço: ~30min (texto)**

Criar tabela clara com:
| Arm | Init weights | Pre-train data | Pre-train epochs | Fine-tune data | Fine-tune epochs | Val set (early stop) | Test set | Seeds |

---

### P2.6 Nota sobre aspect ratio na composição
**Revisão §6.1 | Esforço: ~1h (análise + texto)**

- [ ] Calcular distribuição de aspect ratio dos crops vs bboxes alvo
- [ ] Se houver distorção > 2:1, documentar e discutir impacto
- [ ] Adicionar frase na Seção 4.2.2 sobre limitação de aspect ratio
- [ ] Considerar filtro de compatibilidade AR no script v4

---

### P2.7 Cautela na interpretação de catastrophic forgetting
**Revisão §10 | Esforço: ~30min (texto)**

- [ ] Trocar "can be attributed to catastrophic forgetting" →
      "is consistent with catastrophic forgetting or harmful 
      intermediate feature adaptation"
- [ ] Idealmente: avaliar modelo no COCO val antes e depois do 
      pré-treino em InaTechShips (mostra perda de features genéricas)
- [ ] Se viável: Grad-CAM em 2-3 imagens mostrando mudança de atenção

---

## P3 — REFINAMENTO (qualidade editorial)

### P3.1 Consistência de inglês
**Revisão §18 | Esforço: ~1h**

- [ ] Padronizar para British English (Ocean Engineering aceita ambos,
      mas deve ser consistente): behaviour, optimisation, modelling, etc.
- [ ] Ou padronizar American English e manter consistente

### P3.2 Suavizar tom ao longo do texto
**Revisão §18 | Esforço: ~1h**

- [ ] Revisar todas as frases categóricas (grep por "clearly", 
      "proves", "demonstrates", "shows that", "insufficient")
- [ ] Substituir por linguagem mais cautelosa: "suggests", "indicates",
      "is consistent with", "appears to"

### P3.3 Auditar referências
**Revisão §17 | Esforço: ~1h**

- [ ] Adicionar DOI em todas as referências que possuem
- [ ] Ultralytics YOLO: citar como software com versão e data de acesso
- [ ] Verificar se Nemati 2025 (arXiv) foi publicado em journal
- [ ] Dataset InaTechShips: citar também como data reference
- [ ] Adicionar URL completa + data de acesso em web references

### P3.4 Melhorar análise CLIP (se tempo permitir)
**Revisão §11 | Esforço: ~4h**

- [ ] Testar faixas de similaridade: top 5%, 20%, 50%, random
- [ ] Calcular correlação CLIP score médio vs mAP final
- [ ] Calcular FID/KID entre datasets
- [ ] Suavizar conclusão: "global CLIP similarity, when used alone, 
      may fail to capture structural properties"

### P3.5 Reestruturar seções (opcional)
**Revisão §15 | Esforço: ~2h**

Considerar estrutura revisada:
1. Introduction
2. Related work  
3. Datasets and domain-gap analysis
4. Proposed method: in-place synthetic composition
5. Experimental protocol
6. Results
7. Ablation studies
8. Discussion
9. Limitations and threats to validity
10. Conclusion

### P3.6 Conclusão mais cautelosa
**Revisão §19 | Esforço: ~30min**

- [ ] Trocar "general and can be applied to other domains" →
      "Although evaluated on a single maritime dataset, the results 
      suggest that structurally aligned synthetic composition may be 
      applicable to other domains with similar distributional mismatch"

---

## Cronograma sugerido

### Semana 1 (computação pesada)
| Dia | Tarefa | Tempo |
|---|---|---|
| Seg | P1.1: Rodar v4 (geração sintética corrigida) | ~6h GPU |
| Ter | P1.1: Retreinar A' (3 seeds) | ~6h GPU |
| Ter | P1.3: Braço B seed 123 (paralelo se 2 sessões) | ~2h GPU |
| Qua | P1.3: Braço B seed 2024 | ~2h GPU |
| Qua | P2.1: Extrair métricas (P, R, AP_small, PR curves) | ~2h |
| Qui | P2.2: Baseline copy-paste random (1 seed) | ~4h GPU |
| Sex | P2.3: Ablação variações (1,3,5,10,13) × 1 seed | ~8h GPU |

### Semana 2 (escrita e refinamento)
| Dia | Tarefa | Tempo |
|---|---|---|
| Seg | P1.2: Reenquadrar civil + P1.4: suavizar afirmações | ~3h |
| Ter | P1.5: highlights + P1.7: declarações Elsevier | ~1h |
| Ter | P2.4: descrição CITRA-3D + P2.5: tabela protocolo | ~2h |
| Qua | P2.6: aspect ratio + P2.7: cautela forgetting | ~2h |
| Qui | P3.1-P3.3: inglês + tom + referências | ~3h |
| Sex | P1.6: permissões imagens + revisão final | ~2h |
| Sex | Enviar para Leandro revisar | — |

### Semana 3
| Dia | Tarefa |
|---|---|
| Seg-Qua | Incorporar feedback do Leandro |
| Qui | Revisão final de formatação |
| Sex | **Submissão** |

---

## Checklist de submissão Ocean Engineering

- [ ] Cover letter
- [ ] Highlights (arquivo separado, ≤85 chars/item)
- [ ] Manuscript (double-column, CAS template)
- [ ] Figures (arquivos separados, alta resolução)
- [ ] Tables (dentro do manuscrito)
- [ ] References (author-year, cas-model2-names.bst)
- [ ] CRediT author statement
- [ ] Declaration of competing interest
- [ ] Data availability statement
- [ ] Declaration of AI use
- [ ] Funding statement
- [ ] Suggested reviewers (3-5 nomes com email)
- [ ] Excluded reviewers (opcional)
- [ ] Graphical abstract (opcional mas recomendado)
