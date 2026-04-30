# Visual Similarity Is Not Enough: Domain-Adapted Synthetic Data for Maritime Vessel Detection

**Autora:** Daniela L. Freire (ICMC/USP)
**Colaborador:** Eduardo H. Teixeira (INATEL)
**Supervisor:** Leandro Aparecido Simal Moreira
**Contexto:** Projeto CASNAV/DMarSup, Marinha do Brasil (Termo 66/2025)

---

## 1. Problema

Sistemas de vigilância marítima dependem de detecção automática de embarcações em imagens capturadas por câmeras operacionais. Treinar detectores robustos exige grandes volumes de dados anotados, mas datasets operacionais são tipicamente pequenos e sigilosos. O dataset utilizado neste trabalho (CITRA-3D-Real) contém apenas 2.081 imagens com 7.003 bounding boxes.

Simultaneamente, existem datasets públicos de embarcações com dezenas de milhares de imagens. O InaTechShips (Teixeira et al., 2025) disponibiliza ~28 mil imagens do shipspotting.com com anotações automáticas via PointRend.

## 2. Pergunta de pesquisa

**Como utilizar efetivamente imagens públicas de embarcações para melhorar a detecção em cenário operacional marítimo com dados limitados?**

## 3. Resultados finais

| Rank | Estratégia | Seeds | mAP50 | mAP50-95 | P | R | F1 | Δ vs B2 |
|---|---|---|---|---|---|---|---|---|
| **1** | **Joint balanced (50/50)** | **3** | **0.8451 ± 0.0033** | **0.5206 ± 0.0017** | **0.857** | **0.805** | **0.830** | **+1.00 pp** |
| 2 | B2 (COCO) | 3 | 0.8351 ± 0.0020 | 0.5055 ± 0.0022 | 0.857 | 0.783 | 0.818 | ref |
| 3 | Frozen backbone (100ep) | 3 | 0.8342 ± 0.0039 | 0.5074 ± 0.0035 | 0.855 | 0.774 | 0.812 | −0.09 pp |
| 3 | Synthetic 20ep | 3 | 0.8344 ± 0.0033 | 0.5011 ± 0.0065 | 0.855 | 0.773 | 0.812 | −0.08 pp |
| 5 | A' v4 seq (100ep) | 3 | 0.8221 ± 0.0085 | 0.4933 ± 0.0059 | 0.828 | 0.769 | 0.797 | −1.31 pp |
| 6 | B1 (random init) | 3 | 0.8008 ± 0.0061 | 0.4742 ± 0.0006 | 0.829 | 0.750 | 0.787 | −3.43 pp |
| 7 | B (random direct) | 3 | 0.7945 ± 0.0046 | 0.4728 ± 0.0045 | 0.858 | 0.742 | 0.796 | −4.06 pp |
| 8 | A (curated direct) | 3 | 0.7936 ± 0.0049 | 0.4692 ± 0.0017 | 0.834 | 0.735 | 0.781 | −4.15 pp |

## 4. Narrativa

1. **Tentativa direta falhou:** pré-treino no InaTechShips (curado por CLIP ou aleatório) causa negative transfer de −4.15 pp. O gap é estrutural (escala 80% vs 3%, densidade 1 vs 3.37, contexto profissional vs operacional).

2. **Diagnóstico confirmou catastrophic forgetting:** ablation de épocas mostra degradação monotônica. Curado ≈ aleatório (A ≈ B) confirma que o problema é o domínio, não a curadoria.

3. **Composição in-place reduz mas não resolve em regime sequencial:** sintéticas adaptadas ao domínio reduzem o gap de −4.15 para −1.31 pp, mas pré-treino sequencial ainda causa forgetting.

4. **O regime de treino importa tanto quanto os dados:** frozen backbone (−0.09 pp) e 20 épocas (−0.08 pp) neutralizam o forgetting (≈ B2). Joint balanced (+1.00 pp) é o único que **supera** B2 — o modelo vê reais e sintéticas intercalados em cada batch, sem esquecer nenhum.

5. **Recall é o maior ganho operacional:** de 0.783 (B2) para 0.805 (joint balanced) = +2.8%. O modelo detecta mais navios sem aumentar falsos positivos. Em vigilância marítima, falso negativo é mais custoso que falso positivo.

## 5. Contribuições

1. **Evidência empírica de negative transfer em detecção marítima** com ablation quantificada e comparação curado vs aleatório (3 seeds cada).

2. **Método de adaptação de domínio por composição in-place** que transforma imagens públicas em dados de treino compatíveis com o cenário operacional.

3. **Demonstração de que o regime de treino importa tanto quanto os dados sintéticos:** sequential causa forgetting, frozen neutraliza, joint balanced supera o baseline.

4. **Demonstração de que CLIP similarity é proxy insuficiente para transferibilidade** em detecção de objetos.
