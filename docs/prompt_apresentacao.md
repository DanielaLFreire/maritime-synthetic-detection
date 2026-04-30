# Prompt para Apresentação — Experimento de Detecção Marítima

Use este prompt em qualquer ferramenta de IA (Claude, ChatGPT, Gamma, etc.) para gerar uma apresentação. Copie tudo abaixo do separador.

---

## PROMPT

Crie uma apresentação de slides clara e didática para uma reunião com meu supervisor de pesquisa. O objetivo é que ele entenda o experimento completo: o problema, o que tentamos, por que falhou, o que propusemos, e o resultado final. Use linguagem acessível, evite jargão desnecessário, e conte como uma história com começo, meio e fim.

**Contexto do projeto:**
- Pesquisadora: Daniela L. Freire (ICMC/USP)
- Supervisor: Leandro Aparecido Simal Moreira
- Projeto: CASNAV/DMarSup (Termo 66/2025), Marinha do Brasil
- Colaborador: Eduardo H. Teixeira (INATEL, autor do dataset InaTechShips)
- Detector: YOLOv11m, detecção single-class ("embarcação")
- Artigo alvo: Ocean Engineering (Elsevier)
- Título: "Visual Similarity Is Not Enough: Domain-Adapted Synthetic Data for Maritime Vessel Detection"

**Estrutura da apresentação (15-20 slides):**

### SLIDE 1 — Capa
Título: "Visual Similarity Is Not Enough: Domain-Adapted Synthetic Data for Maritime Vessel Detection"
Autores: Daniela L. Freire, Eduardo H. Teixeira, Leandro A. S. Moreira
Instituições: ICMC/USP, INATEL

### SLIDE 2 — O Problema
- Precisamos detectar embarcações em imagens de câmeras de vigilância marítima
- Temos poucos dados anotados: CITRA-3D-Real tem apenas 2.081 imagens
- Os navios aparecem como objetos pequenos e distantes (71,6% são "small" pelo padrão COCO)
- Média de 3,37 navios por imagem, em condições climáticas variadas
- Pergunta: como melhorar a detecção com tão poucos dados?

### SLIDE 3 — A Ideia Inicial
- Existe um dataset público com ~28 mil imagens de navios (InaTechShips, shipspotting.com)
- Hipótese: pré-treinar o detector nessas imagens públicas deveria melhorar a detecção no cenário operacional
- Usamos CLIP para selecionar as imagens mais similares ao nosso dataset
- Parece intuitivo: mais imagens de navios = melhor detecção de navios

### SLIDE 4 — O Gap de Domínio (visual)
Mostrar lado a lado:
- Imagem InaTechShips: navio grande, close-up, ocupa ~80% da imagem, 1 navio, foto profissional
- Imagem CITRA-3D-Real: navios minúsculos, distantes, ocupam ~3% da imagem, múltiplos navios, câmera de vigilância
- São ambos "imagens de navios", mas são mundos completamente diferentes

### SLIDE 5 — Resultado Surpreendente: Piorou
Tabela simples:
| Estratégia | mAP50 | Resultado |
|---|---|---|
| Sem pré-treino (random init) | 0.8008 | Baseline fraco |
| COCO genérico (B2) | 0.8351 | Baseline forte |
| InaTechShips curado por CLIP (A) | 0.7936 | **Piorou −4,15%** |
| InaTechShips aleatório (B) | 0.7945 | **Piorou −4,06%** |

Mensagem: pré-treinar em 28 mil imagens de navios PIOROU a detecção. Pior que COCO genérico (que nem tem classe "navio"). Pior até que treinar do zero.

### SLIDE 6 — Por que Piorou? Diagnóstico
Três eixos de incompatibilidade:
1. **Escala:** navios ocupam 80% da imagem no InaTechShips vs 3% no CITRA-3D
2. **Densidade:** 1 navio/imagem vs 3,37 navios/imagem
3. **Contexto:** fotos profissionais de porto vs câmeras de vigilância em mar aberto

O modelo aprendeu a detectar navios grandes em fotos bonitas — e esqueceu como detectar objetos pequenos em cenas operacionais.

### SLIDE 7 — Ablation de Épocas: Catastrophic Forgetting
Gráfico mostrando degradação monotônica:
- 0 épocas (COCO puro): mAP50 = 0.8351
- 10 épocas: 0.8200 (já piorou)
- 20 épocas: 0.8171
- 50 épocas: 0.8037
- 100 épocas: 0.8006 (voltou ao nível do random init)

Mensagem: cada época de pré-treino no InaTechShips apaga um pouco mais o conhecimento do COCO. Não existe "dose ideal" — qualquer exposição prejudica.

### SLIDE 8 — CLIP Não Funciona Como Proxy
- Curado por CLIP (A) = 0.7936
- Aleatório (B) = 0.7945
- Diferença: 0.09 pp — praticamente zero
- Mensagem: CLIP mede aparência visual, mas não mede compatibilidade de domínio (escala, densidade, contexto). "Parecer similar" ≠ "ser útil para treino"

### SLIDE 9 — A Solução: Composição Sintética In-Place
Em vez de usar as imagens do InaTechShips diretamente, TRANSFORMAMOS elas:
1. Extraímos os navios com SAM (segmentação precisa, fundo transparente)
2. Redimensionamos para a escala do cenário operacional (~3% da imagem)
3. Posicionamos EXATAMENTE onde navios reais existiam nas cenas do CITRA-3D
4. Resultado: 27.053 imagens sintéticas com navios na escala certa, densidade certa, fundo real

### SLIDE 10 — Composição Visual (Figura)
Mostrar o 2×2:
- (a) Foto InaTechShips original (navio grande)
- (b) Crop SAM (navio isolado, fundo transparente)
- (c) Cena CITRA-3D real (navios distantes)
- (d) Cena sintética (navios do InaTechShips colados na posição certa)

### SLIDE 11 — Integridade dos Dados
- Cada split sintético gerado exclusivamente do split correspondente do CITRA-3D
- Nenhum fundo, anotação ou posição do test set usado durante treino
- Split sintético espelha o split real: train→train, val→val, test→test
- Avaliação final exclusivamente no test set real do CITRA-3D
- Garantia de que nenhum resultado está inflado por acesso indireto ao test set

### SLIDE 12 — Primeiro Resultado (v4, Sequencial): Melhorou Mas Não Superou
| Estratégia | mAP50 | Δ vs B2 |
|---|---|---|
| COCO puro (B2) | 0.8351 | ref |
| A' v4 sequencial (100ep) | 0.8221 | −1.31% |
| InaTechShips direto (A) | 0.7936 | −4.15% |

A composição reduziu o gap de −4.15 para −1.31 — mas o pré-treino sequencial ainda causa forgetting.

### SLIDE 13 — Insight: O Regime de Treino Importa
O problema não são (só) os dados — é COMO usamos eles.
Pré-treino sequencial: COCO → sintético → real
- O modelo "esquece" COCO durante a fase sintética
- Mesmo com dados adaptados, o forgetting persiste

Testamos 3 alternativas:
1. Frozen backbone: congela COCO durante pré-treino sintético
2. 20 épocas: dose mínima de pré-treino
3. Joint balanced: treina real + sintético JUNTOS (50/50)

### SLIDE 14 — Resultado Final: Joint Balanced Vence
| Estratégia | mAP50 | Δ vs B2 |
|---|---|---|
| **Joint balanced (50/50)** | **0.8451 ± 0.0033** | **+1.00 pp** |
| B2 (COCO puro) | 0.8351 ± 0.0020 | ref |
| Frozen backbone | 0.8342 ± 0.0039 | −0.09 pp |
| Synthetic 20ep | 0.8344 ± 0.0033 | −0.08 pp |
| A' sequencial 100ep | 0.8221 ± 0.0085 | −1.31 pp |

Joint balanced é o ÚNICO que supera B2 com separação estatística (intervalos não se sobrepõem).

### SLIDE 15 — Por que Joint Balanced Funciona
- O modelo vê imagens reais E sintéticas em cada batch
- Nunca "esquece" os dados reais — sem catastrophic forgetting
- As sintéticas adicionam diversidade de aparência (navios que o CITRA-3D não tem)
- Recall sobe de 0.783 → 0.805 (+2.8%): detecta mais navios
- Precision mantém 0.857: não aumenta falsos positivos
- F1 = 0.830 — o melhor de todos os braços

### SLIDE 16 — Gráfico de Barras (Figura Final)
Gráfico com todos os 7 braços ordenados por mAP50.
Joint balanced destacado em azul no topo.
B2 como linha de referência.

### SLIDE 17 — Resumo da Narrativa
1. ❌ Pré-treino direto em imagens similares → PIOROU (−4.15 pp)
2. 🔍 Diagnóstico: gap de domínio (escala, densidade, contexto) + catastrophic forgetting
3. 🔧 Composição in-place: adapta as imagens ao domínio operacional
4. ⚠️ Sequencial: melhora mas não resolve (−1.31 pp) — forgetting persiste
5. ✅ Joint balanced: treino conjunto 50/50 → SUPEROU o baseline (+1.00 pp)

Mensagem principal: **tanto os dados quanto o regime de treino importam.**

### SLIDE 18 — Contribuições para o Paper
1. Evidência empírica de negative transfer em detecção marítima (ablation + CLIP vs random)
2. Método de composição in-place (SAM + substituição posicional)
3. Demonstração de que regime de treino importa tanto quanto qualidade dos dados sintéticos
4. CLIP visual similarity ≠ transferibilidade para detecção

### SLIDE 19 — Status do Paper
- Artigo completo, 0 TODOs, ~10 páginas double-column
- Template Ocean Engineering (CAS, Elsevier)
- 4 figuras, 6 tabelas, 23 referências verificadas
- Repositório público: github.com/DanielaLFreire/maritime-synthetic-detection
- Cover letter e lista de reviewers prontos
- Próximo passo: revisão interna → submissão

### SLIDE 20 — Próximos Passos e Discussão
- Revisar paper internamente com o supervisor
- Possíveis extensões: multi-class, outros detectores (RT-DETR), otimizar proporção real/sintético
- Perguntas?

**Estilo visual:**
- Fundo limpo, branco ou azul escuro
- Fonte grande e legível (mínimo 18pt)
- Máximo 5 linhas de texto por slide — prefira imagens e tabelas
- Use setas e cores para guiar a leitura
- Destaque o resultado principal em cada slide
- Tabelas com no máximo 5 linhas
- Tom: profissional mas acessível, como explicando para um colega inteligente que não acompanhou o dia a dia
