# Prompt para Apresentação — Experimento de Detecção Marítima

Use este prompt em qualquer ferramenta de IA (Claude, ChatGPT, Gamma, etc.) para gerar uma apresentação. Copie tudo abaixo do separador.

---

## PROMPT

Crie uma apresentação de slides clara e didática para uma reunião com meu supervisor de pesquisa. O foco é mostrar como o experimento contribui diretamente para melhorar o sistema de detecção de embarcações que estamos desenvolvendo para vigilância marítima. O supervisor quer entender: qual era o problema prático, o que tentamos, o que aprendemos, e como isso melhora o sistema real. A publicação do artigo é uma consequência natural para validar a proposta com a comunidade científica e receber feedback de especialistas.

Conte a história como uma investigação prática: tínhamos um problema real (poucos dados), exploramos uma solução (dados públicos), descobrimos que não funcionava como esperávamos, entendemos por quê, e encontramos o caminho que funciona.

Use as figuras disponíveis nos slides indicados (referência aos PDFs do repositório).

**Contexto do projeto:**
- Pesquisadora: Daniela L. Freire (ICMC/USP)
- Supervisor: Leandro Aparecido Simal Moreira
- Projeto: CASNAV/DMarSup (Termo 66/2025)
- Colaborador: Eduardo H. Teixeira (INATEL, autor do InaTechShips)
- Detector: YOLOv11m, detecção single-class ("embarcação")
- O sistema de detecção faz parte de uma solução maior de vigilância marítima

**Estrutura da apresentação (15-18 slides):**

### SLIDE 1 — Capa
Título: "Como usar imagens públicas de navios para melhorar nosso sistema de detecção"
Subtítulo: Composição sintética adaptada ao domínio operacional
Autores e instituições

### SLIDE 2 — Nosso Desafio
- Estamos desenvolvendo um sistema de detecção de embarcações para vigilância marítima
- O detector precisa funcionar com câmeras fixas em cenário real: navios distantes, pequenos, múltiplos por cena, condições climáticas variadas
- Nosso dataset de treino (CITRA-3D-Real) tem apenas 2.081 imagens
- Com poucos dados, o detector tem limitações — como melhorar sem esperar meses de coleta?

**[Usar: fig_composition_process.pdf, painel (c) — imagem real do CITRA-3D mostrando o cenário operacional]**

### SLIDE 3 — A Oportunidade: Imagens Públicas
- Existe um dataset público com ~28 mil imagens de navios (InaTechShips, do Eduardo/INATEL)
- São fotos do shipspotting.com: navios em close-up, alta qualidade
- Hipótese natural: usar essas imagens como dados de treino para complementar os nossos
- Se funcionar, ganhamos diversidade de aparência sem custo de coleta

**[Usar: fig_composition_process.pdf, painel (a) — imagem do InaTechShips mostrando navio close-up]**

### SLIDE 4 — O Gap: Dois Mundos Diferentes
Mostrar o contraste visual lado a lado:
- InaTechShips: navio ocupa ~80% da imagem, 1 navio, foto profissional
- CITRA-3D: navios ocupam ~3% da imagem, 3-4 navios por cena, câmera de vigilância
- São ambos "imagens de navios", mas as características são completamente diferentes

**[Usar: fig3_scale_distribution.pdf — histogramas mostrando a diferença quantitativa de escala]**

### SLIDE 5 — O Que Aconteceu: Piorou
Tabela simples:
| O que fizemos | mAP50 | Resultado |
|---|---|---|
| Só COCO genérico (baseline) | 0.835 | Referência |
| + 28k imagens públicas (curado CLIP) | 0.794 | **Piorou 4,15 pp** |
| + 28k imagens públicas (aleatório) | 0.795 | **Piorou 4,06 pp** |
| Sem pré-treino nenhum | 0.801 | Pior que COCO |

Mensagem-chave: usar as imagens públicas diretamente **prejudicou** o detector. Pior que não usar nada. Não importa se selecionamos por similaridade ou aleatoriamente — o resultado é o mesmo.

### SLIDE 6 — O Diagnóstico
Por que piorou? Investigamos três eixos:
1. **Escala:** o modelo aprendeu a detectar navios grandes (80% da imagem) e "esqueceu" como detectar objetos pequenos (3%)
2. **Densidade:** aprendeu a encontrar 1 navio por cena, mas no cenário real são 3-4
3. **Contexto:** aprendeu fundos de porto/cais, não oceano aberto com ondas e neblina

Confirmação: ablation de épocas mostra degradação monotônica — quanto mais treina nas imagens públicas, pior fica. Mesmo 10 épocas já prejudica.

**[Usar: fig5_ablation_curve.pdf — curva mostrando degradação com cada incremento]**

### SLIDE 7 — O Insight: Adaptar, Não Usar Direto
O problema não são os navios em si — as fotos públicas mostram navios reais de tipos variados.
O problema é COMO eles aparecem na imagem: escala, posição, contexto.

Solução proposta: transformar as imagens públicas para que pareçam com o nosso cenário operacional.

### SLIDE 8 — O Método: Composição In-Place
Processo em 3 passos:
1. **Recortar:** usar SAM para isolar cada navio do InaTechShips (fundo transparente)
2. **Redimensionar:** ajustar para o tamanho que navios reais têm no CITRA-3D (~3% da imagem)
3. **Posicionar:** colar exatamente onde navios reais existiam nas nossas cenas operacionais

Por que funciona:
- Escala correta (herda do bbox real)
- Posição correta (onde navios reais estavam)
- Fundo real (oceano, costa, condições climáticas reais)
- Labels exatos (mesmas coordenadas das anotações originais)

**[Usar: fig_composition_process.pdf — todos os 4 painéis: (a) original, (b) SAM crop, (c) cena real, (d) resultado sintético]**

### SLIDE 9 — O Dataset Sintético
- 27.053 imagens geradas (13 variações por cena real)
- 91.035 objetos anotados (3,37 por imagem — idêntico ao cenário real)
- Cada split gerado exclusivamente do split correspondente do CITRA-3D
- Zero decisões arbitrárias de posicionamento

### SLIDE 10 — Primeira Tentativa: Sequencial
Treino em duas fases: primeiro no sintético, depois no real
| Estratégia | mAP50 | vs Baseline |
|---|---|---|
| Baseline COCO | 0.835 | ref |
| Sequencial (100 épocas) | 0.822 | −1,31 pp |

Melhorou em relação ao uso direto (de −4,15 para −1,31), mas ainda não superou o baseline.
O problema: o modelo "esquece" o COCO durante a fase sintética (catastrophic forgetting).

### SLIDE 11 — A Solução: Treino Conjunto
Em vez de treinar em fases separadas, misturar real + sintético no MESMO treino:
- 50% imagens reais (oversampling 13×)
- 50% imagens sintéticas
- Em cada batch, o modelo vê os dois tipos intercalados
- Nunca "esquece" os dados reais

### SLIDE 12 — O Resultado: Funciona
| Estratégia | mAP50 | Recall | F1 | vs Baseline |
|---|---|---|---|---|
| **Treino conjunto (50/50)** | **0.845** | **0.805** | **0.830** | **+1,00 pp** |
| Baseline COCO | 0.835 | 0.783 | 0.818 | ref |

Confirmado com 3 seeds independentes. Intervalos de confiança não se sobrepõem.

**[Usar: fig6_comparison_bar.pdf — gráfico de barras com todos os braços]**

### SLIDE 13 — O Ganho Mais Importante: Recall
- Recall subiu de 0.783 para 0.805 (+2,8%)
- Isso significa: o detector encontra MAIS navios na cena
- Precision se manteve em 0.857 — não aumentou falsos alarmes
- Para vigilância marítima, perder um navio (falso negativo) é mais grave que um alarme falso

### SLIDE 14 — O Que Aprendemos (Contribuição para o Projeto)
1. **Imagens públicas são úteis, mas precisam de adaptação.** Usar direto piora. Adaptar e treinar junto melhora.
2. **O regime de treino importa tanto quanto os dados.** Sequencial causa esquecimento. Treino conjunto preserva o conhecimento.
3. **Similaridade visual não garante compatibilidade.** CLIP diz que são parecidas, mas o detector discorda. Escala, densidade e contexto são os fatores reais.
4. **Podemos melhorar o detector sem coletar mais dados operacionais.** As 28 mil imagens públicas, quando adaptadas, complementam nossas 2.081 imagens.

### SLIDE 15 — Impacto Prático para o Sistema
Como isso melhora o sistema que estamos desenvolvendo:
- **Mais detecções corretas** em cenários com múltiplos navios distantes
- **Sem custo adicional de coleta** — usamos dados públicos existentes
- **Pipeline reprodutível** — pode ser reaplicado quando novos dados do InaTechShips ou de outras fontes surgirem
- **Código aberto** — todos os scripts estão no GitHub, prontos para integrar no pipeline de treino do projeto

### SLIDE 16 — Comparação Completa
Tabela com todos os experimentos:
| Estratégia | mAP50 | Δ |
|---|---|---|
| Treino conjunto (50/50) | 0.845 ± 0.003 | +1,00 |
| Backbone congelado | 0.834 ± 0.004 | −0,09 |
| Sequencial 20 épocas | 0.834 ± 0.003 | −0,08 |
| Baseline COCO | 0.835 ± 0.002 | ref |
| Sequencial 100 épocas | 0.822 ± 0.009 | −1,31 |
| Sem pré-treino | 0.801 ± 0.006 | −3,43 |
| InaTechShips aleatório | 0.795 ± 0.005 | −4,06 |
| InaTechShips curado | 0.794 ± 0.005 | −4,15 |

### SLIDE 17 — Próximos Passos
Para o sistema:
- Integrar o treino conjunto no pipeline de atualização do detector
- Testar com novos dados operacionais quando disponíveis
- Explorar detecção multi-classe (tipos de embarcação)
- Investigar proporção ótima real/sintético

Para validação científica:
- Submeter artigo para Ocean Engineering para validar a proposta com revisores especializados e receber feedback da comunidade
- Repositório público para reprodutibilidade

### SLIDE 18 — Discussão
- Dúvidas?
- Sugestões para o próximo ciclo de experimentos?

**Estilo visual:**
- Fundo limpo, branco ou cinza claro
- Fonte grande e legível (mínimo 18pt)
- Máximo 5 linhas de texto por slide — prefira imagens e tabelas
- Use as figuras do artigo onde indicado [fig_*.pdf]
- Tabelas com no máximo 5-6 linhas
- Tom: prático e direto, como um relatório de progresso para o supervisor
- Não usar jargão de ML sem explicação (e.g., explique "catastrophic forgetting" como "o modelo esquece o que aprendeu antes")
- Foco em: "o que isso muda para o nosso sistema?" em vez de "o que isso contribui para a literatura"
