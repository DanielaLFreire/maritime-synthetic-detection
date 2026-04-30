# Roteiro de Falas — Apresentação para o Supervisor

**Tempo total estimado:** 25-30 minutos + discussão
**Tom:** conversa profissional entre colegas, explicando uma investigação prática

---

## SLIDE 1 — Capa
*[Não falar muito, só contextualizar]*

**Fala:**
"Boa tarde, Leandro. Vou apresentar os resultados do experimento que rodamos nas últimas semanas sobre como usar imagens públicas de navios para melhorar o nosso sistema de detecção. Foi uma investigação com algumas surpresas, mas que terminou com um resultado positivo e prático para o projeto."

---

## SLIDE 2 — Nosso Desafio
*[~2 min | Estabelece o problema que o supervisor já conhece]*

**Fala:**
"Vou começar pelo que já sabemos. Nosso sistema de detecção precisa identificar embarcações em imagens de câmeras de vigilância. O cenário é desafiador: os navios aparecem pequenos e distantes, tem vários por cena, e as condições de tempo e iluminação variam bastante.

O CITRA-3D-Real, que é nosso dataset operacional, tem apenas 2.081 imagens. Para deep learning, isso é pouco. A pergunta que motivou este trabalho foi: **como melhorar o detector sem precisar esperar meses para coletar e anotar mais dados operacionais?**"

---

## SLIDE 3 — A Oportunidade: Imagens Públicas
*[~1.5 min | Apresenta a ideia inicial]*

**Fala:**
"Aí identificamos uma oportunidade. O Eduardo Teixeira, lá do INATEL, publicou um dataset chamado InaTechShips com cerca de 28 mil imagens de navios, todas do site shipspotting.com. São fotos de alta qualidade, de diversos tipos de embarcação.

A ideia parecia óbvia: se temos 2 mil imagens de navios e existe um dataset público com 28 mil imagens de navios, por que não usar essas imagens extras para treinar melhor o detector? Mais dados, melhor resultado — essa é a intuição."

---

## SLIDE 4 — O Gap: Dois Mundos Diferentes
*[~2 min | Mostra visualmente o problema — slide importante]*

**Fala:**
"Mas olha a diferença entre os dois datasets. À esquerda, uma imagem típica do InaTechShips: navio fotografado de perto, em alta qualidade, ocupando quase a imagem inteira — cerca de 80% da área. Um navio por foto, fundo limpo.

À direita, uma imagem típica do nosso CITRA-3D: vários navios distantes, cada um ocupando menos de 3% da imagem. Fundo com oceano, neblina, ondas.

Veja os histogramas: a área mediana de um navio no InaTechShips é **54% da imagem**. No CITRA-3D é **0,1%**. Isso é uma diferença de **500 vezes**. São dois mundos completamente diferentes, mesmo que ambos sejam 'imagens de navios'."

*[Pausa — deixa o supervisor absorver o contraste visual]*

---

## SLIDE 5 — O Que Aconteceu: Piorou
*[~2 min | O momento de surpresa — fala com ênfase]*

**Fala:**
"Pois bem. Rodamos o experimento: pré-treinamos o YOLOv11m nas 28 mil imagens do InaTechShips e depois fizemos o fine-tuning no CITRA-3D.

O resultado foi surpreendente. Olha a tabela: o baseline usando só pesos COCO genéricos dá um mAP50 de 0,835. Quando adicionamos as 28 mil imagens de navios como pré-treino... **caiu para 0,794**. Piorou mais de 4 pontos percentuais.

E não é problema da seleção de imagens. Testamos duas abordagens: uma usando CLIP para selecionar as imagens mais parecidas com o nosso dataset, e outra selecionando aleatoriamente. O resultado foi praticamente o mesmo nos dois casos. Ou seja, **o problema não é quais imagens usamos, é o dataset inteiro**.

O COCO genérico, que nem tem uma classe 'navio', foi mais útil que 28 mil fotos de navios. Isso é contraintuitivo."

---

## SLIDE 6 — O Diagnóstico
*[~2.5 min | Explica o porquê — parte técnica mas acessível]*

**Fala:**
"Antes de desistir, investigamos o porquê. Fizemos uma ablation variando quantas épocas de pré-treino o modelo recebe nas imagens do InaTechShips.

Olha o gráfico: a degradação é **monotônica**. Quanto mais o modelo treina nessas imagens, pior fica. Com 10 épocas já piora. Com 100 épocas, volta ao nível de não ter pré-treino nenhum.

O que está acontecendo é o seguinte: o modelo começa com pesos do COCO, que são genéricos — ele sabe detectar bordas, texturas, formas em diversas escalas. Quando treina no InaTechShips, ele se especializa em detectar navios grandes em fotos bonitas. E ao se especializar nisso, ele **esquece** como detectar objetos pequenos em cenas complexas. Na literatura isso se chama catastrophic forgetting — o modelo sobrescreve o que sabia antes.

Então identificamos três eixos do problema: **escala** — 80% versus 3%, **densidade** — 1 navio por imagem versus 3 a 4, e **contexto** — fotos de porto versus câmera de vigilância em mar aberto."

---

## SLIDE 7 — O Insight
*[~1 min | Transição — a virada da narrativa]*

**Fala:**
"Aqui veio o insight principal: o problema não são os navios em si. O InaTechShips tem navios reais, de tipos variados, que seriam úteis para o detector. O problema é **como** eles aparecem na imagem.

Então em vez de descartar esses dados, nos perguntamos: e se transformarmos essas imagens para que **pareçam** com o nosso cenário operacional?"

---

## SLIDE 8 — O Método: Composição In-Place
*[~2.5 min | Explica o método — usar a figura como guia]*

**Fala:**
"Este é o método que desenvolvemos, em três passos.

Primeiro, **recortamos** cada navio do InaTechShips usando o SAM — Segment Anything Model, do Meta. Ele isola o navio com uma máscara precisa, como vemos no painel (b): o navio sem fundo, transparente.

Segundo, **redimensionamos** esse recorte para o tamanho que um navio real teria no nosso cenário — não mais 80% da imagem, mas 3%, que é a escala real.

Terceiro — e esse é o ponto-chave — **posicionamos** o recorte **exatamente** onde um navio real existia em uma das nossas cenas operacionais. Não colocamos em posição aleatória. Olha o painel (d): os navios sintéticos estão nas mesmas posições que os navios reais estavam no painel (c).

Por que isso é importante? Porque elimina três fontes de erro: a escala é correta por construção, a posição é correta porque herda de dados reais, e o fundo é real — oceano, costa, condições climáticas autênticas. Os labels são idênticos aos originais."

---

## SLIDE 9 — O Dataset Sintético
*[~1 min | Números rápidos]*

**Fala:**
"Com esse método, geramos 27 mil imagens sintéticas — 13 variações por cada cena real, cada uma com navios diferentes do InaTechShips. O dataset tem 91 mil objetos anotados, com uma média de 3,37 navios por imagem, que é idêntica à média do dataset real.

Um cuidado importante: cada split sintético — treino, validação e teste — foi gerado exclusivamente a partir do split correspondente do CITRA-3D. Nenhuma informação do teste entrou no treino."

---

## SLIDE 10 — Primeira Tentativa: Sequencial
*[~1.5 min | Mostra que o caminho direto não bastou]*

**Fala:**
"A primeira coisa que testamos foi o pré-treino sequencial: primeiro treina no dataset sintético, depois faz fine-tuning no real. É o protocolo padrão.

O resultado: mAP50 de 0,822. Melhorou em relação ao uso direto — o gap caiu de menos 4,15 para menos 1,31. Mas ainda não superou o baseline COCO de 0,835.

O problema continua sendo o forgetting: mesmo com imagens adaptadas ao domínio, treinar 100 épocas no sintético antes do real faz o modelo perder parte do que aprendeu com o COCO."

---

## SLIDE 11 — A Solução: Treino Conjunto
*[~1.5 min | A virada final]*

**Fala:**
"Então mudamos a abordagem. Em vez de treinar em duas fases separadas, **misturamos** as imagens reais e sintéticas no mesmo treino.

Na prática: pegamos as 1.348 imagens reais de treino, replicamos 13 vezes, e juntamos com as 17.524 sintéticas. Resultado: um dataset combinado com 35 mil imagens, metade real, metade sintética.

A cada batch de treino, o modelo vê imagens dos dois tipos intercaladas. Ele nunca 'esquece' os dados reais porque continua vendo-os o tempo todo. E ao mesmo tempo, ganha diversidade visual das sintéticas — tipos de navio que não existem nas nossas 2 mil imagens."

---

## SLIDE 12 — O Resultado: Funciona
*[~2 min | O clímax — fala com confiança]*

**Fala:**
"E o resultado: **mAP50 de 0,845**, superando o baseline COCO por **1 ponto percentual**. Confirmado com três execuções independentes, com seeds diferentes. Os intervalos de confiança não se sobrepõem — a melhoria é consistente.

Pode parecer pouco — 1 ponto — mas lembre que todas as outras tentativas **pioraram**. Usar direto: menos 4 pontos. Sequencial com dados adaptados: menos 1,3 pontos. Treinar do zero: menos 3,4 pontos. A única configuração que superou o COCO genérico foi essa: composição in-place com treino conjunto balanceado."

---

## SLIDE 13 — O Ganho Mais Importante: Recall
*[~1.5 min | Traduz para impacto operacional]*

**Fala:**
"Mas o número que mais importa para o nosso sistema não é o mAP50. É o **Recall**.

O Recall subiu de 0,783 para 0,805 — um aumento de quase 3%. O que isso significa na prática? O detector encontra **mais navios** na cena. A Precision se manteve em 0,857, então ele não está gerando mais alarmes falsos — está genuinamente detectando embarcações que antes passavam despercebidas.

Para vigilância marítima, isso é o que importa. Um falso negativo — um navio que não foi detectado — é mais grave do que um falso positivo. Esse ganho de Recall é diretamente útil para o sistema."

---

## SLIDE 14 — O Que Aprendemos
*[~2 min | Lições consolidadas]*

**Fala:**
"Quatro lições principais deste experimento.

**Primeira:** imagens públicas são úteis, mas precisam de adaptação. Usar direto é pior que não usar. Adaptar ao domínio e treinar junto funciona.

**Segunda:** o regime de treino importa tanto quanto os dados. Os mesmos dados sintéticos deram resultado negativo em regime sequencial e positivo em treino conjunto. A forma de usar é tão importante quanto a qualidade dos dados.

**Terceira:** similaridade visual não garante compatibilidade. O CLIP dizia que as imagens eram parecidas, mas o detector discordava. O que importa para detecção é escala, densidade e contexto — não aparência global.

**Quarta — e essa é a mais relevante para o projeto:** podemos melhorar o detector sem coletar mais dados operacionais. As 28 mil imagens públicas, quando adaptadas, complementam as nossas 2 mil imagens de forma eficaz."

---

## SLIDE 15 — Impacto Prático para o Sistema
*[~1.5 min | Conecta ao projeto — slide que o supervisor mais valoriza]*

**Fala:**
"Traduzindo para o nosso sistema:

Primeiro: **mais detecções corretas** em cenários difíceis — múltiplos navios distantes, condições de baixa visibilidade.

Segundo: **zero custo adicional de coleta**. Usamos dados que já estão disponíveis publicamente.

Terceiro: o **pipeline é reprodutível**. Quando surgirem novas imagens públicas, ou quando o InaTechShips crescer, podemos reaplicar o mesmo processo.

E quarto: **todo o código está pronto**. São 22 scripts organizados em 6 etapas, com configurações separadas para cada experimento. Pode ser integrado diretamente no pipeline de treino do projeto."

---

## SLIDE 16 — Comparação Completa
*[~1 min | Referência rápida, não precisa detalhar tudo]*

**Fala:**
"Aqui está a comparação completa de tudo que testamos, ordenado por resultado. O treino conjunto balanceado está no topo. Destaco que também testamos backbone congelado e 20 épocas — ambos ficaram estatisticamente equivalentes ao baseline, o que confirma que o problema é o forgetting e que há múltiplas formas de mitigá-lo. Mas só o treino conjunto realmente superou."

---

## SLIDE 17 — Próximos Passos
*[~1.5 min]*

**Fala:**
"Para o sistema, o próximo passo natural é integrar esse treino conjunto no pipeline de atualização do detector. Quando tivermos novos dados operacionais, podemos combinar com as sintéticas e retreinar.

Também quero explorar a detecção multi-classe — separar tipos de embarcação — e investigar se existe uma proporção ótima de imagens reais e sintéticas no treino conjunto.

E como passo de validação científica, preparamos um artigo para a Ocean Engineering, que é a mesma revista onde o Eduardo publicou o InaTechShips. A ideia é submeter para receber feedback de revisores especializados da área e validar a abordagem com a comunidade. O artigo está praticamente pronto e o repositório com todo o código é público."

---

## SLIDE 18 — Discussão
*[Aberto]*

**Fala:**
"É isso. Resumindo em uma frase: **descobrimos que imagens públicas de navios só ajudam quando adaptadas ao domínio operacional e usadas em treino conjunto, não como pré-treino sequencial**.

O que achas, Leandro? Tens sugestões para o próximo ciclo de experimentos?"

---

## DICAS DE ENTREGA

**Ritmo:** fale devagar nos slides 5, 8 e 12 — são os momentos-chave (surpresa, método, resultado). Nos slides de números (9, 16), pode ser mais rápido.

**Postura:** nos slides 5 e 6 (piorou + diagnóstico), mantenha tom analítico, não defensivo. O resultado negativo é uma contribuição, não um erro.

**Pausas:** depois de mostrar que piorou (slide 5), faça uma pausa de 2 segundos antes de ir para o diagnóstico. Deixe o impacto assentar.

**Interação:** se o supervisor interromper com perguntas durante os slides 5-8, é bom sinal — significa que está engajado. Responda e volte ao fluxo.

**Se perguntar "por que só 1 ponto de melhoria?":** responda que o ganho de Recall (+2,8%) é mais relevante operacionalmente, e que o baseline COCO já é muito forte (é pré-treinado em 1,2 milhão de imagens com 80 categorias). Superar ele com dados sintéticos adaptados é significativo.

**Se perguntar sobre custos computacionais:** treino conjunto leva ~1h no Colab com A100. Geração sintética leva ~4h (CPU). O investimento é baixo para o ganho obtido.
