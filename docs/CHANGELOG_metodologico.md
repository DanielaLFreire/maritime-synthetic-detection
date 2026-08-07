
## Rastreabilidade dos pools de crops — manifestos (Fase 1/2 multi-fonte)
O `build_pool` do `08_compose_multisource.py` lista os crops com `glob.glob()`,
que **não ordena** — a ordem vem do sistema de arquivos e varia entre sessões
do Colab. Como o `rng.sample(seed=42)` opera sobre essa lista, o mesmo seed
**não** regenera o mesmo pool em outra sessão. Afeta os pools já construídos
(`synth_inatech`, `synth_both`).

Impacto: **validade interna preservada, regenerabilidade perdida.** O pool é um
sorteio aleatório legítimo no volume controlado, e os 3 seeds de cada braço
compartilham o mesmo pool (o zip é construído uma vez e reutilizado). O que se
perde é a capacidade de terceiros reconstruírem o conjunto exato a partir do
código.

Mitigação adotada: **manifesto por braço** em `docs/manifests/<tag>.json`, com
composição do pool, contagens e SHA-256 da lista ordenada de arquivos do zip.
O artefato passa a ser a fonte de verdade, identificado por hash.

- `synth_both` (Fase 2): ABO 16.016 + InaTech 16.016 = 32.032 crops de pool;
  1.680 imagens CITRA × 13 variações = 21.840 sintéticas; 43.686 arquivos no
  zip (43.680 img+label + 6 entradas de diretório). Contagem confere com a
  paridade do artigo (A′joint) e com o A_joint_ABO.
  SHA-256 da lista: `dfbdda00…f546f9`.

Correção prospectiva: `sorted()` antes do embaralhamento, subamostragem por
prefixo (pools aninhados) e falha explícita quando o volume pedido excede o
pool disponível. Entra a partir do braço InaTech-27.799 — os pools anteriores
**não** são reconstruídos, para não quebrar a paridade entre seeds já treinados.

## Braço `abo_2x` — cancelado (controle inválido)
O controle de volume previsto no cabeçalho do `08_compose_multisource.py`
(`--sources ABO --volume 32032 --tag abo_2x`) **não é construtível**: o
`crops_abo` tem 16.016 crops limpos, então o `build_pool` cai no ramo de aviso
e usa todos — produzindo um braço rotulado "32.032" com pool de 16.016.

Além disso, o ramo de aviso usa a lista na ordem crua do glob, enquanto o braço
de 16K usa uma permutação (`rng.sample` de tamanho igual ao total). Os dois
pools são o **mesmo conjunto**, diferindo só na atribuição crop→posição. Ou
seja, `abo_2x` seria uma **réplica de atribuição** do ABO-16K, não um controle
de volume.

Decisão: braço cancelado. O controle de cardinalidade da Fase 2 passa a ser
**A_joint_InaTech com pool cheio (27.799 crops)** — única fonte que se aproxima
de 32K com fonte única. Esse braço também fecha a atribuição causal declarada
na Seção 5.5 do artigo SIBGRAPI ("crop diversity"), nunca medida: o B2-long
controlou volume de treino (passos de gradiente), não cardinalidade do pool.
