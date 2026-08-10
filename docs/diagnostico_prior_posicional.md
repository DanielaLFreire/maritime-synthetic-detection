# Diagnóstico de prior posicional na composição sintética in-place

**Status:** concluído · resultado negativo (hipótese não confirmada)
**Data:** agosto de 2026
**Branch:** `diag-prior-posicional`
**Script:** `scripts/03_analysis/analisar_prior_posicional.py`
**Painel:** `app/streamlit_prior_posicional.py`
**Resultados:** `results/prior_posicional/`

---

## 1. Hipótese investigada

O pipeline de composição in-place cola os crops sintéticos nas coordenadas exatas
das embarcações reais anotadas. A preocupação: isso poderia levar o detector a
internalizar um prior sobre **onde** embarcações aparecem, em vez de aprender
apenas **como** elas se parecem.

Se o prior existisse, o ganho do braço sintético sobre o baseline estaria
concentrado na banda vertical central (região marítima habitual) e seria nulo ou
negativo nas caudas. E, como o test set do CITRA-3D-Real compartilha a geometria
de câmera do treino, a avaliação in-domain **recompensaria** esse prior — o
número subiria sem ganho real de capacidade.

## 2. Por que a ablação A′joint-rand não responde isso

A ablação existente (§5.5 do artigo SIBGRAPI) compara colocação in-place contra
colocação aleatória. Ela não testa a hipótese acima, por três razões estruturais:

1. **A colocação aleatória não é aleatória no eixo relevante.** O `sea_mask` do
   `gerar_baseline_random_copypaste_v4.py` usa horizonte adaptativo com piso em
   35% da altura. O braço aleatório amostra dentro de aproximadamente a mesma
   banda. Muda a coordenada exata; quase não muda a distribuição marginal de `y`.
2. **O ramo real carrega as coordenadas verdadeiras nos dois braços.** No regime
   joint 50/50, as imagens reais entram oversampleadas ×13 — mesmas bboxes, mesmas
   posições, em ambos os braços. A ablação não isola o que é comum aos dois.
3. **A métrica de comparação é in-domain.** Duas configurações com priors
   posicionais diferentes podem empatar in-domain e divergir fora.

A ablação demonstra robustez **à coordenada exata dentro da banda**, não **à
banda**. São afirmações diferentes.

## 3. Método

AP/AR estratificados por faixa de `y_center` do ground truth no test set do
CITRA-3D-Real (401 imagens, 1.247 objetos), para três braços × três seeds.

**Estratificação.** O `pycocotools` estratifica por tamanho via `params.areaRng`,
com a semântica correta dos dois lados: GT fora da faixa vira `_ignore` (sai do
denominador do recall) e detecções não pareadas fora da faixa também são
ignoradas (não contam como FP). Precisávamos disso no eixo Y, então substituímos
o campo `area` por um proxy `y_center × 1e6` no GT e nas detecções, e definimos
`areaRng` como a faixa do bin. É matematicamente idêntico à estratificação por
tamanho do COCO, apenas com outra variável.

Consequência: `E.summarize()` não pode ser usado (assume as quatro faixas padrão);
AP/AR são lidos de `E.eval['precision']` e `E.eval['recall']`. E este pipeline
**não** produz AP_small/medium/large válidos — para isso, use
`extrair_metricas_detalhadas.py`.

**Controle do confundidor tamanho.** Em câmeras marítimas fixas, posição vertical
pode correlacionar com tamanho aparente. Medimos: **ρ de Spearman = +0,048**
(p = 0,088). Correlação desprezível — no CITRA-3D-Real os eixos são independentes,
provavelmente por variedade de plataformas e distâncias focais. A análise foi
mesmo assim repetida dentro de cada classe de tamanho COCO
(`--size-class small|medium|large`), medida em pixels nativos.

O filtro por classe de tamanho **remove** anotações e detecções fora da faixa em
vez de marcá-las: `COCOeval._prepare` sobrescreve `gt['ignore']` com `iscrowd`,
então o campo `ignore` não é utilizável. Desvio conhecido: uma detecção fora da
classe que pareasse um GT dentro da classe vira FN em vez de TP; com IoU ≥ 0,5 nas
fronteiras 32²/96², efeito de segunda ordem.

**Bordas fixas.** A rodada final usa `--edges 0,0.4499,0.6221,1.0` (terços por
quantil do conjunto completo) em todas as classes, para que as regiões verticais
sejam a mesma parte da imagem em toda comparação. Bordas por quantil dentro de
cada classe produziriam faixas diferentes e resultados não comparáveis.

**Validação.** O ganho global agregado do A′joint reproduz **+1,01 pp** contra o
B2; o artigo reporta +1,00 pp.

**Precisão.** Todos os valores abaixo vêm dos CSVs em `results/prior_posicional/`,
não da saída arredondada do console. A distinção importa: com AP50 truncado em
três casas, o déficit de campo próximo do A′joint em `all` aparecia como empate
em uma das seeds e o sinal consistente passava despercebido.

## 4. Resultado

Δ AP50 vs B2 (pp), por região vertical. `*` = mesmo sinal nas três seeds.

| classe | braço | topo 0–0,45 | banda 0,45–0,62 | frente 0,62–1 | n por faixa |
|---|---|---|---|---|---|
| all | A′joint | +0,09 | +2,05\* | **−1,20\*** | 250 / 747 / 250 |
| all | rand | −0,14 | +1,39 | +0,22 | 250 / 747 / 250 |
| small | A′joint | **+3,28\*** | +2,92\* | −3,69 | 88 / 241 / 66 |
| small | rand | **+4,56\*** | +2,45\* | −1,52 | 88 / 241 / 66 |
| medium | A′joint | −0,30 | +0,75 | +0,09 | 119 / 359 / 123 |
| medium | rand | +0,31 | −0,27 | +0,61 | 119 / 359 / 123 |
| large | A′joint | +1,60 | +1,78 | −1,20 | 43 / 147 / 61 |
| large | rand | +2,65\* | +0,58 | +2,11 | 43 / 147 / 61 |

**A hipótese não se confirma.** Em objetos pequenos — a classe onde o ganho de
fato existe — o ganho na faixa do horizonte (+3,28 pp) é **maior** que na banda
central (+2,92 pp), com sinal consistente nas três seeds. Se houvesse prior de
banda, era na banda que o ganho deveria se concentrar.

Ganho global ponderado por classe:

| classe | A′joint | rand |
|---|---|---|
| all | +1,01 pp | +0,85 pp |
| small | +1,90 pp | +2,26 pp |
| medium | +0,41 pp | +0,02 pp |
| large | +1,02 pp | +1,31 pp |

O ganho é um fenômeno de objetos pequenos, distribuído ao longo da altura da
imagem. Corrobora, por caminho independente, o achado de ARsmall da Tabela 7.

### 4.1 Uma assimetria que sobrevive: déficit de campo próximo

Na faixa inferior (`y_center` > 0,62), o A′joint fica **1,20 pp abaixo do B2 nas
três seeds** com o conjunto agregado (n=250) — enquanto o A′joint-rand não
apresenta déficit (+0,22 pp). Este é o **único ponto de toda a análise em que a
ancoragem in-place se distingue da colocação aleatória**, e é o resultado mais
interessante do diagnóstico depois do resultado negativo principal.

Duas leituras possíveis, não distinguíveis com os dados atuais:

1. **Custo da ancoragem.** Colar crops nas coordenadas reais faria o modelo tratar
   a região de campo próximo de forma mais rígida. Contra essa leitura: o efeito
   não aparece na banda central, onde a densidade de âncoras é muito maior.
2. **Qualidade de crop em âncoras grandes.** A área mediana da bbox no campo
   próximo é 2.737 px², contra ~1.950 nas demais faixas. Crops do SAM colados em
   âncoras maiores sobem de escala e podem ficar borrados. Contra essa leitura: o
   déficit não cresce com a classe de tamanho (large: −1,20 pp, sinais mistos).

Restringindo a objetos pequenos (n=66), os deltas por seed são +4,60 / −9,80 /
−5,90 — não resolvível. Não sabemos qual regime de tamanho dirige o efeito
agregado.

Tem relevância operacional: campo próximo é onde a vigilância menos tolera falhas
de detecção. Resolver exigiria mais seeds, e é a única extensão deste diagnóstico
com retorno claro.

## 5. Falha da estatística inicialmente pré-especificada

O contraste originalmente definido foi a **assimetria centro–cauda**: AP médio nas
faixas centrais menos AP médio nas caudas, um escalar por braço/seed, para evitar
as comparações múltiplas de testes bin a bin. Em `small` ele deu +3,12 pp com
sinal consistente nas três seeds — aparentemente confirmando a hipótese.

Decomposto, o valor se revela artefato da construção:

| seed | Δtopo | Δbanda | Δfrente | Δassimetria |
|---|---|---|---|---|
| 42 | +1,60 | +3,90 | +4,60 | +0,80 |
| 123 | +5,00 | +3,10 | −9,80 | +5,50 |
| 2024 | +3,20 | +1,70 | −5,90 | +3,05 |

A assimetria é positiva porque a cauda **inferior** é negativa, não porque a banda
central ganhe. A estatística soma uma cauda que ganha com uma cauda que perde e
chama o resultado de "concentração central". Foi má operacionalização da
hipótese: o contraste foi definido sem antecipar que as duas caudas pudessem se
comportar de forma oposta.

**Lição metodológica:** um contraste pré-especificado protege contra p-hacking,
mas não contra estar medindo a coisa errada. A decomposição por região era
necessária de todo modo.

## 6. Limitações

- **n = 3 seeds.** O t crítico bilateral é 4,30; nenhum efeito moderado atinge
  p < 0,05. As conclusões se apoiam em consistência de sinal, não em p-valores.
- **Déficit de campo próximo parcialmente resolvido.** O efeito é consistente nas
  três seeds no conjunto agregado (−1,20 pp, n=250), mas não é atribuível a uma
  classe de tamanho: dentro de `small` os sinais são mistos (+4,60 / −9,80 /
  −5,90) sobre 66 objetos. Ver seção 4.1.
- **Um p < 0,05 isolado.** `rand` em `medium` deu −0,73 pp com p = 0,044. São oito
  testes de assimetria na tabela e a magnitude é sub-ponto. Tratado como ruído.
- **Escopo in-domain.** O diagnóstico usa apenas o test set do CITRA-3D-Real. O
  comportamento cross-domain (SMD, SeaShips) é analisado separadamente.
- **Definição de classe de tamanho.** As áreas são medidas em pixels nativos, não
  após redimensionamento para 640. A distribuição resultante (small 31,7%,
  medium 48,2%, large 20,1%) difere da reportada na Tabela 7 do artigo, que usa a
  convenção c640. São populações diferentes sob o mesmo rótulo.

## 7. Conclusão

Não há evidência de que a composição in-place induza dependência posicional do
tipo suposto. O ganho distribui-se por toda a extensão vertical da imagem, é maior
fora da banda central que dentro dela, concentra-se em objetos pequenos, e
persiste após controle por classe de tamanho.

A exceção é o campo próximo (seção 4.1), onde o A′joint perde 1,20 pp de forma
consistente e a colocação aleatória não perde. É um efeito localizado e de sinal
oposto ao previsto pela hipótese original — que previa ganho concentrado no
centro, não perda concentrada na base.

Isto não prova ausência de viés — nenhum experimento prova ausência. O que foi
feito: procurar a assinatura que o viés deixaria, em quatro cortes independentes
dos dados, e não encontrá-la.

**Consequência para o projeto:** a composição in-place é mantida. A objeção fica
documentada e respondida.

## 8. Reprodução

```bash
BASE=/content/drive/MyDrive/PROJETO_MARINHA/Experimento_Dataset_Similar
for cls in all small medium large; do
  python scripts/03_analysis/analisar_prior_posicional.py \
    --citra-root /content/data/CITRA-3D-Real \
    --runs-root $BASE/runs \
    --out $BASE/results/prior_posicional \
    --edges 0,0.4499,0.6221,1.0 --size-class $cls
done
```

As predições ficam cacheadas por braço/seed em `preds_cache/` (fora do git) e são
independentes da classe de tamanho: trocar `--size-class` ou as bordas não custa
nova inferência. Só a primeira rodada usa GPU.

Braços avaliados: `baselines/B2_coco`, `braco_balanced` (A′joint),
`braco_random_copypaste_v4` (A′joint-rand); seeds 42, 123, 2024.
