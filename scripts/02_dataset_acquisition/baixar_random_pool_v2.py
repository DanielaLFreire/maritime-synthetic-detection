"""
baixar_random_pool_v2.py (v2)

Baixa as imagens do random_pool_v2 a partir dos arquivos de IDs sorteados
por gerar_ids_aleatorios.py (rodada 1) ou gerar_ids_rodada2.py (rodada 2).

VERSÃO 2 — mudanças importantes em relação à v1:

  1. **Detecção de HTML disfarçado de JPEG.** O shipspotting CDN retorna
     HTTP 200 com página HTML de erro em vez de HTTP 404 para IDs
     inexistentes. A v1 salvava esses HTMLs como .jpg porque verificava
     apenas status_code == 200 + len(content) > 5000. A v2 verifica os
     "magic bytes" do JPEG (FF D8 FF) antes de salvar e classifica
     respostas inválidas como `invalid_content` (novo outcome).

  2. **Novo outcome `invalid_content`.** Separado de `not_found` (que
     continua reservado para HTTP 404 reais, caso aconteçam) e de
     `errors` (problemas de rede/timeout). Permite distinguir no log
     entre "servidor retornou nada" vs "servidor retornou HTML de erro".

  3. **Flag `--ids-suffix`.** Permite ler arquivos de IDs alternativos.
     Por padrão lê `ids_random_{split}.txt` (rodada 1). Com
     `--ids-suffix rodada2` lê `ids_random_rodada2_{split}.txt`.

Adaptado de download_direto.py, removendo a etapa de filtragem por
similaridade (os IDs já foram estratificados por decis no passo anterior)
e mantendo:

- Sessão HTTP com cookies do Chrome (bypass de Cloudflare)
- Blacklist CLIP para filtrar banners recorrentes (Queen Victoria, etc.)
- Retomada após falha (progresso persistido em JSON)
- Tratamento de 403/timeouts
- Paralelização opcional via --workers N (threads, I/O-bound)
- Log estruturado em JSON

USO

  # ─── RODADA 2 (com IDs novos do gerar_ids_rodada2.py) ───
  python baixar_random_pool_v2.py --ids-suffix rodada2 --workers 4

  # ─── RODADA 1 (compatibilidade reversa) ───
  python baixar_random_pool_v2.py --workers 4

  # Só um split
  python baixar_random_pool_v2.py --split train --workers 4 --ids-suffix rodada2

  # Sequencial seguro (fallback se paralelização falhar)
  python baixar_random_pool_v2.py --workers 1 --ids-suffix rodada2

  # Dry-run
  python baixar_random_pool_v2.py --dry-run --ids-suffix rodada2

MONITORAMENTO (em terminal separado)

  watch -n 30 'for s in train val test; do printf "%-8s %d\\n" "$s:" $(ls ~/PROJETO_MARINHA/random_pool_v2/$s/images/ 2>/dev/null | wc -l); done'

ESTRUTURA DE SAÍDA

  ~/PROJETO_MARINHA/random_pool_v2/
  ├── ids_random_train.txt              (rodada 1)
  ├── ids_random_val.txt                (rodada 1)
  ├── ids_random_test.txt               (rodada 1)
  ├── ids_random_rodada2_train.txt      (rodada 2, gerado pelo gerar_ids_rodada2.py)
  ├── ids_random_rodada2_val.txt        (rodada 2)
  ├── ids_random_rodada2_test.txt       (rodada 2)
  ├── train/
  │   ├── images/{id}.jpg               ← baixadas aqui
  │   └── labels_single_class/          ← preenchido depois
  ├── val/
  ├── test/
  ├── _corrompidas/                     ← quarentena de HTMLs detectados depois
  ├── download_progress.json            (progresso resumível, acumulativo)
  ├── download_report.json              (rodada 1)
  └── download_report_rodada2.json      (rodada 2, se --ids-suffix rodada2)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path

import requests

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════

BASE_DIR = Path.home() / "PROJETO_MARINHA" / "random_pool_v2"
PROGRESS_FILE = BASE_DIR / "download_progress.json"
REPORT_FILE = BASE_DIR / "download_report.json"

# Cookies do Chrome (usados pelo download_direto.py original)
BROWSER_DATA_DIR = Path.home() / "InaTechShips_similar" / "browser_data"

# Blacklist CLIP (preservada do original)
BLACKLIST_DIR = Path.home() / "InaTechShips_similar" / "blacklist"
CACHE_REF = Path.home() / "InaTechShips_similar" / "citra3d_embeddings.npz"
BLACKLIST_THRESHOLD = 0.90
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# HTTP config
MIN_IMG_BYTES = 5000
REQUEST_TIMEOUT = 15

# Rate limits — modo sequencial (workers=1)
DELAY_SEQ_MIN = 0.3
DELAY_SEQ_MAX = 1.0
BATCH_PAUSE_SEQ = 10  # pausa a cada N requests
BATCH_SIZE_SEQ = 100

# Rate limits — modo paralelo (workers >= 2)
# Mais conservador por thread; throughput total ainda é maior
DELAY_PAR_MIN = 0.5
DELAY_PAR_MAX = 1.5

SPLITS = ("train", "val", "test")

# Persistência de progresso thread-safe
_progress_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════
# URL do CDN do shipspotting
# ═══════════════════════════════════════════════════════════════════

def img_url(photo_id: int) -> str:
    """
    Constrói a URL direta da imagem no CDN do shipspotting.
    Padrão: /photos/big/{d1}/{d2}/{d3}/{id}.jpg
    d1, d2, d3 = últimos 3 dígitos do ID, invertidos.
    """
    s = str(photo_id)
    d1 = int(s[-1]) if len(s) >= 1 else 0
    d2 = int(s[-2]) if len(s) >= 2 else 0
    d3 = int(s[-3]) if len(s) >= 3 else 0
    return f"https://www.shipspotting.com/photos/big/{d1}/{d2}/{d3}/{photo_id}.jpg"


# Sanity checks (mesmo que download_direto.py original)
assert img_url(17) == "https://www.shipspotting.com/photos/big/7/1/0/17.jpg"
assert img_url(500) == "https://www.shipspotting.com/photos/big/0/0/5/500.jpg"
assert img_url(100007) == "https://www.shipspotting.com/photos/big/7/0/0/100007.jpg"


# ═══════════════════════════════════════════════════════════════════
# Sessão HTTP com cookies do Chrome
# ═══════════════════════════════════════════════════════════════════

def build_session() -> requests.Session:
    """
    Cria uma sessão HTTP com headers tipo-Chrome e cookies do Chrome
    carregados (se existirem).
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8",
        "Referer": "https://www.shipspotting.com/",
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "same-origin",
    })
    _load_chrome_cookies(session)
    return session


def _load_chrome_cookies(session: requests.Session) -> int:
    """Carrega cookies do Cloudflare da sessão do Chrome, se disponíveis."""
    cookie_db = BROWSER_DATA_DIR / "Default" / "Cookies"
    if not cookie_db.exists():
        cookie_db = BROWSER_DATA_DIR / "Cookies"
    if not cookie_db.exists():
        print(f"   ⚠️  Cookies do Chrome não encontrados em {BROWSER_DATA_DIR}")
        print("       Prosseguindo sem cookies — pode dar 403 se o Cloudflare bloquear.")
        return 0

    tmp = tempfile.mktemp(suffix=".db")
    try:
        shutil.copy2(cookie_db, tmp)
        conn = sqlite3.connect(tmp)
        cursor = conn.execute(
            "SELECT name, value, host_key FROM cookies "
            "WHERE host_key LIKE '%shipspotting%'"
        )
        count = 0
        for name, value, host in cursor.fetchall():
            session.cookies.set(name, value, domain=host)
            count += 1
        conn.close()
        os.remove(tmp)
        print(f"   🍪 {count} cookies carregados do Chrome.")
        return count
    except Exception as exc:
        print(f"   ⚠️  Erro lendo cookies: {exc}")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return 0


# ═══════════════════════════════════════════════════════════════════
# Blacklist CLIP (opcional)
# ═══════════════════════════════════════════════════════════════════

class BlacklistFilter:
    """
    Filtra imagens que são muito similares a uma blacklist conhecida
    (banners, ads, etc.) via embedding CLIP.

    Se PIL, torch, ou open_clip não estiverem disponíveis, ou se a
    blacklist estiver vazia, o filtro é no-op (aceita tudo).
    """

    def __init__(self):
        self.enabled = False
        self.blacklist_embeddings = None
        self.compute_embedding = None

        try:
            import numpy as np  # noqa: F401
            import torch
            import open_clip
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            print(f"   ⚠️  Blacklist desativada: faltam dependências ({exc})")
            return

        if not BLACKLIST_DIR.exists():
            print(f"   ⚠️  Blacklist desativada: {BLACKLIST_DIR} não existe")
            return

        bl_files = [
            p for p in BLACKLIST_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if not bl_files:
            print(f"   ℹ️  Blacklist vazia em {BLACKLIST_DIR} — filtro no-op")
            return

        print(f"   📦 Carregando CLIP ({CLIP_MODEL})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        clip_model = clip_model.to(device).eval()
        print(f"      {CLIP_MODEL} | {device}")

        from PIL import Image
        import numpy as np

        def _embed(pil_img):
            t = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                e = clip_model.encode_image(t)
            e = e / e.norm(dim=-1, keepdim=True)
            return e.cpu().numpy().flatten()

        # Computa embeddings da blacklist
        bl_embs = []
        for p in bl_files:
            try:
                img = Image.open(p).convert("RGB")
                bl_embs.append(_embed(img))
                print(f"      🚫 {p.name}")
            except Exception as exc:
                print(f"      ⚠️  Falha em {p.name}: {exc}")

        if not bl_embs:
            print("   ⚠️  Nenhum embedding de blacklist foi computado")
            return

        self.blacklist_embeddings = np.array(bl_embs)
        self.compute_embedding = _embed
        self.enabled = True
        print(f"   ✅ Blacklist ativa com {len(bl_embs)} entradas "
              f"(threshold={BLACKLIST_THRESHOLD})")

    def is_blacklisted(self, pil_img) -> bool:
        if not self.enabled:
            return False
        try:
            emb = self.compute_embedding(pil_img)
            max_sim = float((self.blacklist_embeddings @ emb).max())
            return max_sim >= BLACKLIST_THRESHOLD
        except Exception:
            return False  # erro → assume não blacklisted


# ═══════════════════════════════════════════════════════════════════
# Progresso persistente
# ═══════════════════════════════════════════════════════════════════

def load_progress() -> dict:
    """Carrega progresso do disco, se existir. Thread-safe."""
    with _progress_lock:
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, "r") as f:
                progress = json.load(f)
            # Backfill: se o JSON antigo (v1) não tem invalid_content, adiciona
            for split in SPLITS:
                if split in progress.get("counters", {}):
                    if "invalid_content" not in progress["counters"][split]:
                        progress["counters"][split]["invalid_content"] = 0
            return progress
        return {
            "started_at": datetime.now().isoformat(),
            "processed": {split: [] for split in SPLITS},
            "counters": {
                split: {
                    "ok": 0, "not_found": 0, "forbidden": 0,
                    "blacklisted": 0, "too_small": 0, "errors": 0,
                    "invalid_content": 0,
                }
                for split in SPLITS
            },
        }


def save_progress(progress: dict) -> None:
    """Salva progresso no disco. Thread-safe."""
    with _progress_lock:
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = PROGRESS_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(progress, f, indent=2)
        tmp.replace(PROGRESS_FILE)


def mark_processed(progress: dict, split: str, photo_id: int, outcome: str) -> None:
    """Marca um ID como processado e atualiza contador. Thread-safe."""
    with _progress_lock:
        progress["processed"][split].append(photo_id)
        if outcome in progress["counters"][split]:
            progress["counters"][split][outcome] += 1


# ═══════════════════════════════════════════════════════════════════
# Download de um único ID
# ═══════════════════════════════════════════════════════════════════

def download_one(
    session: requests.Session,
    photo_id: int,
    output_path: Path,
    blacklist: BlacklistFilter,
    delay_min: float,
    delay_max: float,
) -> str:
    """
    Baixa uma única imagem. Retorna um dos outcomes:
      'ok'              — imagem JPEG válida salva
      'not_found'       — HTTP 404 real (raro no shipspotting)
      'invalid_content' — HTTP 200 mas conteúdo NÃO é JPEG (HTML disfarçado)
      'forbidden'       — HTTP 403 mesmo após retry
      'blacklisted'     — passou no JPEG check mas similar à blacklist CLIP
      'too_small'       — JPEG válido mas < MIN_IMG_BYTES
      'errors'          — exceção de rede/timeout/IO
    """
    time.sleep(random.uniform(delay_min, delay_max))

    url = img_url(photo_id)
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)

        if r.status_code == 404:
            return "not_found"

        if r.status_code == 403:
            # Possível Cloudflare — espera e tenta de novo
            time.sleep(30)
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                return "forbidden"

        if r.status_code != 200:
            return "errors"

        if len(r.content) < MIN_IMG_BYTES:
            return "too_small"

        # ─── CHECK CRÍTICO v2: magic bytes JPEG ───
        # O shipspotting retorna HTTP 200 + HTML de erro para IDs inexistentes.
        # Verificamos os primeiros bytes do conteúdo: JPEG real começa com FF D8 FF.
        # Se não for JPEG, classificamos como invalid_content e NÃO salvamos
        # o arquivo no disco (evita criar HTML disfarçado de .jpg).
        if not r.content[:2] == b"\xff\xd8":
            return "invalid_content"
        # Sanity adicional: terceiro byte tipicamente também é FF, mas alguns
        # encoders usam outros valores válidos. O check de FF D8 nos dois primeiros
        # bytes é o suficiente para distinguir JPEG de HTML/texto.

        # Blacklist (se ativa) — só roda em conteúdo que já passou no JPEG check
        if blacklist.enabled:
            try:
                from PIL import Image
                pil = Image.open(BytesIO(r.content)).convert("RGB")
                if blacklist.is_blacklisted(pil):
                    return "blacklisted"
            except Exception:
                pass  # se falhou no PIL, salva mesmo assim

        # Salva imagem
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(r.content)
        return "ok"

    except requests.exceptions.RequestException:
        return "errors"
    except Exception:
        return "errors"


# ═══════════════════════════════════════════════════════════════════
# Download de um split inteiro
# ═══════════════════════════════════════════════════════════════════

def download_split(
    split: str,
    ids_file: Path,
    output_dir: Path,
    session: requests.Session,
    blacklist: BlacklistFilter,
    progress: dict,
    workers: int,
    save_every: int,
) -> None:
    """Baixa todas as imagens de um split, com paralelização opcional."""

    # Lê IDs
    ids_text = ids_file.read_text().strip().split("\n")
    all_ids = [int(x.strip()) for x in ids_text if x.strip()]

    # Filtra os já processados
    processed_set = set(progress["processed"][split])

    # Também considera como "já feito" qualquer imagem que já exista em disco
    # (pode ter sobrado de execução anterior que crashou antes de salvar progresso)
    existing_set = set()
    if output_dir.is_dir():
        for p in output_dir.iterdir():
            if p.suffix.lower() == ".jpg":
                try:
                    existing_set.add(int(p.stem))
                except ValueError:
                    pass

    pending = [i for i in all_ids if i not in processed_set and i not in existing_set]

    print(f"\n>> Split: {split}")
    print(f"   Arquivo IDs:     {ids_file}")
    print(f"   Output:          {output_dir}")
    print(f"   Total de IDs:    {len(all_ids):,}")
    print(f"   Já processados:  {len(processed_set):,}")
    print(f"   Já em disco:     {len(existing_set):,}")
    print(f"   Pendentes:       {len(pending):,}")

    if not pending:
        print("   ✅ Nada a fazer neste split.")
        return

    # Escolhe rate limit conforme modo
    if workers == 1:
        delay_min, delay_max = DELAY_SEQ_MIN, DELAY_SEQ_MAX
        print(f"   Modo:            sequencial (delay {delay_min}-{delay_max}s)")
    else:
        delay_min, delay_max = DELAY_PAR_MIN, DELAY_PAR_MAX
        print(f"   Modo:            paralelo com {workers} workers "
              f"(delay {delay_min}-{delay_max}s por thread)")

    counters = progress["counters"][split]
    t0 = time.time()
    done_since_save = 0

    def _task(photo_id: int) -> tuple[int, str]:
        output_path = output_dir / f"{photo_id}.jpg"
        outcome = download_one(
            session, photo_id, output_path, blacklist, delay_min, delay_max,
        )
        return (photo_id, outcome)

    def _print_progress(completed: int) -> None:
        elapsed = time.time() - t0
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = len(pending) - completed
        eta_sec = remaining / rate if rate > 0 else 0
        eta_min = eta_sec / 60
        print(
            f"   [{completed:>6,}/{len(pending):,}] "
            f"ok={counters['ok']:>6,} "
            f"inv={counters.get('invalid_content', 0):>5,} "
            f"nf={counters['not_found']:>4,} "
            f"bl={counters['blacklisted']:>4,} "
            f"err={counters['errors']:>4,} "
            f"| {rate:.1f}/s ETA {eta_min:.0f}min",
            flush=True,
        )

    completed = 0

    try:
        if workers == 1:
            # Modo sequencial
            for photo_id in pending:
                _, outcome = _task(photo_id)
                mark_processed(progress, split, photo_id, outcome)
                completed += 1
                done_since_save += 1

                if done_since_save >= save_every:
                    save_progress(progress)
                    done_since_save = 0

                if completed % 50 == 0:
                    _print_progress(completed)

                # Pausa entre lotes
                if completed % BATCH_SIZE_SEQ == 0:
                    time.sleep(BATCH_PAUSE_SEQ + random.uniform(0, 5))

        else:
            # Modo paralelo
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_task, pid): pid for pid in pending}
                for fut in as_completed(futures):
                    try:
                        photo_id, outcome = fut.result()
                        mark_processed(progress, split, photo_id, outcome)
                    except Exception:
                        pass
                    completed += 1
                    done_since_save += 1

                    if done_since_save >= save_every:
                        save_progress(progress)
                        done_since_save = 0

                    if completed % 50 == 0:
                        _print_progress(completed)

    except KeyboardInterrupt:
        print("\n   ⚠️  Interrompido pelo usuário, salvando progresso...")
        save_progress(progress)
        raise

    save_progress(progress)
    _print_progress(completed)
    print(f"   ✅ Split {split} concluído. Resumo:")
    for outcome, n in counters.items():
        print(f"      {outcome:<14} {n:>6,}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Download do random_pool_v2")
    parser.add_argument(
        "--split", choices=SPLITS + ("all",), default="all",
        help="Qual split baixar (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Número de workers para paralelização (1 = sequencial). Default: 4",
    )
    parser.add_argument(
        "--save-every", type=int, default=50,
        help="Salvar progresso a cada N downloads. Default: 50",
    )
    parser.add_argument(
        "--ids-suffix", type=str, default="",
        help="Sufixo para os arquivos de IDs. Vazio = ids_random_{split}.txt (rodada 1). "
             "'rodada2' = ids_random_rodada2_{split}.txt (rodada 2 com IDs novos).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Lista quantos IDs existem por split, sem baixar",
    )
    args = parser.parse_args()

    # Define o nome dos arquivos de IDs e do relatório com base no sufixo
    if args.ids_suffix:
        suffix_str = f"_{args.ids_suffix}"
        report_file = BASE_DIR / f"download_report_{args.ids_suffix}.json"
    else:
        suffix_str = ""
        report_file = REPORT_FILE

    print("=" * 70)
    print("  Download random_pool_v2 (v2 — com magic-bytes JPEG check)")
    print("=" * 70)
    print(f"  Base dir:        {BASE_DIR}")
    print(f"  Splits:          {args.split}")
    print(f"  Workers:         {args.workers}")
    print(f"  Save every:      {args.save_every} downloads")
    print(f"  IDs suffix:      '{args.ids_suffix}' "
          f"({'rodada 2' if args.ids_suffix == 'rodada2' else 'rodada 1 / default'})")
    print(f"  Report file:     {report_file.name}")
    print("=" * 70)

    if not BASE_DIR.exists():
        print(f"ERRO: {BASE_DIR} não existe")
        sys.exit(1)

    splits_to_run = SPLITS if args.split == "all" else (args.split,)

    # Verifica que os arquivos de IDs existem
    ids_files = {}
    for split in splits_to_run:
        ids_file = BASE_DIR / f"ids_random{suffix_str}_{split}.txt"
        if not ids_file.exists():
            print(f"ERRO: arquivo de IDs não encontrado: {ids_file}")
            if args.ids_suffix:
                print(f"      Você rodou `gerar_ids_rodada2.py` antes?")
            sys.exit(1)
        ids_files[split] = ids_file
        n = sum(1 for line in ids_file.read_text().split("\n") if line.strip())
        print(f"  {split}: {n:,} IDs em {ids_file.name}")

    if args.dry_run:
        print("\n>> --dry-run: nada será baixado")
        return

    # Cria estrutura de pastas
    for split in splits_to_run:
        (BASE_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (BASE_DIR / split / "labels_single_class").mkdir(parents=True, exist_ok=True)

    # Setup HTTP
    print("\n🍪 Configurando sessão HTTP...")
    session = build_session()

    # Teste rápido de conexão
    print("   Testando acesso...")
    try:
        r = session.get(img_url(17), timeout=10)
        if r.status_code == 200 and len(r.content) > MIN_IMG_BYTES:
            # Verifica também os magic bytes do teste
            if r.content[:2] == b"\xff\xd8":
                print(f"   ✅ Acesso OK! (ID 17: {len(r.content):,} bytes, JPEG válido)")
            else:
                print(f"   ⚠️  ID 17 retornou 200 + {len(r.content)} bytes mas NÃO é JPEG!")
                print("       Algo está errado com o CDN ou com cookies.")
        else:
            print(f"   ⚠️  Status {r.status_code}, {len(r.content)} bytes")
            print("       Pode haver problema com Cloudflare — prosseguindo assim mesmo.")
    except Exception as exc:
        print(f"   ⚠️  Erro no teste: {exc}")

    # Setup blacklist
    print("\n🚫 Configurando blacklist CLIP...")
    blacklist = BlacklistFilter()

    # Carrega progresso
    progress = load_progress()

    # Processa cada split
    t_start = time.time()
    try:
        for split in splits_to_run:
            output_dir = BASE_DIR / split / "images"
            download_split(
                split=split,
                ids_file=ids_files[split],
                output_dir=output_dir,
                session=session,
                blacklist=blacklist,
                progress=progress,
                workers=args.workers,
                save_every=args.save_every,
            )
    finally:
        save_progress(progress)

    total_elapsed = time.time() - t_start

    # Relatório final
    all_outcomes = ("ok", "not_found", "invalid_content", "forbidden",
                    "blacklisted", "too_small", "errors")
    report = {
        "generated_at": datetime.now().isoformat(),
        "ids_suffix": args.ids_suffix,
        "total_elapsed_seconds": total_elapsed,
        "total_elapsed_human": f"{total_elapsed/60:.1f} min",
        "workers": args.workers,
        "splits_run": list(splits_to_run),
        "counters": progress["counters"],
        "totals_by_outcome": {
            outcome: sum(progress["counters"][s].get(outcome, 0) for s in splits_to_run)
            for outcome in all_outcomes
        },
    }
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print()
    print("=" * 70)
    print("  RELATÓRIO FINAL")
    print("=" * 70)
    print(f"  Tempo total: {total_elapsed/60:.1f} min")
    print()
    print(f"  {'split':<8}{'ok':>8}{'404':>6}{'invalid':>9}{'403':>6}"
          f"{'bl':>5}{'small':>7}{'err':>6}")
    print("  " + "-" * 55)
    for split in splits_to_run:
        c = progress["counters"][split]
        print(
            f"  {split:<8}{c.get('ok', 0):>8,}{c.get('not_found', 0):>6,}"
            f"{c.get('invalid_content', 0):>9,}{c.get('forbidden', 0):>6,}"
            f"{c.get('blacklisted', 0):>5,}{c.get('too_small', 0):>7,}"
            f"{c.get('errors', 0):>6,}"
        )
    print()
    print(f"  Totais:")
    for k, v in report["totals_by_outcome"].items():
        print(f"    {k:<16} {v:>8,}")
    print()
    print(f"  Relatório salvo em: {report_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
