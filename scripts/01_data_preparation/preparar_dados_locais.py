"""
preparar_dados_locais.py

SUBSTITUI gerar_data_yaml.py.

Por que?
  A versão anterior usava links simbólicos no /content/data/ apontando
  para o Drive. Isso parecia uma boa ideia (zero duplicação, idempotente),
  mas o Ultralytics YOLO descobriu-se ter um bug com symlinks: ao seguir
  o link de `images/`, ele resolve o caminho real (Path.resolve()) ANTES
  de aplicar a substituição /images/→/labels/. Resultado: ele acaba
  lendo os labels do path REAL no Drive (que é a pasta `labels/` com as
  9 classes originais), em vez dos `labels_single_class/` apontados
  pelo link.

  O sintoma é silencioso e devastador: o Ultralytics descarta TODAS as
  imagens do treino com a mensagem "ignoring corrupt image/label: Label
  class N exceeds dataset class count 1". O treino roda em zero imagens,
  produz mAP aleatório, e a gente perde várias horas debugando.

  A solução é cópia física para o disco local do Colab. Vantagens:
    1. Resolve o bug do Ultralytics definitivamente.
    2. I/O LOCAL é 5-10x mais rápido que o Drive durante o treino —
       cada época vai ser significativamente mais curta.
    3. Continua sendo idempotente (re-rodadas só copiam o que falta).

  Custo: ~5 minutos para CITRA (2 GB) e ~15-25 minutos para dataset_25k
  (10-15 GB). Como o /content/ é apagado entre sessões, é necessário
  rodar este script no início de cada nova sessão do Colab.

  Os yamls em si são gerados no Drive e são persistentes entre sessões.

USO

  python preparar_dados_locais.py                    # ambos os datasets
  python preparar_dados_locais.py --only citra       # só CITRA-3D-Real
  python preparar_dados_locais.py --only inatech     # só dataset_25k
  python preparar_dados_locais.py --force            # recopiar mesmo se existir

SAÍDAS

  Persistentes (Drive):
    /content/drive/.../Datasets/configs/citra3d_single_class.yaml
    /content/drive/.../Datasets/configs/dataset_25k_single_class.yaml
    /content/drive/.../Datasets/configs/data_setup_log.json

  Voláteis (Colab local, refazer por sessão):
    /content/data/CITRA-3D-Real-SC/{train,val,test}/{images,labels}/
    /content/data/dataset_25k-SC/{train,val,test}/{images,labels}/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

# Datasets de origem (Drive)
CITRA_DRIVE_ROOT = Path("/content/drive/MyDrive/PROJETO_MARINHA/Datasets/CITRA-3D-Real")
INATECH_DRIVE_ROOT = Path("/content/drive/MyDrive/InaTechShips/dataset_25k")

# Subpasta de origem dos labels (já colapsados para classe única)
CITRA_LABELS_SOURCE = "labels_single_class"
INATECH_LABELS_SOURCE = "labels_single_class"

# Estrutura local no Colab (cópia física, não link)
LOCAL_DATA_ROOT = Path("/content/data")
CITRA_LOCAL_ROOT = LOCAL_DATA_ROOT / "CITRA-3D-Real-SC"
INATECH_LOCAL_ROOT = LOCAL_DATA_ROOT / "dataset_25k-SC"

# Onde salvar os yamls (Drive, persistente)
CONFIGS_DIR = Path("/content/drive/MyDrive/PROJETO_MARINHA/Datasets/configs")

CITRA_YAML_NAME = "citra3d_single_class.yaml"
INATECH_YAML_NAME = "dataset_25k_single_class.yaml"

SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def fmt_seconds(n: float) -> str:
    if n < 60:
        return f"{n:.0f}s"
    return f"{n/60:.1f}min"


def remove_if_link_or_dir(path: Path) -> None:
    """Remove se for link, arquivo ou diretório."""
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def count_files(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.iterdir())


def needs_copy(src: Path, dst: Path) -> bool:
    """
    True se a cópia precisa ser feita (não existe ainda, ou está incompleta).
    Comparação leve: se número de arquivos bater, assume que está OK.
    """
    if not dst.exists() or not dst.is_dir():
        return True
    if dst.is_symlink():
        return True  # link → precisa virar cópia
    n_src = count_files(src)
    n_dst = count_files(dst)
    return n_src != n_dst


def copy_dir_files(src: Path, dst: Path) -> tuple[int, int]:
    """
    Copia todos os arquivos de src para dst (não-recursivo).
    Retorna (n_arquivos, total_bytes).
    """
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    total_bytes = 0
    for item in src.iterdir():
        if item.is_file():
            target = dst / item.name
            shutil.copy2(item, target)
            n += 1
            try:
                total_bytes += item.stat().st_size
            except OSError:
                pass
    return n, total_bytes


# ---------------------------------------------------------------------------
# Cópia de um dataset
# ---------------------------------------------------------------------------

def copy_dataset(
    drive_root: Path,
    local_root: Path,
    labels_source: str,
    name: str,
    force: bool,
) -> dict:
    """
    Copia images/ e labels_single_class/ do Drive para o local,
    renomeando labels_single_class → labels.
    """
    print(f"\n>> Preparando dataset: {name}")
    print(f"   origem (Drive):  {drive_root}")
    print(f"   destino (local): {local_root}")
    print(f"   labels source:   {labels_source}/ → labels/")

    result = {
        "name": name,
        "ok": True,
        "splits": {},
        "errors": [],
    }

    if not drive_root.exists():
        msg = f"drive root não existe: {drive_root}"
        result["ok"] = False
        result["errors"].append(msg)
        print(f"   ERRO: {msg}")
        return result

    grand_total_files = 0
    grand_total_bytes = 0
    grand_total_seconds = 0.0

    for split in SPLITS:
        src_images = drive_root / split / "images"
        src_labels = drive_root / split / labels_source

        if not src_images.exists():
            msg = f"[{split}] src images não existe: {src_images}"
            result["ok"] = False
            result["errors"].append(msg)
            print(f"   ERRO: {msg}")
            continue
        if not src_labels.exists():
            msg = f"[{split}] src labels não existe: {src_labels}"
            result["ok"] = False
            result["errors"].append(msg)
            print(f"   ERRO: {msg}")
            continue

        dst_split_dir = local_root / split
        dst_images = dst_split_dir / "images"
        dst_labels = dst_split_dir / "labels"

        # Limpa qualquer link simbólico ou cópia parcial antiga
        if force or dst_images.is_symlink() or dst_labels.is_symlink():
            remove_if_link_or_dir(dst_images)
            remove_if_link_or_dir(dst_labels)

        split_stat = {
            "src_images": str(src_images),
            "src_labels": str(src_labels),
            "dst_images": str(dst_images),
            "dst_labels": str(dst_labels),
        }

        # Imagens
        if needs_copy(src_images, dst_images):
            print(f"   [{split}] copiando imagens...", flush=True)
            t0 = time.time()
            n, b = copy_dir_files(src_images, dst_images)
            elapsed = time.time() - t0
            split_stat["images_copied"] = n
            split_stat["images_bytes"] = b
            split_stat["images_time_seconds"] = elapsed
            grand_total_files += n
            grand_total_bytes += b
            grand_total_seconds += elapsed
            print(f"      {n} arquivos, {fmt_bytes(b)} em {fmt_seconds(elapsed)}")
        else:
            n = count_files(dst_images)
            split_stat["images_copied"] = 0
            split_stat["images_already_present"] = n
            print(f"   [{split}] imagens já presentes ({n} arquivos), pulando")

        # Labels
        if needs_copy(src_labels, dst_labels):
            print(f"   [{split}] copiando labels...", flush=True)
            t0 = time.time()
            n, b = copy_dir_files(src_labels, dst_labels)
            elapsed = time.time() - t0
            split_stat["labels_copied"] = n
            split_stat["labels_bytes"] = b
            split_stat["labels_time_seconds"] = elapsed
            grand_total_files += n
            grand_total_bytes += b
            grand_total_seconds += elapsed
            print(f"      {n} arquivos, {fmt_bytes(b)} em {fmt_seconds(elapsed)}")
        else:
            n = count_files(dst_labels)
            split_stat["labels_copied"] = 0
            split_stat["labels_already_present"] = n
            print(f"   [{split}] labels já presentes ({n} arquivos), pulando")

        # Sanity check de pareamento
        n_imgs_final = count_files(dst_images)
        n_lbls_final = count_files(dst_labels)
        split_stat["final_n_images"] = n_imgs_final
        split_stat["final_n_labels"] = n_lbls_final
        if n_imgs_final != n_lbls_final:
            msg = (f"[{split}] desbalanço pós-cópia: "
                   f"{n_imgs_final} imgs vs {n_lbls_final} labels")
            result["errors"].append(msg)
            result["ok"] = False
            print(f"   AVISO: {msg}")

        result["splits"][split] = split_stat

    result["grand_total_files"] = grand_total_files
    result["grand_total_bytes"] = grand_total_bytes
    result["grand_total_time_seconds"] = grand_total_seconds
    print(f"\n   Total {name}: {grand_total_files} arquivos, "
          f"{fmt_bytes(grand_total_bytes)} em {fmt_seconds(grand_total_seconds)}")

    return result


# ---------------------------------------------------------------------------
# Geração e validação dos yamls
# ---------------------------------------------------------------------------

YAML_TEMPLATE = """\
# {comment}
# Gerado automaticamente por preparar_dados_locais.py
#
# IMPORTANTE: este yaml aponta para uma estrutura de COPIAS FISICAS em
# /content/data/ do Colab. O /content/ é apagado entre sessoes — em toda
# nova sessao do Colab é necessario rodar `preparar_dados_locais.py`
# novamente para recopiar. O yaml em si nao precisa ser regenerado.
#
# Por que copia fisica e nao link simbolico?
#   O Ultralytics YOLO tem um bug com symlinks: ao seguir o link de
#   images/, ele resolve o caminho real (Path.resolve) antes de aplicar
#   a substituicao /images/→/labels/. Resultado: acaba lendo os labels
#   originais (9 classes) em vez de labels_single_class (1 classe), e
#   descarta TODAS as imagens do treino com a mensagem "Label class N
#   exceeds dataset class count 1". Bug silencioso.

path: {path}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['embarcacao']
"""


def generate_yaml(local_root: Path, comment: str, dst_path: Path) -> str:
    content = YAML_TEMPLATE.format(comment=comment, path=str(local_root))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(content, encoding="utf-8")
    return content


def validate_yaml(yaml_path: Path) -> tuple[str, dict]:
    """
    Validação leve do yaml + verificação anti-symlink.
    Retorna (status, info).
    """
    info: dict = {}
    try:
        import yaml
    except ImportError:
        return "skipped (PyYAML não instalado)", info

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        return f"FAIL: parse error: {exc}", info

    required = {"path", "train", "val", "test", "nc", "names"}
    missing = required - set(data.keys())
    if missing:
        return f"FAIL: campos faltando: {missing}", info

    if data["nc"] != 1:
        return f"FAIL: nc esperado=1, obtido={data['nc']}", info

    if data["names"] != ["embarcacao"]:
        return f"FAIL: names esperado=['embarcacao'], obtido={data['names']}", info

    base = Path(data["path"])
    if not base.is_dir():
        return f"FAIL: path não existe: {base}", info

    for split in SPLITS:
        split_rel = data[split]
        images_dir = base / split_rel
        if not images_dir.exists():
            return f"FAIL: {split} images não existe: {images_dir}", info

        # ANTI-SYMLINK CHECK: garantir que NÃO é link
        if images_dir.is_symlink():
            return f"FAIL: {split} images é symlink (vai quebrar Ultralytics): {images_dir}", info

        labels_dir = Path(str(images_dir).replace("/images", "/labels"))
        if not labels_dir.exists():
            return f"FAIL: {split} labels não existe: {labels_dir}", info
        if labels_dir.is_symlink():
            return f"FAIL: {split} labels é symlink (vai quebrar Ultralytics): {labels_dir}", info

        # Conferir contagens
        n_imgs = sum(1 for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        n_lbls = sum(1 for p in labels_dir.iterdir() if p.suffix.lower() == ".txt")
        info[split] = {"n_images": n_imgs, "n_labels": n_lbls}
        if n_imgs != n_lbls:
            return f"FAIL: {split} desbalanço: {n_imgs} imgs vs {n_lbls} labels", info
        if n_imgs == 0:
            return f"FAIL: {split} está vazio", info

        # Sanity check no conteúdo de UM label aleatório:
        # garantir que está com classe 0
        for lbl_file in labels_dir.iterdir():
            if lbl_file.suffix == ".txt":
                first_line = lbl_file.read_text().strip().split("\n")[0] if lbl_file.read_text().strip() else ""
                if first_line:
                    first_class = first_line.split()[0] if first_line.split() else ""
                    if first_class != "0":
                        return (f"FAIL: {split} label {lbl_file.name} tem classe '{first_class}' "
                                f"em vez de '0' — labels NÃO são single-class!"), info
                break  # só checa um por split

    return "OK", info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Preparar dados locais (cópia física)")
    parser.add_argument(
        "--only", choices=["citra", "inatech"],
        help="Copiar apenas um dos datasets",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Forçar recópia mesmo se as pastas já existirem (apaga e recopia)",
    )
    args = parser.parse_args()

    print(">> preparar_dados_locais.py")
    print(f"   destino local:  {LOCAL_DATA_ROOT}")
    print(f"   yamls (Drive):  {CONFIGS_DIR}")
    print(f"   force:          {args.force}")
    print()

    LOCAL_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    setup_results = {}

    # Dataset 1: CITRA-3D-Real
    if args.only != "inatech":
        setup_results["citra"] = copy_dataset(
            CITRA_DRIVE_ROOT, CITRA_LOCAL_ROOT, CITRA_LABELS_SOURCE,
            "CITRA-3D-Real", args.force,
        )

    # Dataset 2: dataset_25k
    if args.only != "citra":
        setup_results["inatech"] = copy_dataset(
            INATECH_DRIVE_ROOT, INATECH_LOCAL_ROOT, INATECH_LABELS_SOURCE,
            "dataset_25k", args.force,
        )

    # Geração dos yamls
    print(f"\n>> Gerando arquivos data.yaml em {CONFIGS_DIR}")

    yamls_generated = {}
    if args.only != "inatech":
        citra_yaml = CONFIGS_DIR / CITRA_YAML_NAME
        generate_yaml(
            CITRA_LOCAL_ROOT,
            "CITRA-3D-Real — single class (embarcacao) — dominio operacional alvo",
            citra_yaml,
        )
        yamls_generated["citra"] = str(citra_yaml)
        print(f"   {citra_yaml}")

    if args.only != "citra":
        inatech_yaml = CONFIGS_DIR / INATECH_YAML_NAME
        generate_yaml(
            INATECH_LOCAL_ROOT,
            "dataset_25k (InaTechShips curado por CLIP) — single class — subset A",
            inatech_yaml,
        )
        yamls_generated["inatech"] = str(inatech_yaml)
        print(f"   {inatech_yaml}")

    # Validação anti-symlink
    print("\n>> Validando yamls (parsing + paths + ANTI-SYMLINK + sample label)")

    validations = {}
    for key, yaml_path_str in yamls_generated.items():
        status, info = validate_yaml(Path(yaml_path_str))
        validations[key] = {"status": status, "info": info}
        marker = "✓" if status == "OK" else "✗"
        print(f"   {marker} {Path(yaml_path_str).name}: {status}")
        for split, counts in info.items():
            print(f"        {split}: {counts['n_images']} imgs, {counts['n_labels']} labels")

    # Salva log
    log = {
        "generated_at": datetime.now().isoformat(),
        "setup": setup_results,
        "yamls": yamls_generated,
        "validations": validations,
    }
    log_path = CONFIGS_DIR / "data_setup_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))

    # Resumo final
    print()
    print("=" * 72)
    print("RESUMO")
    print("=" * 72)

    overall_ok = (
        all(r["ok"] for r in setup_results.values())
        and all(v["status"] == "OK" for v in validations.values())
    )
    print(f"Status geral: {'✓ TUDO OK' if overall_ok else '✗ HÁ PROBLEMAS'}")
    print()
    print(f"Log salvo em: {log_path}")
    print()
    print("Como usar nos treinos:")
    print("  !python treinar_baselines.py --baseline B2 --seed 42")
    print()
    print("ATENÇÃO: ao iniciar uma nova sessão do Colab, rode este script")
    print("novamente para recopiar os dados (o /content/ é volátil).")
    print("Os yamls em si são persistentes no Drive.")


if __name__ == "__main__":
    main()
