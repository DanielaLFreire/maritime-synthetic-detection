"""
analisar_escala_citra3d.py

Analisa a distribuição de tamanhos dos bounding boxes no CITRA-3D-Real
para calibrar o Scale-Aware Copy-Paste.

CONTEXTO

  O braço A do experimento mostrou que pré-treino direto no InaTechShips
  causa catastrophic forgetting (mAP50 = 0.7936 vs B2 = 0.8351). A
  principal diferença visual entre os datasets é a ESCALA: navios no
  InaTechShips são grandes e próximos (~80% da imagem), enquanto no
  CITRA-3D-Real são pequenos e distantes (~5-15% da imagem).

  Para implementar o Scale-Aware Copy-Paste, precisamos saber:
    1. Qual a distribuição de tamanhos relativos dos bboxes no CITRA-3D?
    2. Qual a distribuição de aspect ratios?
    3. Quantos objetos por imagem?
    4. Onde na imagem os objetos aparecem (distribuição espacial)?

  Este script extrai essas estatísticas dos labels YOLO do CITRA-3D-Real.

O QUE ESTE SCRIPT FAZ

  1. Lê todos os labels YOLO do CITRA-3D-Real (formato: class x_center
     y_center width height, coordenadas normalizadas 0-1).
  2. Calcula métricas por bbox: largura relativa, altura relativa, área
     relativa, aspect ratio, posição do centro.
  3. Gera estatísticas descritivas: percentis, média, mediana, DP.
  4. Salva relatório JSON com toda a análise.
  5. Opcionalmente gera histogramas (se matplotlib disponível).

PRÉ-REQUISITOS

  - CITRA-3D-Real com labels single-class em qualquer local acessível.
  - Pode rodar no Colab ou na máquina local.

USO

  python analisar_escala_citra3d.py
  python analisar_escala_citra3d.py --labels-dir /caminho/para/labels
  python analisar_escala_citra3d.py --plot   # gera histogramas PNG

SAÍDA

  escala_citra3d_report.json (no mesmo diretório do script ou --output)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# Configuração — caminhos padrão (ajustar conforme necessidade)
# ═══════════════════════════════════════════════════════════════════

# Tenta vários caminhos conhecidos do CITRA-3D-Real
DEFAULT_PATHS = [
    Path("/content/drive/MyDrive/PROJETO_MARINHA/Datasets/CITRA-3D-Real"),
    Path("/content/data/CITRA-3D-Real"),
    Path.home() / "PROJETO_MARINHA" / "CITRA-3D-Real",
]

SPLITS = ("train", "val", "test")


def find_citra3d_root() -> Path | None:
    """Procura o CITRA-3D-Real nos caminhos conhecidos."""
    for p in DEFAULT_PATHS:
        if p.exists() and (p / "train").exists():
            return p
    return None


def read_yolo_labels(labels_dir: Path) -> list[dict]:
    """
    Lê labels YOLO de um diretório.

    Formato YOLO: class_id x_center y_center width height
    (coordenadas normalizadas 0-1)

    Retorna lista de dicts com campos:
      class_id, x_center, y_center, width, height, image_id
    """
    bboxes = []
    if not labels_dir.exists():
        return bboxes

    for label_file in sorted(labels_dir.iterdir()):
        if label_file.suffix != ".txt":
            continue
        image_id = label_file.stem
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    bboxes.append({
                        "image_id": image_id,
                        "class_id": int(parts[0]),
                        "x_center": float(parts[1]),
                        "y_center": float(parts[2]),
                        "width": float(parts[3]),
                        "height": float(parts[4]),
                    })
                except (ValueError, IndexError):
                    continue
    return bboxes


def compute_percentiles(values: list[float], percentiles: list[int]) -> dict[str, float]:
    """Calcula percentis de uma lista de valores."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    result = {}
    for p in percentiles:
        idx = int(p / 100.0 * (n - 1))
        result[f"p{p}"] = round(sorted_vals[idx], 6)
    return result


def compute_stats(values: list[float]) -> dict:
    """Calcula estatísticas descritivas básicas."""
    if not values:
        return {}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 6),
        "median": round(statistics.median(values), 6),
        "stdev": round(statistics.stdev(values), 6) if len(values) > 1 else 0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        **compute_percentiles(values, [5, 10, 25, 50, 75, 90, 95]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Análise de escala do CITRA-3D-Real")
    parser.add_argument("--root", type=Path, default=None,
                        help="Root do CITRA-3D-Real (auto-detecta se não fornecido)")
    parser.add_argument("--labels-subfolder", default="labels_single_class",
                        help="Subfolder dos labels (default: labels_single_class)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Caminho do relatório JSON")
    parser.add_argument("--plot", action="store_true",
                        help="Gera histogramas PNG (requer matplotlib)")
    args = parser.parse_args()

    # ── Encontra o CITRA-3D-Real ──
    root = args.root
    if root is None:
        root = find_citra3d_root()
        if root is None:
            print("✗ CITRA-3D-Real não encontrado nos caminhos padrão.")
            print("  Use --root /caminho/para/CITRA-3D-Real")
            sys.exit(1)

    print("=" * 72)
    print("  Análise de escala dos bounding boxes — CITRA-3D-Real")
    print("=" * 72)
    print(f"  Root:    {root}")
    print(f"  Labels:  {args.labels_subfolder}")
    print("=" * 72)

    # ── Coleta todos os bboxes ──
    all_bboxes = []
    bboxes_per_split = {}

    for split in SPLITS:
        labels_dir = root / split / args.labels_subfolder
        if not labels_dir.exists():
            # Tenta "labels" como fallback
            labels_dir = root / split / "labels"
        bboxes = read_yolo_labels(labels_dir)
        bboxes_per_split[split] = bboxes
        all_bboxes.extend(bboxes)
        print(f"  {split}: {len(bboxes):,} bboxes de {labels_dir.name}/")

    print(f"\n  Total: {len(all_bboxes):,} bboxes")

    if not all_bboxes:
        print("\n✗ Nenhum bbox encontrado. Verifique o caminho e a subfolder.")
        sys.exit(1)

    # ── Análise de escala ──
    print(f"\n>> Análise de escala (coordenadas normalizadas 0-1)")

    widths = [b["width"] for b in all_bboxes]
    heights = [b["height"] for b in all_bboxes]
    areas = [b["width"] * b["height"] for b in all_bboxes]
    aspect_ratios = [b["width"] / b["height"] if b["height"] > 0 else 0 for b in all_bboxes]
    x_centers = [b["x_center"] for b in all_bboxes]
    y_centers = [b["y_center"] for b in all_bboxes]

    stats_width = compute_stats(widths)
    stats_height = compute_stats(heights)
    stats_area = compute_stats(areas)
    stats_ar = compute_stats(aspect_ratios)
    stats_xcenter = compute_stats(x_centers)
    stats_ycenter = compute_stats(y_centers)

    # ── Distribuição de objetos por imagem ──
    objects_per_image = defaultdict(int)
    for b in all_bboxes:
        objects_per_image[b["image_id"]] += 1

    n_images = len(objects_per_image)
    objs_counts = list(objects_per_image.values())
    stats_objs = compute_stats(objs_counts)

    # ── Classificação por tamanho (COCO-style) ──
    # COCO define: small < 32², medium < 96², large >= 96² (em pixels para 640×640)
    # Em coordenadas normalizadas para 640×640:
    #   small: area_norm < (32/640)² = 0.0025
    #   medium: area_norm < (96/640)² = 0.0225
    #   large: area_norm >= 0.0225
    small_threshold = (32 / 640) ** 2   # 0.0025
    medium_threshold = (96 / 640) ** 2  # 0.0225

    n_small = sum(1 for a in areas if a < small_threshold)
    n_medium = sum(1 for a in areas if small_threshold <= a < medium_threshold)
    n_large = sum(1 for a in areas if a >= medium_threshold)

    # ── Output ──
    print(f"\n   {'Métrica':<25}{'Média':>10}{'Mediana':>10}{'P10':>10}{'P90':>10}{'Min':>10}{'Max':>10}")
    print(f"   {'-'*75}")
    for name, stats in [("Largura (norm)", stats_width),
                        ("Altura (norm)", stats_height),
                        ("Área (norm)", stats_area),
                        ("Aspect ratio (W/H)", stats_ar),
                        ("Centro X (norm)", stats_xcenter),
                        ("Centro Y (norm)", stats_ycenter),
                        ("Objetos/imagem", stats_objs)]:
        print(f"   {name:<25}{stats['mean']:>10.4f}{stats['median']:>10.4f}"
              f"{stats['p10']:>10.4f}{stats['p90']:>10.4f}"
              f"{stats['min']:>10.4f}{stats['max']:>10.4f}")

    # Tamanhos em pixels (assumindo 640×640)
    print(f"\n>> Tamanhos em pixels (para imgsz=640)")
    print(f"   Largura média: {stats_width['mean']*640:.1f} px "
          f"(P10={stats_width['p10']*640:.1f}, P90={stats_width['p90']*640:.1f})")
    print(f"   Altura média:  {stats_height['mean']*640:.1f} px "
          f"(P10={stats_height['p10']*640:.1f}, P90={stats_height['p90']*640:.1f})")
    print(f"   Área média:    {stats_area['mean']*640*640:.1f} px² "
          f"({stats_area['mean']*100:.2f}% da imagem)")

    print(f"\n>> Distribuição COCO-style (para imgsz=640)")
    print(f"   Small  (< 32²px):    {n_small:>5,} ({n_small/len(areas)*100:.1f}%)")
    print(f"   Medium (32²-96²px):  {n_medium:>5,} ({n_medium/len(areas)*100:.1f}%)")
    print(f"   Large  (> 96²px):    {n_large:>5,} ({n_large/len(areas)*100:.1f}%)")

    print(f"\n>> Objetos por imagem")
    print(f"   Média:   {stats_objs['mean']:.2f}")
    print(f"   Mediana: {stats_objs['median']:.1f}")
    print(f"   Min:     {stats_objs['min']:.0f}")
    print(f"   Max:     {stats_objs['max']:.0f}")
    print(f"   P10:     {stats_objs['p10']:.1f}")
    print(f"   P90:     {stats_objs['p90']:.1f}")

    # ── Relatório JSON ──
    report = {
        "generated_at": datetime.now().isoformat(),
        "root": str(root),
        "labels_subfolder": args.labels_subfolder,
        "n_bboxes_total": len(all_bboxes),
        "n_images": n_images,
        "bboxes_per_split": {s: len(bboxes_per_split[s]) for s in SPLITS},
        "scale_analysis": {
            "width_normalized": stats_width,
            "height_normalized": stats_height,
            "area_normalized": stats_area,
            "aspect_ratio": stats_ar,
            "x_center": stats_xcenter,
            "y_center": stats_ycenter,
        },
        "objects_per_image": stats_objs,
        "coco_size_distribution": {
            "small_count": n_small,
            "small_pct": round(n_small / len(areas) * 100, 1),
            "medium_count": n_medium,
            "medium_pct": round(n_medium / len(areas) * 100, 1),
            "large_count": n_large,
            "large_pct": round(n_large / len(areas) * 100, 1),
            "thresholds_note": "COCO-style para imgsz=640: small < 32²px, medium < 96²px, large >= 96²px",
        },
        "copy_paste_recommendations": {
            "target_width_range": [
                round(stats_width["p10"], 4),
                round(stats_width["p90"], 4),
            ],
            "target_height_range": [
                round(stats_height["p10"], 4),
                round(stats_height["p90"], 4),
            ],
            "target_aspect_ratio_range": [
                round(stats_ar["p10"], 4),
                round(stats_ar["p90"], 4),
            ],
            "target_objects_per_image_range": [
                int(stats_objs["p10"]),
                int(stats_objs["p90"]),
            ],
            "target_y_center_range": [
                round(stats_ycenter["p10"], 4),
                round(stats_ycenter["p90"], 4),
            ],
            "note": "Usar estes ranges como parâmetros do copy-paste para gerar "
                    "imagens com distribuição de escala similar ao CITRA-3D-Real.",
        },
    }

    output_path = args.output
    if output_path is None:
        output_path = root / "escala_citra3d_report.json"

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n   ✓ Relatório: {output_path}")

    # ── Histogramas (opcional) ──
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(15, 9))
            fig.suptitle("Distribuição de escala — CITRA-3D-Real", fontsize=14)

            axes[0, 0].hist(widths, bins=50, color="#534AB7", alpha=0.8, edgecolor="white")
            axes[0, 0].set_title("Largura normalizada")
            axes[0, 0].set_xlabel("width (0-1)")
            axes[0, 0].axvline(stats_width["median"], color="red", linestyle="--", label=f"mediana={stats_width['median']:.3f}")
            axes[0, 0].legend(fontsize=8)

            axes[0, 1].hist(heights, bins=50, color="#1D9E75", alpha=0.8, edgecolor="white")
            axes[0, 1].set_title("Altura normalizada")
            axes[0, 1].set_xlabel("height (0-1)")
            axes[0, 1].axvline(stats_height["median"], color="red", linestyle="--", label=f"mediana={stats_height['median']:.3f}")
            axes[0, 1].legend(fontsize=8)

            axes[0, 2].hist(areas, bins=50, color="#D85A30", alpha=0.8, edgecolor="white")
            axes[0, 2].set_title("Área normalizada")
            axes[0, 2].set_xlabel("area (0-1)")
            axes[0, 2].axvline(stats_area["median"], color="red", linestyle="--", label=f"mediana={stats_area['median']:.4f}")
            axes[0, 2].legend(fontsize=8)

            axes[1, 0].hist(aspect_ratios, bins=50, color="#D4537E", alpha=0.8, edgecolor="white")
            axes[1, 0].set_title("Aspect ratio (W/H)")
            axes[1, 0].set_xlabel("ratio")
            axes[1, 0].axvline(stats_ar["median"], color="red", linestyle="--", label=f"mediana={stats_ar['median']:.2f}")
            axes[1, 0].legend(fontsize=8)

            axes[1, 1].hist(objs_counts, bins=range(0, max(objs_counts) + 2),
                           color="#888780", alpha=0.8, edgecolor="white", align="left")
            axes[1, 1].set_title("Objetos por imagem")
            axes[1, 1].set_xlabel("n_objetos")

            # Scatter: posição dos centros
            sample = all_bboxes[:2000]  # amostra para legibilidade
            axes[1, 2].scatter([b["x_center"] for b in sample],
                              [b["y_center"] for b in sample],
                              s=3, alpha=0.3, color="#378ADD")
            axes[1, 2].set_title("Posição dos centros (amostra)")
            axes[1, 2].set_xlabel("x_center")
            axes[1, 2].set_ylabel("y_center")
            axes[1, 2].invert_yaxis()
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].set_ylim(1, 0)

            plt.tight_layout()
            plot_path = output_path.with_suffix(".png")
            plt.savefig(plot_path, dpi=150)
            print(f"   ✓ Histogramas: {plot_path}")
            plt.close()

        except ImportError:
            print("   ⚠ matplotlib não disponível, histogramas não gerados")

    print(f"\n{'='*72}")
    print(f"  ANÁLISE CONCLUÍDA")
    print(f"  Próximo passo: usar os ranges de copy_paste_recommendations")
    print(f"  no script de geração de imagens sintéticas.")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
