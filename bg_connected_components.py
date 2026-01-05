import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


if "__file__" in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASE_DIR = os.getcwd()

print("BASE_DIR =", BASE_DIR)


# DATASET YOLLARI (BG)

CSV_PATH = os.path.join(BASE_DIR, "dataset", "coins_count_values.csv")
IMG_DIR  = os.path.join(BASE_DIR, "dataset", "coins_images", "coins_images", "all_coins")

OUT_ROOT = os.path.join(BASE_DIR, "outputs")
OUT_OVER = os.path.join(OUT_ROOT, "overlays")
OUT_TAB  = os.path.join(OUT_ROOT, "tables")
OUT_FIG  = os.path.join(OUT_ROOT, "figures")

os.makedirs(OUT_OVER, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

# AYARLAR

MIN_AREA = 120                    
SHOW_ONE_EXAMPLE = False          
ENABLE_INFOGRAPHIC_FILTER = True  

# Deneyler: Otsu/Adaptive + kernel 3/5
experiments = [
    {"name": "otsu_k3", "use_adaptive": False, "k": 3},
    {"name": "otsu_k5", "use_adaptive": False, "k": 5},
    {"name": "adp_k3",  "use_adaptive": True,  "k": 3},
    {"name": "adp_k5",  "use_adaptive": True,  "k": 5},
]

# GT OKU (coins_count_values.csv)
# columns: folder,image_name,coins_count

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError("CSV bulunamadı: " + CSV_PATH)

GT = {}
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        GT[row["image_name"]] = int(row["coins_count"])

img_names = list(GT.keys())


all_rows = []
summary_abs = {e["name"]: [] for e in experiments}
summary_acc = {e["name"]: [] for e in experiments}
summary_n   = {e["name"]: 0  for e in experiments}

skipped_infographic = 0
missing_files = 0


saved_report_figs = False

# ------------------------------------------------
# ANA DÖNGÜ

for exp in experiments:

    exp_name = exp["name"]
    use_adaptive = exp["use_adaptive"]
    k = exp["k"]

    exp_out_dir = os.path.join(OUT_OVER, exp_name)
    os.makedirs(exp_out_dir, exist_ok=True)

    for fname in img_names:

        img_path = os.path.join(IMG_DIR, fname)
        original = cv2.imread(img_path)

        if original is None:
            missing_files += 1
            continue

        gt_count = GT.get(fname, -1)

        # --- GRAY & RGB ---
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        rgb  = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # --- THRESHOLD ---
        if use_adaptive:
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                51, 2
            )
            method = "adaptive"
        else:
            _, binary = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            method = "otsu"

        # --- MORFOLOJİ ---
        kernel = np.ones((k, k), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

        # --- INFOGRAFIK/METIN FILTRE ---
        if ENABLE_INFOGRAPHIC_FILTER:
            n_tmp, _, _, _ = cv2.connectedComponentsWithStats(closing, 8)
            if (n_tmp - 1) > 200:
                skipped_infographic += 1
                continue

        # --- CONNECTED COMPONENTS ---
        num, labels, stats, _ = cv2.connectedComponentsWithStats(closing, 8)

        components = []
        for lab in range(1, num):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area >= MIN_AREA:
                x = int(stats[lab, cv2.CC_STAT_LEFT])
                y = int(stats[lab, cv2.CC_STAT_TOP])
                w = int(stats[lab, cv2.CC_STAT_WIDTH])
                h = int(stats[lab, cv2.CC_STAT_HEIGHT])
                components.append((lab, x, y, w, h, area))

        pred_count = len(components)

        # --- BASIT HATA SINIFLANDIRMA (alan-temelli) ---
        areas = [c[5] for c in components]
        med = np.median(areas) if len(areas) else 1.0
        small_th = 0.60 * med
        large_th = 1.55 * med

        cls_counts = {"small": 0, "normal": 0, "large_stuck": 0}
        overlay = rgb.copy()

        for lab, x, y, w, h, area in components:

            # dairesellik (opsiyonel bilgi)
            mask = (labels == lab).astype(np.uint8)
            cnts, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) > 0:
                cmax = max(cnts, key=cv2.contourArea)
                p = cv2.arcLength(cmax, True) + 1e-6
                circ = float(4 * np.pi * area / (p * p))
            else:
                circ = 0.0

            if area < small_th:
                cls = "small"
                color = (255, 0, 0)         # kırmızı
            elif area > large_th:
                cls = "large_stuck"
                color = (255, 255, 0)       # sarı
            else:
                cls = "normal"
                color = (0, 255, 0)         # yeşil

            cls_counts[cls] += 1

            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                overlay, f"{cls} A={area} C={circ:.2f}",
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

        # --- METRIKLER ---
        if gt_count > 0:
            abs_err = abs(pred_count - gt_count)
            acc = 1.0 - (abs_err / gt_count)
        else:
            abs_err = -1
            acc = -1

        summary_abs[exp_name].append(abs_err)
        summary_acc[exp_name].append(acc)
        summary_n[exp_name] += 1

        # --- OVERLAY KAYDET ---
        cv2.imwrite(
            os.path.join(exp_out_dir, fname),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

        # --- results.csv satırı ---
        all_rows.append({
            "experiment": exp_name,
            "image": fname,
            "method": method,
            "kernel": k,
            "pred_count": pred_count,
            "gt_count": gt_count,
            "abs_error": abs_err,
            "count_accuracy": acc,
            "small": cls_counts["small"],
            "normal": cls_counts["normal"],
            "large_stuck": cls_counts["large_stuck"],
        })

  
        if not saved_report_figs:
           

            cv2.imwrite(os.path.join(OUT_FIG, "fig_original.png"), original)
            cv2.imwrite(os.path.join(OUT_FIG, "fig_binary.png"), binary)
            cv2.imwrite(os.path.join(OUT_FIG, "fig_closing.png"), closing)
            cv2.imwrite(
                os.path.join(OUT_FIG, "fig_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            )

         
            if SHOW_ONE_EXAMPLE:
                plt.figure(figsize=(14, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(rgb)
                plt.title("Original (RGB)")
                plt.axis("off")

                plt.subplot(1, 2, 2)
                plt.imshow(overlay)
                plt.title("Overlay Result")
                plt.axis("off")
                plt.show()

            saved_report_figs = True


# CSV ÇIKTILARI

if len(all_rows) == 0:
    raise RuntimeError(
        "Hiç sonuç üretilmedi. Muhtemelen IMG_DIR yolu yanlış veya filtre çok sert.\n"
        "Kontrol et: " + IMG_DIR
    )

results_csv = os.path.join(OUT_TAB, "results.csv")
with open(results_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    writer.writeheader()
    writer.writerows(all_rows)

summary_rows = []
for e in experiments:
    name = e["name"]
    mae = float(np.mean(summary_abs[name])) if len(summary_abs[name]) else -1
    mean_acc = float(np.mean(summary_acc[name])) if len(summary_acc[name]) else -1
    summary_rows.append({
        "experiment": name,
        "method": "adaptive" if e["use_adaptive"] else "otsu",
        "kernel": e["k"],
        "n_images_used": summary_n[name],
        "MAE": mae,
        "mean_count_accuracy": mean_acc
    })

summary_csv = os.path.join(OUT_TAB, "summary.csv")
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

print("\nBİTTİ ✅")
print("Eksik dosya uyarısı (imread okuyamadı):", missing_files)
print("Atlanan infografik görüntü:", skipped_infographic)
print("Overlay klasörü:", OUT_OVER)
print("Figür klasörü:", OUT_FIG)
print("results.csv:", results_csv)
print("summary.csv:", summary_csv)
