from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "Pillow is required to run this script. Install it with `pip install pillow`." 
        + f"\nOriginal error: {exc}"
    )

SOURCE_FILES = [
    Path("figures/bifurcation_mu.png"),
    Path("figures/bifurcation_beta_mu04.png"),
    Path("figures/bifurcation_W1_mu04.png"),
    Path("figures/bifurcation_rho_mu04.png"),
]
OUTPUT_FILE = Path("figures/bifurcation_panel.png")


def combine_panel(source_files, output_file):
    existing_files = [p for p in source_files if p.exists()]
    missing_files = [p for p in source_files if not p.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing source files: {', '.join(str(p) for p in missing_files)}"
        )

    images = [Image.open(path) for path in source_files]
    widths, heights = zip(*(img.size for img in images))
    target_width = min(widths)
    target_height = min(heights)
    images = [img.resize((target_width, target_height), Image.LANCZOS) for img in images]

    combined = Image.new("RGB", (target_width * 2, target_height * 2), (255, 255, 255))
    combined.paste(images[0], (0, 0))
    combined.paste(images[1], (target_width, 0))
    combined.paste(images[2], (0, target_height))
    combined.paste(images[3], (target_width, target_height))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_file, dpi=(100, 100))
    print(f"Created combined image: {output_file}")


if __name__ == "__main__":
    combine_panel(SOURCE_FILES, OUTPUT_FILE)
