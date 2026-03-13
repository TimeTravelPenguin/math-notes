import subprocess
from pathlib import Path
import shutil

print("Exporting notebooks to HTML with WebAssembly...")

out_root = Path("docs/_build/notebooks").resolve()
shutil.rmtree(out_root, ignore_errors=True)
out_root.mkdir(parents=True, exist_ok=True)

nb_root = Path("notebooks").resolve(strict=True)
notebook_dirs = [
    (nb_root / p).resolve(strict=True)
    for p in [
        "odes",
    ]
]

# Export each notebook to the output directory
for notebook_dir in notebook_dirs:
    print(f"Processing notebook directory: {notebook_dir}")
    for notebook in notebook_dir.glob("*.py"):
        print(f"  - Processing notebook: {notebook}")

        out_path = out_root / notebook.relative_to(nb_root).with_suffix("")
        print(f"    -> Out path: {out_path}")

        subprocess.run(
            [
                "uv",
                "run",
                "marimo",
                "--quiet",
                "export",
                "html-wasm",
                str(notebook),
                "-o",
                str(out_path),
                "--force",
                "--sandbox",
            ],
            check=True,
        )
