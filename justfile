_export-marimo:
    python3 scripts/export_marimo.py


build: _export-marimo
  @echo "Building the project..."
  uv run mkdocs build

serve: build
  @echo "Serving the project..."

  uv run mkdocs serve --livereload
