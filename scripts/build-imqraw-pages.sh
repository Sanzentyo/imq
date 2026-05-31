#!/usr/bin/env bash
set -euo pipefail

version="${1:?usage: build-imqraw-pages.sh VERSION OUT_DIR}"
out_dir="${2:?usage: build-imqraw-pages.sh VERSION OUT_DIR}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pkg_dir="$repo_root/target/imqraw-wasm-pkg"
version_dir="$out_dir/imqraw/$version"
latest_dir="$out_dir/imqraw/latest"
version_number="${version#v}"

rm -rf "$pkg_dir" "$version_dir" "$latest_dir"
mkdir -p "$version_dir" "$latest_dir"

wasm-pack build "$repo_root/crates/imqraw-wasm" \
  --target web \
  --release \
  --out-dir "$pkg_dir" \
  --out-name imqraw_wasm

copy_dist() {
  local dest="$1"
  cp "$pkg_dir/imqraw_wasm.js" "$dest/"
  cp "$pkg_dir/imqraw_wasm_bg.wasm" "$dest/"
  cp "$pkg_dir/imqraw_wasm.d.ts" "$dest/"
  cp "$repo_root/packages/imqraw-js/imqraw.js" "$dest/"
  cp "$repo_root/packages/imqraw-js/imqraw.d.ts" "$dest/"
  printf '%s\n' "$version" > "$dest/VERSION"
  cat > "$dest/package.json" <<JSON
{
  "name": "@sanzentyo/imqraw-web",
  "version": "$version_number",
  "type": "module",
  "exports": {
    ".": "./imqraw.js",
    "./wasm": "./imqraw_wasm.js"
  },
  "files": [
    "imqraw.js",
    "imqraw.d.ts",
    "imqraw_wasm.js",
    "imqraw_wasm.d.ts",
    "imqraw_wasm_bg.wasm",
    "VERSION"
  ]
}
JSON
}

copy_dist "$version_dir"
cp -R "$version_dir/." "$latest_dir/"

cat > "$out_dir/index.html" <<HTML
<!doctype html>
<meta charset="utf-8">
<title>imqraw web distribution</title>
<h1>imqraw web distribution</h1>
<ul>
  <li><a href="./imqraw/latest/">latest</a></li>
  <li><a href="./imqraw/$version/">$version</a></li>
</ul>
HTML

cat > "$version_dir/index.html" <<HTML
<!doctype html>
<meta charset="utf-8">
<title>imqraw $version</title>
<script type="module">
  import { init, encodeRgba8, imqraw_image_count } from "./imqraw.js";
  await init();
  const bytes = encodeRgba8(new Uint8Array([0, 0, 0, 255]), 1, 1, {
    label: "smoke",
    tags: ["web"],
  });
  document.body.textContent = "imqraw $version smoke image count: " + imqraw_image_count(bytes);
</script>
HTML
cp "$version_dir/index.html" "$latest_dir/index.html"
