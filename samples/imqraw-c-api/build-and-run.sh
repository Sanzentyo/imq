#!/usr/bin/env sh
set -eu

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
target_dir="$repo_root/target/debug"
sample_bin="$repo_root/target/imqraw-c-api-sample"

cargo build -p imqraw-capi

case "$(uname -s)" in
  Darwin)
    cc "$repo_root/samples/imqraw-c-api/main.c" \
      -I"$repo_root/crates/imqraw-capi/include" \
      -L"$target_dir" \
      -Wl,-rpath,"$target_dir" \
      -limqraw_capi \
      -o "$sample_bin"
    ;;
  Linux)
    cc "$repo_root/samples/imqraw-c-api/main.c" \
      -I"$repo_root/crates/imqraw-capi/include" \
      -L"$target_dir" \
      -Wl,-rpath,"$target_dir" \
      -limqraw_capi \
      -o "$sample_bin"
    ;;
  *)
    cc "$repo_root/samples/imqraw-c-api/main.c" \
      -I"$repo_root/crates/imqraw-capi/include" \
      -L"$target_dir" \
      -limqraw_capi \
      -o "$sample_bin"
    ;;
esac

"$sample_bin"
