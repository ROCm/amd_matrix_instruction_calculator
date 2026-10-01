#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Runs the delta test against a base git ref and against the current working tree, then diffs
# the two outputs. The current delta_test.py is used for both runs so the outputs are comparable.
#
# Exit status: 0 if the outputs are identical, 1 if they differ, 2 on error.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [-c cores] [-k] [base_ref]

Compare delta test output of base_ref against the current working tree.

  base_ref   Git ref to compare against. Default: merge-base of HEAD and main.
  -c cores   Number of parallel test jobs to pass to delta_test.py. Default: all cores.
  -k         Keep the output directory even when the outputs are identical.
  -h         Print this help.
EOF
}

cores=-1
keep=0
while getopts "c:kh" opt; do
    case "$opt" in
        c) cores="$OPTARG" ;;
        k) keep=1 ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
if [ $# -gt 1 ]; then
    usage >&2
    exit 2
fi

repo_root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
if [ $# -eq 1 ]; then
    base_ref="$1"
else
    base_ref="$(git -C "$repo_root" merge-base HEAD main)"
fi
base_sha="$(git -C "$repo_root" rev-parse --verify --quiet "${base_ref}^{commit}")" || {
    echo "ERROR: '$base_ref' is not a valid git ref." >&2
    exit 2
}
base_desc="$(git -C "$repo_root" rev-parse --short "$base_sha")"
if [ $# -eq 1 ]; then
    base_desc="$base_ref ($base_desc)"
fi

out_dir="$(mktemp -d "${TMPDIR:-/tmp}/delta_diff.XXXXXX")"
worktree="$out_dir/base_worktree"
cleanup() {
    git -C "$repo_root" worktree remove --force "$worktree" 2>/dev/null || true
    git -C "$repo_root" worktree prune
}
trap cleanup EXIT

echo "Checking out $base_desc into a temporary worktree..."
git -C "$repo_root" worktree add --quiet --detach "$worktree" "$base_sha"
cp "$repo_root/test/delta_test.py" "$worktree/test/delta_test.py"

echo "Running delta test on $base_desc..."
"$worktree/test/delta_test.py" -o -c "$cores" "$out_dir/base.txt" || {
    echo "ERROR: delta test failed on $base_desc. Outputs are in $out_dir" >&2
    exit 2
}
echo "Running delta test on the working tree..."
"$repo_root/test/delta_test.py" -o -c "$cores" "$out_dir/new.txt" || {
    echo "ERROR: delta test failed on the working tree. Outputs are in $out_dir" >&2
    exit 2
}

# The tester echoes each command with the tool's absolute path, which differs between the two
# checkouts. Strip each checkout's root so only real output changes show up in the diff.
strip_root() {
    local escaped
    escaped="$(printf '%s' "$1/" | sed 's/[][\.*^$|]/\\&/g')"
    sed "s|$escaped||g" "$2" > "$2.tmp" && mv "$2.tmp" "$2"
}
strip_root "$worktree" "$out_dir/base.txt"
strip_root "$repo_root" "$out_dir/new.txt"

if diff -u "$out_dir/base.txt" "$out_dir/new.txt" > "$out_dir/delta.diff"; then
    echo "No differences from $base_desc."
    if [ "$keep" -eq 1 ]; then
        echo "Outputs are in $out_dir"
    else
        rm -rf "$out_dir"
    fi
    exit 0
fi

echo "Outputs differ from $base_desc:"
# Skip the two ---/+++ header lines when counting
echo "  $(tail -n +3 "$out_dir/delta.diff" | grep -c '^-' || true) lines removed," \
     "$(tail -n +3 "$out_dir/delta.diff" | grep -c '^+' || true) lines added"
echo "  Full diff: $out_dir/delta.diff"
echo "  Outputs:   $out_dir/base.txt, $out_dir/new.txt"
exit 1
