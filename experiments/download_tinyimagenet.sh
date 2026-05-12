#!/bin/bash

# This script is based on the download script from:
# https://github.com/eth-sri/CTBench/blob/main/scripts/examples/tinyimagenet/download_tinyimagenet.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
data_root="${repo_root}/data-tinyimagenet"

mkdir -p "${data_root}"
cd "${data_root}"
rm -rf tiny-imagenet-200 tiny-imagenet-200.zip

# download and unzip dataset
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

current="$(pwd)/tiny-imagenet-200"

# training data
echo "preparing training data..."
cd "$current/train"
for DIR in *; do
   cd "$DIR"
   rm -f ./*.txt
   mv images/* .
   rm -rf images
   cd ..
done

# validation data
echo "preparing validation data..."
cd "$current/val"
annotate_file="val_annotations.txt"
length=$(wc -l < "$annotate_file")
for i in $(seq 1 "$length"); do
    # fetch i th line
    line=$(sed -n "${i}p" "$annotate_file")
    # get file name and directory name
    file=$(echo "$line" | cut -f1 -d" " )
    directory=$(echo "$line" | cut -f2 -d" ")
    mkdir -p "$directory"
    mv "images/$file" "$directory"
done
rm -rf images
echo "done"
