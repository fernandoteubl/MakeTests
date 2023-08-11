#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage:"
    echo "  $(basename "$0") <PDF's file>"
    exit
fi

PDF_FILE_INPUT=$1
filename=$(basename -- "$PDF_FILE_INPUT")
extension="${filename##*.}"
filename="${filename%.*}_img.${extension}"
PDF_FILE_OUTPUT="$filename"

mkdir temp
cp ${PDF_FILE_INPUT} temp
cd temp
convert -density 150 ${PDF_FILE_INPUT} -quality 90 output.jpg
convert output*.jpg ${PDF_FILE_OUTPUT}
mv ${PDF_FILE_OUTPUT} ..
cd ..
rm -rf temp
