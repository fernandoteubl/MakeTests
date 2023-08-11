#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage:"
    echo "  $(basename "$0") <PDF's file>"
    exit
elif [[ ! -f "$1" ]]; then
    echo "The file '$1' didn't exists."
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
echo "Step 1/2: generating images of each page of '$PDF_FILE_INPUT'..."
convert -density 150 ${PDF_FILE_INPUT} -quality 90 output.jpg
echo "Step 2/2: merging the images into the '$PDF_FILE_OUTPUT'..."
convert output*.jpg ${PDF_FILE_OUTPUT}
mv ${PDF_FILE_OUTPUT} ..
cd ..
rm -rf temp
