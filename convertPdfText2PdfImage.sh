#!/bin/bash

PDF_FILE_INPUT="Provas.pdf"
PDF_FILE_OUTPUT="Provas_img.pdf"

cd "$(dirname "$0")"
mkdir temp
cp ${PDF_FILE_INPUT} temp
cd temp
convert -density 150 ${PDF_FILE_INPUT} -quality 90 output.jpg
convert output*.jpg ${PDF_FILE_OUTPUT}
mv ${PDF_FILE_OUTPUT} ..
cd ..
rm -rf temp

if [ "$(uname)" == "Darwin" ]; then
	osascript -e 'tell application "Terminal" to close first window' & exit
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    echo GNU/Linux platform
    
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
    echo Windows NT platform
fi
