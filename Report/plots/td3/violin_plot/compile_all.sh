#!/bin/bash

# Set your compiler (pdflatex is default)
COMPILER=pdflatex

# Enable shell-escape if needed (uncomment if required)
FLAGS="--shell-escape"

# Loop through all .tex files in current folder
for file in *.tex; do
    if [ -f "$file" ]; then
        echo "Compiling $file ..."
        $COMPILER $FLAGS -interaction=nonstopmode "$file"
        echo "------------------------"
    fi
done

echo "✅ All .tex files compiled."

