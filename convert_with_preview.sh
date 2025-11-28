#!/bin/bash

# Convert all PDFs to PNG using sips (built-in macOS tool)
# Usage: ./convert_with_preview.sh

find Report/plots -name "*.pdf" | while read pdf_file; do
    png_file="${pdf_file%.pdf}.png"
    echo "Converting: $pdf_file -> $png_file"
    
    # Use sips to convert (macOS built-in)
    sips -s format png "$pdf_file" --out "$png_file" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Success"
    else
        echo "  ✗ Failed"
    fi
done

echo "Conversion complete!"
