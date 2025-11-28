"""
Script to convert all PDF plots to PNG format for GitHub README compatibility.
Requires: pip install pdf2image pillow
On macOS: brew install poppler
On Ubuntu: sudo apt-get install poppler-utils
On Windows: Download poppler from http://blog.alivate.com.au/poppler-windows/
"""

import os
from pathlib import Path
from pdf2image import convert_from_path

def convert_pdfs_to_png(root_dir='Report/plots', dpi=300):
    """
    Recursively convert all PDF files to PNG in the given directory.
    
    Args:
        root_dir: Root directory to search for PDFs
        dpi: Resolution for PNG output (300 is good for README)
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Error: Directory {root_dir} does not exist")
        return
    
    # Find all PDF files
    pdf_files = list(root_path.rglob('*.pdf'))
    
    if not pdf_files:
        print(f"No PDF files found in {root_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to convert")
    
    for pdf_path in pdf_files:
        try:
            print(f"Converting: {pdf_path.relative_to(root_path)}")
            
            # Convert PDF to images
            images = convert_from_path(str(pdf_path), dpi=dpi)
            
            # Save as PNG (take first page only for plots)
            png_path = pdf_path.with_suffix('.png')
            images[0].save(png_path, 'PNG')
            
            print(f"  ✓ Saved: {png_path.name}")
            
        except Exception as e:
            print(f"  ✗ Error converting {pdf_path.name}: {e}")
    
    print(f"\n✓ Conversion complete!")

if __name__ == '__main__':
    # Convert all plots
    convert_pdfs_to_png('Report/plots')
