# Nelson Books - PDF to Text Conversion

This repository contains pediatric medical reference materials from Nelson's Textbook of Pediatrics, converted from PDF format to plain text format for easier processing and analysis.

## Contents

### Original PDF Files (23 files)
- Allergic Disorder.pdf
- Behavioural & pyschatrical disorder.pdf
- Bone and Joint Disorders.pdf
- Digestive system.pdf
- Diseases of the Blood.pdf
- Ear .pdf
- Fluid &electrolyte disorder.pdf
- Growth development & behaviour.pdf
- Gynecologic History and Physical Examination.pdf
- Humangenetics.pdf
- Rehabilitation Medicine.pdf
- Rheumatic Disease.pdf
- Skin.pdf
- The Cardiovascular System.pdf
- The Endocrine System.pdf
- The Nervous System.pdf
- The Respiratory System .pdf
- Urology.pdf
- aldocent medicine.pdf
- cancer & benign tumor.pdf
- immunology.pdf
- learning & developmental disorder.pdf
- metabolic disorder.pdf

### Converted Text Files (23 files)
All PDF files have been successfully converted to text format and are located in the `txt_files/` directory. The text files maintain the same naming convention as the original PDFs but with `.txt` extension.

**Total size of converted text files: ~18MB**

## Conversion Process

The conversion was performed using `pdftotext` from the poppler-utils package, which provides high-quality text extraction from PDF documents while preserving the structure and readability of the content.

### Conversion Command Used:
```bash
pdftotext "input.pdf" "txt_files/output.txt"
```

## Usage

The text files can now be used for:
- Text analysis and natural language processing
- Search and indexing
- Machine learning training data
- Content extraction for medical knowledge bases
- Integration with AI/ML systems for medical question answering

## Quality Notes

- All 23 PDF files were successfully converted to text format
- The conversion preserved the medical terminology and structure
- Some formatting elements (tables, images, special characters) may have been simplified during conversion
- Font weight warnings during conversion are normal and do not affect text quality

## File Structure
```
Nelson-books/
├── README.md
├── [Original PDF files]
└── txt_files/
    ├── Allergic Disorder.txt
    ├── Behavioural & pyschatrical disorder.txt
    ├── [... all other converted text files]
    └── metabolic disorder.txt
```

