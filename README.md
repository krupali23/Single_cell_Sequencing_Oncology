https://singlecellsequencingoncology-krupalipoharkar.streamlit.app/

🧬 Immunotherapy Response Explorer

Predict who may benefit from immunotherapy and explain why—by turning complex single-cell and genomic data into simple, actionable visuals.

Table of contents

Overview

Features

Live demo / Screenshots

Quick start

Project structure

Required data & models

How to use the app

Troubleshooting

Roadmap

Contributing

License

Medical disclaimer

Citation

Acknowledgments

Overview

The Immunotherapy Response Explorer helps non-technical users (clinicians, researchers, decision makers) make sense of millions of single-cell and genomics measurements per patient. It provides:

A clean summary of complex signals,

Model predictions of responder vs non-responder,

Explainability (which features/pathways matter),

Visual tools to compare cells, genes, and patients.

Features

Clean visuals from complex patient & single-cell data

Immune cell highlights (e.g., which cells look active vs exhausted)

Responders vs Non-responders comparisons (e.g., CD8 T cells)

Pathways & Hallmarks linked to success or resistance

AI predictions with explanations (feature importance, optional SHAP)

Chat helper that answers plain-English questions (e.g., “Is patient P7 a responder?”)


Quick start
1) Prerequisites

Python 3.9+

Recommended: a virtual environment: python -m venv .venv && source .venv/bin/activate (Windows: .venv\Scripts\activate)

2) Install dependencies
pip install -r requirements.txt


If you don’t have a requirements.txt yet, start with:

streamlit
numpy
pandas
scikit-learn
joblib
plotly
pyarrow       # for parquet (optional but recommended)
shap          # optional; enables SHAP explanations if available

3) Prepare data & models

Put your data under: C:\Users\krupa\Desktop\Bootcamp\Final_project\Data

Or point the app to another folder using the env var DATA_DIR.

Put your .joblib models under: <parent_of_DATA_DIR>/models
(By default: C:\Users\krupa\Desktop\Bootcamp\Final_project\models)

4) Run
streamlit run app_combined.py


Open the URL Streamlit prints (usually http://localhost:8501).

Project structure
repo-root/
├─ app_combined.py
├─ requirements.txt
├─ README.md
├─ Data/                          # DATA_DIR (can be changed via env var)
│  ├─ patient_features.csv
│  ├─ patient_response_binary.csv (or *_cleaned_with_mixed.csv/.xlsx)
│  ├─ sc_annot.csv                # UMAP + metadata per cell
│  ├─ sc_expr.parquet OR sc_expr.csv  # gene expression matrix (cell_id + genes)
│  ├─ markers/
│  │  └─ per_group_top50/         # per-cluster marker files (.csv/.xlsx)
│  ├─ patient_features_with_hallmark.csv         # optional (all cells)
│  ├─ patient_features_with_hallmark_CD8.csv     # optional (CD8/TNK)
│  ├─ gsea_prerank_cd8*.csv       # optional cached GSEA results
│  └─ slides_intro/               # optional intro slides (JPG/PNG)
└─ models/                        # MODELS_DIR
   └─ <your_model>.joblib

Required data & models

Minimum to run predictions

Data/patient_features.csv — rows = patient_id, columns = features

models/<your_model>.joblib — can contain the estimator and (optionally) feature_names & final_threshold

To show labels & evaluation metrics

One of:

Data/patient_response_binary.csv

Data/patient_response_cleaned_with_mixed.csv

(CSV/XLSX variants supported)

Must include columns for patient id and response (R/NR or 1/0)

To enable Cell Map & Gene Explorer

Data/sc_annot.csv — must include cell_id, umap1, umap2, and a cluster label column (the app auto-detects common names)

Data/sc_expr.parquet (preferred) or Data/sc_expr.csv — cell_id + gene columns

To enable marker panels

Data/markers/per_group_top50/*.csv|.xlsx — each file lists marker genes (first column)

To enable Hallmark Explorer & GSEA (optional)

Data/patient_features_with_hallmark.csv

Data/patient_features_with_hallmark_CD8.csv

Data/gsea_prerank_cd8*.csv

To enable Intro slides (optional)

Place .jpg/.png in Data/slides_intro/

How to use the app

Background
A simple intro to why immunotherapy response varies and why single-cell/genomics data help us see what’s active or silent in each patient.

Performance

Select a model; adjust the Decision threshold to trade precision vs recall.

If labels are present, you’ll see Accuracy, Sensitivity, Specificity, ROC-AUC, PR-AUC and a confusion matrix.

Download predictions as CSV.

Cell Map

UMAP of all cells. Color by Cell type, Responder status, or Gene expression.

Gene Explorer

Pick a cluster and its marker panel.

Compare gene expression between Responders vs Non-responders (violin/box).

View per-cluster means, heatmaps, UMAP colored by gene, and co-expression.

Comparison

“Traffic-light” view for a single patient.

Compare the patient’s features to group means for responders and non-responders.

What Drives Response

Feature importances (SHAP if available), Hallmark pathway differences, and optional CD8 GSEA tables.

Chat

Ask plain-English questions like “Is patient P7 a responder?”

Ask what metrics mean (precision, recall, ROC-AUC, etc.).

Summary

Clear takeaways and a Thank you slide for presentations.

Troubleshooting

“No .joblib models found”
Ensure your model file is under models/ and ends with .joblib.

“No images found in Data/slides_intro” (Intro)
Create the folder and add at least one JPG/PNG. (Optional feature.)

Blank Performance metrics
You need a label file (see Required data & models) with a response column mapped to 1/0 or R/NR.

Gene matrix not available
Add sc_expr.parquet (preferred) or sc_expr.csv with cell_id + genes.

Markers not loading
Check Data/markers/per_group_top50/*.csv|.xlsx and ensure first column contains gene symbols.

Feature mismatch
The app aligns features using feature_names from the model bundle if present. Missing features are filled with 0.0; ensure training and inference schemas match as much as possible.

Roadmap

Export full PDF reports (figures + text)

More page-level tooltips and mini-tours

Add cohort stratification (cancer type, line of therapy)

Model comparison panel

Contributing

PRs welcome! Please:

Open a small, focused Pull Request.

Include a short description and screenshots.

Add/update docstrings and comments where needed.

(Optional) Add CONTRIBUTING.md and a CODE_OF_CONDUCT.md.

License

This project is licensed under the MIT License. See LICENSE.

Medical disclaimer

This software is for research and educational purposes only and not intended for clinical use or to replace professional medical judgment.

Citation

If you use this app, please cite:

Your Name (2025). Immunotherapy Response Explorer.
GitHub: https://github.com/<user>/<repo>


(Replace with your repo URL. Consider adding a CITATION.cff for GitHub’s “Cite this repository” button.)

Acknowledgments

Open-source libraries: Streamlit, scikit-learn, Plotly, SHAP

Colleagues and mentors who contributed feedback

Any datasets or tools you built upon
