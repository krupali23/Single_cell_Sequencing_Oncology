# app_combined.py — Immunotherapy Response Explorer (single file, consolidated)
import os, json, joblib, base64, warnings, re, io, datetime, glob
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)

# ================== BASIC CONFIG ==================
st.set_page_config(page_title="Immunotherapy Response Explorer", layout="wide")

# ---- constants / paths
DATA_DIR   = os.environ.get("DATA_DIR", r"C:\Users\krupa\Desktop\Bootcamp\Final_project\Data")
MODELS_DIR = os.path.join(os.path.dirname(DATA_DIR), "models")

SC_ANNOT = os.path.join(DATA_DIR, "sc_annot.csv")        # UMAP + metadata
SC_EXPR_PARQUET = os.path.join(DATA_DIR, "sc_expr.parquet")
SC_EXPR_CSV     = os.path.join(DATA_DIR, "sc_expr.csv")

# Marker lists
MARKERS_DIR         = os.path.join(DATA_DIR, "markers")
MARKERS_TOP50_DIR   = os.path.join(MARKERS_DIR, "per_group_top50")  # << your folder

# Intro slides folder (put your JPG/PNG images here)
SLIDES_DIR = os.path.join(DATA_DIR, "slides_intro")

# ================== HERO + TOP NAV ==================
st.markdown(
    """
    <style>
    /* Make the top radio look like a bold tab bar */
    div[data-testid="stRadio"] > label p {font-size: 0; margin: 0;}
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        font-size:1.05rem; font-weight:800; padding:8px 12px; margin:4px 8px 20px 0;
        border-radius:10px; border:1px solid #d9e2ec; background:#fafcfe; cursor:pointer;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover {background:#f3f7ff;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='margin-bottom:0.2rem'>🧬 Immunotherapy Response Explorer</h1>"
    "<p style='font-size:1.05rem; color:#2f4f4f; margin-top:0'>"
    "Predict responders, visualize single cells, and explore marker genes — all in one place."
    "</p>",
    unsafe_allow_html=True,
)

# ================== TOP NAV (reordered + renamed) ==================
nav_labels = [
    "Background",              # first page
    "Performance",
    "Cell Map",
    "Gene Explorer",
    "Comparison",
    "What Drives Response",    # (renamed from Explainability)
    "Chat",
    "Summary"                  # new last page
]
page = st.radio("Navigation", nav_labels, index=0, horizontal=True, label_visibility="collapsed")

# ================== LANGUAGE (top of sidebar) + i18n for upload text ==================
LANGS = {
    "en":{"name":"English",
          "upload_header":"Score custom CSV",
          "upload_caption":"CSV indexed by patient_id with same feature columns. Add `true_label` (R/NR) to see metrics.",
          "upload_btn":"Upload features CSV"},
    "de":{"name":"Deutsch",
          "upload_header":"Eigenes CSV auswerten",
          "upload_caption":"CSV mit patient_id als Index und identischen Merkmalen. `true_label` (R/NR) hinzufügen für Metriken.",
          "upload_btn":"CSV mit Merkmalen hochladen"},
    "hi":{"name":"हिन्दी",
          "upload_header":"कस्टम CSV स्कोर करें",
          "upload_caption":"patient_id इंडेक्स वाला CSV, वही फीचर कॉलम. मेट्रिक्स के लिए `true_label` (R/NR) जोड़ें।",
          "upload_btn":"फ़ीचर CSV अपलोड करें"},
    "es":{"name":"Español",
          "upload_header":"Evaluar CSV propio",
          "upload_caption":"CSV indexado por patient_id con las mismas columnas. Añade `true_label` (R/NR) para ver métricas.",
          "upload_btn":"Subir CSV de características"}
}
def L(key, lang): return LANGS.get(lang, LANGS["en"]).get(key, LANGS["en"][key])

st.sidebar.header("Language")
lang_code = st.sidebar.selectbox("Language", ["en","de","hi","es"], index=0,
                                 format_func=lambda c: LANGS[c]["name"])

# ================== UPLOAD (immediately under Language) ==================
st.sidebar.header(L("upload_header", lang_code))
st.sidebar.caption(L("upload_caption", lang_code))
upload = st.sidebar.file_uploader(L("upload_btn", lang_code), type=["csv"])
X_custom = None
if upload is not None:
    try:
        X_custom = pd.read_csv(upload, index_col=0)
        st.sidebar.success(f"Uploaded shape: {X_custom.shape}")
    except Exception as e:
        st.sidebar.error("Failed to read uploaded CSV.")
        st.sidebar.exception(e)

# ================== HELPERS ==================
def listdir_safe(p):
    try: return sorted(os.listdir(p))
    except Exception as e: return [f"ERR: {e}"]

@st.cache_data(show_spinner=False)
def load_features():
    p=os.path.join(DATA_DIR,"patient_features.csv")
    if not os.path.exists(p): raise FileNotFoundError(f"Missing features file: {p}")
    return pd.read_csv(p, index_col=0).fillna(0.0)

@st.cache_data(show_spinner=False)
def load_labels():
    for name in ["patient_response_binary.csv","patient_response_cleaned_with_mixed.csv",
                 "patient_response_binary.xlsx","patient_response_cleaned_with_mixed.xlsx"]:
        p=os.path.join(DATA_DIR,name)
        if not os.path.exists(p): continue
        try:
            df = pd.read_excel(p) if p.lower().endswith(".xlsx") else pd.read_csv(p)
            cols=[c.strip().lower() for c in df.columns]; df.columns=cols
            pid=next((c for c in cols if ("patient" in c) or ("sample" in c)), None)
            resp=next((c for c in cols if ("response" in c) or ("label" in c) or ("responder" in c) or ("binary" in c)), None)
            if pid is None or resp is None: continue
            out = df[[pid,resp]].rename(columns={pid: "patient_id", resp: "response"})
            if out["response"].dtype==object:
                m={"responder":1,"r":1,"cr":1,"pr":1,"non-responder":0,"nr":0,"sd":0,"pd":0,"1":1,"0":0}
                out["response"]=out["response"].astype(str).str.strip().str.lower().map(m)
            out=out.dropna(subset=["patient_id","response"]).copy()
            out["patient_id"]=out["patient_id"].astype(str).str.strip()
            out["response"]=out["response"].astype(int)
            return out.groupby("patient_id")["response"].max().to_frame()
        except Exception:
            continue
    return None

def parse_model_bundle(obj):
    if isinstance(obj,dict):
        model=obj.get("model",obj); feat=obj.get("feature_names",None); thr=obj.get("final_threshold",obj.get("threshold",0.5))
    else:
        model,feat,thr=obj,None,0.5
    try: thr=float(thr)
    except Exception: thr=0.5
    return model,feat,thr

@st.cache_resource(show_spinner=False)
def load_model(path):
    obj=joblib.load(path); return parse_model_bundle(obj)

def probas(est, X):
    if hasattr(est,"predict_proba"): return est.predict_proba(X)[:,1]
    if hasattr(est,"decision_function"): z=est.decision_function(X); return 1/(1+np.exp(-z))
    p=est.predict(X).astype(float); return np.clip(p,0.0,1.0)

def align(X, feat_names): return X if feat_names is None else X.reindex(columns=feat_names).fillna(0.0)

def feature_importance_series(est, feat_names):
    if hasattr(est,"feature_importances_"):
        return pd.Series(est.feature_importances_, index=feat_names).sort_values(ascending=False)
    coef=getattr(est,"coef_",None)
    if coef is None and hasattr(est,"named_steps"):
        for step in est.named_steps.values():
            if hasattr(step,"coef_"): coef=step.coef_; break
    if coef is not None:
        return pd.Series(np.abs(coef).ravel(), index=feat_names).sort_values(ascending=False)
    return None

def correlation_importance(model, X, feat_names):
    try:
        s=probas(model, X[feat_names])
        vals = {}
        for f in feat_names:
            col = pd.to_numeric(X[f], errors="coerce").fillna(0.0)
            vals[f] = (np.corrcoef(col.values, s)[0,1] if col.std(ddof=0)>0 else 0.0)
        return pd.Series(vals).abs().sort_values(ascending=False)
    except Exception:
        return None

# ---- sc_annot loader (UMAP + metadata)
@st.cache_data(show_spinner=False)
def load_sc_annot(path_csv: str):
    if not os.path.exists(path_csv): return None
    df = pd.read_csv(path_csv)
    candidates = [c for c in df.columns if "umap" in c.lower()] or [c for c in df.columns if c.lower() in ["x","y","dim1","dim2"]]
    if len(candidates) >= 2:
        xcol, ycol = candidates[0], candidates[1]
    else:
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        xcol, ycol = (nums[0], nums[1]) if len(nums) >=2 else (df.columns[0], df.columns[1])
    df = df.rename(columns={xcol:"umap1", ycol:"umap2"})
    if "cell_id" not in df.columns: df["cell_id"] = df.index.astype(str)
    if "celltype" in df.columns and "cell_type" not in df.columns:
        df = df.rename(columns={"celltype":"cell_type"})
    label_col = None
    for k in ["cell_type","cluster","clusters","leiden","louvain","cluster_name","annot"]:
        if k in df.columns:
            label_col = k; break
    if label_col is None: df["cluster_label"] = "Cluster_" + df.index.astype(str)
    else: df["cluster_label"] = df[label_col].astype(str)
    for k in ["patient_id","patient","sample","donor","case","subject","pt"]:
        if k in df.columns:
            df["patient_id"] = df[k].astype(str); break
    if "patient_id" not in df.columns: df["patient_id"] = "unknown"
    keep = ["cell_id","umap1","umap2","cluster_label","patient_id"]
    more = [c for c in df.columns if c not in keep]
    return df[keep + more]

sc_df = load_sc_annot(SC_ANNOT)

# ---- expression helper
@st.cache_data(show_spinner=False)
def available_gene_list():
    if os.path.exists(SC_EXPR_PARQUET):
        try:
            import pyarrow.parquet as pq
            return [c for c in pq.ParquetFile(SC_EXPR_PARQUET).schema.names if c!="cell_id"]
        except Exception:
            try:
                return [c for c in pd.read_parquet(SC_EXPR_PARQUET, engine="pyarrow").columns if c!="cell_id"]
            except Exception:
                pass
    if os.path.exists(SC_EXPR_CSV):
        with open(SC_EXPR_CSV,"r",encoding="utf-8") as f:
            header=f.readline().strip().split(",")
        return [h for h in header if h!="cell_id"]
    return []

ALL_GENES = available_gene_list()
ALL_GENES_LUT = {g.lower(): g for g in ALL_GENES}  # case-insensitive mapper

def map_to_available_genes(markers: list[str]) -> list[str]:
    mapped = []
    for g in markers:
        key = str(g).strip().lower()
        if key in ALL_GENES_LUT:
            mapped.append(ALL_GENES_LUT[key])
    return mapped

def read_gene_cols(genes):
    genes = map_to_available_genes(genes)
    if not genes: return None
    if os.path.exists(SC_EXPR_PARQUET):
        try: return pd.read_parquet(SC_EXPR_PARQUET, columns=["cell_id"]+genes)
        except Exception: pass
    if os.path.exists(SC_EXPR_CSV):
        try: return pd.read_csv(SC_EXPR_CSV, usecols=["cell_id"]+genes)
        except Exception: pass
    return None

# ---- marker files loader (top-50 per cluster)
@st.cache_data(show_spinner=False)
def load_top50_markers(markers_dir: str):
    cluster2genes, union = {}, set()
    if not os.path.isdir(markers_dir):
        return cluster2genes, []
    files = glob.glob(os.path.join(markers_dir, "*.xlsx")) + glob.glob(os.path.join(markers_dir, "*.csv"))
    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        cluster = re.sub(r'[_-]*top50$', '', re.sub(r'^celltype[_-]*', '', base)).strip()
        try:
            df = pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        gcol = next((c for c in df.columns if str(c).lower() in
                    ["gene","genes","symbol","gene_symbol","marker","features","feature"]), df.columns[0])
        genes_raw = (df[gcol].dropna().astype(str).str.strip().tolist())[:50]
        cluster2genes[cluster] = genes_raw
        union.update(genes_raw)
    return cluster2genes, sorted(union)

CLUSTER2MARKERS, ALL_MARKER_GENES_RAW = load_top50_markers(MARKERS_TOP50_DIR)

# ================== LOAD PATIENT FEATURES / LABELS / MODEL ==================
try:
    X_all = load_features()
    st.success(f"Loaded features: {X_all.shape[0]} patients × {X_all.shape[1]} features")
except Exception as e:
    st.error("Failed to load features file.")
    st.exception(e)
    st.stop()

labels_df = load_labels()
if labels_df is not None:
    matched = labels_df.index.intersection(X_all.index)
    labels_df = labels_df.loc[matched].astype(int)
    st.caption(f"Labels matched to features: {len(labels_df)}")

available_models=[m for m in listdir_safe(MODELS_DIR) if isinstance(m,str) and m.endswith(".joblib")]
if not available_models:
    st.error("No .joblib models found in MODELS_DIR.")
    st.stop()
model_file=st.sidebar.selectbox("Model (.joblib)", available_models, index=0, key="model_select_main")
MODEL_FILE=os.path.join(MODELS_DIR, model_file)

try:
    model, feat_names, default_thr = load_model(MODEL_FILE)
    if feat_names is None: feat_names=list(X_all.columns)
    st.sidebar.success(f"Model loaded: {os.path.basename(MODEL_FILE)}")
except Exception as e:
    st.sidebar.error("Failed to load model.")
    st.exception(e)
    st.stop()

# <- NEW: show features vs model uses
st.sidebar.caption(f"Features file: **{X_all.shape[1]}** columns • Model uses: **{len(feat_names)}**")

thr = st.sidebar.slider("Decision threshold", 0.0, 1.0, float(default_thr), 0.01, key="thr_main")

st.sidebar.header("Cohort")
sel_patients = st.sidebar.multiselect("Select patients (optional)", list(X_all.index),
                                      default=list(X_all.index), key="patients_multiselect")

# ================== SHARED SMALL UTILS ==================
def get_predictions_dataframe():
    X_use = X_custom if X_custom is not None else X_all.loc[sel_patients]
    X_use = align(X_use, feat_names)
    scores = probas(model, X_use)
    preds  = (scores >= thr).astype(int)
    out = pd.DataFrame({"probability":scores,"prediction":preds}, index=X_use.index)
    out = out.sort_values("probability", ascending=False)
    return out

def attach_response(df: pd.DataFrame, labels_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Robustly attach 'response' by patient_id even if patient_id is an index in either DF.
    Avoids: ValueError: 'patient_id' is both an index level and a column label.
    """
    if labels_df is None:
        df2 = df.copy()
        if "response" not in df2.columns:
            df2["response"] = np.nan
        return df2

    # normalize left DF to have a 'patient_id' column (not index)
    left = df.copy()
    if "patient_id" not in left.columns:
        if left.index.name == "patient_id" or ("patient_id" in (left.index.names if hasattr(left.index, "names") else [])):
            left = left.reset_index()
        else:
            left["response"] = np.nan
            return left

    # normalize labels to have 'patient_id' column
    lab = labels_df.copy()
    if lab.index.name == "patient_id" or ("patient_id" in (lab.index.names if hasattr(lab.index, "names") else [])):
        lab = lab.reset_index()
    if "patient_id" not in lab.columns:
        lab = lab.reset_index().rename(columns={"index":"patient_id"})

    left["patient_id"] = left["patient_id"].astype(str).str.strip()
    lab["patient_id"]  = lab["patient_id"].astype(str).str.strip()

    merged = left.merge(lab[["patient_id","response"]], on="patient_id", how="left")
    return merged

# ================== KNOWLEDGE-BASE ANSWERS (NEW) ==================
def kb_answer(question: str, page: str | None = None) -> str:
    q = (question or "").strip().lower()

    # metrics / glossary
    if any(k in q for k in ["what are features", "what is feature", "features", "input features"]):
        return ("**Features** are the numeric inputs to the model (per patient). "
                "Examples: proportions of cell types, marker scores, clinical covariates.")
    if any(k in q for k in ["f1", "f1-score", "f1 score"]):
        return ("**F1-score** = harmonic mean of precision and recall (2·P·R/(P+P)). "
                "Balances false positives and false negatives.")
    if any(k in q for k in ["sensitivity", "recall", "tpr", "true positive rate"]):
        return ("**Sensitivity (Recall, TPR)** = TP/(TP+FN). Of the true responders, how many did we catch?")
    if any(k in q for k in ["specificity", "tnr", "true negative rate"]):
        return ("**Specificity (TNR)** = TN/(TN+FP). Of the true non-responders, how many did we correctly reject?")
    if any(k in q for k in ["precision", "ppv"]):
        return ("**Precision (PPV)** = TP/(TP+FP). Of the predicted responders, how many were actually responders?")
    if "confusion" in q and "matrix" in q:
        return ("A **confusion matrix** is a 2×2 table of TN, FP, FN, TP (true rows, predicted columns). "
                "It shows error types at a glance.")
    if "roc" in q:
        return ("**ROC curve** plots TPR vs FPR across thresholds; **AUC** measures discrimination (1 best, 0.5 random).")
    if "precision-recall" in q or "pr curve" in q or "pr-auc" in q:
        return ("**Precision–Recall curve** shows precision vs recall; **PR-AUC** summarizes performance on imbalanced data.")
    if any(k in q for k in ["what is a cluster", "what is cluster", "cluster", "cell type"]):
        return ("A **cluster/cell type** groups similar cells by expression; labels are assigned via annotation.")
    if any(k in q for k in ["responder status", "responders", "non-responders", "label"]):
        return ("**Responder status** is the patient outcome (1=Responder, 0=Non-responder).")

    # page-specific summaries
    if page == "Performance":
        if any(k in q for k in ["what is this page", "overview", "help"]):
            return ("**Performance** reports Accuracy, Sensitivity, Specificity, ROC-AUC, PR-AUC, and the confusion matrix. "
                    "If you upload labeled data, it evaluates on that too.")
        if "threshold" in q:
            return ("The **decision threshold** converts probabilities to 0/1. Higher threshold → fewer predicted responders "
                    "but higher precision (and lower recall).")
    if page == "Cell Map":
        if any(k in q for k in ["what is this page", "overview", "help"]):
            return ("**Cell Map** is a UMAP of all cells. Color by cluster, responder status, or a gene to see where signals live.")
    if page == "Gene Explorer":
        if "group distribution" in q or "violin" in q or "box" in q:
            return ("**Group distribution (Violin/Box)** compares a gene’s expression between Responders and Non-responders.")
        if "expression by cell type" in q or "by cluster" in q or "means" in q:
            return ("**Expression by cell type/cluster** shows per-cluster violins and a mean-by-cluster table/bar.")
        if "heatmap" in q and ("group" in q or "mean" in q):
            return ("**Heatmap — group mean expression** aggregates cells by response group for selected marker genes.")
        if "umap colored" in q:
            return ("**UMAP colored by gene** paints the UMAP by that gene’s expression.")
        if "co-expression" in q or "coexpression" in q:
            return ("**Co-expression scatter** plots Gene-1 vs Gene-2 per cell; a diagonal suggests joint expression.")
        if any(k in q for k in ["what is this page", "overview", "help"]):
            return ("**Gene Explorer** loads a cluster’s top-50 markers and lets you visualize violins/means/heatmaps/UMAP/co-expression.")
    if page == "Comparison":
        if any(k in q for k in ["what is this page", "overview", "help", "patient vs group"]):
            return ("**Patient vs Group Averages** shows one patient’s features vs responder and non-responder means. "
                    "Bars show Patient−Mean differences; the traffic light is the model prediction.")
    if page == "What Drives Response":
        if any(k in q for k in ["what is this page", "overview", "help", "which features matter"]):
            return ("**What Drives Response** ranks features by importance (SHAP if available; otherwise model/correlation importances), "
                    "and includes Hallmark Explorer and optional GSEA (CD8) results if present.")
    if page == "Summary":
        if any(k in q for k in ["what is this page", "overview", "help"]):
            return ("**Summary** recaps how the app helps interpret complex single-cell and genomic data for treatment decisions.")

    # overall (Chat page)
    if page in [None, "Chat"]:
        if any(k in q for k in ["what is this app", "overall", "all pages", "how to use", "guide"]):
            return ("Overall guide:\n"
                    "• **Performance**: how good the model is.\n"
                    "• **Cell Map**: UMAP—color by cluster/label/gene.\n"
                    "• **Gene Explorer**: marker panels and multiple plots.\n"
                    "• **Comparison**: one patient vs group means.\n"
                    "• **What Drives Response**: which features/pathways mattered.\n"
                    "• **Background**/**Summary**: overview & takeaways.")
    return ("Ask about metrics (F1, sensitivity, specificity, ROC/PR), clusters, responder status, gene expression, "
            "or what each section on this page means.")

# ----- tiny chat helper shown at bottom of every page (UPDATED) -----
def render_page_chat(page_name: str):
    st.markdown("---")
    with st.expander("💬 Help on this page"):
        key = f"mini_chat_{page_name}"
        if key not in st.session_state:
            st.session_state[key] = []
        q = st.text_input("Ask about this page or your data", key=f"input_{key}")
        if st.button("Ask", key=f"btn_{key}"):
            st.session_state[key].append(("You", q or ""))
            ans = kb_answer(q, page=page_name)
            st.session_state[key].append(("App", ans))
        for who, msg in st.session_state[key]:
            st.markdown(f"**{who}:** {msg}")

# ================== PAGE: PERFORMANCE ==================
if page == "Performance":
    st.header("📊 Performance")

    out = get_predictions_dataframe()

    # --- EVALUATION FIRST (internal labels or uploaded)
    if labels_df is not None and X_custom is None and not out.empty:
        st.subheader("Evaluation on labeled patients")
        common = out.index.intersection(labels_df.index)
        if len(common) > 0:
            y_pred = out.loc[common,"prediction"].astype(int).values
            y_prob = out.loc[common,"probability"].values
            y_true = labels_df.loc[common,"response"].astype(int).values

            tn,fp,fn,tp = confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
            acc  = (tp+tn)/(tp+tn+fp+fn)
            sens = tp/(tp+fn) if (tp+fn) else 0.0
            spec = tn/(tn+fp) if (tn+fp) else 0.0
            rocA = roc_auc_score(y_true,y_prob)
            prA  = average_precision_score(y_true,y_prob)

            c1,c2,c3 = st.columns(3)
            c1.metric("Accuracy",     f"{acc:.3f}")
            c2.metric("Sensitivity",  f"{sens:.3f}")
            c3.metric("Specificity",  f"{spec:.3f}")
            st.write(f"ROC-AUC: **{rocA:.3f}** · PR-AUC: **{prA:.3f}**")

            with st.expander("Show ROC, PR and Confusion Matrix"):
                fpr,tpr,_ = roc_curve(y_true,y_prob)
                prec,rec,_= precision_recall_curve(y_true,y_prob)
                figROC = go.Figure(); figROC.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                figROC.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="chance"))
                figROC.update_layout(title=f"ROC (AUC={auc(fpr,tpr):.3f})", xaxis_title="FPR", yaxis_title="TPR", height=400)
                st.plotly_chart(figROC, use_container_width=True)

                figPR  = go.Figure(); figPR.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
                figPR.update_layout(title="Precision–Recall", xaxis_title="Recall", yaxis_title="Precision", height=400)
                st.plotly_chart(figPR,  use_container_width=True)

                cm2 = confusion_matrix(y_true, y_pred, labels=[0,1])
                figCM = px.imshow(cm2, text_auto=True, color_continuous_scale="Blues",
                                  labels=dict(x="Predicted", y="True", color="Count"),
                                  x=["0 (NR)","1 (R)"], y=["0 (NR)","1 (R)"], title="Confusion Matrix")
                st.plotly_chart(figCM, use_container_width=True)

            with st.expander("Classification report (text)"):
                st.text(classification_report(y_true, y_pred, target_names=["0 (NR)", "1 (R)"], digits=3))
        else:
            st.info("No overlap between selected patients and label table.")

    if X_custom is not None:
        st.subheader("Evaluation on uploaded patients")
        tmp = align(X_custom, feat_names)
        scores = probas(model, tmp)
        preds  = (scores >= thr).astype(int)
        user_df = X_custom.copy()
        if "true_label" in user_df.columns:
            y_true = user_df["true_label"].map({"NR":0,"R":1,"0":0,"1":1}).astype(float).dropna().astype(int)
            valid_ids = y_true.index.intersection(tmp.index)
            if len(valid_ids) > 0:
                y = y_true.loc[valid_ids].values
                p = preds[np.isin(tmp.index, valid_ids)]
                s = scores[np.isin(tmp.index, valid_ids)]

                tn,fp,fn,tp = confusion_matrix(y, p, labels=[0,1]).ravel()
                acc  = (tp+tn)/(tp+tn+fp+fn)
                sens = tp/(tp+fn) if (tp+fn) else 0.0
                spec = tn/(tn+fp) if (tn+fp) else 0.0
                rocA = roc_auc_score(y, s)
                prA  = average_precision_score(y, s)

                c1,c2,c3 = st.columns(3)
                c1.metric("Accuracy",     f"{acc:.3f}")
                c2.metric("Sensitivity",  f"{sens:.3f}")
                c3.metric("Specificity",  f"{spec:.3f}")
                st.write(f"ROC-AUC: **{rocA:.3f}** · PR-AUC: **{prA:.3f}**")

                with st.expander("Show Confusion Matrix (uploaded)"):
                    figCMu = px.imshow(confusion_matrix(y, p, labels=[0,1]), text_auto=True, color_continuous_scale="Greens",
                                    labels=dict(x="Predicted", y="True", color="Count"),
                                    x=["0 (NR)","1 (R)"], y=["0 (NR)","1 (R)"], title="Confusion Matrix (uploaded)")
                    st.plotly_chart(figCMu, use_container_width=True)
            else:
                st.info("Uploaded file had no matching indices for labels.")
        else:
            st.info("Upload a file with a `true_label` column (R/NR) to see performance metrics.")

    # ---- Predictions table AFTER metrics
    st.subheader("Predictions table")
    out_all = get_predictions_dataframe()
    tbl = out_all.copy()
    tbl.insert(0, "traffic", tbl["prediction"].map(lambda p:"🟢" if p==1 else "🔴"))
    tbl.insert(0, "patient_id", tbl.index)
    st.dataframe(
        tbl[["patient_id","traffic","probability","prediction"]]
            .style.format({"probability":"{:.3f}"}),
        use_container_width=True
    )
    st.download_button("Download predictions (CSV)", data=out_all.to_csv().encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv")

    render_page_chat("Performance")

# ================== PAGE: CELL MAP ==================
elif page == "Cell Map":
    st.header("🗺️ Cell Map")
    st.caption("Each dot is a single cell; closer dots mean more similar biology.")
    if sc_df is None:
        st.info("sc_annot.csv not found in DATA_DIR.")
    else:
        df_plot = sc_df.copy()
        df_plot = attach_response(df_plot, labels_df)

        mode = st.radio("Color by", ["Cell type / Cluster","Responder status","Gene expression"], horizontal=True, key="cellmap_mode")
        if mode == "Cell type / Cluster":
            fig = px.scatter(df_plot, x="umap1", y="umap2", color="cluster_label", render_mode="webgl", height=650)
        elif mode == "Responder status":
            lab = df_plot["response"].map({1:"Responders",0:"Non-responders"}).fillna("Unknown")
            fig = px.scatter(df_plot, x="umap1", y="umap2", color=lab,
                             color_discrete_map={"Responders":"#2ecc71","Non-responders":"#e74c3c","Unknown":"#95a5a6"},
                             render_mode="webgl", height=650)
        else:
            if not ALL_GENES:
                st.warning("Gene expression matrix not available (sc_expr.* missing).")
                fig = px.scatter(df_plot, x="umap1", y="umap2", color="cluster_label", render_mode="webgl", height=650)
            else:
                gene = st.selectbox("Gene", ALL_GENES, index=0, key="cellmap_gene")
                expr = read_gene_cols([gene])
                if expr is not None:
                    df_plot = df_plot.merge(expr, on="cell_id", how="left")
                    fig = px.scatter(df_plot, x="umap1", y="umap2", color=gene, color_continuous_scale="RdBu_r",
                                     render_mode="webgl", height=650)
                else:
                    st.warning("Could not read expression for this gene.")
                    fig = px.scatter(df_plot, x="umap1", y="umap2", color="cluster_label", render_mode="webgl", height=650)

        try:
            cent = df_plot.groupby("cluster_label")[["umap1","umap2"]].median().reset_index()
            fig.add_trace(go.Scatter(x=cent["umap1"], y=cent["umap2"], mode="text",
                                     text=cent["cluster_label"], textposition="middle center",
                                     textfont=dict(size=12,color="black"), showlegend=False, hoverinfo="skip"))
        except Exception:
            pass
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    render_page_chat("Cell Map")

# ================== PAGE: GENE EXPLORER ==================
elif page == "Gene Explorer":
    st.title("🧪 Gene Explorer — expression by group and cluster")

    if not ALL_GENES:
        st.info("Single-cell expression matrix not found (sc_expr.parquet).")
        st.stop()

    with st.expander("🔎 Pick a cluster & its markers", expanded=True):
        clusters_for_picker = sorted(list(CLUSTER2MARKERS.keys()))
        if not clusters_for_picker:
            st.warning("No marker files found under /Data/markers (CSV/XLSX).")
            st.stop()

        cluster_pick = st.selectbox("Cluster", clusters_for_picker, index=0, key="gx_cluster")
        cluster_genes_raw = CLUSTER2MARKERS.get(cluster_pick, [])
        cluster_genes_present = [g for g in cluster_genes_raw if g in ALL_GENES]

        st.caption(
            f"Found **{len(cluster_genes_raw)}** markers in files · "
            f"**{len(cluster_genes_present)}** present in expression matrix"
        )

        use_all = st.checkbox("Use all markers from this cluster", value=True, key="gx_useall")
        if use_all:
            panel_genes = cluster_genes_present
        else:
            panel_genes = st.multiselect(
                "Panel genes (for heatmap/UMAP coloring)",
                options=cluster_genes_present if cluster_genes_present else ALL_GENES,
                default=cluster_genes_present[:10] if cluster_genes_present else [],
                key="panel_genes_multiselect"
            )

        gene_options = cluster_genes_present if cluster_genes_present else ALL_GENES
        g1 = st.selectbox("Primary gene", options=gene_options, index=0, key="gx_g1")
        g2 = st.selectbox("Second gene (for co-expression)", options=["(none)"] + gene_options, index=0, key="gx_g2")

    need = []
    if g1 in ALL_GENES: need.append(g1)
    if g2 != "(none)" and g2 in ALL_GENES: need.append(g2)
    need += [g for g in panel_genes if g in ALL_GENES]
    need = list(dict.fromkeys(need))
    if not need:
        st.warning("Selected genes are not present in the matrix.")
        st.stop()

    expr = read_gene_cols(need)
    if expr is None or expr.empty:
        st.warning("Could not read expression; check sc_expr.parquet.")
        st.stop()

    df = sc_df.merge(expr, on="cell_id", how="left")
    df = attach_response(df, labels_df)

    label_map = {1:"Responders", 0:"Non-responders"}
    lab = df["response"].map(label_map).fillna("Unknown")
    green, red = "#2ecc71", "#e74c3c"
    colmap = {"Responders":green, "Non-responders":red, "Unknown":"#95a5a6"}

    with st.expander("🎯 Group distribution: Responders vs Non-responders (Violin / Box)", expanded=False):
        tabV, tabB = st.tabs(["Violin","Box"])
        with tabV:
            figv = px.violin(df, x=lab, y=g1, color=lab, box=True, points="all", height=440, color_discrete_map=colmap)
            figv.update_layout(xaxis_title="", yaxis_title=g1, legend_title_text="")
            st.plotly_chart(figv, use_container_width=True)
        with tabB:
            figb = px.box(df, x=lab, y=g1, color=lab, points="all", height=440, color_discrete_map=colmap)
            figb.update_layout(xaxis_title="", yaxis_title=g1, legend_title_text="")
            st.plotly_chart(figb, use_container_width=True)

    with st.expander("🧩 Expression by cell type / cluster (Violin + Means)", expanded=False):
        cluster_col = "cluster_label"
        figct = px.violin(df, x=cluster_col, y=g1, color=cluster_col, box=True, height=520)
        figct.update_layout(xaxis_title="", yaxis_title=g1, legend_title_text="")
        st.plotly_chart(figct, use_container_width=True)

        means = df.groupby(cluster_col)[g1].mean().sort_values(ascending=False)
        st.dataframe(means.to_frame(f"{g1} mean"), use_container_width=True)
        figbar = px.bar(means, title=f"{g1} mean by cluster", height=420)
        st.plotly_chart(figbar, use_container_width=True)

    with st.expander("🔥 Heatmap — group mean expression for selected markers", expanded=False):
        if panel_genes:
            exprH = read_gene_cols(panel_genes)
            if exprH is not None and not exprH.empty:
                dfH = sc_df.merge(exprH, on="cell_id", how="left")
                dfH = attach_response(dfH, labels_df)
                grp_lab = dfH["response"].map(label_map).fillna("Unknown")
                mat = dfH.groupby(grp_lab)[panel_genes].mean().sort_index()
                figH = px.imshow(mat, color_continuous_scale="RdBu_r", aspect="auto", height=420)
                figH.update_layout(xaxis_title="Gene", yaxis_title="Group", legend_title_text="")
                st.plotly_chart(figH, use_container_width=True)
            else:
                st.info("Selected heatmap genes were not found in the expression matrix.")
        else:
            st.caption("Pick genes above to render a heatmap.")

    with st.expander("🗺️ UMAP colored by gene", expanded=False):
        gene_for_umap = st.selectbox(
            "Gene to color UMAP",
            options=[g1] + [g for g in panel_genes if g != g1],
            index=0, key="gx_umap_gene"
        )
        exprU = read_gene_cols([gene_for_umap])
        if exprU is not None:
            dfU = sc_df.merge(exprU, on="cell_id", how="left")
            figU = px.scatter(dfU, x="umap1", y="umap2", color=gene_for_umap,
                              color_continuous_scale="RdBu_r", render_mode="webgl", height=650)
            figU.update_layout(legend_title_text="")
            st.plotly_chart(figU, use_container_width=True)
        else:
            st.info("Could not read expression for this gene.")

    with st.expander("🔗 Co-expression (scatter)", expanded=False):
        if g2 != "(none)" and (g2 in df.columns):
            fig2 = px.scatter(df, x=g1, y=g2, color=lab, trendline="ols", height=450,
                              color_discrete_map=colmap)
            fig2.update_layout(xaxis_title=g1, yaxis_title=g2, legend_title_text="")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("Pick a second gene above to enable co-expression.")

    render_page_chat("Gene Explorer")

# ================== PAGE: COMPARISON ==================
elif page == "Comparison":
    st.header("🆚 Patient vs Group Averages — traffic light")
    out = get_predictions_dataframe()
    if out.empty:
        st.info("Run predictions first.")
    else:
        psel = st.selectbox("Choose a patient", options=out.index.tolist(), index=0, key="cmp_patient")
        row  = out.loc[psel]
        traffic = "🟢" if int(row["prediction"])==1 else "🔴"
        st.metric("Traffic Light (prediction)", f"{traffic}  {int(row['prediction'])}", delta=f"{row['probability']:.3f} prob")

        feat_common=X_all.columns.intersection(feat_names)
        vx=X_all.loc[[psel], feat_common]
        if labels_df is not None:
            dfj=X_all.join(labels_df, how="inner")
            mu_R=dfj[dfj["response"]==1][feat_common].mean()
            mu_N=dfj[dfj["response"]==0][feat_common].mean()
            comp=pd.DataFrame({"patient":vx.squeeze(),"mean_R":mu_R,"mean_N":mu_N})
            comp["delta_R"]=comp["patient"]-comp["mean_R"]; comp["delta_N"]=comp["patient"]-comp["mean_N"]
            k=st.slider("Show top N differing features",5,min(30,comp.shape[0]),10, key="cmp_topN")
            top=comp.reindex(comp["delta_R"].abs().sort_values(ascending=False).head(k).index)
            figC=go.Figure()
            figC.add_trace(go.Bar(name="Patient − Mean(R)", x=top.index, y=top["delta_R"]))
            figC.add_trace(go.Bar(name="Patient − Mean(NR)", x=top.index, y=top["delta_N"]))
            figC.update_layout(barmode="group", xaxis_title="Feature", yaxis_title="Difference", height=450)
            st.plotly_chart(figC, use_container_width=True)
            with st.expander("Raw table"):
                st.dataframe(top, use_container_width=True)
        else:
            st.info("Labels not available; group averages require response labels.")

    render_page_chat("Comparison")

# ================== PAGE: WHAT DRIVES RESPONSE (Explainability) ==================
elif page == "What Drives Response":
    st.header("🧠 What Drives Response")

    with st.expander("Show feature importances and pathway panels", expanded=False):
        # ---- Feature Importance ----
        fi=feature_importance_series(model, feat_names)
        try:
            import shap; shap_available=True
        except Exception:
            shap_available=False

        if shap_available:
            try:
                explainer=shap.Explainer(model, X_all[feat_names])
                sample=X_all[feat_names].sample(min(500,X_all.shape[0]), random_state=42)
                sv=explainer(sample)
                shap_df=pd.DataFrame(np.abs(sv.values).mean(axis=0), index=feat_names, columns=["mean|SHAP|"]).sort_values("mean|SHAP|", ascending=False)
                topk=st.slider("Top K",5,min(30,len(shap_df)),10, key="wd_topk_shap")
                figImp=px.bar(shap_df.head(topk), height=420)
                st.plotly_chart(figImp, use_container_width=True)
            except Exception:
                fi = fi or correlation_importance(model, X_all, feat_names)
                if fi is not None:
                    topk=st.slider("Top features",5,min(30,len(fi)),10, key="wd_topk_corr1")
                    figImp=px.bar(fi.head(topk), height=420)
                    st.plotly_chart(figImp, use_container_width=True)
                else:
                    st.info("No importances available.")
        else:
            if fi is None: fi=correlation_importance(model, X_all, feat_names)
            if fi is not None:
                topk=st.slider("Top features",5,min(30,len(fi)),10, key="wd_topk_corr2")
                figImp=px.bar(fi.head(topk), height=420)
                st.plotly_chart(figImp, use_container_width=True)
            else:
                st.info("No importances available.")

        # ---- Hallmark Explorer with TOGGLE (NEW) ----
        st.subheader("Hallmark Explorer — pathway scores by response")

        # Toggle for source
        h_src = st.radio("Hallmark source", ["All cells Hallmarks", "CD8/TNK Hallmarks"],
                         horizontal=True, key="hallmark_source")

        H_ALL_FILE = os.path.join(DATA_DIR, "patient_features_with_hallmark.csv")
        H_CD8_FILE = os.path.join(DATA_DIR, "patient_features_with_hallmark_CD8.csv")
        h_path = H_CD8_FILE if h_src == "CD8/TNK Hallmarks" else H_ALL_FILE

        h_df = None
        if os.path.exists(h_path):
            try:
                h_df = pd.read_csv(h_path, index_col=0)
            except Exception as e:
                st.warning(f"Could not read {os.path.basename(h_path)}")
                st.exception(e)
        else:
            st.info(f"Missing file: {os.path.basename(h_path)} under Data. Please add it to enable this source.")

        if h_df is not None and not h_df.empty:
            hallmark_cols = [c for c in h_df.columns
                             if str(c).upper().startswith("H_") or str(c).upper().startswith("HALLMARK_")]
            if len(hallmark_cols) == 0:
                st.info("No Hallmark columns (prefix 'H_' or 'HALLMARK_') found in the selected table.")
            else:
                h_df2 = h_df.copy()
                if "patient_id" not in h_df2.columns:
                    h_df2 = h_df2.reset_index().rename(columns={"index":"patient_id"})
                h_df2 = attach_response(h_df2, labels_df)

                if "response" not in h_df2.columns or h_df2["response"].isna().all():
                    st.info("No response labels matched these patients; cannot compare groups.")
                else:
                    chosen = st.selectbox("Choose a Hallmark pathway", sorted(hallmark_cols), key="wd_hallmark_pick")
                    vals0 = pd.to_numeric(h_df2.loc[h_df2["response"]==0, chosen], errors="coerce")
                    vals1 = pd.to_numeric(h_df2.loc[h_df2["response"]==1, chosen], errors="coerce")

                    figHmk = go.Figure()
                    figHmk.add_trace(go.Box(y=vals0, name="Non-responders (0)"))
                    figHmk.add_trace(go.Box(y=vals1, name="Responders (1)"))
                    figHmk.update_layout(title=chosen.replace("HALLMARK_", "").replace("H_", "").replace("_", " "),
                                         yaxis_title="Hallmark score (feature scale)", height=420)
                    st.plotly_chart(figHmk, use_container_width=True)

                    diff = (np.nanmean(vals1) - np.nanmean(vals0)) if (len(vals0)>0 and len(vals1)>0) else np.nan
                    st.caption(f"Mean difference (R − NR): {diff:.3f} (higher → more active in responders)")
        else:
            st.info("No Hallmark table loaded for the selected source.")

        # ---- GSEA (CD8) — now in a DROPDOWN (expander) ----
        with st.expander("🧪 GSEA (CD8) — preranked (from cached results)", expanded=False):
            def load_cached_gsea_csv(path_csv: str) -> pd.DataFrame | None:
                if not os.path.exists(path_csv):
                    return None
                try:
                    df = pd.read_csv(path_csv)
                    low = {c: c.lower() for c in df.columns}
                    df = df.rename(columns=low)
                    ren = {
                        "fdr q-val":"fdr", "fdr_q_val":"fdr", "q":"fdr", "padj":"fdr", "adj_pval":"fdr",
                        "nom p-val":"pval", "nom_p_val":"pval", "p":"pval"
                    }
                    df = df.rename(columns=ren)
                    if "term" not in df.columns:
                        if "name" in df.columns:
                            df["term"] = df["name"]
                    keep = [c for c in ["term","nes","fdr","pval"] if c in df.columns]
                    df = df[keep].copy()
                    df = df.loc[:,~df.columns.duplicated()].copy()
                    for c in ["nes","fdr","pval"]:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                    return df
                except Exception:
                    return None

            gsea_candidates = [
                os.path.join(DATA_DIR, "gsea_prerank_cd8_hallmark.csv"),
                os.path.join(DATA_DIR, "gsea_prerank_cd8.csv"),
                os.path.join(DATA_DIR, "gsea_prerank_results_cd8.csv"),
            ]
            gsea_df = None
            for pth in gsea_candidates:
                gsea_df = load_cached_gsea_csv(pth)
                if gsea_df is not None and not gsea_df.empty:
                    st.caption(f"Loaded cached GSEA results: {os.path.basename(pth)}")
                    break

            if gsea_df is None or gsea_df.empty:
                st.info("No cached GSEA CSV found (looked for gsea_prerank_cd8_hallmark.csv / gsea_prerank_cd8.csv).")
            else:
                sig_col = "fdr" if "fdr" in gsea_df.columns else ("pval" if "pval" in gsea_df.columns else None)
                if sig_col is None:
                    st.warning(f"Could not find FDR/p columns. Columns were: {list(gsea_df.columns)}")
                else:
                    alpha = 0.25 if sig_col == "fdr" else 0.05
                    up = gsea_df[gsea_df["nes"]>0].sort_values([sig_col,"nes"], ascending=[True,False])
                    dn = gsea_df[gsea_df["nes"]<0].sort_values([sig_col,"nes"], ascending=[True,True])

                    only_sig = st.checkbox(f"Show only significant sets (alpha = {alpha})", value=True, key="wd_gsea_sig")
                    if only_sig:
                        up = up[up[sig_col] <= alpha]
                        dn = dn[dn[sig_col] <= alpha]

                    # brief summary so users know if anything was found
                    st.caption(f"Responder-enriched (NES>0) passing alpha: **{len(up)}** · "
                               f"Non-responder-enriched (NES<0) passing alpha: **{len(dn)}**")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Responder-enriched (NES > 0)**")
                        if up.empty:
                            st.info("No responder-enriched pathways under the current filter.")
                        else:
                            st.dataframe(up[["term","nes"] + ([sig_col] if sig_col in up.columns else [])].reset_index(drop=True),
                                         use_container_width=True)
                    with c2:
                        st.markdown("**Non-responder-enriched (NES < 0)**")
                        if dn.empty:
                            st.info("No non-responder-enriched pathways under the current filter.")
                        else:
                            st.dataframe(dn[["term","nes"] + ([sig_col] if sig_col in dn.columns else [])].reset_index(drop=True),
                                         use_container_width=True)

    render_page_chat("What Drives Response")

# ================== PAGE: CHAT (UPDATED) ==================
elif page == "Chat":
    st.markdown("<h2 style='display:flex;align-items:center;gap:8px'>💬 <span>How can I help?</span></h2>", unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    def parse_patient_id_from_text(text: str, known_ids: list[str]) -> str | None:
        """
        Pull a patient id from free text:
        'patient 20', 'patient P20', 'is P7 a responder', 'tell me about P003'
        """
        t = (text or "").strip().lower()
        for pid in known_ids:
            if pid.lower() == t:
                return pid
        m = re.search(r"(?:^|[^a-z0-9])patient\s*([a-z0-9\-_]+)", t)
        token = m.group(1) if m else None
        if token is None:
            m2 = re.search(r"\b(p[\-_]?\d+)\b", t)
            token = m2.group(1) if m2 else None
        if token is None:
            m3 = re.search(r"\b(\d+)\b", t)
            token = m3.group(1) if m3 else None
        if token is None:
            return None
        cand = token.replace("_", "").replace("-", "")
        for pid in known_ids:
            if pid.lower().replace("_", "").replace("-", "") == cand:
                return pid
        for pid in known_ids:
            if cand in pid.lower().replace("_", "").replace("-", ""):
                return pid
        return None

    q = st.text_input("Ask anything about the app, metrics, clusters, genes, or patients (e.g., 'Is patient P7 a responder?')", key="chat_input")
    if st.button("Ask", key="chat_btn"):
        st.session_state["chat"].append(("You", q))
        ql = (q or "").strip().lower()
        ans = None

        def has(*kw): return any(k in ql for k in kw)

        # ---- Patient-level responder question ----
        if any(k in ql for k in ["responder", "respond", "responding", "will respond", "likely to respond"]):
            preds_df = get_predictions_dataframe()
            if preds_df is not None and preds_df.shape[0] > 0:
                pid = parse_patient_id_from_text(q, preds_df.index.astype(str).tolist())
                if pid is not None and pid in preds_df.index:
                    p = float(preds_df.loc[pid, "probability"])
                    y = int(preds_df.loc[pid, "prediction"])
                    label = "Responder (1)" if y == 1 else "Non-responder (0)"
                    ans = f"**{pid}** → **{label}** with probability **{p:.3f}** (threshold = {thr:.2f})."
                else:
                    ans = ("I couldn't find a patient id in your question. Try e.g. "
                           "`Is patient P7 a responder?` or `Will patient 20 respond?`")

        # keep your functional intents
        if ans is None and has("cluster names","cell types","list clusters"):
            if sc_df is None:
                ans = "Single-cell table not loaded."
            else:
                names = sc_df["cluster_label"].astype(str).unique().tolist()
                st.dataframe(pd.Series(names, name="clusters"), use_container_width=True)
                ans = f"I listed {len(names)} cluster names above."
        elif ans is None and "average expression of gene" in ql:
            m = re.search(r"average expression of gene ([a-z0-9\-_\.]+)", ql)
            if m and ALL_GENES:
                g = next((x for x in ALL_GENES if x.lower()==m.group(1).lower()), None)
                if g:
                    expr = read_gene_cols([g])
                    if expr is not None and sc_df is not None:
                        df = sc_df.merge(expr, on="cell_id", how="left")
                        df = attach_response(df, labels_df)
                        grp = df.groupby(df["response"].map({1:"Responders",0:"Non-responders"}).fillna("Unknown"))[g].mean()
                        st.bar_chart(grp, use_container_width=True)
                        ans = f"Plotted average {g} by response group."
                else:
                    ans = "That gene is not in the expression matrix."
            else:
                ans = "Gene matrix not available."
        elif ans is None and has("top features","most important"):
            fi = feature_importance_series(model, feat_names) or correlation_importance(model, X_all, feat_names)
            if fi is not None:
                top = fi.head(12)
                st.dataframe(top.to_frame("importance"), use_container_width=True)
                st.plotly_chart(px.bar(top.sort_values(), orientation="h"), use_container_width=True)
                ans = "Top features shown above."
            else:
                ans = "No importances available."
        elif ans is None and has("how many cells","total cells","number of cells"):
            ans = f"There are **{0 if sc_df is None else sc_df.shape[0]:,}** cells."
        elif ans is None and has("how many patients","number of patients"):
            ans = f"There are **{X_all.shape[0]}** patients with features."

        # fallback to knowledge base
        if ans is None:
            ans = kb_answer(q, page="Chat")

        st.session_state["chat"].append(("App", ans))

    with st.expander("Common questions"):
        cols = st.columns(3)
        examples = [
            "What are features?",
            "What is F1 score?",
            "What is sensitivity?",
            "What is specificity?",
            "What is a confusion matrix?",
            "What is a cluster?",
            "What does the Cell Map show?",
            "What is co-expression?",
            "What is Patient vs Group Averages?",
            "Which features matter on What Drives Response?",
            "What is this app (overall)?",
            "Is patient P7 a responder?"
        ]
        for i, e in enumerate(examples):
            if cols[i % 3].button(e, key=f"qa_{i}"):
                st.session_state["chat"].append(("You", e))
                if "patient p7" in e.lower():
                    preds_df = get_predictions_dataframe()
                    if preds_df is not None and "P7" in preds_df.index:
                        p = float(preds_df.loc["P7","probability"])
                        y = int(preds_df.loc["P7","prediction"])
                        ans = f"**P7** → **{'Responder (1)' if y==1 else 'Non-responder (0)'}** with probability **{p:.3f}** (threshold = {thr:.2f})."
                    else:
                        ans = "I couldn't find P7 in the current cohort."
                else:
                    ans = kb_answer(e, page="Chat")
                st.session_state["chat"].append(("App", ans))
                st.rerun()

    for who, msg in st.session_state["chat"]:
        st.markdown(f"**{who}:** {msg}")

# ================== PAGE: BACKGROUND (INTRO SLIDES + APP OVERVIEW) ==================
elif page == "Background":
    st.header("📘 Introduction")

    # ---- CSS for integrated slide card (image + text in one card)
    st.markdown("""
    <style>
      .slide-card{
        display:flex; gap:24px; align-items:center;
        background:#ffffff; border-radius:18px; padding:18px 20px;
        border:1px solid #e6eef7; box-shadow:0 6px 22px rgba(0,0,0,0.06);
      }
      .slide-img{
        width:58%; height:430px; border-radius:14px; overflow:hidden;
        background:#eef3fb;
      }
      .slide-img img{ width:100%; height:100%; object-fit:cover; display:block; }
      .slide-text{ flex:1; font-size:1.12rem; line-height:1.65; }
      .slide-text h2{ font-size:1.9rem; margin:0 0 10px 0; }
      .slide-text ul{ margin:0 0 0 1.1rem; }
      .slide-controls{ display:flex; gap:14px; margin-top:10px; }
    </style>
    """, unsafe_allow_html=True)

    # ---- slide builder (returns list of (title, img_path, markdown_text))
    def build_intro_slides(slides_dir: str):
        import glob
        imgs = glob.glob(os.path.join(slides_dir, "*.png")) + \
               glob.glob(os.path.join(slides_dir, "*.jpg")) + \
               glob.glob(os.path.join(slides_dir, "*.jpeg"))
        byname = {os.path.basename(p).lower(): p for p in imgs}

        def pick(*keys):
            for k in keys:
                for name, path in byname.items():
                    if k in name:
                        return path
            return None

        img_world  = pick("world")
        img_doctor = pick("doctor")
        img_dna    = pick("dna")

        text1 = (
            "## Immunotherapy Response Explorer\n"
            "- Cancer is still one of the leading causes of death.\n"
            "- **~20 million** people diagnosed with cancer each year\n"
            "- **9.7 million** deaths annually\n"
            "- **~35 million** new cases projected each year by **2050**\n\n"
            "*That’s not just numbers — it’s families, choices, and time we can’t waste.*"
        )
        text2 = (
            "### The Current Reality\n"
            "Treatments like immunotherapy can be life-saving but are also **painful** and carry side effects: fatigue, "
            "nausea, hair loss, infections, and weakened immunity.\n\n"
            "- Not every patient responds to these treatments.\n"
            "- Doctors often try different treatments repeatedly, causing more suffering."
        )
        text3 = (
            "### What If We Knew Beforehand?\n"
            "Imagine if we could predict which patients will respond to immunotherapy.\n\n"
            "This would:\n"
            "- **Reduce** patient suffering\n"
            "- **Avoid** unnecessary side effects\n"
            "- **Save** critical time in cancer treatment\n"
            "- Make **personalized** treatment possible\n\n"
            "To address this, we built an app that turns complex immune-cell and gene data into a **clear, explainable** "
            "prediction of who is more likely to benefit from immunotherapy."
        )

        slides = [
            ("The Scale of the Problem", img_world,  text1),
            ("Why It’s Hard Today",      img_doctor, text2),
            ("Imagine + Our Solution",   img_dna,    text3),
        ]
        return slides

    def encode_image_b64(path: str):
        if not path or not os.path.exists(path): return None, None
        ext = os.path.splitext(path)[1].lower()
        mime = "jpeg" if ext in [".jpg", ".jpeg"] else "png"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return mime, b64

    def strip_first_heading(md: str) -> str:
        lines = (md or "").strip().splitlines()
        if lines and (lines[0].startswith("##") or lines[0].startswith("###")):
            return "\n".join(lines[1:])
        return md

    def md_to_html(md: str) -> str:
        """Very light markdown → HTML for headings/bullets/paragraphs."""
        lines = (md or "").strip().splitlines()
        html, in_ul = [], False
        for line in lines:
            if line.startswith("## "):
                if in_ul: html.append("</ul>"); in_ul=False
                html.append(f"<h2>{line[3:].strip()}</h2>")
            elif line.startswith("### "):
                if in_ul: html.append("</ul>"); in_ul=False
                html.append(f"<h3>{line[4:].strip()}</h3>")
            elif line.strip().startswith("- "):
                if not in_ul:
                    html.append("<ul>"); in_ul=True
                html.append(f"<li>{line.strip()[2:].strip()}</li>")
            elif line.strip()=="":
                if in_ul: html.append("</ul>"); in_ul=False
                html.append("<p>&nbsp;</p>")
            else:
                if in_ul: html.append("</ul>"); in_ul=False
                html.append(f"<p>{line.strip()}</p>")
        if in_ul: html.append("</ul>")
        return "\n".join(html)

    slides = build_intro_slides(SLIDES_DIR)

    if "intro_idx" not in st.session_state:
        st.session_state["intro_idx"] = 0

    any_img = any((p and os.path.exists(p)) for _, p, _ in slides)
    if not any_img:
        st.warning(f"No images found in {SLIDES_DIR}. Add JPG/PNG files to show slide images. "
                   "Text slides are still displayed below.")

    # Ensure index in range
    idx = max(0, min(st.session_state["intro_idx"], len(slides) - 1))
    title, img_path, md_text = slides[idx]
    mime, b64 = encode_image_b64(img_path)
    body_html = md_to_html(strip_first_heading(md_text))

    # Render single integrated slide card
    if b64:
        st.markdown(
            f"""
            <div class="slide-card">
              <div class="slide-img">
                <img src="data:image/{mime};base64,{b64}" alt="{title}" />
              </div>
              <div class="slide-text">
                <h2>{title}</h2>
                {body_html}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="slide-card">
              <div class="slide-img" style="display:flex;align-items:center;justify-content:center;color:#667;">
                <span>Image not found</span>
              </div>
              <div class="slide-text">
                <h2>{title}</h2>
                {body_html}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Prev / Next controls (thumbnails removed)
    c1, c2, c3 = st.columns([0.24, 0.24, 0.52])
    with c1:
        if st.button("⬅️ Previous", use_container_width=True, disabled=(idx == 0)):
            st.session_state["intro_idx"] = idx - 1
            st.rerun()
    with c2:
        if st.button("Next ➡️", use_container_width=True, disabled=(idx == len(slides) - 1)):
            st.session_state["intro_idx"] = idx + 1
            st.rerun()
    with c3:
        st.caption(f"Slide {idx+1} / {len(slides)}")

    # ------- What this app does (before Page guide) -------
    st.markdown("### What this app does")
    st.markdown(
        "- Turns complex patient & single-cell data into **clean visuals**.\n"
        "- Highlights which **immune cells** are **active** or **exhausted**.\n"
        "- Compares **responders vs non-responders** (e.g., **CD8 T cells**).\n"
        "- Surfaces **biological pathways** linked to success or resistance.\n"
        "- Provides **AI predictions** with explanations."
    )

    # ------- Page guide (kept) -------
    st.markdown("### Page guide (what to look for)")
    st.markdown(
        "- **📊 Performance** — Metrics, ROC/PR curves, confusion matrix.  \n"
        "  *Tip:* Tune the threshold to balance precision vs recall.  \n"
        "- **🗺️ Cell Map** — UMAP colored by cluster, responder status, or gene expression.  \n"
        "  *Tip:* Are responder-enriched clusters spatially coherent?  \n"
        "- **🧪 Gene Explorer** — Violin/box by group, cluster means, heatmaps, UMAP coloring, co-expression.  \n"
        "  *Tip:* Check CD8 cytotoxic (**GZMB/PRF1**) vs memory-like (**IL7R/TCF7**) genes.  \n"
        "- **🆚 Comparison** — One patient vs group means with a traffic-light prediction.  \n"
        "  *Tip:* Inspect features driving the patient’s deviation.  \n"
        "- **🧠 What Drives Response** — SHAP/model importances, **Hallmark Explorer**, and optional **CD8 GSEA**."
    )

    render_page_chat("Background")

# ================== PAGE: SUMMARY ==================
elif page == "Summary":
    st.header("🌟 Summary")
    st.markdown("""
The **Immunotherapy Response Explorer** is designed to make complex science simple and meaningful.
It brings together patient data, immune cell information, and artificial intelligence to help us
understand how each person’s immune system responds to cancer treatment.

By studying the immune system at the level of single cells, we can:
- Detect early signs of response or resistance  
- Guide the design of smarter, more targeted therapies  
- Support personalized treatment, giving each patient care that fits their biology  
- Help doctors make more confident, evidence-based treatment choices

This approach moves us closer to **personalized medicine** — where each patient gets the treatment
that fits their biology, not a one-size-fits-all approach.
""")
    # Thank you at the end (a bit larger)
    st.markdown("<h2 style='text-align:center; margin-top: 1.25rem;'>Thank you</h2>", unsafe_allow_html=True)

    render_page_chat("Summary")

# ================== REPORT EXPORT (appears on all pages) ==================
st.markdown("---")
st.subheader("📄 Download quick report")
def build_report_html():
    out = get_predictions_dataframe()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts=[f"<h1>Immunotherapy Response — Report</h1><p>Generated: {ts}</p>"]

    try:
        common = out.index.intersection(labels_df.index) if labels_df is not None else []
        if len(common)>0:
            y_pred = out.loc[common,"prediction"].astype(int).values
            y_prob = out.loc[common,"probability"].values
            y_true = labels_df.loc[common,"response"].astype(int).values
            tn,fp,fn,tp = confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
            acc  = (tp+tn)/(tp+tn+fp+fn)
            sens = tp/(tp+fn) if (tp+fn) else 0.0
            spec = tn/(tn+fp) if (tn+fp) else 0.0
            rocA = roc_auc_score(y_true,y_prob)
            prA  = average_precision_score(y_true,y_prob)
            parts.append(f"<h2>Internal evaluation</h2><ul>"
                         f"<li>Accuracy: {acc:.3f}</li>"
                         f"<li>Sensitivity: {sens:.3f}</li>"
                         f"<li>Specificity: {spec:.3f}</li>"
                         f"<li>ROC-AUC: {rocA:.3f}</li>"
                         f"<li>PR-AUC: {prA:.3f}</li></ul>")
    except Exception:
        pass

    try:
        parts.append("<h2>Predictions (top 50)</h2>")
        parts.append(get_predictions_dataframe().head(50).to_html())
    except Exception:
        pass

    return "".join(parts).encode("utf-8")

st.download_button("Download HTML report", data=build_report_html(),
                   file_name="immunotherapy_report.html", mime="text/html")

# ================== SIDEBAR (bottom): Appearance + Clear cache ==================
def set_background(kind="gradient", color="#dff3c4", color2="#b7e39c", image_bytes=None):
    if kind=="image" and image_bytes is not None:
        b64=base64.b64encode(image_bytes).decode()
        css=f'<style>.stApp{{background:url("data:image/png;base64,{b64}") no-repeat center center fixed;background-size:cover;}}</style>'
    elif kind=="solid":
        css=f"<style>.stApp{{background:{color} !important;}}</style>"
    else:
        css=f"<style>.stApp{{background:linear-gradient(135deg,{color} 0%,{color2} 100%) !important;}}</style>"
    st.markdown(css, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.header("Appearance")
bg_mode = st.sidebar.selectbox("Background", ["gradient","solid","image"], index=0, key="bg_mode")
if bg_mode=="solid":
    set_background("solid", color=st.sidebar.color_picker("Background color", "#f7fbff", key="bg_col"))
elif bg_mode=="gradient":
    c1=st.sidebar.color_picker("Gradient start", "#eef7ff", key="bg_g1")
    c2=st.sidebar.color_picker("Gradient end",   "#dbeeff", key="bg_g2")
    set_background("gradient", c1, c2)
else:
    up_bg=st.sidebar.file_uploader("Upload background image", type=["png","jpg","jpeg"], key="bg_upload")
    set_background("image", image_bytes=up_bg.read()) if up_bg else set_background("gradient")

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear app cache", key="clear_cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
