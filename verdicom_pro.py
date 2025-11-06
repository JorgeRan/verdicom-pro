"""
VERDICOM PRO - Interactive DICOM Viewer in Streamlit
Ready to run:
    streamlit run verdicom.py
Requirements:
    pip install streamlit pydicom numpy matplotlib pillow
"""

# ==============================
# 🧩 IMPORTS
# ==============================
import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import io
import base64
from datetime import datetime

# ==============================
# 🧩 ESTILO (CSS INLINE)
# ==============================
_BASE_CSS = """
<style>
:root{
  --kairos-blue: #0d6efd;
  --kairos-red: #ec4c4c;
  --surface: #f6f8fa;
  --card-radius: 12px;
  --text-size: 16px;
  --muted: #6c757d;
}

/* container and cards */
main .block-container {
  max-width: 1200px;
  padding-top: 1.5rem;
  padding-bottom: 2rem;
  font-family: "Helvetica Neue", "Segoe UI", Arial, sans-serif;
  font-size: var(--text-size);
  color: #111827;
}

.viewer-card {
  background: white;
  border-radius: var(--card-radius);
  padding: 14px;
  box-shadow: 0 6px 18px rgba(17,24,39,0.06);
}

/* rounded images */
.viewer-image img {
  border-radius: 10px;
  box-shadow: 0 8px 24px rgba(2,6,23,0.06);
  border: 1px solid rgba(13,110,253,0.06);
}

/* sliders and buttons */
.stSlider > div:nth-child(1) {
  accent-color: var(--kairos-red);
}

/* minor text */
.muted { color: var(--muted); font-size: 0.9rem; }

/* metadata key */
.meta-key { color: #334155; font-weight: 600; margin-right: 6px; }

/* download button accent */
.stButton>button {
  border-radius: 10px;
}

/* small responsive tweak */
@media (max-width: 900px) {
  main .block-container { padding-left: 1rem; padding-right: 1rem; }
}

/* Dark mode automatic (via prefers-color-scheme) */
@media (prefers-color-scheme: dark) {
  :root{
    --surface: #0b1220;
    --text-size: 16px;
    --muted: #9aa4b2;
  }
  main .block-container { color: #e6eef8; }
  .viewer-card { background: #071223; box-shadow: 0 6px 20px rgba(0,0,0,0.8); border: 1px solid rgba(255,255,255,0.03); }
  .viewer-image img { border: 1px solid rgba(255,255,255,0.03); }
}
</style>
"""

st.set_page_config(page_title="VERDICOM PRO", layout="wide", initial_sidebar_state="auto")
st.markdown(_BASE_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_dicom(file) -> pydicom.dataset.FileDataset:
    """
    Reads a DICOM file from Streamlit's uploader and returns the dataset.
    file: UploadedFile or file-like
    """
    try:
        ds = pydicom.dcmread(file, force=False)
        return ds
    except Exception as e:
        file.seek(0)
        raw = file.read()
        bio = io.BytesIO(raw)
        ds = pydicom.dcmread(bio, force=True)
        return ds

def get_pixel_array(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Extracts the pixel array and converts it to float (normalized internally).
    """
    arr = ds.pixel_array.astype(np.float32)
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    arr = arr * slope + intercept
    return arr

def default_window(ds: pydicom.dataset.FileDataset):
    """
    Gets WindowCenter and WindowWidth from DICOM if available; returns (center, width).
    If missing, computes them from percentiles of the pixel array.
    """
    wc = None; ww = None
    if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        if isinstance(wc, pydicom.multival.MultiValue):
            wc = float(wc[0])
        else:
            wc = float(wc)
        if isinstance(ww, pydicom.multival.MultiValue):
            ww = float(ww[0])
        else:
            ww = float(ww)
        return wc, ww
    # fallback: use center=median, width = (p99 - p1)
    arr = get_pixel_array(ds)
    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)
    center = (p99 + p1) / 2.0
    width = max(1.0, p99 - p1)
    return float(center), float(width)

def apply_window(img: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Applies windowing to the image and returns uint8 (0-255).
    Formula: map [c-w/2, c+w/2] -> [0,255], with clipping.
    """
    lower = center - (width / 2.0)
    upper = center + (width / 2.0)
    img_w = np.clip(img, lower, upper)
    img_norm = (img_w - lower) / (upper - lower)
    img_8 = np.uint8(np.round(img_norm * 255.0))
    return img_8

def to_pil(img_array: np.ndarray) -> Image.Image:
    """
     Converts uint8 array (H,W) to a PIL image in L mode.
    Handles 3D arrays with color channels if needed.
    """
    if img_array.ndim == 2:
        pil = Image.fromarray(img_array, mode='L')
    elif img_array.ndim == 3 and img_array.shape[2] == 3:
        pil = Image.fromarray(img_array)
    else:
        # fallback: flatten
        arr = np.squeeze(img_array)
        arr = np.uint8(np.clip(arr, 0, 255))
        pil = Image.fromarray(arr)
    return pil

def pil_to_bytes(pil_img: Image.Image, fmt="PNG") -> bytes:
    bio = io.BytesIO()
    pil_img.save(bio, format=fmt, optimize=True)
    bio.seek(0)
    return bio.read()

def show_image(img, caption=None):
    """
    Compatible display handler for different Streamlit versions.
    img can be PIL.Image, numpy array, or file path.
    """
    try:
        st.image(img, caption=caption, use_column_width=True)
    except TypeError:
        # versiones antiguas: use width en su lugar
        st.image(img, caption=caption, width=700)

def safe_get(ds, key, default="—"):
    try:
        val = getattr(ds, key)
        if val is None:
            return default
        # convert multiple values elegantly
        if isinstance(val, (list, pydicom.multival.MultiValue, tuple)):
            return str(val[0])
        return str(val)
    except Exception:
        return default

# ==============================
# 🧩 UI PRINCIPAL
# ==============================
def main():
    st.markdown("<h1 style='margin-bottom:0.2rem'>VERDICOM PRO</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Visor DICOM clínico — brillo, contraste, histograma y exportación</div>", unsafe_allow_html=True)
    st.write("")  # separator

    # Layout: 2 columnas (izquierda metadata, derecha visor)
    col_meta, col_view = st.columns([1, 2], gap="large")

    with col_meta:
        st.markdown("<div class='viewer-card'>", unsafe_allow_html=True)
        st.markdown("### 📁 Load DICOM")
        uploaded = st.file_uploader("Choose a .dcm file", type=["dcm", "dicom"], accept_multiple_files=False)
        st.markdown("---")
        st.markdown("### 🔎 Main Metadata", unsafe_allow_html=True)

        # placeholders for metadata
        meta_placeholder = st.empty()
        st.markdown("---")
        st.markdown("### 🔬 Technical Details", unsafe_allow_html=True)
        tech_placeholder = st.empty()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_view:
        st.markdown("<div class='viewer-card viewer-image'>", unsafe_allow_html=True)
        image_area = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    # Si no hay archivo, mostrar instrucciones útiles
    if not uploaded:
        with col_meta:
            st.info("Upload a DICOM file (.dcm) here. You can also drag and drop.")
            st.write("Tip: if 'streamlit run verdicom.py' doesn't work, try 'python -m streamlit run verdicom.py' in PowerShell/CMD.")
        return

    # Cargar dataset
    try:
        ds = load_dicom(uploaded)
    except Exception as e:
        st.error(f"Could not read DICOM file: {e}")
        return

    # Mostrar metadatos
    patient_name = safe_get(ds, "PatientName")
    patient_id = safe_get(ds, "PatientID")
    patient_age = safe_get(ds, "PatientAge")
    patient_sex = safe_get(ds, "PatientSex")
    study_date = safe_get(ds, "StudyDate")
    study_time = safe_get(ds, "StudyTime")
    modality = safe_get(ds, "Modality")
    institution = safe_get(ds, "InstitutionName")

    # Feature: formatea fecha/hora si es posible
    def fmt_dt(dstr, tstr):
        try:
            if dstr is None or dstr == "—":
                return "—"
            # DICOM fecha: YYYYMMDD
            if len(dstr) >= 8:
                dt = datetime.strptime(dstr[:8], "%Y%m%d")
                date_part = dt.strftime("%Y-%m-%d")
            else:
                date_part = dstr
            t = ""
            if tstr and tstr != "—":
                # intentar formatear HHMMSS
                tclean = str(tstr).split(".")[0]
                if len(tclean) >= 6:
                    t = datetime.strptime(tclean[:6], "%H%M%S").strftime("%H:%M:%S")
                elif len(tclean) >= 4:
                    t = datetime.strptime(tclean[:4], "%H%M").strftime("%H:%M")
                else:
                    t = tclean
            return f"{date_part} {t}".strip()
        except Exception:
            return f"{dstr} {tstr}"

    study_dt = fmt_dt(study_date, study_time)

    meta_md = f"""
**Patient:** <span class='meta-key'>{patient_name}</span>  
**ID:** <span class='meta-key'>{patient_id}</span>  
**Age:** <span class='meta-key'>{patient_age}</span>  
**Sex:** <span class='meta-key'>{patient_sex}</span>  

**Study:** <span class='meta-key'>{study_dt}</span>  
**Modality:** <span class='meta-key'>{modality}</span>  
**Institution:** <span class='meta-key'>{institution}</span>  
"""
    meta_placeholder.markdown(meta_md, unsafe_allow_html=True)

    # Detalles técnicos (pixel spacing, dimensions, bits)
    try:
        pixel_spacing = safe_get(ds, "PixelSpacing")
    except Exception:
        pixel_spacing = "—"
    rows = getattr(ds, 'Rows', '—')
    cols = getattr(ds, 'Columns', '—')
    bits_allocated = getattr(ds, 'BitsAllocated', '—')
    bits_stored = getattr(ds, 'BitsStored', '—')
    photometric = getattr(ds, 'PhotometricInterpretation', '—')
    samples_per_pixel = getattr(ds, 'SamplesPerPixel', '—')
    transfer_syntax = "—"
    try:
        transfer_syntax = ds.file_meta.TransferSyntaxUID
    except Exception:
        pass

    tech_md = f"""
- **Dimensions:** {rows} × {cols}  
- **Pixel Spacing:** {pixel_spacing}  
- **Bits Allocated / Stored:** {bits_allocated} / {bits_stored}  
- **Photometric:** {photometric}  
- **SamplesPerPixel:** {samples_per_pixel}  
- **TransferSyntaxUID:** {transfer_syntax}  
"""
    tech_placeholder.markdown(tech_md, unsafe_allow_html=True)

    # Obtener pixel array y valores default window
    try:
        arr = get_pixel_array(ds)
    except Exception as e:
        st.error(f"Could not extract pixel_array: {e}")
        return

    # obtener valores por defecto de WC/WW
    default_center, default_width = default_window(ds)

    # Sliders para Window Center / Width (brillo/contraste)
    st.sidebar.markdown("## 🛠 Image Controls")
    wc_slider = st.sidebar.slider("Brightness — Window Center", min_value=float(np.min(arr)), max_value=float(np.max(arr)),
                                  value=float(default_center), step=(float(default_width) / 100.0 if default_width else 1.0))
    ww_slider = st.sidebar.slider("Contrast — Window Width", min_value=1.0, max_value=float(np.max(arr) - np.min(arr) + 1.0),
                                  value=float(default_width), step=max(1.0, float(default_width) / 100.0))

    # Opción de invertir (para imágenes invertidas)
    invert = st.sidebar.checkbox("Invert scale (White/Black)", value=False)
    # opción de clahe (ecualización adaptativa) como extra
    clahe_opt = st.sidebar.checkbox("Apply adaptive equalization (CLAHE)", value=False)
    # botón para restablecer a DICOM WC/WW (si existen)
    if st.sidebar.button("Reset WC/WW to DICOM values"):
        wc_slider = default_center
        ww_slider = default_width

    # Aplicar windowing
    img_8 = apply_window(arr, wc_slider, ww_slider)
    if invert:
        img_8 = 255 - img_8

    # CLAHE (simple): usar Pillow ImageOps.equalize como alternativa ligera
    pil_img = to_pil(img_8)
    if clahe_opt:
        try:
            # PIL no tiene CLAHE nativo sin OpenCV; usar equalize como simplificación
            pil_img = ImageOps.equalize(pil_img)
        except Exception:
            pass

    # Mostrar imagen y controles en la columna derecha
    with col_view:
        st.markdown("<div class='viewer-card viewer-image'>", unsafe_allow_html=True)
        st.markdown("### 🖼 DICOM Image")
        show_image(pil_img, caption=f"WC={wc_slider:.1f}  WW={ww_slider:.1f}")
        st.markdown("---")

        # Histogramas y stats
        with st.expander("📊 Intensity Histogram"):
            fig, ax = plt.subplots(figsize=(6, 2.6))
            ax.hist(np.asarray(pil_img).ravel(), bins=256, alpha=0.8)
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Frequency")
            ax.set_title("Histogram (adjusted image)")
            plt.tight_layout()
            st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # Descarga de la imagen ajustada (PNG)
    png_bytes = pil_to_bytes(pil_img, fmt="PNG")
    b64 = base64.b64encode(png_bytes).decode()
    filename = f"verdicom_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    st.sidebar.download_button("⬇️ Download Image (PNG)", data=png_bytes, file_name=filename, mime="image/png")

    # Mostrar algunos valores numéricos y percentiles
    with st.expander("📈 Quick Statistics"):
        arr_flat = arr.ravel()
        p10 = np.percentile(arr_flat, 10)
        p50 = np.percentile(arr_flat, 50)
        p90 = np.percentile(arr_flat, 90)
        mn = np.min(arr_flat); mx = np.max(arr_flat); mean = np.mean(arr_flat); std = np.std(arr_flat)
        stats_md = f"""
- Min / Max: **{mn:.2f}** / **{mx:.2f}**  
- Mean / Std: **{mean:.2f}** / **{std:.2f}**  
- P10 / P50 / P90: **{p10:.2f}** / **{p50:.2f}** / **{p90:.2f}**  
- Applied Window: **Center={wc_slider:.2f}**, **Width={ww_slider:.2f}**
"""
        st.markdown(stats_md)

    # Mostrar raw DICOM tags relevantes en un expander (limitado)
    with st.expander("🧾 DICOM Tags (selection)"):
        keys = ["StudyDescription", "SeriesDescription", "Manufacturer", "InstitutionName",
                "ProtocolName", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]
        table_md = "|Tag|Valor|\n|---:|:---|\n"
        for k in keys:
            table_md += f"|{k}|{safe_get(ds, k)}|\n"
        st.markdown(table_md)

    # Footer (pequeña ayuda)
    st.markdown("---")

# ==============================
# 🧩 PUNTO DE ENTRADA
# ==============================
if __name__ == "__main__":
    main()
