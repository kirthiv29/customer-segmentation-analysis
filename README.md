# 🎯 Customer Segmentation — Streamlit Web App

Convert your ML pipeline script into a fully interactive web app using Streamlit.

---

## 📁 Project Structure

```
your-project/
├── csamp2.py
└── requirements.txt
```

---

## ⚙️ Installation

### Step 1 — Install dependencies using the correct Python

> ⚠️ **Windows users:** Always use `python -m pip` instead of `pip` directly to avoid launcher errors.

```bash
C:\Python313\python.exe -m pip install streamlit scikit-learn matplotlib pandas numpy
```

Or if `python` is recognized in your terminal:

```bash
python -m pip install -r requirements.txt
```

---

### Step 2 — Run the app

```bash
python -m streamlit run csamp2.py
```

Or with full path if needed:

```bash
C:\Python313\python.exe -m streamlit run csamp2.py
```

> ✅ **Always use `python -m streamlit`** instead of the `streamlit.exe` launcher to avoid the *"Unable to create process"* error on Windows.

---

## 🐛 Common Error Fix

### ❌ Error
```
Fatal error in launcher: Unable to create process using
'"C:\Python313\python.exe" "...streamlit.exe" run app.py':
The system cannot find the file specified.
```

### ✅ Fix
The `.exe` launcher points to a broken Python path. Bypass it entirely:

```bash
python -m streamlit run csamp2.py
```

---

## 🖥️ App Features

| Feature | Description |
|---|---|
| 📂 File Upload | Drag & drop any CSV file |
| 🔢 Feature Selection | Pick numeric columns to cluster on |
| 🎚️ Cluster Slider | Choose 2–10 clusters dynamically |
| 📊 EDA Tab | Data preview, stats, missing values |
| 🔢 Segmentation Tab | Cluster assignments + summary table |
| 📈 Visualization Tab | Scatter plot (Before vs After) + Pie chart |
| ⬇️ Download | Export segmented CSV with cluster labels |

---

## 🌐 Deploy Online (Free)

Deploy your app publicly using **Streamlit Community Cloud**:

1. Push your project to a GitHub repository
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **"New app"** → connect your GitHub repo
4. Set the main file as `app.py`
5. Click **Deploy** — it reads `requirements.txt` automatically

---

## 📦 Requirements

```
streamlit>=1.32.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
```

---

## 🔄 Original vs Streamlit Comparison

| Original Script | Streamlit App |
|---|---|
| `input()` prompts in terminal | Sidebar file uploader + multiselect |
| `print()` output | `st.dataframe()`, `st.metric()` |
| `plt.show()` popup | `st.pyplot(fig)` inline |
| Single top-to-bottom flow | 3 tabs: EDA / Segmentation / Visualizations |
| No export | Download segmented CSV button |

---

*Built with Python · Streamlit · scikit-learn · matplotlib*
