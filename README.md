# Diabetic-Retinopathy-Severity-Classification
Academic Project on Diabetic Retinopathy Severity Classification using Pyspark, Spark ML, and PyTorch on GCP.

---

## 🔍 Overview

1. **Metadata Extraction**  
   Scan GCS folders, extract image paths, class labels, file sizes & timestamps, then save per-split CSVs.
2. **Feature Extraction**  
   Load images from GCS, run through pretrained ResNet50 (no classification head) to get 2048-d bottleneck vectors, save as CSVs.
3. **Dataset Integration**  
   Join metadata and feature CSVs via a shared row index in PySpark, producing a unified DataFrame containing image path, ResNet features, and class label.
4. **Modeling**  
   - **Spark ML Random Forest** on ResNet features (80/20 train/test split).  
   - **EfficientNet-B0 CNN** fine-tuning in PyTorch with class-weighted loss.
5. **Scaling**  
   - GCS for data storage  
   - Dataproc for distributed Spark jobs  
   - (Optional) Vertex AI for batch inference

---

## 🚀 Quickstart

### Prerequisites

- **GCP Project** with Cloud Storage & Dataproc APIs enabled  
- Service account JSON key with Storage- and Dataproc-access  
- Python 3.8+, Java 11, Spark 3.5  

```bash
git clone https://github.com/your-org/retinopathy-classification.git
cd retinopathy-classification
pip install -r requirements.txt
