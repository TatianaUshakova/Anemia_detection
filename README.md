# Anemia Detection using Deep Learning

Anemia detection using deep learning techniques, designed for deployment in rural areas of India where anemia-related complications, especially among pregnant women, are common and access to medical help is limited. The project includes automated tools for:

- Color correction from a color card
- Image segmentation
- Data cleaning and exploration
- Multiple predictive models ranging from classical regression on mean colors to convolutional neural networks (CNNs)

## 📊 Project Overview
The goal of this project is to provide a reliable tool for anemia detection in regions with limited medical resources. The original patient data used for model training involved Indian patients and remains private. For demonstration purposes, publicly available images similar to those used in the original project are included. The images analyzed include the conjunctiva, nails, palms, and tongue, as paleness in these areas can indicate anemia.

## 📦 Project Structure
- `src/` - Source code including preprocessing, models, and utilities
- `data/` - Contains public images for demonstration only
- `notebooks/` - Jupyter notebooks for exploratory analysis
- `tests/` - Unit tests for verifying code components
- `requirements.txt` - Python package dependencies

## 🚀 How to Run the Project
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd anemia-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main pipeline:
   ```bash
   python src/main.py
   ```

## 📈 Results
The project outputs include:
- Processed images with color correction and segmentation
- Trained models for anemia detection
- Evaluation metrics and performance reports

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

**Note:** This project is shared for educational purposes and uses public sample data for demonstration. The original patient data remains private to ensure compliance with data privacy standards.
