
# Diabetic Patient Readmission Prediction using Machine Learning and Neural Networks

## Project Overview
This project predicts the **30-day hospital readmission risk** for diabetic patients. It utilizes **data preprocessing**, **class balancing**, and advanced machine learning techniques, including **Neural Networks** and **SVM models**, to analyze a real-world medical dataset. The implementation aligns with findings from the research paper titled:

*"An Integrated Data Mining Algorithms and Meta-Heuristic Technique to Predict the Readmission Risk of Diabetic Patients."*

---

## Features

### 1. Data Preprocessing and Balancing
- Handles missing values using **mode imputation**.
- Encodes categorical features using **Label Encoding**.
- Balances the dataset using **undersampling** to handle class imbalance issues.
- Scales the data using **StandardScaler**.

### 2. Exploratory Data Analysis (EDA)
- Visualizes feature distributions, correlations, and target class balance.
- Boxplots and heatmaps highlight numerical feature relationships with `readmitted` status.

### 3. Machine Learning Models
The following models were implemented and evaluated:
- **Support Vector Machines (SVM)** with:
   - Linear Kernel
   - RBF Kernel
   - Polynomial Kernels (degrees 3 and 5)
- **Feedforward Neural Network**:
   - Layers: Fully Connected Dense layers with **ReLU** activation.
   - Optimizer: **Adam** optimizer.
   - Loss: **Binary Cross-Entropy** for classification.

### 4. Performance Evaluation
The models were evaluated using:
- **10-Fold Stratified Cross-Validation**
- Metrics:
   - **Accuracy**
   - **Precision**
   - **Recall**
   - **F1-Score**
- Confusion matrices were generated for detailed class performance analysis.

---

## Dataset
- **Source**: Diabetic patient readmission dataset.
- **Target Variable**: `readmitted`
   - `0`: No readmission
   - `1`: Readmission within 30 days.

### Selected Features:
- Demographics: `age`, `gender`, `race`
- Admission details: `admission_type_id`, `time_in_hospital`
- Lab results: `num_lab_procedures`, `num_medications`
- Diabetes-specific features: `insulin`, `change`, `diabetesMed`

---

## Technologies Used
- **Python**  
- **Libraries**:
   - Pandas, NumPy
   - Scikit-learn
   - TensorFlow/Keras
   - Matplotlib, Seaborn (Visualizations)

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/diabetic-readmission-prediction.git
   cd diabetic-readmission-prediction
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook class_weights_balanced.ipynb
   ```

4. Follow the notebook steps to preprocess the data, train models, and evaluate their performance.

---

## Results
The implemented models demonstrated their ability to predict diabetic patient readmission with competitive accuracy and recall scores. The Neural Network implementation, in particular, showed promising results due to its ability to capture complex feature interactions.

---

## License
This project is open-source under the [MIT License](LICENSE).
