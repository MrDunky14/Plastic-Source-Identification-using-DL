# **🌎 Deep Earth Sentinel: AI-Powered Microplastic Pollution Source Identification**

## **Project Overview**

"Deep Earth Sentinel" is a deep learning-driven initiative aimed at combating microplastic pollution by accurately identifying the geographical origin of plastic bottle waste. This project develops and deploys a robust Artificial Intelligence model that predicts the scan\_country (country of origin) of a plastic bottle based on its product attributes. By precisely pinpointing the source, environmental agencies and policymakers can implement targeted strategies to reduce plastic waste at its origin.

## **Problem Statement**

Microplastic pollution poses a severe threat to global ecosystems, marine life, and human health. Plastic bottles are a significant contributor to this problem. Identifying the source of this pollution is a critical step towards effective waste management and environmental policy. Traditional methods of tracing waste are often manual, time-consuming, and inefficient. This project leverages the power of deep learning to automate and enhance this identification process, providing actionable insights for environmental protection.

## **Features**

* **Data Aggregation:** Merges large-scale plastic bottle waste data from multiple CSV files.  
* **Intelligent Preprocessing:** Handles mixed data types, missing values, and transforms raw features into a machine-learning-ready format using techniques like One-Hot Encoding and StandardScaler.  
* **Deep Learning Model:** Implements a Multi-Layer Perceptron (MLP) using TensorFlow/Keras for robust classification of scan\_country.  
* **GPU Accelerated Training:** Utilizes GPU resources for efficient model training and optimization.  
* **Model Interpretability:** Employs SHAP (SHapley Additive exPlanations) to provide insights into which product attributes are most influential in predicting the scan\_country.  
* **Interactive Web Application:** Deploys the trained model as an intuitive Streamlit web application, allowing users to input bottle attributes and get real-time predictions.

## **Data Source**

The project utilizes a substantial dataset comprising **2.7 million rows** of plastic bottle waste data, aggregated from multiple CSV files. Each record includes features such as:

* product\_label  
* product\_size  
* brand\_name  
* manufacturer\_country  
* manufacturer\_name  
* bottle\_count  
* scan\_country (Target Variable)

The data undergoes meticulous cleaning and preprocessing to ensure quality and compatibility with the deep learning model.

## **Methodology & Technical Stack**

### **Data Preprocessing**

The raw data is processed using scikit-learn's ColumnTransformer for efficient and consistent transformations:

* **Numerical Features (bottle\_count):** Scaled using StandardScaler to normalize their range.  
* **Categorical Features (product\_label, brand\_name, etc.):** Transformed using OneHotEncoder to convert them into a numerical format, suitable for neural networks. handle\_unknown='ignore' is used to gracefully handle categories not seen during training.  
* **Label Encoding:** The scan\_country (target variable) is converted from string labels to numerical integers using LabelEncoder.

### **Model Architecture (Deep Learning \- MLP)**

A Sequential Multi-Layer Perceptron (MLP) built with TensorFlow 2.x and Keras is used for classification:

* **Input Layer:** Automatically configured to accept the high-dimensional, preprocessed feature vector (e.g., 4481 features after One-Hot Encoding).  
* **Hidden Layers:** Multiple Dense layers (e.g., 256, 128, 64 neurons) with ReLU (Rectified Linear Unit) activation functions to introduce non-linearity and learn complex patterns. The decreasing number of neurons encourages feature compression.  
* **Regularization:** Dropout layers (e.g., 0.3, 0.2 rates) are strategically placed after hidden layers to prevent overfitting by randomly deactivating neurons during training.  
* **Output Layer:** A Dense layer with softmax activation and num\_classes (number of unique scan\_country values) neurons. softmax outputs a probability distribution over all possible scan\_country classes.

### **Training & Optimization**

* **Optimizer:** Adam optimizer is used, known for its adaptive learning rate capabilities and efficiency.  
* **Loss Function:** SparseCategoricalCrossentropy is employed, suitable for multi-class classification with integer-encoded labels.  
* **Metrics:** Model performance is monitored using accuracy.  
* **Callbacks:**  
  * EarlyStopping: Monitors validation loss and stops training if no improvement is observed for a certain number of epochs (patience), preventing overfitting and saving computational resources.  
  * ModelCheckpoint: Automatically saves the model weights (or the entire model) corresponding to the best validation accuracy achieved during training.

### **Model Interpretability (SHAP)**

* **SHAP (SHapley Additive exPlanations):** Used to explain the model's predictions by quantifying the contribution of each feature to the output. This helps in understanding the "why" behind the model's decisions, revealing which product attributes are most critical for identifying a scan\_country.  
* Visualizations (Beeswarm and Bar plots) clearly show overall feature importance and how individual feature values influence predictions.

### **Deployment (Streamlit)**

* The trained model and its preprocessor components (ColumnTransformer, LabelEncoder) are loaded into a user-friendly web application built with Streamlit.  
* Users can input product attributes through interactive widgets and receive real-time predictions of the scan\_country.  
* The app displays the predicted country and the probability distribution across top classes.

### **GPU Acceleration**

The entire deep learning pipeline, from data preprocessing (where applicable with RAPIDS/CuPy, though mostly handled by Scikit-learn on CPU for ColumnTransformer) to model training, leverages GPU capabilities provided by TensorFlow's CUDA integration, significantly speeding up computation for large datasets.

## **Key Results & Findings**

*(After you have successfully run the project, fill in these sections with your actual results)*

* **Model Accuracy:** Achieved a test accuracy of \[X.XX\]% on unseen data.  
* **Top Contributing Features (from SHAP analysis):**  
  * \[Feature 1 Name\] (e.g., manufacturer\_country): Most influential, indicating a strong correlation between the manufacturer's country and scan country.  
  * \[Feature 2 Name\] (e.g., brand\_name\_xyz): Highly important, suggesting specific brands are strongly associated with particular regions.  
  * \[Feature 3 Name\] (e.g., product\_label\_type): Also plays a significant role, showing certain product types are prevalent in specific locations.  
  * *(Elaborate on 2-3 key insights you gained from the SHAP plots)*

## **Project Structure**

.  
├── app.py                      \# Streamlit interactive web application  
├── training\_notebook.ipynb     \# (Optional) Jupyter Notebook for data exploration, preprocessing, training, and SHAP analysis  
├── best\_deep\_earth\_sentinel\_model.keras \# Saved Keras model (architecture \+ weights \+ optimizer state)  
├── fitted\_preprocessor.joblib  \# Saved ColumnTransformer object  
├── fitted\_label\_encoder.joblib \# Saved LabelEncoder object  
├── merged\_plastic\_bottle\_waste.csv \# The combined dataset  
├── README.md                   \# This file  
└── requirements.txt            \# Python dependencies

## **Requirements**

To run this project, you'll need Python 3.8+ and the following libraries. It's highly recommended to use a virtual environment.

\# requirements.txt  
pandas\>=2.0.0  
scikit-learn\>=1.0.0  
tensorflow\>=2.10.0 \# Or specific version you used (e.g., 2.15.0)  
keras\>=2.10.0      \# If Keras is separate from TensorFlow  
streamlit\>=1.0.0  
matplotlib\>=3.0.0  
seaborn\>=0.11.0  
joblib\>=1.0.0  
shap\>=0.40.0  
numpy\>=1.20.0  
\# Add specific CUDA/cuDNN requirements if running locally with GPU  
\# E.g., for TensorFlow 2.10: NVIDIA GPU drivers, CUDA Toolkit 11.2, cuDNN 8.1

You can install all required packages using pip:

pip install \-r requirements.txt

## **How to Run the Project**

Follow these steps to set up and run the "Deep Earth Sentinel" project:

### **1\. Clone the Repository**

git clone \<your-repository-url\>  
cd Plastic\\ Source\\ Identification\\ using\\ DL

### **2\. Set Up Virtual Environment (Recommended)**

python \-m venv venv  
source venv/bin/activate  \# On Windows: venv\\Scripts\\activate

### **3\. Install Dependencies**

pip install \-r requirements.txt

*(If you are using a GPU for training, ensure your NVIDIA drivers, CUDA Toolkit, and cuDNN are compatible with your TensorFlow version. Refer to TensorFlow's official documentation for specific version compatibility.)*

### **4\. Prepare the Data**

Ensure the merged\_plastic\_bottle\_waste.csv file is present in the root directory of your project. This file is crucial for both training and the Streamlit app's preprocessor initialization.

### **5\. Train the Model and Save Components**

Run your training script (e.g., training\_script.py if you extracted the training logic, or execute the relevant cells in your Jupyter Notebook if using Customer\_Churn\_Prediction.ipynb as a base for your training). This step will:

* Preprocess the data.  
* Train the deep learning model.  
* Save the best-performing model as best\_deep\_earth\_sentinel\_model.keras.  
* Save the fitted ColumnTransformer as fitted\_preprocessor.joblib.  
* Save the fitted LabelEncoder as fitted\_label\_encoder.joblib.

Ensure these three saved files are in the same directory as your app.py file.

### **6\. Run the Streamlit Application**

streamlit run app.py

This command will open your web browser to the Streamlit application (usually at http://localhost:8501), where you can interact with your model.

## **Author**

**Krishna Singh**

* [MY GitHub Profile](https://github.com/MrDunky14)
