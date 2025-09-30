 Exoplanet Candidate Classifier
 
   A **Streamlit web application** for classifying exoplanet candidates from NASA’s Kepler dataset
   using multiple **machine learning models**.
   This project enables users to **upload datasets**, **train classifiers**, **compare models**, explore
   **feature importance**, and **predict new candidates** interactively.
   
Features
  • **Upload Dataset**: Supports CSV uploads (e.g., NASA Kepler exoplanet archive).
  • **Automatic Preprocessing**:- Drops irrelevant or empty columns.- Handles missing values with mean imputation.- Scales features with `StandardScaler`.
  • **Model Training & Comparison**:- Logistic Regression- Random Forest- Gradient Boosting- SVM (Support Vector Machine)
  • **Model Accuracy Visualization**: Interactive Plotly bar charts.
  • **Feature Importance Analysis**: For tree-based models (Random Forest, Gradient Boosting).
  • **Candidate Prediction**: Input values for top features and predict if a candidate is
     **CONFIRMED** or **FALSE POSITIVE**.
     
 Tech Stack
  • **Frontend/UI**: Streamlit
  • **Data Handling**: Pandas, NumPy
  • **Machine Learning**: scikit-learn
  • **Visualization**: Plotly Express
  
 Project Structure
 ```
 ```
  ### 1
   app.py # Main Streamlit application
   requirements.txt # Dependencies
   README.md # Project documentation
   
 Installation & Setup
    Clone the repository
    ```bash
   git clone https://github.com/your-username/exoplanet-classifier.git
   cd exoplanet-classifier
    ```
    
 ### 2
   ```
  bash
  Install dependencies
 pip install -r requirements.txt
   ```

 ### 3
    ```bash
     Run the Streamlit app
     streamlit run app.py
  
    ```
 Usage
 1. Upload a **Kepler dataset CSV** (with a `disposition` column).
 2. The app automatically preprocesses the dataset.
 3. View **model accuracies** and identify the **best-performing model**.
 4. Explore **feature importance** (if available).
 5. Enter values for key features and predict whether a candidate is a **CONFIRMED exoplanet** or
 a **FALSE POSITIVE**.


 Example Workflow
 1. Upload NASA dataset (CSV).
 2. App trains ML models → Displays accuracies.
 3. Best model auto-selected (highlighted in UI).
 4. Explore important features.
 5. Input values for top features → Get prediction instantly.

    
 Future Improvements
 • Add **deep learning models** (e.g., Neural Networks).
 • Allow **hyperparameter tuning** via UI.
 • Deploy on **Streamlit Cloud / Hugging Face Spaces**.
 • Support for **real-time candidate streaming data**.

 
 Author
 Developed with love
 by **Raju Kushwaha** using **Python, Streamlit, and scikit-learn** with the help of some friends and best friend open AI.
