# Financial-Fraud
This Python program includes two scripts for fraud detection. The **fraud_data.py** script processes data, trains machine learning models (Decision Tree, MLP, Random Forest), and saves the best model. The **web_fraud.py** script creates a Flask app that predicts fraud from user input using the pre-trained model, accessible online via Ngrok.

1. **Data Loading and Preparation:**
   - Mounts Google Drive to access data stored in it.
   - Inspects the dataset by displaying its structure and summary statistics.
   - Sorts the data based on the `isFraud` column and creates a new indexed column to manage the data.

2. **Feature Engineering:**
   - Encodes categorical variables, such as the `type` column, using label encoding.
   - Prepares the dataset by selecting relevant features and removing unnecessary columns.

3. **Data Splitting:**
   - Splits the data into training and testing sets using `train_test_split`.

4. **Model Training:**
   - Trains three different models: Decision Tree, Multi-Layer Perceptron (MLP) Classifier, and Random Forest Classifier.
   - Evaluates the models using accuracy scores and classification reports.

5. **Model Saving:**
   - Saves the trained MLP model and datasets into pickle files for later use.

### **web_fraud.py**
This script sets up a web application using Flask to interact with the fraud detection model trained in `fraud_data.py`. The script performs the following tasks:

1. **Environment Setup:**
   - Installs necessary Python packages, including Flask, Flask-Bootstrap, Flask-Ngrok, and Requests-HTML.
   - Configures Google Colab and Ngrok for running the Flask app.

2. **Web Application Configuration:**
   - Loads the pre-trained fraud detection model from a pickle file.
   - Sets up the Flask application, specifying the template and static file directories.

3. **Routes and Views:**
   - Defines routes for the web application, including the main page and a test page.
   - Implements a route that handles form submissions, processes input data, and uses the model to predict whether a transaction is fraudulent.
   - Displays the prediction result on the web page.

4. **Ngrok Integration:**
   - Integrates Ngrok to expose the Flask app to the internet, making it accessible via a public URL.

Both scripts are designed to work together, with `fraud_data.py` handling the data preparation and model training, and `web_fraud.py` providing a user-friendly interface for making predictions with the trained model.
