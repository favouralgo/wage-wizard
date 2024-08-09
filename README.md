--- 

# WageWizard

WageWizard is a salary prediction tool utilizing a multilinear regression model created from scratch. It predicts salaries based on 15 different features and provides an interactive web interface for users to input their information and receive salary estimates.

## Features

- **Multilinear Regression Model**: A custom-built model to predict salaries based on multiple features.
- **15 Features**: The model uses 15 features for salary prediction.
- **Interactive Interface**: Built with Streamlit for easy and user-friendly access.

## Video Demo

Check out our project demonstration video here: [WageWizard Demo](https://youtu.be/SBE2ESUOH40)

## Dataset

To understand and use the project, you need the following datasets:

- **[Datasets Link](https://drive.google.com/drive/folders/1XGs1wVbuDvRNBDeL6RhpLe2R3PXZ1GvI?usp=sharing)**: Contains the data used for training and testing the model. Ensure you have access to this dataset for a better understanding of how the model works.

## Requirements

To run the WageWizard project, follow these steps:

### 1. Check for Streamlit Installation

Before running the app, ensure Streamlit is installed on your machine:

```bash
pip show streamlit
```

If Streamlit is not installed, you will need to install it. 

### 2. Set Up a Virtual Environment

To run the program, You may use the Spyder Environment in Anaconda  of follow the steps below to run the program in a virtual environment 
For a clean environment and to avoid conflicts, it’s recommended to use a virtual environment:

1. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   ```

   Replace `venv` with your preferred environment name.

2. **Activate the Virtual Environment**

   On **Windows**:

   ```bash
   venv\Scripts\activate
   ```

   On **macOS/Linux**:

   ```bash
   source venv/bin/activate
   ```

3. **Install Dependencies**

   Install the necessary packages using:

   ```bash
   pip install -r requirements.txt
   ```

### 3. Running the App

1. **Navigate to the Project Directory**

   Change to the directory containing your Streamlit script:

   ```bash
   cd path-to-your-project-directory
   ```

2. **Run the Streamlit App**

   Start the Streamlit app by running:

   ```bash
   streamlit run app.py
   ```

   Replace `app.py` with the name of your Streamlit script if it differs.

3. **Access the Web Interface**

   After running the app, it should open in your default web browser. If it doesn’t open automatically, visit:

   ```
   http://localhost:8501
   ```

## Using the Interface

1. **Input Information**

   On the web interface, select the information that applies to you from the available options.

2. **Click Predict**

   Click the "Predict" button to receive a salary estimate based on the features you selected.

3. **Limitations**

   - **Version 1.0**: This initial release is limited to specific companies and locations as we are still gathering more data.
   - If you do not find your information or if the prediction is not accurate, please check back later as we work on expanding the model to include more data.

## Contributing

We welcome contributions to improve WageWizard. To contribute, please fork the repository and submit a pull request with your changes. Contributions that enhance accuracy, functionality, or user experience are particularly appreciated.


