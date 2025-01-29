# Samsaru - Dynamic Pricing of Video Game Consoles (Sale) - Group A1

## Introduction  
As part of this project, we are working with **Samsaru** to develop a dynamic pricing model for selling video game consoles.  
The goal is to identify and predict optimal price points by analyzing market data and using machine learning models.  
Our group (**A1**) focuses on identifying sale prices for **Xbox consoles**.

---

## Installation and Usage

### 1. Clone Repository  
Clone this repository into an empty folder:

```bash
https://github.com/amaxharraj/Samsaru.git
```

## 2. Install Dependencies
Open the folder and install the following dependencies:

```bash
pip install scrapy numpy pandas
```

Open the notebook that is inside the project and install following packages.

```bash
pip install gradio scikit-learn pandas
```

## 3. Start Application

### Step 1: Start the Scraping Process
The first step is to start the scraping process by running the `app.py` file. Once you run the file, the scraping process will start, and a CSV file containing the data will be generated.

### Step 2: Run the Notebook
After the CSV file is generated, you can run the notebook. However, before running the notebook, you need to specify the path where the CSV file is saved.

1. **Copy the path** of the CSV file.
2. **Insert the path into the `data_file` variable** in the first cell of the notebook.
Example:
```bash
data_file = "C:\\Users\\auron\\Documents\\CodeProjekte\\asgoodasnew_products.csv"
```

Once the path is set, you can run the notebook by selecting "Run All" (or any other option that runs all cells).

### Step 3: Use the Gradio Interface
At the bottom of the notebook, you will see the Gradio interface, where you can choose the model of your choice.

**Important:**  
Make sure you first select a model or change the current one. After that, you can choose the storage capacity and the condition of the console.

Once selected, the notebook will display:

- The **predicted price** for the respective console.
- The **R² value** of the model's prediction, which indicates how well the model fits the data.

