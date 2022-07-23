## Description

Machine learning project using logistic regression, written from scratch (without any librairy like scikit-learn).
You've got two versions :
- One on jupyter notebook : cancer_prediction.ipynb
- One using streamlit, which is a web interface : cancer_web.py

## Installation

In order to launch this project you'll need to install python, go to this official link https://www.python.org/downloads/.
- For the first version
  You can easily follow this link which explains very well how to install Jupyter Notebook, with the terminal : https://medium.com/analytics-vidhya/how-to-install-jupyter-notebook-using-pip-e597b5038bb1
- For the second
  Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the different packages.
  ```bash
  pip install streamlit
  pip install pandas
  pip install numpy
  pip install sklearn
  ```
  
## Usage
Before starting, you can notice that the processing part is quite short, the reason being that I focused a lot more on the algorithm part knowing that the dataset was already prepared.
- For the first version
  You'll need to import both "cancer_prediction.ipynb" and "survey lung cancer.csv" into a document.
  After that, just launch Jupyter Notebook, and then open the cancer.ipynb file !<br>
  The file is separated into two parts :
  - The processing part which consists in analyzing the data
  - The Logistic regression part where you'll find the code for this algorithm, from scratch (i hope that you've some math knowledge :))
- For the second
  Import "cancer_web.py" and "survey lung cancer.csv" into a document, enter in it, and launch the application with the following command:
  ```bash
  streamlit run cancer_web.py
  ```
