

\# AI Disease Prediction System



An AI-powered web application that predicts possible diseases based on symptoms selected by users. It uses Machine Learning and Flask to provide an interactive disease prediction experience.



\## Live Demo



https://ai-disease-prediction-system-c366.onrender.com



\## Features



\- User registration and login

\- Password hashing

\- Symptom selection

\- Machine Learning-based disease prediction

\- Prediction history

\- User dashboard

\- SQLite database integration

\- Responsive user interface



\## Technologies Used



\- Python

\- Flask

\- Scikit-learn

\- Pandas

\- NumPy

\- SQLite

\- HTML

\- CSS

\- Bootstrap

\- Git and GitHub

\- Render



\## Project Structure



```text

AI-Disease-Prediction-System/

│

├── app.py

├── requirements.txt

├── .python-version

├── .gitignore

│

├── dataset/

│   ├── cleaned\_disease\_dataset.csv

│   └── symbipredict\_2022.csv

│

├── model/

│   ├── best\_model.pkl

│   ├── train\_model.py

│   ├── data\_preprocessing.py

│   ├── model\_validation.py

│   └── other analysis files

│

├── static/

│   └── style.css

│

├── templates/

│   ├── base.html

│   ├── index.html

│   ├── login.html

│   ├── register.html

│   ├── dashboard.html

│   ├── predict.html

│   └── history.html

│

└── README.md

```



\## Installation and Setup



\### 1. Clone the repository



```bash

git clone https://github.com/YashBhand/AI-Disease-Prediction-System.git

```



\### 2. Open the project folder



```bash

cd AI-Disease-Prediction-System

```



\### 3. Create a virtual environment



```bash

python -m venv venv

```



\### 4. Activate the virtual environment



For Windows:



```bash

venv\\Scripts\\activate

```



\### 5. Install dependencies



```bash

pip install -r requirements.txt

```



\### 6. Run the application



```bash

python app.py

```



\### 7. Open in your browser



http://127.0.0.1:5000



\## Deployment



The application is deployed using Render.



Live application:



https://ai-disease-prediction-system-c366.onrender.com



\## Security



\- The virtual environment is excluded from GitHub.

\- The SQLite database is excluded from GitHub.

\- Set a strong SECRET\_KEY environment variable for deployment.

\- Never upload passwords or private credentials.



\## Disclaimer



This project is developed for educational and academic purposes. Predictions are based on the dataset and trained Machine Learning model. This application is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.



\## Author



\*\*Yash Bhand\*\*



GitHub: https://github.com/YashBhand



