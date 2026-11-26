
import joblib
import pandas as pd
import sqlite3
import os

from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash


# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE = os.path.join(BASE_DIR, "database", "disease_prediction.db")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "cleaned_disease_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")


# Create Flask application
app = Flask(__name__)

# Use environment variable for deployment
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "temporary-local-development-key"
)


# Database connection
def get_database_connection():
    os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
    return sqlite3.connect(DATABASE)


# Login Required Decorator
def login_required(route_function):

    @wraps(route_function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:
            flash("Please login first!")
            return redirect(url_for("login"))

        return route_function(*args, **kwargs)

    return wrapper


# Create Database
def create_database():

    conn = get_database_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symptoms TEXT NOT NULL,
            predicted_disease TEXT NOT NULL,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


# Home Page
@app.route("/")
@login_required
def home():
    return render_template("index.html")


# Registration Page
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        full_name = request.form["full_name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if password != confirm_password:
            flash("Passwords do not match!")
            return redirect(url_for("register"))

        if len(password) < 8:
            flash("Password must contain at least 8 characters!")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        try:
            conn = get_database_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO users (full_name, email, password)
                VALUES (?, ?, ?)
            """, (full_name, email, hashed_password))

            conn.commit()

            flash("Registration successful! Please login.")
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            flash("Email already registered!")

        except sqlite3.Error:
            flash("Database error. Please try again.")

        finally:
            if "conn" in locals():
                conn.close()

    return render_template("register.html")


# Login Page
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"].strip().lower()
        password = request.form["password"]

        conn = get_database_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM users WHERE email = ?
        """, (email,))

        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):

            session["user_id"] = user[0]
            session["user_name"] = user[1]

            flash("Login successful!")
            return redirect(url_for("home"))

        else:
            flash("Invalid email or password!")

    return render_template("login.html")


# Logout
@app.route("/logout")
def logout():

    session.clear()

    flash("You have been logged out successfully!")

    return redirect(url_for("login"))


# Disease Prediction
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():

    df = pd.read_csv(DATASET_PATH)
    symptoms = df.drop("prognosis", axis=1).columns.tolist()

    if request.method == "POST":

        selected_symptoms = request.form.getlist("symptoms")

        if not selected_symptoms:
            flash("Please select at least one symptom!")
            return redirect(url_for("predict"))

        # Reject values that are not valid dataset symptoms
        if not set(selected_symptoms).issubset(set(symptoms)):
            flash("Invalid symptom selection!")
            return redirect(url_for("predict"))

        input_data = pd.DataFrame(
            [[1 if symptom in selected_symptoms else 0
              for symptom in symptoms]],
            columns=symptoms
        )

        model = joblib.load(MODEL_PATH)
        predicted_disease = model.predict(input_data)[0]

        conn = get_database_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO prediction_history
            (user_id, symptoms, predicted_disease)
            VALUES (?, ?, ?)
        """, (
            session["user_id"],
            ", ".join(selected_symptoms),
            str(predicted_disease)
        ))

        conn.commit()
        conn.close()

        return render_template(
            "predict.html",
            symptoms=symptoms,
            prediction=predicted_disease
        )

    return render_template("predict.html", symptoms=symptoms)


# Prediction History Page
@app.route("/history")
@login_required
def history():

    conn = get_database_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, symptoms, predicted_disease, prediction_date
        FROM prediction_history
        WHERE user_id = ?
        ORDER BY prediction_date DESC
    """, (session["user_id"],))

    history_records = cursor.fetchall()
    conn.close()

    return render_template("history.html", history=history_records)


# Dashboard
@app.route("/dashboard")
@login_required
def dashboard():

    conn = get_database_connection()
    cursor = conn.cursor()

    user_id = session.get("user_id")

    cursor.execute("""
        SELECT COUNT(*)
        FROM prediction_history
        WHERE user_id = ?
    """, (user_id,))

    total_predictions = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(DISTINCT predicted_disease)
        FROM prediction_history
        WHERE user_id = ?
    """, (user_id,))

    different_diseases = cursor.fetchone()[0]

    cursor.execute("""
        SELECT symptoms, predicted_disease, prediction_date
        FROM prediction_history
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 5
    """, (user_id,))

    recent_history = cursor.fetchall()
    conn.close()

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        different_diseases=different_diseases,
        recent_history=recent_history
    )


# Create database when application starts
create_database()


# Run locally
if __name__ == "__main__":
    app.run(debug=True)
