import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Title
st.title("📩 Spam Email Detection App")

# Load Dataset
df = pd.read_csv("spam_nb.csv")

# Features and Target
X = df[["EmailLength", "Links", "SpamWords"]]
y = df["Spam"]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = GaussianNB()
model.fit(x_train, y_train)

# Accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

st.write("### Model Accuracy:", round(accuracy * 100, 2), "%")

st.write("## 🔍 Enter Email Details")

# User Input
email_length = st.number_input("Email Length", min_value=0)
links = st.number_input("Number of Links", min_value=0)
spam_words = st.number_input("Number of Spam Words", min_value=0)

# Prediction Button
if st.button("Predict"):
    prediction = model.predict([[email_length, links, spam_words]])

    if prediction[0] == 1:
        st.error("⚠️ This Email is Spam!")
    else:
        st.success("✅ This Email is Not Spam")