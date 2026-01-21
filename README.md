# 🛡️ PhishGuard - AI-Powered Phishing Classifier

**PhishGuard** is a Machine Learning security tool that detects phishing attempts in SMS messages. It features an **Active Learning** pipeline, allowing the model to retrain itself based on user feedback while preventing data poisoning via an Admin "Quarantine" Review system.

## 🚀 Features
- **Hybrid Detection:** Trained on SMS (UCI) datasets.
- **Active Learning:** Users can flag false positives/negatives.
- **Security Quarantine:** Admin panel (`admin_review.py`) to verify feedback before retraining.
- **Adversarial Defense:** Prevents malicious actors from poisoning the training data.

## 🛠️ Installation

1. **Clone the repo:**
        git clone [https://github.com/YOUR-USERNAME/PhishGuard.git](https://github.com/YOUR-USERNAME/PhishGuard.git)
        cd PhishGuard

2. **Create & Activate Virtual Environment:**
        python -m venv .venv
        - **macOS/Linux:** source .venv/bin/activate
        - **Windows(Command Prompt):** .venv\Scripts\activate
        - **Windows(PowerShell):** .venv\Scripts\Activate.ps1

2. **Install dependencies:**
        pip install -r requirements.txt

3. **Initialize the Brain:**
        python training_model.py

4. **Run the Interface:**
        streamlit run app.py

5. **Review Feedback (Admin Only):**
        python admin_review.py

## 🏗️ Architecture
        User Input -> Streamlit App -> Prediction -> Feedback -> Quarantine (CSV) -> Admin Verification -> Retraining