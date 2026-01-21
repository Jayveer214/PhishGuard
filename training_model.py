import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
VERIFIED_DATA_FILE = 'verified_data.csv'
MODEL_FILE = 'spam_detector_model.pkl'
VECTORIZER_FILE = 'vectorizer.pkl'

def train_system():
    print("\n--- 🚀 STARTING TRAINING ENGINE ---")
    
    # --- 1. LOAD SOURCE A: SMS DATA ---
    print("Loading SMS Data...")
    url_sms = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df_sms = pd.read_csv(url_sms, sep='\t', header=None, names=['label', 'message'])
    
    # --- 3. LOAD SOURCE C: VERIFIED FEEDBACK (From Admin Panel) ---
    if os.path.exists(VERIFIED_DATA_FILE):
        print("Found Verified Feedback Data. Merging into Brain...")
        df_verified = pd.read_csv(VERIFIED_DATA_FILE)
    else:
        df_verified = pd.DataFrame(columns=['label', 'message'])

    # --- 4. COMBINE ALL KNOWLEDGE ---
    full_data = pd.concat([df_sms, df_verified], ignore_index=True)
    full_data = full_data.dropna()
    
    print(f"Total training examples: {len(full_data)}")
    print(full_data.head()) # show first 5 rows
    
    # --- 5. PREPROCESSING & TRAINING ---
    full_data['label_num'] = full_data.label.map({'ham': 0, 'spam': 1})

    # Split the data: 80% for training (studying), 20% for testing (the final exam)
    X = full_data['message']
    y = full_data['label_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorizer (min_df=1 ensures we learn rare words from feedback)
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    # Fit the vectorizer to our training data and transform it into a matrix of numbers
    X_train_dtm = vectorizer.fit_transform(X_train)

    # Don't fit on test data! Only transform. (We can't "peek" at the test answers)
    X_test_dtm = vectorizer.transform(X_test)

    # --- STEP 4: TRAINING THE MODEL ---
    # We use Multinomial Naive Bayes - the classic choice for text classification
    model = MultinomialNB()
    model.fit(X_train_dtm, y_train)

    # --- STEP 5: EVALUATION ---
    # Let's see how well we did
    y_pred_class = model.predict(X_test_dtm)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred_class) * 100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred_class))
    
    # --- 7. SAVE ---
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print("✅ Model trained and saved successfully.")

# This ensures the script runs if you click it directly, 
# but DOES NOT run if you just import it elsewhere.
if __name__ == "__main__":
    train_system()