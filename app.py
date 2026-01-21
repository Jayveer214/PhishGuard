import streamlit as st
import joblib
import pandas as pd
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="PhishGuard+ | Active Learning", page_icon="🛡️")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('spam_detector_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_assets()

# --- HEADER ---
st.title("🛡️ PhishGuard+: Adaptive Threat Detector")
st.caption("Now with Active Learning: Your feedback trains the next version.")

# --- INPUT SECTION ---
user_input = st.text_area("Analyze your message:", height=150)

# We use 'session_state' to remember the prediction so the buttons don't disappear
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None
if 'user_input_history' not in st.session_state:
    st.session_state['user_input_history'] = ""

if st.button("Analyze"):
    if user_input.strip():
        # Save input to history so we don't lose it when buttons are clicked
        st.session_state['user_input_history'] = user_input
        
        # Predict
        input_data = vectorizer.transform([user_input])
        pred_id = model.predict(input_data)[0]
        st.session_state['prediction'] = "SPAM" if pred_id == 1 else "HAM"
    else:
        st.warning("Please enter text.")

# --- DISPLAY RESULT & FEEDBACK ---
if st.session_state['prediction']:
    result = st.session_state['prediction']
    
    # Show the colorful badge
    if result == "SPAM":
        st.error("🚨 POTENTIAL PHISHING / SPAM DETECTED")
        st.write("Take caution: This message contains patterns common in malicious attacks.")
    else:
        st.success("✅ MESSAGE SEEMS SAFE")
        st.write("This message appears to be legitimate based on our training data.")

    st.markdown("---")
    st.write("### 📝 Was this correct?")
    st.info("Help us improve! If the AI was wrong, tell us below.")

    col1, col2 = st.columns(2)

    true_label = None
    
    # BUTTON 1: YES (Correct)
    if col1.button("👍 Yes, Correct"):
        true_label = "spam" if result == "SPAM" else "ham"

        # We can log this as a "confirmed" data point if we want
        st.toast("Thanks for the feedback!", icon="🎉")
        # Optional: Save confirmed data too
        
    # BUTTON 2: NO (Incorrect)
    if col2.button("👎 No, Incorrect"):
        # If model said SPAM but it was HAM, the true label is HAM (and vice versa)
        true_label = "ham" if result == "SPAM" else "spam"
        
        st.toast(f"Correction saved! We've marked this as {true_label}.", icon="💾")
        
    if true_label:
        # Save to CSV
        feedback_data = pd.DataFrame([[true_label, st.session_state['user_input_history']]], 
                                    columns=['label', 'message'])
            
        # Append to a 'feedback.csv' file (create if doesn't exist)
        header = not os.path.exists('feedback.csv')
        feedback_data.to_csv('feedback.csv', mode='a', header=header, index=False)
            
        st.success("Feedback recorded. This example will be used in the next training cycle.")