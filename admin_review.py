import pandas as pd
import os
import training_model  # <--- IMPORTING YOUR TRAINING SCRIPT

# --- CONFIGURATION ---
FEEDBACK_FILE = 'feedback.csv'
VERIFIED_DATA_FILE = 'verified_data.csv'

def review_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        print("No feedback to review.")
        return

    try:
        feedback_df = pd.read_csv(FEEDBACK_FILE)
    except pd.errors.EmptyDataError:
        print("Feedback file is empty.")
        return

    if feedback_df.empty:
        print("No new feedback entries.")
        return

    print(f"--- 🛡️ ADMIN PANEL: Found {len(feedback_df)} reports to review ---\n")

    approved_entries = []
    
    for index, row in feedback_df.iterrows():
        print(f"Message: {row['message']}")
        print(f"User says it should be: {row['label'].upper()}")
        
        choice = input("Approve this correction? (y/n/skip): ").lower().strip()
        
        if choice == 'y':
            approved_entries.append(row)
            print(">> Approved.")
        elif choice == 'n':
            print(">> Rejected (Data Poisoning blocked).")
        else:
            print(">> Skipped.")
        print("-" * 20)

    if approved_entries:
        new_data = pd.DataFrame(approved_entries)
        
        # Append to verified storage
        header = not os.path.exists(VERIFIED_DATA_FILE)
        new_data.to_csv(VERIFIED_DATA_FILE, mode='a', header=header, index=False)
        print(f"\nSaved {len(approved_entries)} new verified examples.")
        
        if input("\nRetrain model now? (y/n): ").lower() == 'y':
            # Instead of rewriting code, we just call the function.
            training_model.train_system()
    
    # Clear the feedback file
    print("\nResetting feedback queue...")
    pd.DataFrame(columns=['label', 'message']).to_csv(FEEDBACK_FILE, index=False)
    
    print("Quarantine cleared. Process complete.")

if __name__ == "__main__":
    review_feedback()