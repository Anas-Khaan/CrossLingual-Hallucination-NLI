import pandas as pd
import re
import os

def is_roman_urdu(text):
    """
    Checks if a given string contains any Latin alphabetic characters (a-z, A-Z).
    If it does, we classify it as Roman Urdu or English-mixed.
    """
    if pd.isna(text):
        return False
    return bool(re.search(r'[a-zA-Z]', str(text)))

def split_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'new_data', 'xnli_urdu.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path)
    
    # A row is considered Roman Urdu if EITHER the Claim or Evidence contains Latin characters
    is_roman = df['Claim'].apply(is_roman_urdu) | df['evidence'].apply(is_roman_urdu)
    
    roman_df = df[is_roman]
    pure_df = df[~is_roman]
    
    pure_path = os.path.join(base_dir, 'new_data', 'xnli_pure_urdu.csv')
    roman_path = os.path.join(base_dir, 'new_data', 'xnli_roman_urdu.csv')
    
    pure_df.to_csv(pure_path, index=False, encoding='utf-8-sig')
    roman_df.to_csv(roman_path, index=False, encoding='utf-8-sig')
    
    print(f"Split complete!")
    print(f"Total Original rows: {len(df):,}")
    print(f"Pure Urdu rows:      {len(pure_df):,}  -> saved to xnli_pure_urdu.csv")
    print(f"Roman Urdu rows:     {len(roman_df):,}  -> saved to xnli_roman_urdu.csv")

if __name__ == "__main__":
    split_dataset()
