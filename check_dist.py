
import pandas as pd
import os

pkl_path = r'c:\Users\Admin\Desktop\hbm-project\hbm-backend\models\BATCH_20260131_144222.pkl'
try:
    data = pd.read_pickle(pkl_path)
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
        
    print(f"Total rows: {len(df)}")
    
    # Clean failure_type if needed
    if 'failure_type' in df.columns:
        df['failure_type'] = df['failure_type'].apply(
            lambda x: x[0][0] if isinstance(x, list) and len(x) > 0 and isinstance(x[0], (list, tuple)) and len(x[0]) > 0 
            else (x[0] if isinstance(x, list) and len(x) > 0 else (x if isinstance(x, str) else str(x)))
        )
        df['failure_type'] = df['failure_type'].astype(str)
        print("\nValue Counts:")
        print(df['failure_type'].value_counts())
    else:
        print("No failure_type column")
        
except Exception as e:
    print(e)
