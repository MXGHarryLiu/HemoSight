import pandas as pd
import hashlib

def load_dataset(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['is_public'] = df['rel_path'].str.contains('PBC_dataset_normal_DIB').astype(bool)
    # create uid for each image
    # df['uid'] = df['rel_path'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
    return df
