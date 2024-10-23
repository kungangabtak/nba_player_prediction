# verify_label_encoder.py

import pickle
import os

def load_label_encoder():
    label_encoder_path = os.path.join('models', 'label_encoder.pkl')
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder file '{label_encoder_path}' not found.")
    with open(label_encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)
    return label_encoder

def main():
    try:
        label_encoder = load_label_encoder()
        print("Label Encoder Classes:")
        print(label_encoder.classes_)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()