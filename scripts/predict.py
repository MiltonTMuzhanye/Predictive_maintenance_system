"""
Prediction script for single customer.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import json
from app.inference.predictor import Predictor

def main():
    parser = argparse.ArgumentParser(description="Make customer prediction")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--data', type=str, required=True,
                       help='Customer data JSON file')
    args = parser.parse_args()
    
    # Load data
    with open(args.data, 'r') as f:
        data = json.load(f)
        
    # Make prediction
    predictor = Predictor(args.model)
    result = predictor.predict(data)
    
    # Print result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()