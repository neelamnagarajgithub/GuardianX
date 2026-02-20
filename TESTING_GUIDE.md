# Network Intelligence Model Testing Guide

## Overview
This guide explains how to test your trained Network Intelligence fraud detection model.

## Testing Methods

### 1. Comprehensive Model Testing
Use `test_network.py` for full model evaluation:

```bash
python test_network.py
```

**What it does:**
- Loads your trained model, scaler, and feature columns
- Extracts network features from test data
- Makes predictions and evaluates performance
- Generates detailed performance metrics
- Saves predictions to `artifacts/predictions/`

**Output:**
- AUC and Average Precision scores
- Classification report
- Confusion matrix  
- Risk bucket analysis
- Prediction files (parquet format)

### 2. Real-time Inference
Use `inference.py` for individual transaction scoring:

```python
from inference import NetworkFraudScorer

# Initialize scorer
scorer = NetworkFraudScorer()

# Score single transaction
result = scorer.score_transaction(
    cust_id='C123456789',
    dest_id='M987654321', 
    amount=1000.0
)
print(f"Fraud probability: {result['fraud_probability']}")
print(f"Risk bucket: {result['risk_bucket']}")
```

### 3. Batch Scoring
For production batch processing:

```python
# Score batch of transactions
df = pd.DataFrame({
    'cust_id': ['C1', 'C2', 'C3'],
    'dest_id': ['M1', 'M2', 'M3'], 
    'amount': [100, 5000, 25]
})

results = scorer.score_batch(df)
```

## Test Data Formats

### Required Columns
Your test data must have:
- `cust_id` (string): Customer identifier
- `dest_id` (string): Destination identifier  
- `amount` (float): Transaction amount

### Optional Columns
- `label` (int): True fraud label (0/1) for evaluation
- `dataset` (string): Dataset identifier

### Example Test Data
```csv
cust_id,dest_id,amount,label
C724282573,M987654321,150.00,0
C555666777,M123456789,10000.00,1
C999888777,M555666777,25.50,0
```

## Model Artifacts Required

Ensure these files exist in `artifacts/models/`:
- `lgbm_model.txt` - Trained LightGBM model
- `scaler.pkl` - Feature scaler
- `feature_columns.txt` - List of model features

## Performance Interpretation

### Metrics
- **AUC**: Area Under ROC Curve (0.5-1.0, higher better)
- **AP**: Average Precision (0.0-1.0, higher better)
- **Precision**: True positives / (True + False positives)
- **Recall**: True positives / (True positives + False negatives)

### Risk Buckets
- **HIGH**: Fraud probability ≥ 0.8
- **MEDIUM**: Fraud probability 0.5-0.8  
- **LOW**: Fraud probability < 0.5

## Network Features Tested

The model tests these 33 features:

### Customer Features (16):
- `cust_degree_centrality` - Network connectivity
- `cust_betweenness_centrality` - Bridge between clusters
- `cust_eigenvector_centrality` - Influence score
- `cust_pagerank` - PageRank score
- `cust_closeness_centrality` - Average distance to others
- `cust_clustering_coefficient` - Local clustering
- `cust_triangle_count` - Triangle connections
- `cust_core_number` - K-core level
- `cust_local_efficiency` - Local network efficiency
- `cust_in_degree_weighted` - Incoming money flow
- `cust_out_degree_weighted` - Outgoing money flow
- `cust_in_degree_count` - Incoming transaction count
- `cust_out_degree_count` - Outgoing transaction count
- `cust_flow_ratio` - Out-flow ratio
- `cust_community_id` - Community assignment
- `cust_community_size` - Community size

### Destination Features (16):
Same as customer features with `dest_` prefix

### Transaction Feature (1):
- `amount_log` - Log-transformed amount

## Production Deployment

### API Integration Example
```python
from inference import NetworkFraudScorer
from flask import Flask, request, jsonify

app = Flask(__name__)
scorer = NetworkFraudScorer()

@app.route('/score', methods=['POST'])
def score_transaction():
    data = request.json
    result = scorer.score_transaction(
        cust_id=data['cust_id'],
        dest_id=data['dest_id'],
        amount=data['amount']
    )
    return jsonify(result)
```

### Monitoring Recommendations
- Track prediction distributions over time
- Monitor feature drift in network metrics
- Retrain model when performance degrades
- Log high-risk transactions for investigation

## Troubleshooting

### Common Issues
1. **Missing model files**: Ensure training completed successfully
2. **Feature mismatch**: Check test data has required columns
3. **Performance issues**: Large networks may need optimization
4. **Memory errors**: Reduce test data size or use batch processing

### Performance Optimization
- Cache network features for frequent nodes
- Pre-compute features for known entities
- Use sampling for very large test networks
- Consider feature selection for faster scoring

## Example Test Commands

```bash
# Full model test with evaluation
python test_network.py

# Demo inference 
python inference.py

# Custom test data
python test_network.py --input_path "path/to/test_data.parquet"

# Batch scoring only (no evaluation)
python -c "
from test_network import batch_score
batch_score('input.parquet', 'output.parquet')
"
```