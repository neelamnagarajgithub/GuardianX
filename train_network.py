# ============================================================
# ADVANCED NETWORK INTELLIGENCE MODEL
# Graph-based Fraud Detection using Network Topology
# ============================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from loguru import logger
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ARTIFACT_DIR = Path("artifacts")
RAW_DIR = ARTIFACT_DIR / "raw"
NORM_DIR = ARTIFACT_DIR / "norm"
FEAT_DIR = ARTIFACT_DIR / "features"
MODEL_DIR = ARTIFACT_DIR / "models"

for d in [NORM_DIR, FEAT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(lambda msg: print(msg, end=""))

# ============================================================
# NETWORK INTELLIGENCE FEATURE ENGINEERING
# ============================================================

class NetworkIntelligenceEngine:
    def __init__(self):
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()
        self.node_features = {}
        
    def build_transaction_network(self, transactions):
        """Build comprehensive transaction networks"""
        logger.info("Building transaction networks...")
        
        # Undirected graph for mutual connections
        edges = transactions[['cust_id', 'dest_id', 'amount']].copy()
        
        for _, row in edges.iterrows():
            if self.graph.has_edge(row['cust_id'], row['dest_id']):
                self.graph[row['cust_id']][row['dest_id']]['weight'] += row['amount']
                self.graph[row['cust_id']][row['dest_id']]['count'] += 1
            else:
                self.graph.add_edge(row['cust_id'], row['dest_id'], 
                                  weight=row['amount'], count=1)
        
        # Directed graph for flow analysis
        for _, row in edges.iterrows():
            if self.directed_graph.has_edge(row['cust_id'], row['dest_id']):
                self.directed_graph[row['cust_id']][row['dest_id']]['weight'] += row['amount']
                self.directed_graph[row['cust_id']][row['dest_id']]['count'] += 1
            else:
                self.directed_graph.add_edge(row['cust_id'], row['dest_id'], 
                                           weight=row['amount'], count=1)
        
        logger.info(f"Network built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self
    
    def compute_centrality_features(self):
        """Compute various centrality measures - optimized for large networks"""
        logger.info("Computing centrality features...")
        
        num_nodes = self.graph.number_of_nodes()
        
        # Always compute degree centrality - fast
        logger.info("Computing degree centrality...")
        degree_centrality = nx.degree_centrality(self.graph)
        
        # For very large networks, skip expensive centrality measures
        if num_nodes > 100000:
            logger.info(f"Large network detected ({num_nodes} nodes). Skipping expensive centrality measures.")
            # Use simple degree-based approximations
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            
            betweenness = {node: degrees[node] / max_degree for node in self.graph.nodes()}
            eigenvector = {node: degrees[node] / max_degree for node in self.graph.nodes()}
            closeness = {node: 1.0 / (degrees[node] + 1) for node in self.graph.nodes()}
        else:
            # Betweenness centrality (sample for large graphs)
            logger.info("Computing betweenness centrality...")
            if num_nodes > 10000:
                k_sample = min(1000, num_nodes // 10)  # Much smaller sample
                betweenness = nx.betweenness_centrality(self.graph, k=k_sample, seed=RANDOM_STATE)
            else:
                betweenness = nx.betweenness_centrality(self.graph)
            
            # Eigenvector centrality with timeout
            logger.info("Computing eigenvector centrality...")
            try:
                eigenvector = nx.eigenvector_centrality(self.graph, max_iter=100, tol=1e-3)
            except:
                # Fallback to degree-based approximation
                degrees = dict(self.graph.degree())
                max_degree = max(degrees.values()) if degrees else 1
                eigenvector = {node: degrees[node] / max_degree for node in self.graph.nodes()}
            
            # Closeness centrality (skip for large graphs)
            logger.info("Computing closeness centrality...")
            if num_nodes > 20000:
                closeness = {node: 0 for node in self.graph.nodes()}
            else:
                closeness = nx.closeness_centrality(self.graph)
        
        # PageRank - usually fast
        logger.info("Computing PageRank...")
        try:
            pagerank = nx.pagerank(self.directed_graph, max_iter=50, tol=1e-3)
        except:
            # Fallback to uniform distribution
            pagerank = {node: 1.0/num_nodes for node in self.directed_graph.nodes()}
        
        self.node_features.update({
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness,
            'eigenvector_centrality': eigenvector,
            'pagerank': pagerank,
            'closeness_centrality': closeness
        })
        
        return self
    
    def compute_structural_features(self):
        """Compute structural network features - optimized for large networks"""
        logger.info("Computing structural features...")
        
        num_nodes = self.graph.number_of_nodes()
        
        # Clustering coefficient - can be expensive for large graphs
        logger.info("Computing clustering coefficient...")
        if num_nodes > 100000:
            # Sample nodes for clustering coefficient
            sample_nodes = np.random.choice(list(self.graph.nodes()), 
                                          min(10000, num_nodes), 
                                          replace=False)
            clustering_sample = nx.clustering(self.graph, nodes=sample_nodes)
            # Extend to all nodes with average value for non-sampled
            avg_clustering = np.mean(list(clustering_sample.values())) if clustering_sample else 0
            clustering = {node: clustering_sample.get(node, avg_clustering) for node in self.graph.nodes()}
        else:
            clustering = nx.clustering(self.graph)
        
        # Triangle count
        logger.info("Computing triangle count...")
        triangles = nx.triangles(self.graph)
        
        # Core number (k-core decomposition) - usually fast
        logger.info("Computing core numbers...")
        core_numbers = nx.core_number(self.graph)
        
        # Local efficiency - skip for very large networks as it's very expensive
        logger.info("Computing local efficiency...")
        if num_nodes > 50000:
            logger.info("Skipping local efficiency for large network...")
            local_efficiency = {node: 0 for node in self.graph.nodes()}
        else:
            local_efficiency = {}
            for i, node in enumerate(self.graph.nodes()):
                if i % 10000 == 0 and i > 0:
                    logger.info(f"Local efficiency progress: {i}/{num_nodes}")
                
                neighbors = list(self.graph.neighbors(node))
                if len(neighbors) > 1:
                    subgraph = self.graph.subgraph(neighbors)
                    if subgraph.number_of_edges() > 0:
                        local_efficiency[node] = nx.global_efficiency(subgraph)
                    else:
                        local_efficiency[node] = 0
                else:
                    local_efficiency[node] = 0
        
        self.node_features.update({
            'clustering_coefficient': clustering,
            'triangle_count': triangles,
            'core_number': core_numbers,
            'local_efficiency': local_efficiency
        })
        
        return self
    
    def compute_flow_features(self):
        """Compute money flow and transaction features"""
        logger.info("Computing flow features...")
        
        # In-degree and out-degree (weighted)
        in_degree_weighted = {}
        out_degree_weighted = {}
        in_degree_count = {}
        out_degree_count = {}
        
        for node in self.directed_graph.nodes():
            # Weighted degrees
            in_weight = sum([data['weight'] for _, _, data in self.directed_graph.in_edges(node, data=True)])
            out_weight = sum([data['weight'] for _, _, data in self.directed_graph.out_edges(node, data=True)])
            
            # Transaction counts
            in_count = sum([data['count'] for _, _, data in self.directed_graph.in_edges(node, data=True)])
            out_count = sum([data['count'] for _, _, data in self.directed_graph.out_edges(node, data=True)])
            
            in_degree_weighted[node] = in_weight
            out_degree_weighted[node] = out_weight
            in_degree_count[node] = in_count
            out_degree_count[node] = out_count
        
        # Flow ratios
        flow_ratio = {}
        for node in self.directed_graph.nodes():
            in_w = in_degree_weighted.get(node, 0)
            out_w = out_degree_weighted.get(node, 0)
            total_w = in_w + out_w
            if total_w > 0:
                flow_ratio[node] = out_w / total_w
            else:
                flow_ratio[node] = 0.5
        
        self.node_features.update({
            'in_degree_weighted': in_degree_weighted,
            'out_degree_weighted': out_degree_weighted,
            'in_degree_count': in_degree_count,
            'out_degree_count': out_degree_count,
            'flow_ratio': flow_ratio
        })
        
        return self
    
    def compute_community_features(self):
        """Compute community detection features"""
        logger.info("Computing community features...")
        
        try:
            # Louvain community detection
            communities = nx.community.louvain_communities(self.graph, seed=RANDOM_STATE)
            
            # Create node to community mapping
            node_to_community = {}
            community_sizes = {}
            
            for i, community in enumerate(communities):
                community_sizes[i] = len(community)
                for node in community:
                    node_to_community[node] = i
            
            # Community size for each node
            node_community_size = {}
            for node, comm_id in node_to_community.items():
                node_community_size[node] = community_sizes[comm_id]
            
            self.node_features.update({
                'community_id': node_to_community,
                'community_size': node_community_size
            })
            
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            # Fallback: assign all nodes to single community
            fallback_community = {node: 0 for node in self.graph.nodes()}
            fallback_size = {node: self.graph.number_of_nodes() for node in self.graph.nodes()}
            self.node_features.update({
                'community_id': fallback_community,
                'community_size': fallback_size
            })
        
        return self
    
    def extract_network_features(self, transactions):
        """Main method to extract all network features"""
        self.build_transaction_network(transactions)
        self.compute_centrality_features()
        self.compute_structural_features()
        self.compute_flow_features()
        self.compute_community_features()
        
        # Convert to DataFrame
        feature_df = pd.DataFrame()
        all_nodes = set()
        
        # Collect all unique nodes
        for feature_dict in self.node_features.values():
            all_nodes.update(feature_dict.keys())
        
        # Create features DataFrame
        for feature_name, feature_dict in self.node_features.items():
            values = [feature_dict.get(node, 0) for node in all_nodes]
            if not feature_df.empty:
                feature_df[feature_name] = values
            else:
                feature_df = pd.DataFrame({
                    'node_id': list(all_nodes),
                    feature_name: values
                })
        
        return feature_df

# ============================================================
# DATA LOADING AND NORMALIZATION
# ============================================================

def load_and_normalize_datasets():
    """Load and normalize all available datasets"""
    logger.info("Loading and normalizing datasets...")
    
    all_transactions = []
    
    # Load Nigerian dataset
    if (RAW_DIR / "nigerian.parquet").exists():
        logger.info("Loading Nigerian dataset...")
        df = pd.read_parquet(RAW_DIR / "nigerian.parquet")
        
        # Normalize Nigerian dataset using schema
        df_norm = pd.DataFrame({
            'cust_id': df['sender_account'].astype(str),
            'dest_id': df['receiver_account'].astype(str),
            'amount': pd.to_numeric(df['amount_ngn'], errors='coerce').fillna(0),
            'label': df['is_fraud'].astype(int),
            'dataset': 'nigerian'
        })
        all_transactions.append(df_norm)
        logger.info(f"Nigerian dataset: {len(df_norm)} transactions")
    
    # Load PaySim dataset
    if (RAW_DIR / "paysim.parquet").exists():
        logger.info("Loading PaySim dataset...")
        df = pd.read_parquet(RAW_DIR / "paysim.parquet")
        
        df_norm = pd.DataFrame({
            'cust_id': df['nameOrig'].astype(str),
            'dest_id': df['nameDest'].astype(str),
            'amount': pd.to_numeric(df['amount'], errors='coerce').fillna(0),
            'label': df['isFraud'].astype(int),
            'dataset': 'paysim'
        })
        all_transactions.append(df_norm)
        logger.info(f"PaySim dataset: {len(df_norm)} transactions")
    
    # Load CIFER dataset
    if (RAW_DIR / "cifer_sample.parquet").exists():
        logger.info("Loading CIFER dataset...")
        df = pd.read_parquet(RAW_DIR / "cifer_sample.parquet")
        
        # Assume CIFER has similar structure to PaySim (adjust as needed)
        if 'nameOrig' in df.columns:
            df_norm = pd.DataFrame({
                'cust_id': df['nameOrig'].astype(str),
                'dest_id': df['nameDest'].astype(str),
                'amount': pd.to_numeric(df['amount'], errors='coerce').fillna(0),
                'label': df['isFraud'].astype(int) if 'isFraud' in df.columns else 0,
                'dataset': 'cifer'
            })
            all_transactions.append(df_norm)
            logger.info(f"CIFER dataset: {len(df_norm)} transactions")
    
    if not all_transactions:
        raise ValueError("No datasets found in raw directory!")
    
    # Combine all datasets
    combined_df = pd.concat(all_transactions, ignore_index=True)
    
    # Remove invalid transactions
    combined_df = combined_df[
        (combined_df['amount'] > 0) & 
        (combined_df['cust_id'] != combined_df['dest_id'])
    ].reset_index(drop=True)
    
    logger.info(f"Combined dataset: {len(combined_df)} transactions")
    logger.info(f"Fraud rate: {combined_df['label'].mean():.4f}")
    
    return combined_df

# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_network_intelligence_model():
    """Train the network intelligence model"""
    logger.info("Starting Network Intelligence Model Training...")
    
    # Load data
    transactions = load_and_normalize_datasets()
    
    # Sample if dataset is too large
    if len(transactions) > 500000:
        logger.info("Sampling large dataset for training...")
        fraud_txns = transactions[transactions['label'] == 1]
        normal_txns = transactions[transactions['label'] == 0].sample(n=min(400000, len(transactions[transactions['label'] == 0])), random_state=RANDOM_STATE)
        transactions = pd.concat([fraud_txns, normal_txns]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    # Extract network features
    network_engine = NetworkIntelligenceEngine()
    network_features = network_engine.extract_network_features(transactions)
    
    # Merge features with transactions
    # For customers (senders)
    cust_features = network_features.copy()
    cust_features.columns = ['cust_id'] + [f'cust_{col}' for col in cust_features.columns if col != 'node_id']
    cust_features.rename(columns={'node_id': 'cust_id'}, inplace=True)
    
    # For destinations (receivers)
    dest_features = network_features.copy()
    dest_features.columns = ['dest_id'] + [f'dest_{col}' for col in dest_features.columns if col != 'node_id']
    dest_features.rename(columns={'node_id': 'dest_id'}, inplace=True)
    
    # Merge with transactions
    transactions_with_features = transactions.merge(cust_features, on='cust_id', how='left')
    transactions_with_features = transactions_with_features.merge(dest_features, on='dest_id', how='left')
    
    # Fill missing values with 0 (for nodes not in network)
    feature_columns = [col for col in transactions_with_features.columns 
                      if col.startswith(('cust_', 'dest_')) and col not in ('cust_id', 'dest_id')]
    transactions_with_features[feature_columns] = transactions_with_features[feature_columns].fillna(0)
    
    # Add amount-based features
    transactions_with_features['amount_log'] = np.log1p(transactions_with_features['amount'])
    
    # Prepare training data
    feature_columns = feature_columns + ['amount_log']
    X = transactions_with_features[feature_columns]
    y = transactions_with_features['label']
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Features: {feature_columns}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    # Calculate class weights
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    # LightGBM parameters optimized for network features
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 64,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'scale_pos_weight': scale_pos_weight,
        'seed': RANDOM_STATE,
        'verbosity': -1,
        'device_type': 'gpu'  # Change to 'gpu' if available
    }
    
    # Train model
    logger.info("Training LightGBM model...")
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(50)
        ]
    )
    
    # Evaluate model
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    ap_score = average_precision_score(y_test, y_pred_proba)
    
    logger.info(f"Model Performance:")
    logger.info(f"AUC Score: {auc_score:.4f}")
    logger.info(f"AP Score: {ap_score:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 Most Important Features:")
    logger.info(feature_importance.head(10).to_string())
    
    # Save model and artifacts
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    model.save_model(str(MODEL_DIR / "lgbm_model.txt"))
    feature_importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    
    # Save feature columns for inference
    with open(MODEL_DIR / "feature_columns.txt", 'w') as f:
        for col in feature_columns:
            f.write(f"{col}\n")
    
    logger.info(f"Model and artifacts saved to {MODEL_DIR}")
    
    return model, scaler, feature_columns

if __name__ == "__main__":
    train_network_intelligence_model()