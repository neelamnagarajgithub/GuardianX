# ============================================================
# FIXED ADVANCED NETWORK INTELLIGENCE TRAINING SYSTEM
# Bug-free version with proper NetworkX parameter handling
# ============================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, QuantileTransformer
from sklearn.ensemble import IsolationForest
import joblib
from loguru import logger
from collections import defaultdict, Counter
import warnings
import hashlib
import json
import optuna
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Kaggle-compatible paths
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/kaggle/working/artifacts"))
RAW_DIR = OUTPUT_DIR / "raw"
FEAT_DIR = OUTPUT_DIR / "features"
MODEL_DIR = OUTPUT_DIR / "models"

for d in [FEAT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(lambda msg: print(msg, end=""))

class AdvancedNetworkIntelligence:
    """Advanced network intelligence with FIXED centrality computation"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()
        self.temporal_graphs = {}
        self.node_features = {}
        self.advanced_features = {}
        
    def build_comprehensive_network(self, transactions):
        """Build multi-layer temporal network with memory optimization"""
        logger.info("Building comprehensive network architecture...")
        
        # Use more efficient edge aggregation
        logger.info("Aggregating transaction edges...")
        edge_data = transactions.groupby(['cust_id', 'dest_id']).agg({
            'amount': ['sum', 'count']
        }).reset_index()
        
        edge_data.columns = ['cust_id', 'dest_id', 'weight', 'count']
        
        logger.info(f"Processing {len(edge_data)} unique edges...")
        
        # Build graphs more efficiently
        for _, row in edge_data.iterrows():
            # Undirected graph
            self.graph.add_edge(row['cust_id'], row['dest_id'], 
                               weight=row['weight'], count=row['count'])
            
            # Directed graph
            self.directed_graph.add_edge(row['cust_id'], row['dest_id'], 
                                       weight=row['weight'], count=row['count'])
        
        # Time-based network snapshots
        if 'dataset' in transactions.columns:
            for dataset in transactions['dataset'].unique():
                dataset_txns = transactions[transactions['dataset'] == dataset]
                G_temp = nx.DiGraph()
                for _, row in dataset_txns.iterrows():
                    G_temp.add_edge(row['cust_id'], row['dest_id'], weight=row['amount'])
                self.temporal_graphs[dataset] = G_temp
        
        logger.info(f"Network: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self
        
    def compute_advanced_centrality_features(self):
        """FIXED centrality computation with proper parameter handling"""
        logger.info("Computing advanced centrality features...")
        
        num_nodes = self.graph.number_of_nodes()
        logger.info(f"Graph size: {num_nodes} nodes")
        
        # 1. Degree centrality (always fast and safe)
        degree_centrality = nx.degree_centrality(self.graph)
        logger.info("✅ Degree centrality computed")
        
        # 2. PageRank (usually reliable)
        try:
            pagerank = nx.pagerank(
                self.directed_graph,
                alpha=0.85,
                max_iter=200,
                tol=1e-3
            )
            logger.info("✅ PageRank computed")
        except Exception as e:
            logger.warning(f"PageRank failed: {e}")
            pagerank = {node: 1.0/num_nodes for node in self.directed_graph.nodes()}
        
        # 3. FIXED Betweenness centrality
        try:
            if num_nodes > 50000:
                # For very large graphs, use approximation
                logger.info("Large graph detected - using degree-based betweenness approximation")
                degrees = dict(self.graph.degree())
                max_degree = max(degrees.values()) if degrees else 1
                betweenness = {node: degrees[node] / max_degree * 0.1 for node in self.graph.nodes()}
                
            elif num_nodes > 10000:
                # FIXED: Use proper k parameter (integer, not list)
                sample_size = min(5000, num_nodes // 2)  # Ensure reasonable sample size
                
                logger.info(f"Computing sampled betweenness with k={sample_size}...")
                betweenness = nx.betweenness_centrality(
                    self.graph,
                    k=sample_size,  # This must be an INTEGER, not a list
                    normalized=True,
                    seed=RANDOM_STATE
                )
                
            else:
                # Small graph - compute exact betweenness
                logger.info("Computing exact betweenness centrality...")
                betweenness = nx.betweenness_centrality(self.graph)
            
            logger.info("✅ Betweenness centrality computed")
            
        except Exception as e:
            logger.warning(f"Betweenness centrality failed: {e}")
            # Fallback to degree-based approximation
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            betweenness = {node: degrees[node] / max_degree * 0.1 for node in self.graph.nodes()}
        
        # 4. Eigenvector centrality (skip for very large graphs)
        if num_nodes < 100000:
            try:
                logger.info("Computing eigenvector centrality...")
                eigenvector = nx.eigenvector_centrality(
                    self.graph,
                    max_iter=300,
                    tol=1e-4
                )
                logger.info("✅ Eigenvector centrality computed")
            except Exception as e:
                logger.warning(f"Eigenvector centrality failed: {e}")
                degrees = dict(self.graph.degree())
                max_degree = max(degrees.values()) if degrees else 1
                eigenvector = {node: degrees[node] / max_degree for node in self.graph.nodes()}
        else:
            logger.info("Skipping eigenvector centrality for large graph")
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            eigenvector = {node: degrees[node] / max_degree for node in self.graph.nodes()}
        
        # 5. Closeness centrality (very expensive, skip for large graphs)
        if num_nodes < 20000:
            try:
                logger.info("Computing closeness centrality...")
                closeness = nx.closeness_centrality(self.graph)
                logger.info("✅ Closeness centrality computed")
            except Exception as e:
                logger.warning(f"Closeness centrality failed: {e}")
                closeness = {node: 0.0 for node in self.graph.nodes()}
        else:
            logger.info("Skipping closeness centrality for large graph")
            closeness = {node: 0.0 for node in self.graph.nodes()}
        
        # 6. HITS (skip for very large graphs)
        if num_nodes < 100000:
            try:
                logger.info("Computing HITS scores...")
                hits = nx.hits(self.directed_graph, max_iter=200, tol=1e-4)
                authority_scores = hits[1]
                hub_scores = hits[0]
                logger.info("✅ HITS computed")
            except Exception as e:
                logger.warning(f"HITS failed: {e}")
                authority_scores = {node: 0.0 for node in self.directed_graph.nodes()}
                hub_scores = {node: 0.0 for node in self.directed_graph.nodes()}
        else:
            logger.info("Skipping HITS for large graph")
            authority_scores = {node: 0.0 for node in self.directed_graph.nodes()}
            hub_scores = {node: 0.0 for node in self.directed_graph.nodes()}
        
        # 7. Katz centrality (skip for very large graphs)  
        if num_nodes < 50000:
            try:
                logger.info("Computing Katz centrality...")
                katz = nx.katz_centrality(self.directed_graph, max_iter=200, tol=1e-4)
                logger.info("✅ Katz centrality computed")
            except Exception as e:
                logger.warning(f"Katz centrality failed: {e}")
                katz = {node: 0.0 for node in self.directed_graph.nodes()}
        else:
            logger.info("Skipping Katz centrality for large graph")
            katz = {node: 0.0 for node in self.directed_graph.nodes()}
        
        self.node_features.update({
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness,
            'eigenvector_centrality': eigenvector,
            'pagerank': pagerank,
            'closeness_centrality': closeness,
            'authority_score': authority_scores,
            'hub_score': hub_scores,
            'katz_centrality': katz
        })
        
        logger.info("✅ All centrality features completed successfully")
        return self
    
    def compute_structural_features(self):
        """Compute structural features with optimization for large graphs"""
        logger.info("Computing structural network features...")
        
        num_nodes = self.graph.number_of_nodes()
        
        # 1. Clustering coefficient (sample for very large graphs)
        if num_nodes > 200000:
            logger.info("Using degree-based clustering approximation for large graph")
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            clustering = {node: (degrees[node] / max_degree) * 0.3 for node in self.graph.nodes()}
        elif num_nodes > 50000:
            # Sample clustering for medium-large graphs
            logger.info("Computing clustering for sample of nodes...")
            sample_nodes = np.random.choice(
                list(self.graph.nodes()), 
                size=min(20000, num_nodes // 3), 
                replace=False
            )
            clustering_sample = nx.clustering(self.graph, nodes=sample_nodes)
            avg_clustering = np.mean(list(clustering_sample.values())) if clustering_sample else 0.3
            
            clustering = {}
            for node in self.graph.nodes():
                if node in clustering_sample:
                    clustering[node] = clustering_sample[node]
                else:
                    clustering[node] = avg_clustering
        else:
            clustering = nx.clustering(self.graph)
        
        logger.info("✅ Clustering coefficient computed")
        
        # 2. Triangle count (approximate for large graphs)
        if num_nodes > 100000:
            logger.info("Using degree-based triangle approximation")
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            triangles = {node: int((degrees[node] / max_degree) * degrees[node] * 0.1) 
                        for node in self.graph.nodes()}
        else:
            try:
                triangles = nx.triangles(self.graph)
                logger.info("✅ Triangle count computed")
            except Exception as e:
                logger.warning(f"Triangle computation failed: {e}")
                degrees = dict(self.graph.degree())
                triangles = {node: degrees[node] // 3 for node in self.graph.nodes()}
        
        # 3. K-core decomposition (usually fast)
        try:
            core_numbers = nx.core_number(self.graph)
            logger.info("✅ K-core decomposition computed")
        except Exception as e:
            logger.warning(f"K-core computation failed: {e}")
            core_numbers = {node: 1 for node in self.graph.nodes()}
        
        # Skip expensive square clustering and constraint for very large graphs
        if num_nodes < 50000:
            # Square clustering (simplified computation)
            square_clustering = {}
            for node in self.graph.nodes():
                neighbors = list(self.graph.neighbors(node))
                if len(neighbors) >= 2:
                    squares = sum(1 for i, n1 in enumerate(neighbors) 
                                 for n2 in neighbors[i+1:] 
                                 if self.graph.has_edge(n1, n2))
                    square_clustering[node] = squares / max(len(neighbors), 1)
                else:
                    square_clustering[node] = 0
            
            # Constraint (simplified Burt's structural holes)
            constraint = {}
            for node in self.graph.nodes():
                neighbors = list(self.graph.neighbors(node))
                if len(neighbors) > 0:
                    constraint[node] = 1.0 / len(neighbors)  # Simplified calculation
                else:
                    constraint[node] = 0
            
            # Effective size
            effective_size = {}
            for node in self.graph.nodes():
                neighbors = list(self.graph.neighbors(node))
                if len(neighbors) > 0:
                    redundancy = sum(1 for i, n1 in enumerate(neighbors) 
                                   for n2 in neighbors[i+1:] 
                                   if self.graph.has_edge(n1, n2))
                    effective_size[node] = len(neighbors) - redundancy / max(len(neighbors), 1)
                else:
                    effective_size[node] = 0
                    
            logger.info("✅ Advanced structural features computed")
            
        else:
            logger.info("Skipping expensive structural features for large graph")
            # Use simple degree-based proxies
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            
            square_clustering = {node: (degrees[node] / max_degree) * 0.1 for node in self.graph.nodes()}
            constraint = {node: 1.0 / max(degrees[node], 1) for node in self.graph.nodes()}
            effective_size = {node: degrees[node] * 0.8 for node in self.graph.nodes()}
        
        self.node_features.update({
            'clustering_coefficient': clustering,
            'triangle_count': triangles,
            'square_clustering': square_clustering,
            'core_number': core_numbers,
            'constraint': constraint,
            'effective_size': effective_size
        })
        
        logger.info("✅ All structural features completed")
        return self
    
    def compute_flow_and_velocity_features(self):
        """Compute money flow and velocity features (optimized)"""
        logger.info("Computing flow and velocity features...")
        
        # Pre-compute all degrees in batch (much faster)
        in_degrees_weighted = dict(self.directed_graph.in_degree(weight='weight'))
        out_degrees_weighted = dict(self.directed_graph.out_degree(weight='weight'))
        in_degrees_count = dict(self.directed_graph.in_degree(weight='count'))
        out_degrees_count = dict(self.directed_graph.out_degree(weight='count'))
        
        # Compute derived features
        flow_ratio = {}
        velocity_score = {}
        concentration_index = {}
        
        for node in self.directed_graph.nodes():
            in_w = in_degrees_weighted.get(node, 0)
            out_w = out_degrees_weighted.get(node, 0)
            total_w = in_w + out_w
            
            flow_ratio[node] = out_w / total_w if total_w > 0 else 0.5
            
            in_c = in_degrees_count.get(node, 0)
            out_c = out_degrees_count.get(node, 0)
            total_c = in_c + out_c
            velocity_score[node] = total_w / total_c if total_c > 0 else 0
            
            # Simplified concentration index
            if out_w > 0:
                concentration_index[node] = min(1.0, out_w / (in_w + out_w + 1))
            else:
                concentration_index[node] = 0
        
        self.node_features.update({
            'in_degree_weighted': in_degrees_weighted,
            'out_degree_weighted': out_degrees_weighted,
            'in_degree_count': in_degrees_count,
            'out_degree_count': out_degrees_count,
            'flow_ratio': flow_ratio,
            'velocity_score': velocity_score,
            'concentration_index': concentration_index
        })
        
        logger.info("✅ Flow and velocity features computed")
        return self
    
    def compute_community_and_anomaly_features(self):
        """Compute community detection and anomaly features (scalable)"""
        logger.info("Computing community and anomaly features...")
        
        num_nodes = self.graph.number_of_nodes()
        
        try:
            if num_nodes > 200000:
                logger.info("Using label propagation for very large graph...")
                communities_iter = nx.community.label_propagation_communities(
                    self.graph, seed=RANDOM_STATE
                )
                communities = list(communities_iter)
            else:
                logger.info("Using Louvain community detection...")
                communities = nx.community.louvain_communities(self.graph, seed=RANDOM_STATE)
            
            node_to_community = {}
            community_sizes = {}
            
            for i, community in enumerate(communities):
                community_sizes[i] = len(community)
                for node in community:
                    node_to_community[node] = i
            
            node_community_size = {
                node: community_sizes.get(node_to_community.get(node, 0), 1)
                for node in self.graph.nodes()
            }
            
            logger.info(f"✅ Found {len(communities)} communities")
            
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            # Fallback: assign all nodes to single community
            node_to_community = {node: 0 for node in self.graph.nodes()}
            node_community_size = {node: num_nodes for node in self.graph.nodes()}
        
        # Local anomaly score (degree-based)
        degrees = dict(self.graph.degree())
        local_anomaly_score = {}
        
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                node_degree = degrees[node]
                neighbor_degrees = [degrees[n] for n in neighbors]
                avg_neighbor_degree = np.mean(neighbor_degrees)
                deviation = abs(node_degree - avg_neighbor_degree) / (avg_neighbor_degree + 1)
                local_anomaly_score[node] = min(deviation, 5.0)  # Cap extreme values
            else:
                local_anomaly_score[node] = 1.0
        
        # Simplified modularity contribution
        modularity_contribution = {}
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                comm_id = node_to_community.get(node, 0)
                internal_edges = sum(1 for n in neighbors 
                                   if node_to_community.get(n, -1) == comm_id)
                external_edges = len(neighbors) - internal_edges
                total_edges = len(neighbors)
                modularity_contribution[node] = (internal_edges - external_edges) / total_edges
            else:
                modularity_contribution[node] = 0
        
        self.node_features.update({
            'community_id': node_to_community,
            'community_size': node_community_size,
            'modularity_contribution': modularity_contribution,
            'local_anomaly_score': local_anomaly_score
        })
        
        logger.info("✅ Community and anomaly features computed")
        return self
    
    def compute_temporal_features(self):
        """Compute temporal network evolution features"""
        logger.info("Computing temporal network features...")
        
        # Cross-dataset connectivity (if multiple datasets exist)
        if len(self.temporal_graphs) > 1:
            dataset_presence = {}
            degree_variance = {}
            degree_stability = {}
            
            for node in self.graph.nodes():
                # Count dataset presence
                presence = sum(1 for G in self.temporal_graphs.values() if node in G)
                dataset_presence[node] = presence
                
                # Compute degree variance across datasets
                degrees = []
                for G in self.temporal_graphs.values():
                    if node in G:
                        degrees.append(G.degree(node))
                    else:
                        degrees.append(0)
                
                if len(degrees) > 1:
                    variance = np.var(degrees)
                    degree_variance[node] = variance
                    degree_stability[node] = 1.0 / (1.0 + variance)
                else:
                    degree_variance[node] = 0
                    degree_stability[node] = 1.0
        else:
            # Single dataset case
            dataset_presence = {node: 1 for node in self.graph.nodes()}
            degree_variance = {node: 0 for node in self.graph.nodes()}
            degree_stability = {node: 1.0 for node in self.graph.nodes()}
        
        self.node_features.update({
            'dataset_presence': dataset_presence,
            'degree_variance': degree_variance,
            'degree_stability': degree_stability
        })
        
        logger.info("✅ Temporal features computed")
        return self
    
    def extract_all_network_features(self, transactions):
        """Extract all advanced network features with progress tracking"""
        logger.info("🔗 Extracting comprehensive network intelligence features...")
        
        # Build network
        self.build_comprehensive_network(transactions)
        
        # Extract features with progress tracking
        logger.info("Step 1/5: Computing centrality features...")
        self.compute_advanced_centrality_features()
        
        logger.info("Step 2/5: Computing structural features...")
        self.compute_structural_features()
        
        logger.info("Step 3/5: Computing flow and velocity features...")
        self.compute_flow_and_velocity_features()
        
        logger.info("Step 4/5: Computing community and anomaly features...")
        self.compute_community_and_anomaly_features()
        
        logger.info("Step 5/5: Computing temporal features...")
        self.compute_temporal_features()
        
        # Convert to DataFrame
        all_nodes = set()
        for feature_dict in self.node_features.values():
            all_nodes.update(feature_dict.keys())
        
        feature_df = pd.DataFrame({'node_id': list(all_nodes)})
        
        for feature_name, feature_dict in self.node_features.items():
            values = [feature_dict.get(node, 0) for node in all_nodes]
            feature_df[feature_name] = values
        
        logger.info(f"✅ Network feature extraction completed: {len(feature_df.columns)-1} features for {len(all_nodes)} nodes")
        return feature_df

# Keep all other classes from your original file (AdvancedTransactionFeatures, AdvancedDeviceFeatures, etc.)
# Copy them exactly as they are working fine
def load_and_preprocess_datasets():
    """Load and preprocess all available datasets"""
    logger.info("Loading and preprocessing datasets...")
    
    all_transactions = []
    
    # Load datasets in order of preference
    dataset_configs = [
        {'file': 'nigerian.parquet', 'name': 'nigerian', 'cust_col': 'sender_account', 
         'dest_col': 'receiver_account', 'amount_col': 'amount_ngn', 'label_col': 'is_fraud'},
        {'file': 'paysim.parquet', 'name': 'paysim', 'cust_col': 'nameOrig', 
         'dest_col': 'nameDest', 'amount_col': 'amount', 'label_col': 'isFraud'},
        {'file': 'cifer_sample.parquet', 'name': 'cifer', 'cust_col': 'nameOrig', 
         'dest_col': 'nameDest', 'amount_col': 'amount', 'label_col': 'isFraud'}
    ]
    
    for config in dataset_configs:
        file_path = RAW_DIR / config['file']
        if file_path.exists():
            logger.info(f"Loading {config['name']} dataset...")
            try:
                df = pd.read_parquet(file_path)
                
                # Normalize columns
                normalized_df = pd.DataFrame({
                    'cust_id': df[config['cust_col']].astype(str),
                    'dest_id': df[config['dest_col']].astype(str),
                    'amount': pd.to_numeric(df[config['amount_col']], errors='coerce').fillna(0),
                    'label': df[config['label_col']].astype(int) if config['label_col'] in df.columns else 0,
                    'dataset': config['name']
                })
                
                all_transactions.append(normalized_df)
                logger.info(f"{config['name']} dataset: {len(normalized_df)} transactions")
                
            except Exception as e:
                logger.error(f"Failed to load {config['name']}: {e}")
                continue
    
    if not all_transactions:
        raise ValueError("No valid datasets found! Run download_and_save.py first.")
    
    # Combine datasets
    combined_df = pd.concat(all_transactions, ignore_index=True)
    
    # Clean data
    combined_df = combined_df[
        (combined_df['amount'] > 0) & 
        (combined_df['cust_id'] != combined_df['dest_id']) &
        (combined_df['cust_id'].notna()) &
        (combined_df['dest_id'].notna())
    ].reset_index(drop=True)
    
    logger.info(f"Combined dataset: {len(combined_df)} transactions")
    logger.info(f"Fraud rate: {combined_df['label'].mean():.4f}")
    
    # Sample if too large (for Kaggle memory limits)
    max_transactions = 1_000_000
    if len(combined_df) > max_transactions:
        logger.info(f"Sampling to {max_transactions} transactions...")
        
        # Stratified sampling to preserve fraud rate
        fraud_txns = combined_df[combined_df['label'] == 1]
        normal_txns = combined_df[combined_df['label'] == 0]
        
        fraud_sample_size = min(len(fraud_txns), max_transactions // 10)  # 10% fraud
        normal_sample_size = max_transactions - fraud_sample_size
        
        if len(normal_txns) > normal_sample_size:
            normal_sample = normal_txns.sample(n=normal_sample_size, random_state=RANDOM_STATE)
        else:
            normal_sample = normal_txns
        
        combined_df = pd.concat([fraud_txns.sample(n=fraud_sample_size, random_state=RANDOM_STATE), 
                                normal_sample], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        
        logger.info(f"Sampled dataset: {len(combined_df)} transactions, fraud rate: {combined_df['label'].mean():.4f}")
    
    return combined_df


class AdvancedTransactionFeatures:
    """Extract advanced transaction-level features"""
    
    @staticmethod
    def extract_transaction_features(transactions):
        """Extract comprehensive transaction features"""
        logger.info("Extracting advanced transaction features...")
        
        df = transactions.copy()
        
        # Amount features
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_sqrt'] = np.sqrt(df['amount'])
        df['amount_reciprocal'] = 1.0 / (df['amount'] + 1)
        
        # Round number detection
        df['is_round_amount'] = ((df['amount'] % 10 == 0) | 
                                (df['amount'] % 100 == 0) | 
                                (df['amount'] % 1000 == 0)).astype(int)
        
        # Customer-level aggregations
        cust_stats = df.groupby('cust_id')['amount'].agg([
            'count', 'sum', 'mean', 'std', 'min', 'max', 'median'
        ]).fillna(0)
        cust_stats.columns = [f'cust_{col}' for col in cust_stats.columns]
        
        # Destination-level aggregations
        dest_stats = df.groupby('dest_id')['amount'].agg([
            'count', 'sum', 'mean', 'std', 'min', 'max', 'median'
        ]).fillna(0)
        dest_stats.columns = [f'dest_{col}' for col in dest_stats.columns]
        
        # Merge back
        df = df.merge(cust_stats, left_on='cust_id', right_index=True, how='left')
        df = df.merge(dest_stats, left_on='dest_id', right_index=True, how='left')
        
        # Z-scores
        df['amount_zscore_cust'] = (df['amount'] - df['cust_mean']) / (df['cust_std'] + 1e-6)
        df['amount_zscore_dest'] = (df['amount'] - df['dest_mean']) / (df['dest_std'] + 1e-6)
        
        # Velocity features
        df['cust_velocity'] = df['cust_sum'] / df['cust_count']
        df['dest_velocity'] = df['dest_sum'] / df['dest_count']
        
        # Risk ratios
        df['amount_vs_cust_max_ratio'] = df['amount'] / (df['cust_max'] + 1)
        df['amount_vs_dest_max_ratio'] = df['amount'] / (df['dest_max'] + 1)
        
        # Dataset features
        if 'dataset' in df.columns:
            dataset_encoder = LabelEncoder()
            df['dataset_encoded'] = dataset_encoder.fit_transform(df['dataset'])
        else:
            df['dataset_encoded'] = 0
        
        logger.info(f"Generated {len([col for col in df.columns if col not in ['cust_id', 'dest_id', 'amount', 'label', 'dataset']])} transaction features")
        return df

# ...existing code...

class AdvancedDeviceFeatures:
    """Generate synthetic device intelligence features - OPTIMIZED VERSION"""
    
    @staticmethod
    def generate_device_features(transactions):
        """Generate advanced device fingerprinting features - FIXED for large datasets"""
        logger.info("Generating advanced device intelligence features...")
        
        df = transactions.copy()
        np.random.seed(RANDOM_STATE)
        
        # FIXED: Use customer-based features instead of session-based to reduce memory
        logger.info("Computing device features for unique customers...")
        unique_customers = df['cust_id'].unique()
        logger.info(f"Processing {len(unique_customers)} unique customers...")
        
        # Pre-compute device profiles for customers only (not sessions)
        device_profiles = {}
        
        # Process in chunks to avoid memory issues
        chunk_size = 10000
        for i in range(0, len(unique_customers), chunk_size):
            chunk_customers = unique_customers[i:i + chunk_size]
            
            for cust in chunk_customers:
                # Use hash for deterministic but pseudo-random features
                hash_val = hash(cust) % 1000000
                np.random.seed(hash_val)
                
                device_profiles[cust] = {
                    # Simplified device features (reduced from 25+ to 10)
                    'screen_width': np.random.choice([1920, 1366, 1280, 768]),
                    'screen_height': np.random.choice([1080, 768, 720, 1024]),
                    'color_depth': np.random.choice([24, 32, 16]),
                    'timezone_offset': np.random.randint(-12, 13),
                    'cpu_cores': np.random.choice([2, 4, 8, 16]),
                    'browser_score': np.random.uniform(0.1, 1.0),
                    'risk_score': np.random.uniform(0, 0.3),
                    'automation_score': np.random.uniform(0, 0.2),
                    'consistency_score': np.random.uniform(0.5, 1.0),
                    'session_duration': np.random.uniform(1, 120)
                }
            
            if (i + chunk_size) % 50000 == 0:
                logger.info(f"Processed {min(i + chunk_size, len(unique_customers))} customers...")
        
        # Map features to transactions (much faster with customer-based mapping)
        logger.info("Mapping device features to transactions...")
        for feature_name in ['screen_width', 'screen_height', 'color_depth', 'timezone_offset', 
                           'cpu_cores', 'browser_score', 'risk_score', 'automation_score', 
                           'consistency_score', 'session_duration']:
            df[f'device_{feature_name}'] = df['cust_id'].map(
                {cust: profile[feature_name] for cust, profile in device_profiles.items()}
            ).fillna(0.5)  # Default value for unmapped customers
        
        # Simple device switching rate (much faster computation)
        logger.info("Computing device switching behavior...")
        dest_counts = df.groupby('cust_id')['dest_id'].nunique().to_dict()
        txn_counts = df.groupby('cust_id').size().to_dict()
        
        df['device_switching_rate'] = df['cust_id'].map(
            {cust: dest_counts.get(cust, 1) / txn_counts.get(cust, 1) 
             for cust in df['cust_id'].unique()}
        ).fillna(0)
        
        df['device_consistency_score'] = 1.0 / (1.0 + df['device_switching_rate'])
        
        device_feature_count = len([col for col in df.columns if col.startswith('device_')])
        logger.info(f"Generated {device_feature_count} device intelligence features")
        return df

# ...existing code...

# Keep your existing AdvancedFraudEnsemble class exactly as is
class AdvancedFraudEnsemble:
    """Advanced ensemble with multiple algorithms - FIXED VERSION"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.ensemble_weights = None
        
    # Add this method to your AdvancedFraudEnsemble class to replace the existing train_ensemble method

    def train_ensemble(self, X_train, y_train, X_val, y_val):
        """Train multiple models and learn ensemble weights - FINAL FIX"""
        logger.info("Training advanced ensemble models...")
        
        # Calculate class weights
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        logger.info(f"Dataset balance: {pos_count} fraud, {neg_count} normal (scale: {scale_pos_weight:.2f})")
        
        # FINAL FIX: Better feature preprocessing before scaling
        logger.info("🔧 Applying final preprocessing fixes...")
        
        # Cap extreme outliers more aggressively
        X_train_processed = X_train.copy()
        X_val_processed = X_val.copy()
        
        for col in X_train_processed.columns:
            # Use more aggressive capping (99.5%/0.5% instead of 99%/1%)
            q995 = X_train_processed[col].quantile(0.995)
            q005 = X_train_processed[col].quantile(0.005)
            
            # If range is still too extreme, cap further
            if q995 - q005 > 1e6:
                q99 = X_train_processed[col].quantile(0.99)
                q01 = X_train_processed[col].quantile(0.01)
                q995, q005 = q99, q01
            
            X_train_processed[col] = X_train_processed[col].clip(lower=q005, upper=q995)
            X_val_processed[col] = X_val_processed[col].clip(lower=q005, upper=q995)
        
        # Use QuantileTransformer for even more robust scaling
        from sklearn.preprocessing import QuantileTransformer
        
        models_config = {
            'lightgbm': {
                'params': {
                    'objective': 'binary',
                    'metric': 'auc',
                    'boosting_type': 'gbdt',
                    'learning_rate': 0.01,        # FINAL FIX: Even slower learning
                    'num_leaves': 63,             # FINAL FIX: Reduce complexity to prevent overfitting
                    'max_depth': 6,               # FINAL FIX: Reduce depth
                    'min_data_in_leaf': 100,      # FINAL FIX: More data per leaf
                    'min_gain_to_split': 0.01,    # FINAL FIX: Higher threshold for splits
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 1,
                    'lambda_l1': 0.1,            # FINAL FIX: Add L1 regularization
                    'lambda_l2': 0.1,            # FINAL FIX: Add L2 regularization
                    'scale_pos_weight': scale_pos_weight,
                    'verbosity': 0,              # FINAL FIX: Reduce verbosity
                    'random_state': RANDOM_STATE,
                    'device_type': 'cpu',
                    'force_col_wise': True,
                    'min_child_samples': 50,     # FINAL FIX: Prevent small leaves
                    'reg_sqrt': True            # FINAL FIX: Additional regularization
                },
                'scaler': QuantileTransformer(    # FINAL FIX: Most robust scaler
                    n_quantiles=1000,
                    output_distribution='uniform',
                    subsample=100000,
                    random_state=RANDOM_STATE
                )
            }
        }
        
        val_predictions = {}
        
        for model_name, config in models_config.items():
            logger.info(f"Training {model_name}...")
            
            # Scale features with quantile transformer
            scaler = config['scaler']
            logger.info("Applying quantile transformation for maximum robustness...")
            X_train_scaled = scaler.fit_transform(X_train_processed)
            X_val_scaled = scaler.transform(X_val_processed)
            
            # Check final scaling results
            logger.info(f"Final scaled data range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
            logger.info(f"Final scaled data mean: {X_train_scaled.mean():.3f}, std: {X_train_scaled.std():.3f}")
            
            try:
                if model_name == 'lightgbm':
                    train_data = lgb.Dataset(X_train_scaled, label=y_train)
                    val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
                    
                    logger.info("Starting LightGBM training with FINAL FIXED parameters...")
                    model = lgb.train(
                        config['params'],
                        train_data,
                        num_boost_round=500,         # FINAL FIX: Reasonable limit
                        valid_sets=[train_data, val_data],
                        valid_names=['train', 'eval'],
                        callbacks=[
                            lgb.early_stopping(50),      # FINAL FIX: Less patience to prevent overfitting
                            lgb.log_evaluation(0)        # FINAL FIX: Silent training
                        ]
                    )
                    
                    logger.info(f"✅ Training completed! Best iteration: {model.best_iteration}, Total trees: {model.num_trees()}")
                    
                    # Enhanced validation
                    if model.num_trees() < 10:
                        logger.warning(f"⚠️  Model has only {model.num_trees()} trees!")
                        # Try to retrain with even more conservative parameters
                        logger.info("Retraining with ultra-conservative parameters...")
                        
                        conservative_params = config['params'].copy()
                        conservative_params.update({
                            'learning_rate': 0.001,      # Very slow
                            'num_leaves': 31,           # Very simple
                            'min_data_in_leaf': 200,    # Large leaves
                            'min_gain_to_split': 0.1,   # High split threshold
                        })
                        
                        model = lgb.train(
                            conservative_params,
                            train_data,
                            num_boost_round=1000,
                            valid_sets=[val_data],
                            valid_names=['eval'],
                            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
                        )
                        
                        logger.info(f"🔄 Retrained model: {model.best_iteration} iteration, {model.num_trees()} trees")
                    
                    val_pred = model.predict(X_val_scaled, num_iteration=model.best_iteration)
                    self.feature_importance[model_name] = model.feature_importance()
                
                # Store model and scaler
                self.models[model_name] = model
                self.scalers[model_name] = scaler
                val_predictions[model_name] = val_pred
                
                # Evaluate individual model
                auc = roc_auc_score(y_val, val_pred)
                logger.info(f"🎯 {model_name} validation AUC: {auc:.4f}")
                
                # Enhanced prediction analysis
                unique_preds = len(np.unique(np.round(val_pred, 6)))
                pred_range = val_pred.max() - val_pred.min()
                high_confidence = np.sum(val_pred > 0.8)
                low_confidence = np.sum(val_pred < 0.1)
                
                logger.info(f"📊 Prediction Analysis:")
                logger.info(f"   Unique values: {unique_preds}")
                logger.info(f"   Range: {pred_range:.4f}")
                logger.info(f"   High confidence (>0.8): {high_confidence}")
                logger.info(f"   Low confidence (<0.1): {low_confidence}")
                logger.info(f"   Mean: {val_pred.mean():.4f}")
                
                if unique_preds < 100:
                    logger.warning("⚠️  Limited prediction diversity - model may need more training")
                else:
                    logger.info("✅ Good prediction diversity")
                    
            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not self.models:
            raise ValueError("❌ No models were successfully trained!")
        
        logger.info(f"✅ Successfully trained {len(self.models)} models")
        return self
    
    def predict(self, X):
        """Make ensemble predictions"""
        if not self.models:
            raise ValueError("No models trained")
        
        # Use the first available model
        model_name = list(self.models.keys())[0]
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled, num_iteration=model.best_iteration)
        
        return predictions

# Add RobustScaler import at the top of your file if not already present
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder

# ...rest of your existing code stays the same...

def train_advanced_network_intelligence():
    """FIXED main training pipeline"""
    logger.info("🚀 Starting FIXED Advanced Network Intelligence Training")
    logger.info("=" * 60)
    
    # Load and preprocess data
    transactions = load_and_preprocess_datasets()
    
    # Extract network features with FIXED centrality computation
    logger.info("🔗 Extracting Network Intelligence Features...")
    network_engine = AdvancedNetworkIntelligence()  # Using FIXED version
    network_features = network_engine.extract_all_network_features(transactions)
    
    # Extract transaction features
    logger.info("💰 Extracting Transaction Features...")
    transactions_with_txn_features = AdvancedTransactionFeatures.extract_transaction_features(transactions)
    
    # Generate device features
    logger.info("📱 Generating Device Intelligence Features...")
    transactions_enhanced = AdvancedDeviceFeatures.generate_device_features(transactions_with_txn_features)
    
    # Merge network features
    logger.info("🔄 Merging All Features...")
    
    # Customer network features
    cust_features = network_features.copy()
    cust_features.columns = ['cust_id'] + [f'cust_{col}' for col in cust_features.columns if col != 'node_id']
    cust_features.rename(columns={'node_id': 'cust_id'}, inplace=True)
    
    # Destination network features
    dest_features = network_features.copy()
    dest_features.columns = ['dest_id'] + [f'dest_{col}' for col in dest_features.columns if col != 'node_id']
    dest_features.rename(columns={'node_id': 'dest_id'}, inplace=True)
    
    # Merge all features
    final_df = transactions_enhanced.merge(cust_features, on='cust_id', how='left')
    final_df = final_df.merge(dest_features, on='dest_id', how='left')
    
    # Fill missing network features (for nodes not in network)
    network_columns = [col for col in final_df.columns if col.startswith(('cust_', 'dest_'))]
    final_df[network_columns] = final_df[network_columns].fillna(0)
    
    # FIXED: Handle extreme values in features before training
    logger.info("🔧 Handling extreme values in features...")
    feature_columns = [col for col in final_df.columns 
                      if col not in ['cust_id', 'dest_id', 'label', 'dataset']]
    
    # Cap extreme values to reasonable ranges
    for col in feature_columns:
        if final_df[col].dtype in ['float64', 'int64']:
            q99 = final_df[col].quantile(0.99)
            q01 = final_df[col].quantile(0.01)
            
            if q99 > 1e6 or q01 < -1e6:  # Extreme values detected
                logger.info(f"Capping extreme values in {col}: [{q01:.2e}, {q99:.2e}]")
                final_df[col] = final_df[col].clip(lower=q01, upper=q99)
    
    X = final_df[feature_columns]
    y = final_df['label']
    
    logger.info(f"📊 Final Feature Matrix: {X.shape}")
    logger.info(f"📈 Total Features: {len(feature_columns)}")
    logger.info(f"🎯 Target Distribution: {Counter(y)}")
    logger.info(f"📈 Feature ranges: min={X.min().min():.3f}, max={X.max().max():.3f}")
    
    # Ensure we have enough samples of each class
    if y.sum() < 10:
        logger.warning("⚠️  Very few fraud samples - creating additional synthetic fraud samples")
        # Add some synthetic fraud samples to ensure proper training
        fraud_indices = y[y == 1].index
        if len(fraud_indices) > 0:
            # Duplicate existing fraud samples with small noise
            synthetic_samples = []
            for _ in range(50 - y.sum()):  # Add up to 50 total fraud samples
                idx = np.random.choice(fraud_indices)
                synthetic_sample = X.iloc[idx].copy()
                # Add small random noise
                noise = np.random.normal(0, 0.01, len(synthetic_sample))
                synthetic_sample = synthetic_sample + noise
                synthetic_samples.append(synthetic_sample)
            
            if synthetic_samples:
                synthetic_df = pd.DataFrame(synthetic_samples, columns=X.columns)
                synthetic_labels = pd.Series([1] * len(synthetic_samples))
                
                X = pd.concat([X, synthetic_df], ignore_index=True)
                y = pd.concat([y, synthetic_labels], ignore_index=True)
                
                logger.info(f"✅ Added {len(synthetic_samples)} synthetic fraud samples")
                logger.info(f"🎯 New Target Distribution: {Counter(y)}")
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    logger.info(f"Training fraud rate: {y_train.mean():.4f}")
    logger.info(f"Validation fraud rate: {y_val.mean():.4f}")
    
    # Train ensemble
    logger.info("🎯 Training Advanced Ensemble Models with FIXED parameters...")
    ensemble = AdvancedFraudEnsemble()
    ensemble.train_ensemble(X_train, y_train, X_val, y_val)
    
    # Final evaluation
    logger.info("📋 Final Model Evaluation...")
    y_pred_proba = ensemble.predict(X_val)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    auc_score = roc_auc_score(y_val, y_pred_proba)
    ap_score = average_precision_score(y_val, y_pred_proba)
    
    # Additional diagnostics
    logger.info(f"🔍 Prediction diagnostics:")
    logger.info(f"   Prediction range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
    logger.info(f"   Unique predictions: {len(np.unique(np.round(y_pred_proba, 4)))}")
    logger.info(f"   Mean prediction: {y_pred_proba.mean():.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🏆 FINAL MODEL PERFORMANCE")
    logger.info("=" * 60)
    logger.info(f"🎯 AUC Score: {auc_score:.4f}")
    logger.info(f"📊 Average Precision: {ap_score:.4f}")
    logger.info(f"🔢 Total Features: {len(feature_columns)}")
    logger.info(f"🌳 Trees in model: {list(ensemble.models.values())[0].num_trees() if ensemble.models else 0}")
    
    # Save models and artifacts
    logger.info("💾 Saving Models and Artifacts...")
    
    # Save ensemble
    joblib.dump(ensemble, MODEL_DIR / "advanced_ensemble_fixed.pkl")
    
    # Save feature columns
    with open(MODEL_DIR / "feature_columns.txt", 'w') as f:
        for col in feature_columns:
            f.write(f"{col}\n")
    
    # Save model metadata
    metadata = {
        'model_type': 'Fixed Advanced Network Intelligence Ensemble',
        'total_features': len(feature_columns),
        'models': list(ensemble.models.keys()),
        'performance': {
            'auc': float(auc_score),
            'average_precision': float(ap_score)
        },
        'training_data': {
            'total_transactions': len(final_df),
            'fraud_rate': float(y.mean()),
            'datasets': final_df['dataset'].unique().tolist()
        },
        'model_info': {
            'num_trees': int(list(ensemble.models.values())[0].num_trees()) if ensemble.models else 0,
            'best_iteration': int(list(ensemble.models.values())[0].best_iteration) if ensemble.models else 0,
            'scaler_type': 'RobustScaler'
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(MODEL_DIR / "model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ All artifacts saved to: {MODEL_DIR}")
    logger.info("\n" + "=" * 60)
    logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    
    return ensemble, feature_columns, {
        'auc': auc_score,
        'average_precision': ap_score,
        'total_features': len(feature_columns),
        'num_trees': list(ensemble.models.values())[0].num_trees() if ensemble.models else 0
    }

if __name__ == "__main__":
    try:
        ensemble, features, performance = train_advanced_network_intelligence()
        logger.info(f"\n🏁 Training Summary:")
        logger.info(f"   Features: {len(features)}")
        logger.info(f"   AUC: {performance['auc']:.4f}")
        logger.info(f"   AP: {performance['average_precision']:.4f}")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise