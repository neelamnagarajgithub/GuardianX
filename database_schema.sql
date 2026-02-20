-- ============================================================
-- TRAVEL PLATFORM DATABASE SCHEMA
-- PostgreSQL schema for network intelligence and fraud detection
-- ============================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- ============================================================
-- CORE BUSINESS TABLES
-- ============================================================

-- Users table
CREATE TABLE users (
    user_id VARCHAR(50) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    country_code VARCHAR(2),
    age_group VARCHAR(20),
    customer_segment VARCHAR(20) DEFAULT 'regular',
    total_bookings INTEGER DEFAULT 0,
    total_spent DECIMAL(12,2) DEFAULT 0.00,
    last_activity TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'active',
    kyc_status VARCHAR(20) DEFAULT 'pending',
    risk_score DECIMAL(5,4) DEFAULT 0.0000,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Travel agencies table
CREATE TABLE agencies (
    agency_id VARCHAR(50) PRIMARY KEY,
    agency_name VARCHAR(255) NOT NULL,
    license_number VARCHAR(100),
    registration_country VARCHAR(2),
    business_type VARCHAR(20) DEFAULT 'online',
    commission_rate DECIMAL(5,4) DEFAULT 0.1000,
    onboarding_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    kyc_status VARCHAR(20) DEFAULT 'pending',
    risk_category VARCHAR(20) DEFAULT 'unknown',
    total_transactions INTEGER DEFAULT 0,
    total_volume DECIMAL(15,2) DEFAULT 0.00,
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Transactions table (main transaction log)
CREATE TABLE transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL REFERENCES users(user_id),
    agency_id VARCHAR(50) NOT NULL REFERENCES agencies(agency_id),
    booking_type VARCHAR(20) NOT NULL,
    amount DECIMAL(12,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    booking_date TIMESTAMP WITH TIME ZONE NOT NULL,
    travel_date DATE,
    destination VARCHAR(3),
    source_country VARCHAR(3),
    payment_method VARCHAR(50),
    device_fingerprint VARCHAR(255),
    ip_address INET,
    session_id VARCHAR(255),
    user_agent TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- ============================================================
-- NETWORK INTELLIGENCE TABLES
-- ============================================================

-- Network nodes (users + agencies)
CREATE TABLE network_nodes (
    node_id VARCHAR(50) PRIMARY KEY,
    node_type VARCHAR(20) NOT NULL, -- 'user' or 'agency'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Centrality metrics
    degree_centrality DECIMAL(8,6) DEFAULT 0.000000,
    betweenness_centrality DECIMAL(8,6) DEFAULT 0.000000,
    eigenvector_centrality DECIMAL(8,6) DEFAULT 0.000000,
    pagerank DECIMAL(8,6) DEFAULT 0.000000,
    closeness_centrality DECIMAL(8,6) DEFAULT 0.000000,
    
    -- Structural metrics
    clustering_coefficient DECIMAL(8,6) DEFAULT 0.000000,
    triangle_count INTEGER DEFAULT 0,
    core_number INTEGER DEFAULT 0,
    local_efficiency DECIMAL(8,6) DEFAULT 0.000000,
    
    -- Flow metrics
    in_degree_weighted DECIMAL(15,2) DEFAULT 0.00,
    out_degree_weighted DECIMAL(15,2) DEFAULT 0.00,
    in_degree_count INTEGER DEFAULT 0,
    out_degree_count INTEGER DEFAULT 0,
    flow_ratio DECIMAL(8,6) DEFAULT 0.500000,
    
    -- Community metrics
    community_id INTEGER DEFAULT 0,
    community_size INTEGER DEFAULT 1,
    
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Network edges (connections between nodes)
CREATE TABLE network_edges (
    edge_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_node VARCHAR(50) NOT NULL,
    target_node VARCHAR(50) NOT NULL,
    edge_type VARCHAR(20) DEFAULT 'transaction',
    
    -- Aggregated metrics
    total_weight DECIMAL(15,2) DEFAULT 0.00,
    transaction_count INTEGER DEFAULT 0,
    first_interaction TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_interaction TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Edge features
    avg_amount DECIMAL(12,2) DEFAULT 0.00,
    frequency_score DECIMAL(8,6) DEFAULT 0.000000,
    recency_score DECIMAL(8,6) DEFAULT 0.000000,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(source_node, target_node, edge_type)
);

-- ============================================================
-- FRAUD DETECTION TABLES
-- ============================================================

-- Risk scores history
CREATE TABLE risk_scores (
    score_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id VARCHAR(50) NOT NULL REFERENCES transactions(transaction_id),
    user_id VARCHAR(50) NOT NULL,
    agency_id VARCHAR(50) NOT NULL,
    
    -- Model scores
    fraud_probability DECIMAL(8,6) NOT NULL,
    fraud_prediction INTEGER NOT NULL, -- 0 or 1
    risk_bucket VARCHAR(10) NOT NULL,
    model_version VARCHAR(20) DEFAULT 'v1.0',
    
    -- Risk factors
    risk_factors JSONB DEFAULT '[]'::jsonb,
    feature_values JSONB DEFAULT '{}'::jsonb,
    
    -- Processing info
    processing_time_ms DECIMAL(8,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- High risk transactions requiring investigation
CREATE TABLE high_risk_transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    agency_id VARCHAR(50) NOT NULL,
    risk_data JSONB NOT NULL,
    investigation_status VARCHAR(20) DEFAULT 'pending',
    assigned_investigator VARCHAR(100),
    resolution_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Anomaly detection results
CREATE TABLE anomalies (
    anomaly_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(20) NOT NULL, -- 'user', 'agency', 'transaction'
    entity_id VARCHAR(50) NOT NULL,
    anomaly_type VARCHAR(50) NOT NULL,
    severity VARCHAR(10) NOT NULL, -- 'low', 'medium', 'high'
    score DECIMAL(8,6) NOT NULL,
    details JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(20) DEFAULT 'new',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================
-- FEATURE ENGINEERING TABLES
-- ============================================================

-- Pre-computed user features for fast lookup
CREATE TABLE user_features (
    user_id VARCHAR(50) PRIMARY KEY REFERENCES users(user_id),
    
    -- Behavioral features
    avg_transaction_amount DECIMAL(12,2) DEFAULT 0.00,
    transaction_frequency DECIMAL(8,4) DEFAULT 0.0000,
    booking_diversity DECIMAL(8,4) DEFAULT 0.0000,
    preferred_destinations JSONB DEFAULT '[]'::jsonb,
    
    -- Temporal features
    avg_advance_booking_days DECIMAL(6,2) DEFAULT 30.00,
    weekend_booking_ratio DECIMAL(8,4) DEFAULT 0.0000,
    night_booking_ratio DECIMAL(8,4) DEFAULT 0.0000,
    
    -- Device/Security features
    device_count INTEGER DEFAULT 1,
    ip_count INTEGER DEFAULT 1,
    location_consistency_score DECIMAL(8,4) DEFAULT 1.0000,
    
    -- Network features (from network_nodes)
    network_risk_score DECIMAL(8,4) DEFAULT 0.0000,
    
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pre-computed agency features
CREATE TABLE agency_features (
    agency_id VARCHAR(50) PRIMARY KEY REFERENCES agencies(agency_id),
    
    -- Business metrics
    avg_transaction_amount DECIMAL(12,2) DEFAULT 0.00,
    customer_count INTEGER DEFAULT 0,
    repeat_customer_ratio DECIMAL(8,4) DEFAULT 0.0000,
    cancellation_rate DECIMAL(8,4) DEFAULT 0.0000,
    
    -- Quality metrics
    customer_satisfaction_score DECIMAL(8,4) DEFAULT 5.0000,
    complaint_count INTEGER DEFAULT 0,
    response_time_avg DECIMAL(8,2) DEFAULT 24.00,
    
    -- Risk indicators
    chargebacks_count INTEGER DEFAULT 0,
    dispute_ratio DECIMAL(8,4) DEFAULT 0.0000,
    unusual_pattern_count INTEGER DEFAULT 0,
    
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================
-- MONITORING AND ALERTING TABLES
-- ============================================================

-- System alerts
CREATE TABLE alerts (
    alert_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(10) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    entity_type VARCHAR(20),
    entity_id VARCHAR(50),
    alert_data JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(20) DEFAULT 'active',
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model performance tracking
CREATE TABLE model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    evaluation_date DATE NOT NULL,
    
    -- Performance metrics
    auc_score DECIMAL(8,6),
    precision_score DECIMAL(8,6),
    recall_score DECIMAL(8,6),
    f1_score DECIMAL(8,6),
    
    -- Distribution metrics
    total_predictions INTEGER,
    fraud_predictions INTEGER,
    actual_frauds INTEGER,
    
    evaluation_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================

-- Transactions indexes
CREATE INDEX idx_transactions_user_id ON transactions(user_id);
CREATE INDEX idx_transactions_agency_id ON transactions(agency_id);
CREATE INDEX idx_transactions_booking_date ON transactions(booking_date);
CREATE INDEX idx_transactions_amount ON transactions(amount);
CREATE INDEX idx_transactions_status ON transactions(status);
CREATE INDEX idx_transactions_composite ON transactions(user_id, agency_id, booking_date);

-- Network indexes
CREATE INDEX idx_network_nodes_type ON network_nodes(node_type);
CREATE INDEX idx_network_nodes_updated ON network_nodes(updated_at);
CREATE INDEX idx_network_edges_source ON network_edges(source_node);
CREATE INDEX idx_network_edges_target ON network_edges(target_node);
CREATE INDEX idx_network_edges_last_interaction ON network_edges(last_interaction);

-- Risk scoring indexes
CREATE INDEX idx_risk_scores_transaction ON risk_scores(transaction_id);
CREATE INDEX idx_risk_scores_user ON risk_scores(user_id);
CREATE INDEX idx_risk_scores_bucket ON risk_scores(risk_bucket);
CREATE INDEX idx_risk_scores_created ON risk_scores(created_at);

-- Anomalies indexes
CREATE INDEX idx_anomalies_entity ON anomalies(entity_type, entity_id);
CREATE INDEX idx_anomalies_type ON anomalies(anomaly_type);
CREATE INDEX idx_anomalies_severity ON anomalies(severity);
CREATE INDEX idx_anomalies_status ON anomalies(status);
CREATE INDEX idx_anomalies_created ON anomalies(created_at);

-- GIN indexes for JSONB columns
CREATE INDEX idx_transactions_metadata ON transactions USING GIN(metadata);
CREATE INDEX idx_risk_scores_factors ON risk_scores USING GIN(risk_factors);
CREATE INDEX idx_alerts_data ON alerts USING GIN(alert_data);

-- ============================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language plpgsql;

-- Apply update timestamp triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agencies_updated_at BEFORE UPDATE ON agencies 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_network_nodes_updated_at BEFORE UPDATE ON network_nodes 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_network_edges_updated_at BEFORE UPDATE ON network_edges 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to update user stats when transaction is inserted
CREATE OR REPLACE FUNCTION update_user_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update user totals
    UPDATE users 
    SET total_bookings = total_bookings + 1,
        total_spent = total_spent + NEW.amount,
        last_activity = NEW.booking_date
    WHERE user_id = NEW.user_id;
    
    -- Update agency totals
    UPDATE agencies
    SET total_transactions = total_transactions + 1,
        total_volume = total_volume + NEW.amount
    WHERE agency_id = NEW.agency_id;
    
    RETURN NEW;
END;
$$ language plpgsql;

CREATE TRIGGER update_stats_on_transaction AFTER INSERT ON transactions
    FOR EACH ROW EXECUTE FUNCTION update_user_stats();

-- Function to update network edges
CREATE OR REPLACE FUNCTION update_network_edge()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO network_edges (source_node, target_node, total_weight, transaction_count, last_interaction)
    VALUES (NEW.user_id, NEW.agency_id, NEW.amount, 1, NEW.booking_date)
    ON CONFLICT (source_node, target_node, edge_type)
    DO UPDATE SET
        total_weight = network_edges.total_weight + NEW.amount,
        transaction_count = network_edges.transaction_count + 1,
        last_interaction = NEW.booking_date,
        avg_amount = (network_edges.total_weight + NEW.amount) / (network_edges.transaction_count + 1),
        updated_at = NOW();
    
    RETURN NEW;
END;
$$ language plpgsql;

CREATE TRIGGER update_network_on_transaction AFTER INSERT ON transactions
    FOR EACH ROW EXECUTE FUNCTION update_network_edge();

-- ============================================================
-- VIEWS FOR ANALYTICS
-- ============================================================

-- Daily transaction summary
CREATE VIEW daily_transaction_summary AS
SELECT 
    DATE(booking_date) as transaction_date,
    COUNT(*) as total_transactions,
    SUM(amount) as total_volume,
    AVG(amount) as avg_amount,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT agency_id) as unique_agencies,
    COUNT(*) FILTER (WHERE status = 'completed') as completed_transactions,
    COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled_transactions
FROM transactions
WHERE booking_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(booking_date)
ORDER BY transaction_date DESC;

-- High risk entities view
CREATE VIEW high_risk_entities AS
SELECT 
    'user' as entity_type,
    user_id as entity_id,
    risk_score,
    total_bookings,
    total_spent,
    last_activity
FROM users 
WHERE risk_score > 0.7

UNION ALL

SELECT 
    'agency' as entity_type,
    agency_id as entity_id,
    CASE 
        WHEN risk_category = 'high' THEN 0.8
        WHEN risk_category = 'medium' THEN 0.5
        ELSE 0.2
    END as risk_score,
    total_transactions,
    total_volume,
    NULL as last_activity
FROM agencies
WHERE risk_category IN ('high', 'medium')

ORDER BY risk_score DESC;

-- Network statistics view  
CREATE VIEW network_statistics AS
SELECT 
    COUNT(*) as total_nodes,
    COUNT(*) FILTER (WHERE node_type = 'user') as user_nodes,
    COUNT(*) FILTER (WHERE node_type = 'agency') as agency_nodes,
    AVG(degree_centrality) as avg_degree_centrality,
    MAX(degree_centrality) as max_degree_centrality,
    AVG(pagerank) as avg_pagerank,
    COUNT(DISTINCT community_id) as total_communities,
    AVG(community_size) as avg_community_size
FROM network_nodes;

-- ============================================================
-- SAMPLE DATA (for development/testing)
-- ============================================================

-- Insert sample users
INSERT INTO users (user_id, email, country_code, age_group, customer_segment) VALUES
('U001', 'user1@example.com', 'US', '25-34', 'premium'),
('U002', 'user2@example.com', 'UK', '35-44', 'regular'),
('U003', 'user3@example.com', 'DE', '18-24', 'budget');

-- Insert sample agencies
INSERT INTO agencies (agency_id, agency_name, license_number, registration_country, business_type) VALUES
('A001', 'Premium Travel Co', 'LIC001', 'US', 'online'),
('A002', 'Budget Tours Ltd', 'LIC002', 'UK', 'hybrid'),
('A003', 'Adventure Trips', 'LIC003', 'DE', 'offline');

-- Insert sample transactions
INSERT INTO transactions (transaction_id, user_id, agency_id, booking_type, amount, booking_date, travel_date, destination) VALUES
('T001', 'U001', 'A001', 'flight', 1250.00, NOW() - INTERVAL '1 day', CURRENT_DATE + INTERVAL '30 days', 'UK'),
('T002', 'U002', 'A002', 'hotel', 450.00, NOW() - INTERVAL '2 days', CURRENT_DATE + INTERVAL '15 days', 'FR'),
('T003', 'U003', 'A003', 'package', 2100.00, NOW() - INTERVAL '3 days', CURRENT_DATE + INTERVAL '60 days', 'JP');

-- Grant permissions (adjust as needed for your environment)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO travel_platform_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO travel_platform_user;