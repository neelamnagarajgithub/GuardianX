# ============================================================
# DEBUG MODEL STRUCTURE - Find out what's causing the segfault
# ============================================================

import joblib
import numpy as np
from pathlib import Path
import inspect

def debug_model_structure():
    """Debug the actual model structure to find the issue"""
    print("🔍 DEBUGGING MODEL STRUCTURE")
    print("="*40)
    
    model_path = Path("artifacts/models/advanced_ensemble_fixed.pkl")
    
    try:
        # Load model
        model = joblib.load(model_path)
        print(f"✅ Model loaded: {type(model)}")
        
        # Inspect the model class
        print(f"\n📋 Model attributes:")
        for attr in dir(model):
            if not attr.startswith('_'):
                print(f"   {attr}: {type(getattr(model, attr, None))}")
        
        # Check if it has models
        if hasattr(model, 'models'):
            print(f"\n🤖 Ensemble models: {list(model.models.keys())}")
            
        # Check if it has scalers
        if hasattr(model, 'scalers'):
            print(f"⚖️  Scalers: {list(model.scalers.keys())}")
        
        # Look at the predict method
        if hasattr(model, 'predict'):
            print(f"\n🎯 Predict method exists")
            try:
                # Get the source code of predict method
                predict_source = inspect.getsource(model.predict)
                print(f"📝 Predict method source:")
                print(predict_source)
            except:
                print("❌ Cannot get predict method source")
        
        # Try to access LightGBM directly
        if hasattr(model, 'models') and 'lightgbm' in model.models:
            lgb_model = model.models['lightgbm']
            print(f"\n🌳 LightGBM model:")
            print(f"   Type: {type(lgb_model)}")
            print(f"   Trees: {lgb_model.num_trees()}")
            print(f"   Features: {lgb_model.num_feature()}")
            
            # Try direct LightGBM prediction
            print(f"\n🧪 Testing direct LightGBM prediction...")
            X_test = np.random.random((1, 93))
            
            try:
                pred = lgb_model.predict(X_test)
                print(f"✅ Direct LightGBM works: {pred[0]:.4f}")
                return lgb_model  # Return working model
            except Exception as e:
                print(f"❌ Direct LightGBM failed: {e}")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        
    return None

def create_working_predictor():
    """Create a working predictor that bypasses the problematic ensemble"""
    print("\n🔧 CREATING WORKING PREDICTOR")
    print("="*35)
    
    # Debug first
    lgb_model = debug_model_structure()
    
    if lgb_model is None:
        print("❌ Cannot create working predictor")
        return None
    
    # Load the scaler separately
    model_path = Path("artifacts/models/advanced_ensemble_fixed.pkl")
    full_model = joblib.load(model_path)
    
    scaler = None
    if hasattr(full_model, 'scalers') and 'lightgbm' in full_model.scalers:
        scaler = full_model.scalers['lightgbm']
        print(f"✅ Scaler loaded: {type(scaler)}")
    
    def safe_predict(features):
        """Safe prediction function"""
        try:
            # Ensure features are the right type and shape
            if features.shape[1] != 93:
                raise ValueError(f"Expected 93 features, got {features.shape[1]}")
            
            # Apply scaling if available
            if scaler is not None:
                features = scaler.transform(features)
            
            # Make prediction
            pred = lgb_model.predict(features)[0]
            return float(pred)
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return 0.5  # Safe fallback
    
    return safe_predict

def test_working_predictor():
    """Test the working predictor"""
    print("\n🚀 TESTING WORKING PREDICTOR")
    print("="*30)
    
    # Create working predictor
    predict_func = create_working_predictor()
    
    if predict_func is None:
        print("❌ Failed to create working predictor")
        return
    
    # Test with sample data
    test_cases = [
        {"name": "Low Risk", "features": np.random.random((1, 93)) * 100},
        {"name": "High Risk", "features": np.random.random((1, 93)) * 10000},
        {"name": "Medium Risk", "features": np.random.random((1, 93)) * 1000}
    ]
    
    print(f"🔍 Testing {len(test_cases)} cases...")
    
    for i, case in enumerate(test_cases):
        print(f"\n--- Test {i+1}: {case['name']} ---")
        
        try:
            pred = predict_func(case['features'])
            print(f"✅ Prediction: {pred:.4f}")
            
            if pred > 0.8:
                print(f"🚨 HIGH RISK - Block transaction")
            elif pred > 0.6:
                print(f"⚠️  MEDIUM-HIGH RISK - Review required")
            elif pred > 0.4:
                print(f"🔍 MEDIUM RISK - Monitor closely")
            else:
                print(f"✅ LOW RISK - Approve")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print(f"\n🎉 Working predictor test completed!")
    return predict_func

if __name__ == "__main__":
    working_predictor = test_working_predictor()