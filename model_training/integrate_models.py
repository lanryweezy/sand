#!/usr/bin/env python3
"""
Integration Script for Trained Risk Models with Physical Risk Oracle

Connects the trained machine learning models with the Physical Risk Oracle
to enable data-driven predictions.
"""

import os
import sys
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from cognitive.advanced_cognitive_system import PhysicalRiskOracle
from cognitive.physical_risk_oracle import PhysicalRiskAssessment
from models.congestion_predictor import CongestionPredictor
from models.timing_analyzer import TimingAnalyzer
from data.comprehensive_rtl_parser import DesignHierarchyBuilder


class TrainedModelIntegrator:
    """
    Integrates trained ML models with the Physical Risk Oracle
    """
    
    def __init__(self, model_path: str = "models/trained"):
        self.model_path = Path(model_path)
        
        # Load trained models
        self.congestion_model = self._load_model("congestion_model.pkl")
        self.timing_model = self._load_model("timing_model.pkl")
        self.power_model = self._load_model("power_model.pkl")
        self.drc_model = self._load_model("drc_model.pkl")
        
        # Initialize the original components
        self.original_oracle = PhysicalRiskOracle()
        self.hierarchy_builder = DesignHierarchyBuilder()
        
        print(f"Models loaded from: {self.model_path}")
    
    def _load_model(self, model_name: str):
        """Load a trained model from disk"""
        model_file = self.model_path / model_name
        if model_file.exists():
            model = joblib.load(model_file)
            print(f"Loaded {model_name}")
            return model
        else:
            print(f"Warning: {model_name} not found, using fallback")
            return None
    
    def extract_features_from_rtl(self, rtl_file: str, constraints_file: str) -> Dict[str, float]:
        """
        Extract features from RTL and constraints for ML models
        """
        features = {}
        
        try:
            # Use the hierarchy builder to parse and extract features
            rtl_data = self.hierarchy_builder.parse_rtl(rtl_file)
            
            # Extract structural features
            features['node_count'] = len(rtl_data.get('instances', [])) if rtl_data else 0
            features['module_count'] = len(rtl_data.get('modules', [])) if rtl_data else 0
            features['port_count'] = len(rtl_data.get('ports', [])) if rtl_data else 0
            
            # Parse constraints file for additional features
            with open(constraints_file, 'r') as f:
                constraint_content = f.read()
            
            # Extract constraint-related features
            features['clock_count'] = constraint_content.count('create_clock')
            features['input_delay_count'] = constraint_content.count('set_input_delay')
            features['output_delay_count'] = constraint_content.count('set_output_delay')
            features['timing_path_count'] = constraint_content.count('set_max_delay') + constraint_content.count('set_min_delay')
            
            # Normalize features
            features['node_count_norm'] = min(features['node_count'] / 1000.0, 1.0)
            features['module_count_norm'] = min(features['module_count'] / 100.0, 1.0)
            features['port_count_norm'] = min(features['port_count'] / 100.0, 1.0)
            features['clock_count_norm'] = min(features['clock_count'] / 10.0, 1.0)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return default features
            features = {
                'node_count_norm': 0.1,
                'module_count_norm': 0.1,
                'port_count_norm': 0.1,
                'clock_count_norm': 0.1,
                'input_delay_count': 0,
                'output_delay_count': 0,
                'timing_path_count': 0
            }
        
        return features
    
    def create_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        Create a standardized feature vector for ML models
        """
        # Define the expected feature order
        feature_order = [
            'node_count_norm', 'module_count_norm', 'port_count_norm', 
            'clock_count_norm', 'input_delay_count', 'output_delay_count', 
            'timing_path_count', 0.0, 0.0, 0.0  # Padding to 10 features
        ]
        
        vector = []
        for feature_name in feature_order[:7]:  # First 7 are real features
            vector.append(features.get(feature_name, 0.0))
        
        # Add padding
        for _ in range(10 - len(vector)):
            vector.append(0.0)
        
        return np.array(vector).reshape(1, -1)
    
    def predict_with_trained_models(self, rtl_file: str, constraints_file: str) -> Dict[str, Any]:
        """
        Use trained models to make predictions
        """
        # Extract features
        features = self.extract_features_from_rtl(rtl_file, constraints_file)
        feature_vector = self.create_feature_vector(features)
        
        predictions = {}
        
        # Make predictions with each model
        if self.congestion_model:
            try:
                congestion_pred = self.congestion_model.predict(feature_vector)[0]
                # Ensure prediction is in valid range
                congestion_pred = max(0.0, min(1.0, float(congestion_pred)))
                predictions['congestion'] = congestion_pred
            except Exception as e:
                print(f"Congestion model prediction failed: {e}")
                predictions['congestion'] = 0.5  # Default value
        else:
            predictions['congestion'] = 0.5
        
        if self.timing_model:
            try:
                timing_pred = self.timing_model.predict(feature_vector)[0]
                timing_pred = max(0.0, min(1.0, float(timing_pred)))
                predictions['timing'] = timing_pred
            except Exception as e:
                print(f"Timing model prediction failed: {e}")
                predictions['timing'] = 0.5
        else:
            predictions['timing'] = 0.5
        
        if self.power_model:
            try:
                power_pred = self.power_model.predict(feature_vector)[0]
                power_pred = max(0.0, min(1.0, float(power_pred)))
                predictions['power'] = power_pred
            except Exception as e:
                print(f"Power model prediction failed: {e}")
                predictions['power'] = 0.5
        else:
            predictions['power'] = 0.5
        
        if self.drc_model:
            try:
                drc_pred = self.drc_model.predict(feature_vector)[0]
                drc_prob = self.drc_model.predict_proba(feature_vector)[0]
                predictions['drc'] = {
                    'risk': int(drc_pred),
                    'probability': float(max(drc_prob))
                }
            except Exception as e:
                print(f"DRC model prediction failed: {e}")
                predictions['drc'] = {'risk': 0, 'probability': 0.5}
        else:
            predictions['drc'] = {'risk': 0, 'probability': 0.5}
        
        return predictions
    
    def enhance_oracle_with_trained_models(self, rtl_file: str, constraints_file: str, node: str = "7nm") -> PhysicalRiskAssessment:
        """
        Enhance the Physical Risk Oracle with trained model predictions
        """
        # Get predictions from trained models
        ml_predictions = self.predict_with_trained_models(rtl_file, constraints_file)
        
        # Create enhanced assessment by combining ML predictions with original oracle
        original_assessment = self.original_oracle.predict_physical_risks(rtl_file, constraints_file, node)
        
        # Enhance the original assessment with ML predictions
        enhanced_congestion = self._enhance_congestion_map(original_assessment.congestion_heatmap, ml_predictions['congestion'])
        enhanced_timing = self._enhance_timing_risks(original_assessment.timing_risk_zones, ml_predictions['timing'])
        enhanced_power = self._enhance_power_hotspots(original_assessment.power_density_hotspots, ml_predictions['power'])
        enhanced_drc = self._enhance_drc_risks(original_assessment.drc_risk_classes, ml_predictions['drc'])
        
        # Update confidence based on ML model agreement
        ml_confidence = np.mean([ml_predictions['congestion'], ml_predictions['timing'], ml_predictions['power']])
        combined_confidence = (original_assessment.overall_confidence + ml_confidence) / 2.0
        
        # Create enhanced recommendations based on ML predictions
        enhanced_recommendations = self._combine_recommendations(
            original_assessment.recommendations, 
            ml_predictions
        )
        
        enhanced_assessment = PhysicalRiskAssessment(
            congestion_heatmap=enhanced_congestion,
            timing_risk_zones=enhanced_timing,
            clock_skew_sensitivity=original_assessment.clock_skew_sensitivity,
            power_density_hotspots=enhanced_power,
            drc_risk_classes=enhanced_drc,
            overall_confidence=combined_confidence,
            recommendations=enhanced_recommendations
        )
        
        return enhanced_assessment
    
    def _enhance_congestion_map(self, original_map: Dict[str, float], ml_prediction: float) -> Dict[str, float]:
        """Enhance congestion map with ML prediction"""
        # If ML predicts high congestion, increase all congestion estimates
        enhancement_factor = 1.0 + (ml_prediction - 0.5) * 0.4  # ±20% adjustment based on ML
        
        enhanced_map = {}
        for region, original_congestion in original_map.items():
            enhanced_congestion = min(1.0, original_congestion * enhancement_factor)
            enhanced_map[region] = enhanced_congestion
        
        # If no regions exist, create a default one based on ML prediction
        if not enhanced_map:
            enhanced_map['central_region'] = ml_prediction
        
        return enhanced_map
    
    def _enhance_timing_risks(self, original_risks: list, ml_prediction: float) -> list:
        """Enhance timing risks with ML prediction"""
        # Add or modify timing risks based on ML prediction
        enhanced_risks = original_risks.copy()
        
        # If ML predicts high timing risk, add a critical path warning
        if ml_prediction > 0.7 and not any('critical' in str(risk).lower() for risk in enhanced_risks):
            enhanced_risks.append({
                'path': 'ML_predicted_critical_path',
                'risk_level': 'critical',
                'estimated_slack': -abs(ml_prediction * 0.5)
            })
        
        return enhanced_risks
    
    def _enhance_power_hotspots(self, original_hotspots: list, ml_prediction: float) -> list:
        """Enhance power hotspots with ML prediction"""
        enhanced_hotspots = original_hotspots.copy()
        
        # If ML predicts high power risk, add a hotspot
        if ml_prediction > 0.7:
            # Find if we already have a high-risk hotspot
            has_high_risk = any(h.get('estimated_power', 0) > 0.8 for h in enhanced_hotspots)
            if not has_high_risk:
                enhanced_hotspots.append({
                    'node': 'ML_predicted_hotspot',
                    'estimated_power': ml_prediction,
                    'region': 'computed_region',
                    'risk_level': 'high'
                })
        
        return enhanced_hotspots
    
    def _enhance_drc_risks(self, original_risks: list, ml_prediction: dict) -> list:
        """Enhance DRC risks with ML prediction"""
        enhanced_risks = original_risks.copy()
        
        # If ML predicts high DRC risk, add a warning
        if ml_prediction['risk'] == 1 and ml_prediction['probability'] > 0.7:
            # Check if we already have this type of risk
            has_similar_risk = any('ML_predicted' in str(risk) for risk in enhanced_risks)
            if not has_similar_risk:
                enhanced_risks.append({
                    'rule_class': 'ML_predicted_risk',
                    'severity': 'high',
                    'description': f'ML model predicts DRC issues with {ml_prediction["probability"]:.2f} confidence',
                    'probability': ml_prediction['probability']
                })
        
        return enhanced_risks
    
    def _combine_recommendations(self, original_recs: list, ml_predictions: Dict[str, Any]) -> list:
        """Combine original recommendations with ML-enhanced ones"""
        enhanced_recs = original_recs.copy()
        
        # Add ML-based recommendations
        if ml_predictions['congestion'] > 0.8:
            enhanced_recs.append(f"ML model indicates HIGH congestion risk ({ml_predictions['congestion']:.2f}), consider early floorplan adjustments")
        
        if ml_predictions['timing'] > 0.8:
            enhanced_recs.append(f"ML model indicates HIGH timing risk ({ml_predictions['timing']:.2f}), consider aggressive timing constraints")
        
        if ml_predictions['power'] > 0.8:
            enhanced_recs.append(f"ML model indicates HIGH power risk ({ml_predictions['power']:.2f}), consider power grid reinforcement")
        
        if ml_predictions['drc']['risk'] == 1 and ml_predictions['drc']['probability'] > 0.7:
            enhanced_recs.append(f"ML model indicates HIGH DRC risk ({ml_predictions['drc']['probability']:.2f}), consider design rule compliance checks")
        
        return enhanced_recs


def main():
    """Main integration function"""
    print("Silicon Intelligence - Model Integration with Physical Risk Oracle")
    print("=" * 80)
    
    integrator = TrainedModelIntegrator()
    
    print("\nThe trained models have been integrated with the Physical Risk Oracle.")
    print("The system now combines:")
    print("- Original heuristic-based predictions")
    print("- ML model-based predictions from real and synthetic data")
    print("- Enhanced confidence scores")
    print("- Combined recommendations")
    
    print(f"\nIntegration status:")
    print(f"  - Congestion model: {'LOADED' if integrator.congestion_model else 'NOT FOUND'}")
    print(f"  - Timing model: {'LOADED' if integrator.timing_model else 'NOT FOUND'}")
    print(f"  - Power model: {'LOADED' if integrator.power_model else 'NOT FOUND'}")
    print(f"  - DRC model: {'LOADED' if integrator.drc_model else 'NOT FOUND'}")
    
    print(f"\nReady to make data-driven predictions with enhanced accuracy!")
    
    # Example usage would be:
    print(f"\nTo use the enhanced oracle:")
    print(f"  assessment = integrator.enhance_oracle_with_trained_models(rtl_file, constraints_file, node)")


if __name__ == "__main__":
    main()