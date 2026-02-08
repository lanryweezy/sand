"""
Learning Loop with Silicon Feedback

This module implements the learning system that incorporates post-silicon
data to improve the AI models and agent decision-making.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pickle
import os
from datetime import datetime
from silicon_intelligence.utils.logger import get_logger
from silicon_intelligence.agents.base_agent import BaseAgent
from silicon_intelligence.models.congestion_predictor import CongestionPredictor
from silicon_intelligence.models.timing_analyzer import TimingAnalyzer
from silicon_intelligence.models.drc_predictor import DRCPredictor


class SiliconFeedbackProcessor:
    """
    Silicon Feedback Processor - processes post-silicon data to improve models
    """
    
    def __init__(self, model_storage_path: str = "./silicon_feedback_data"):
        self.logger = get_logger(__name__)
        self.model_storage_path = model_storage_path
        self.feedback_data = []
        self.prediction_accuracy_tracker = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(model_storage_path, exist_ok=True)
    
    def process_silicon_bringup_data(self, design_id: str, bringup_results: Dict[str, Any]):
        """
        Process silicon bring-up results to extract learning signals
        
        Args:
            design_id: Unique identifier for the design
            bringup_results: Results from silicon bring-up including measurements
        """
        self.logger.info(f"Processing silicon bring-up data for design {design_id}")
        
        # 1. Input Validation
        if not isinstance(design_id, str) or not design_id:
            self.logger.error("Invalid design_id: Must be a non-empty string.")
            return
        if not isinstance(bringup_results, dict) or not bringup_results:
            self.logger.error(f"Invalid bringup_results for {design_id}: Must be a non-empty dictionary.")
            return

        # 2. Standardize bringup_results format
        standardized_results = self._standardize_bringup_results(bringup_results)
        self.logger.debug(f"Standardized bring-up results for {design_id}.")

        feedback_entry = {
            'design_id': design_id,
            'timestamp': datetime.now().isoformat(),
            'bringup_results': standardized_results, # Use standardized results
            'extracted_features': self._extract_features_from_bringup(standardized_results),
            'performance_metrics': self._extract_performance_metrics(standardized_results)
        }
        
        self.feedback_data.append(feedback_entry)
        self._store_feedback_data(feedback_entry)
        
        self.logger.info(f"Silicon bring-up feedback processed for design {design_id}")
        
        self.logger.info(f"Silicon feedback processed for design {design_id}")
    
    def process_yield_data(self, design_id: str, yield_map: Dict[str, float], 
                         field_failure_data: Optional[Dict] = None):
        """
        Process yield maps and field failure data
        
        Args:
            design_id: Unique identifier for the design
            yield_map: Map of yield across different regions/dies
            field_failure_data: Data about field failures if available
        """
        self.logger.info(f"Processing yield data for design {design_id}")
        
        feedback_entry = {
            'design_id': design_id,
            'timestamp': datetime.now().isoformat(),
            'yield_data': yield_map,
            'field_failures': field_failure_data or {},
            'yield_insights': self._analyze_yield_patterns(yield_map, field_failure_data)
        }
        
        self.feedback_data.append(feedback_entry)
        self._store_feedback_data(feedback_entry)
        
        self.logger.info(f"Yield feedback processed for design {design_id}")
    
    def process_performance_drift_data(self, design_id: str, 
                                    performance_measurements: Dict[str, List[float]]):
        """
        Process performance drift data over time and conditions
        
        Args:
            design_id: Unique identifier for the design
            performance_measurements: Performance measurements over time/conditions
        """
        self.logger.info(f"Processing performance drift data for design {design_id}")
        
        feedback_entry = {
            'design_id': design_id,
            'timestamp': datetime.now().isoformat(),
            'performance_data': performance_measurements,
            'drift_analysis': self._analyze_performance_drift(performance_measurements)
        }
        
        self.feedback_data.append(feedback_entry)
        self._store_feedback_data(feedback_entry)
        
        self.logger.info(f"Performance drift feedback processed for design {design_id}")
    
    def _extract_features_from_bringup(self, bringup_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant features from bring-up results"""
        features = {}
        
        # Extract timing results
        timing_results = bringup_results.get('timing', {})
        features['timing_closure_rate'] = timing_results.get('closure_rate', 0.0)
        features['worst_negative_slack'] = timing_results.get('wns', 0.0)
        features['total_negative_slack'] = timing_results.get('tns', 0.0)
        
        # Extract power results
        power_results = bringup_results.get('power', {})
        features['leakage_power'] = power_results.get('leakage', 0.0)
        features['dynamic_power'] = power_results.get('dynamic', 0.0)
        features['peak_power'] = power_results.get('peak', 0.0)
        
        # Extract area utilization
        area_results = bringup_results.get('area', {})
        features['utilization'] = area_results.get('utilization', 0.0)
        features['congestion'] = bringup_results.get('congestion', 0.0)
        
        # Extract DRC results
        drc_results = bringup_results.get('drc', {})
        features['drc_errors'] = drc_results.get('total_errors', 0)
        features['drc_error_types'] = drc_results.get('error_types', {})
        
        return features
    
    def _extract_performance_metrics(self, bringup_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics from bring-up results"""
        self.logger.debug("Extracting performance metrics.")
        metrics = {}
        
        # --- Timing Accuracy ---
        predicted_timing = bringup_results.get('predicted_timing', {})
        actual_timing = bringup_results.get('actual_timing', {})
        if predicted_timing and actual_timing:
            pred_wns = predicted_timing.get('wns', 0.0)
            actual_wns = actual_timing.get('wns', 0.0)
            # Ensure actual_wns is not zero to avoid division error, or handle cases where WNS might be positive
            timing_error = abs(pred_wns - actual_wns) / max(0.01, abs(actual_wns)) # Relative error
            metrics['timing_prediction_error'] = timing_error
            metrics['timing_accuracy'] = 1.0 / (1.0 + timing_error)
            self.logger.debug(f"Timing accuracy: {metrics['timing_accuracy']:.2f}")

        # --- Power Accuracy ---
        predicted_power = bringup_results.get('predicted_power', {})
        actual_power = bringup_results.get('actual_power', {})
        if predicted_power and actual_power:
            pred_total_power = predicted_power.get('total', 0.0)
            actual_total_power = actual_power.get('total', 0.0)
            power_error = abs(pred_total_power - actual_total_power) / max(0.01, actual_total_power)
            metrics['power_prediction_error'] = power_error
            metrics['power_accuracy'] = 1.0 / (1.0 + power_error)
            self.logger.debug(f"Power accuracy: {metrics['power_accuracy']:.2f}")

        # --- Congestion Accuracy ---
        predicted_congestion = bringup_results.get('predicted_congestion', 0.0)
        actual_congestion = bringup_results.get('actual_congestion', 0.0)
        if predicted_congestion is not None and actual_congestion is not None:
            congestion_error = abs(predicted_congestion - actual_congestion) / max(0.01, actual_congestion)
            metrics['congestion_prediction_error'] = congestion_error
            metrics['congestion_accuracy'] = 1.0 / (1.0 + congestion_error)
            self.logger.debug(f"Congestion accuracy: {metrics['congestion_accuracy']:.2f}")

        # --- DRC Accuracy ---
        predicted_drc = bringup_results.get('predicted_drc', {})
        actual_drc = bringup_results.get('actual_drc', {})
        if predicted_drc and actual_drc:
            pred_drc_errors = predicted_drc.get('total_violations', 0)
            actual_drc_errors = actual_drc.get('total_violations', 0)
            drc_error = abs(pred_drc_errors - actual_drc_errors) / max(1, actual_drc_errors) # Avoid div by zero, use count
            metrics['drc_prediction_error'] = drc_error
            metrics['drc_accuracy'] = 1.0 / (1.0 + drc_error)
            self.logger.debug(f"DRC accuracy: {metrics['drc_accuracy']:.2f}")

        # --- Thermal Accuracy ---
        predicted_thermal = bringup_results.get('predicted_thermal', {})
        actual_thermal = bringup_results.get('actual_thermal', {})
        if predicted_thermal and actual_thermal:
            pred_max_temp = predicted_thermal.get('max_temp_c', 0.0)
            actual_max_temp = actual_thermal.get('max_temp_c', 0.0)
            thermal_error = abs(pred_max_temp - actual_max_temp) / max(0.01, actual_max_temp)
            metrics['thermal_prediction_error'] = thermal_error
            metrics['thermal_accuracy'] = 1.0 / (1.0 + thermal_error)
            self.logger.debug(f"Thermal accuracy: {metrics['thermal_accuracy']:.2f}")

        # --- Yield Accuracy ---
        predicted_yield = bringup_results.get('predicted_yield', 0.0)
        actual_yield = bringup_results.get('actual_yield', 0.0)
        if predicted_yield is not None and actual_yield is not None:
            yield_error = abs(predicted_yield - actual_yield) / max(0.01, actual_yield)
            metrics['yield_prediction_error'] = yield_error
            metrics['yield_accuracy'] = 1.0 / (1.0 + yield_error)
            self.logger.debug(f"Yield accuracy: {metrics['yield_accuracy']:.2f}")

        return metrics    
    def _analyze_yield_patterns(self, yield_map: Dict[str, float], 
                              field_failures: Optional[Dict]) -> Dict[str, Any]:
        """Analyze yield patterns to extract insights"""
        self.logger.debug("Analyzing yield patterns.")
        insights = {}
        
        # Calculate overall yield
        yields = list(yield_map.values())
        insights['average_yield'] = np.mean(yields) if yields else 0.0
        insights['yield_std'] = np.std(yields) if len(yields) > 1 else 0.0
        
        # Identify low-yield regions: significantly below average yield
        low_yield_threshold = insights['average_yield'] - (insights['yield_std'] * 1.5) # 1.5 standard deviations below average
        insights['low_yield_regions'] = [region for region, yield_val in yield_map.items() 
                                       if yield_val < low_yield_threshold]
        
        if insights['low_yield_regions']:
            self.logger.debug(f"Identified {len(insights['low_yield_regions'])} low-yield regions.")
        
        # Analyze field failures if available
        if field_failures:
            insights['failure_modes'] = field_failures.get('modes', [])
            insights['failure_locations'] = field_failures.get('locations', [])
            insights['failure_correlation'] = self._correlate_failures_with_design_attributes(
                yield_map, field_failures
            )
        
        return insights    
    def _analyze_performance_drift(self, performance_measurements: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze performance drift patterns"""
        self.logger.debug("Analyzing performance drift patterns.")
        drift_analysis = {}
        
        if not isinstance(performance_measurements, dict) or not performance_measurements:
            self.logger.warning("No performance measurements provided for drift analysis.")
            return {}

        for metric, values in performance_measurements.items():
            if not isinstance(values, list) or len(values) < 2:
                self.logger.debug(f"Skipping drift analysis for '{metric}' due to insufficient data.")
                continue

            # Calculate drift statistics
            metric_mean = np.mean(values)
            metric_std = np.std(values)
            metric_drift_rate = (values[-1] - values[0]) / len(values) if len(values) > 0 else 0
            metric_variability = np.var(values)

            drift_analysis[f'{metric}_mean'] = metric_mean
            drift_analysis[f'{metric}_std'] = metric_std
            drift_analysis[f'{metric}_drift_rate'] = metric_drift_rate
            drift_analysis[f'{metric}_variability'] = metric_variability

            self.logger.debug(f"Drift analysis for '{metric}': mean={metric_mean:.2f}, std={metric_std:.2f}, drift_rate={metric_drift_rate:.2f}.")
        
        return drift_analysis    
    def _correlate_failures_with_design_attributes(self, yield_map: Dict[str, float], 
                                                 field_failures: Dict) -> Dict[str, Any]:
        """Correlate failures with design attributes to find patterns"""
        correlation_analysis = {}
        
        # This would normally correlate failure locations with design features
        # like density, routing congestion, power consumption, etc.

        # Simplified spatial correlation: Check if field failures concentrate in low-yield regions
        low_yield_regions = [region for region, yield_val in yield_map.items() if yield_val < 0.9] # Example threshold
        
        failure_locations = field_failures.get('locations', [])
        
        correlated_regions = []
        for location in failure_locations:
            # Assuming location can be mapped to a region in yield_map
            if location in low_yield_regions:
                correlated_regions.append(location)
        
        if correlated_regions:
            correlation_analysis['spatial_correlation'] = f"Failures correlate with low-yield regions: {', '.join(set(correlated_regions))}"
        else:
            correlation_analysis['spatial_correlation'] = "No direct spatial correlation found with low-yield regions."

        # Feature correlation remains simplified as we don't have detailed features here
        correlation_analysis['feature_correlation'] = 'pending_detailed_feature_analysis'
        
        return correlation_analysis
        
        return correlation_analysis
    
    def _store_feedback_data(self, feedback_entry: Dict[str, Any]):
        """Store feedback data to persistent storage"""
        filename = f"{feedback_entry['design_id']}_{feedback_entry['timestamp']}.pkl"
        filepath = os.path.join(self.model_storage_path, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(feedback_entry, f)

    def _standardize_bringup_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardizes raw silicon bring-up results into a consistent format.
        This handles different possible input structures from various sources.
        """
        self.logger.debug("Standardizing raw bring-up results.")
        standardized = {
            'timing': {},
            'power': {},
            'area': {},
            'congestion': 0.0,
            'drc': {},
            'thermal': {},
            'yield': 0.0,
            'predicted_timing': {},
            'actual_timing': {},
            'predicted_power': {},
            'actual_power': {},
            'predicted_congestion': 0.0,
            'actual_congestion': 0.0,
            'predicted_drc': {},
            'actual_drc': {}
        }

        # --- Timing Data ---
        # Look for WNS, TNS, Fmax from various possible keys
        timing_keys = ['timing', 'sta_report', 'prime_time_report']
        for key in timing_keys:
            if key in raw_results and isinstance(raw_results[key], dict):
                standardized['timing']['wns'] = raw_results[key].get('wns', raw_results[key].get('worst_negative_slack', None))
                standardized['timing']['tns'] = raw_results[key].get('tns', raw_results[key].get('total_negative_slack', None))
                standardized['timing']['fmax'] = raw_results[key].get('fmax', raw_results[key].get('max_frequency_ghz', None))
                break
        
        # --- Power Data ---
        power_keys = ['power', 'pt_power_report', 'redhawk_report']
        for key in power_keys:
            if key in raw_results and isinstance(raw_results[key], dict):
                standardized['power']['total'] = raw_results[key].get('total_power_mw', raw_results[key].get('total', None))
                standardized['power']['leakage'] = raw_results[key].get('leakage_power_mw', raw_results[key].get('leakage', None))
                standardized['power']['dynamic'] = raw_results[key].get('dynamic_power_mw', raw_results[key].get('dynamic', None))
                break

        # --- Area Data ---
        area_keys = ['area', 'floorplan_report']
        for key in area_keys:
            if key in raw_results and isinstance(raw_results[key], dict):
                standardized['area']['utilization'] = raw_results[key].get('utilization', None)
                standardized['area']['total_area_um2'] = raw_results[key].get('total_area_um2', None)
                break
        
        # --- Congestion Data ---
        congestion_keys = ['congestion', 'route_congestion_report']
        for key in congestion_keys:
            if key in raw_results:
                standardized['congestion'] = raw_results[key].get('average_congestion', raw_results[key])
                break

        # --- DRC Data ---
        drc_keys = ['drc', 'calibre_drc_report', 'ic_validator_report']
        for key in drc_keys:
            if key in raw_results and isinstance(raw_results[key], dict):
                standardized['drc']['total_errors'] = raw_results[key].get('total_violations', raw_results[key].get('total_errors', None))
                standardized['drc']['error_types'] = raw_results[key].get('violation_types', raw_results[key].get('error_types', None))
                break
        
        # --- Thermal Data ---
        thermal_keys = ['thermal', 'redhawk_thermal_report']
        for key in thermal_keys:
            if key in raw_results and isinstance(raw_results[key], dict):
                standardized['thermal']['max_temp_c'] = raw_results[key].get('max_temperature_celsius', None)
                standardized['thermal']['avg_temp_c'] = raw_results[key].get('average_temperature_celsius', None)
                break

        # --- Yield Data ---
        yield_keys = ['yield', 'yield_report']
        for key in yield_keys:
            if key in raw_results:
                standardized['yield'] = raw_results[key].get('overall_yield', raw_results[key])
                break

        # --- Predicted vs Actual (for model accuracy tracking) ---
        # Assuming predicted values are prefixed with 'predicted_' and actual with 'actual_'
        for key, value in raw_results.items():
            if key.startswith('predicted_'):
                standardized[key] = value
            elif key.startswith('actual_'):
                standardized[key] = value

        self.logger.debug("Raw results standardized.")
        return standardized    
    def load_historical_feedback(self) -> List[Dict[str, Any]]:
        """Load historical feedback data from storage"""
        feedback_entries = []
        
        for filename in os.listdir(self.model_storage_path):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.model_storage_path, filename)
                with open(filepath, 'rb') as f:
                    feedback_entries.append(pickle.load(f))
        
        return feedback_entries


class ModelUpdater:
    """
    Model Updater - updates internal models based on silicon feedback
    """
    
    def __init__(self, feedback_processor: SiliconFeedbackProcessor):
        self.feedback_processor = feedback_processor
        self.logger = get_logger(f"{__name__}.model_updater")
    
    def update_congestion_predictor(self, congestion_predictor: CongestionPredictor):
        """Update congestion predictor model with silicon feedback"""
        self.logger.info("Updating congestion predictor with silicon feedback")
        
        # Load historical feedback data
        feedback_data = self.feedback_processor.load_historical_feedback()
        
        # Extract congestion-related feedback
        congestion_training_data = []
        for entry in feedback_data:
            bringup_results = entry.get('bringup_results', {})
            predicted_congestion = bringup_results.get('predicted_congestion', {})
            actual_congestion = bringup_results.get('actual_congestion', {})
            
            if predicted_congestion and actual_congestion:
                # Create training sample comparing predictions to actuals
                training_sample = {
                    'predicted': predicted_congestion,
                    'actual': actual_congestion,
                    'features': entry.get('extracted_features', {}),
                    'error': self._calculate_congestion_error(predicted_congestion, actual_congestion)
                }
                congestion_training_data.append(training_sample)
        
        # Update the model with new data
        if congestion_training_data:
            congestion_predictor.train(congestion_training_data)
            self.logger.info(f"Updated congestion predictor with {len(congestion_training_data)} samples")
            
            # After training, we assume the model has improved, so slightly increase its general weight
            # This is a simplified way to reflect that the model is learning.
            congestion_predictor.model_weights = min(congestion_predictor.model_weights + 0.01, 1.0)
            self.logger.info(f"Congestion predictor model_weights adjusted to: {congestion_predictor.model_weights:.3f}")
        else:
            self.logger.info("No congestion training data available")
    
    def update_timing_analyzer(self, timing_analyzer: TimingAnalyzer):
        """Update timing analyzer model with silicon feedback"""
        self.logger.info("Updating timing analyzer with silicon feedback")
        
        # Load historical feedback data
        feedback_data = self.feedback_processor.load_historical_feedback()
        
        # Extract timing-related feedback
        timing_training_data = []
        for entry in feedback_data:
            bringup_results = entry.get('bringup_results', {})
            predicted_timing = bringup_results.get('predicted_timing', {})
            actual_timing = bringup_results.get('actual_timing', {})
            
            if predicted_timing and actual_timing:
                # Create training sample comparing predictions to actuals
                training_sample = {
                    'predicted': predicted_timing,
                    'actual': actual_timing,
                    'features': entry.get('extracted_features', {}),
                    'error': self._calculate_timing_error(predicted_timing, actual_timing)
                }
                timing_training_data.append(training_sample)
        
        # In a real implementation, this would update the timing analyzer model
        if timing_training_data:
            timing_analyzer.train(timing_training_data)
            self.logger.info(f"Updated timing analyzer with {len(timing_training_data)} samples")
            
            # After training, assume the model has improved and slightly increase its general weight
            timing_analyzer.model_weights = min(timing_analyzer.model_weights + 0.01, 1.0)
            self.logger.info(f"Timing analyzer model_weights adjusted to: {timing_analyzer.model_weights:.3f}")
        else:
            self.logger.info("No timing training data available")
    
    def update_drc_predictor(self, drc_predictor: DRCPredictor):
        """Update DRC predictor model with silicon feedback"""
        self.logger.info("Updating DRC predictor with silicon feedback")
        
        # Load historical feedback data
        feedback_data = self.feedback_processor.load_historical_feedback()
        
        # Extract DRC-related feedback
        drc_training_data = []
        for entry in feedback_data:
            bringup_results = entry.get('bringup_results', {})
            predicted_drc = bringup_results.get('predicted_drc', {})
            actual_drc = bringup_results.get('actual_drc', {})
            
            if predicted_drc and actual_drc:
                # Create training sample comparing predictions to actuals
                training_sample = {
                    'predicted': predicted_drc,
                    'actual': actual_drc,
                    'features': entry.get('extracted_features', {}),
                    'error': self._calculate_drc_error(predicted_drc, actual_drc)
                }
                drc_training_data.append(training_sample)
        
        # Update the DRC predictor model
        # In a real implementation, this would update the internal model parameters
        if drc_training_data:
            self.logger.info(f"Collected {len(drc_training_data)} DRC training samples")
            # Update internal model parameters based on feedback
            self._update_drc_model_parameters(drc_predictor, drc_training_data)
        else:
            self.logger.info("No DRC training data available")
    
    def update_agent_authority(self, agents: List[BaseAgent]):
        """Update agent authority levels based on prediction accuracy"""
        self.logger.info("Updating agent authority based on prediction accuracy")
        
        feedback_data = self.feedback_processor.load_historical_feedback()
        
        # Calculate accuracy for each agent type
        agent_accuracy = {}
        
        for entry in feedback_data:
            bringup_results = entry.get('bringup_results', {})
            
            # Check timing predictions (likely from placement/clock agents)
            if 'predicted_timing' in bringup_results and 'actual_timing' in bringup_results:
                timing_error = self._calculate_timing_error(
                    bringup_results['predicted_timing'], 
                    bringup_results['actual_timing']
                )
                # Associate with placement and clock agents
                agent_accuracy['placement'] = agent_accuracy.get('placement', []) + [1.0 - timing_error]
                agent_accuracy['clock'] = agent_accuracy.get('clock', []) + [1.0 - timing_error]
            
            # Check power predictions (likely from power agent)
            if 'predicted_power' in bringup_results and 'actual_power' in bringup_results:
                power_error = self._calculate_power_error(
                    bringup_results['predicted_power'], 
                    bringup_results['actual_power']
                )
                agent_accuracy['power'] = agent_accuracy.get('power', []) + [1.0 - power_error]
            
            # Check congestion predictions (likely from floorplan/placement agents)
            if 'predicted_congestion' in bringup_results and 'actual_congestion' in bringup_results:
                congestion_error = self._calculate_congestion_error(
                    bringup_results['predicted_congestion'], 
                    bringup_results['actual_congestion']
                )
                agent_accuracy['floorplan'] = agent_accuracy.get('floorplan', []) + [1.0 - congestion_error]
                agent_accuracy['placement'] = agent_accuracy.get('placement', []) + [1.0 - congestion_error]
        
        # Update agent authority based on accuracy
        for agent in agents:
            agent_type = agent.agent_type.value
            if agent_type in agent_accuracy and agent_accuracy[agent_type]:
                avg_accuracy = np.mean(agent_accuracy[agent_type])
                # Update agent's authority based on accuracy
                if avg_accuracy > 0.8:
                    agent.update_authority(True, penalty_factor=0.05)  # Reward good performance
                elif avg_accuracy < 0.6:
                    agent.update_authority(False, penalty_factor=0.1)  # Penalize poor performance
                else:
                    # Neutral performance, slight adjustment
                    if avg_accuracy > agent.get_recent_performance():
                        agent.update_authority(True, penalty_factor=0.02)
                    else:
                        agent.update_authority(False, penalty_factor=0.02)
        
        self.logger.info("Agent authority updated based on silicon feedback")
    
    def _calculate_congestion_error(self, predicted: Dict, actual: Dict) -> float:
        """Calculate error between predicted and actual congestion"""
        # Simple error calculation - in reality would be more sophisticated
        pred_val = predicted.get('average_congestion', 0.0)
        actual_val = actual.get('average_congestion', 0.0)
        return abs(pred_val - actual_val) / (actual_val + 0.001)  # Avoid division by zero
    
    def _calculate_timing_error(self, predicted: Dict, actual: Dict) -> float:
        """Calculate error between predicted and actual timing"""
        pred_wns = predicted.get('wns', 0.0)
        actual_wns = actual.get('wns', 0.0)
        return abs(pred_wns - actual_wns) / (abs(actual_wns) + 0.001)  # Normalize error
    
    def _calculate_power_error(self, predicted: Dict, actual: Dict) -> float:
        """Calculate error between predicted and actual power"""
        pred_total = predicted.get('total', 1.0)
        actual_total = actual.get('total', 1.0)
        return abs(pred_total - actual_total) / actual_total
    
    def _calculate_drc_error(self, predicted: Dict, actual: Dict) -> float:
        """Calculate error between predicted and actual DRC violations"""
        pred_count = predicted.get('total_violations', 0)
        actual_count = actual.get('total_violations', 0)
        return abs(pred_count - actual_count) / (actual_count + 1)  # Add 1 to avoid division by zero
    
    def _update_drc_model_parameters(self, drc_predictor: DRCPredictor, training_data: List[Dict]):
        """Update DRC predictor model parameters based on training data"""
        # This would update the internal model parameters based on feedback
        # For now, we'll just log what would be updated
        self.logger.info(f"Updating DRC model parameters with {len(training_data)} samples")
        
        # Calculate prediction accuracy and adjust model weights
        if training_data:
            accuracies = []
            for sample in training_data:
                error = sample.get('error', 1.0)
                # Ensure error is non-negative and not too small to avoid division by zero or large accuracies
                error = max(0.01, error) 
                accuracy = 1.0 / (1.0 + error)
                accuracies.append(accuracy)
            
            avg_accuracy = np.mean(accuracies) if accuracies else 0.0
            self.logger.info(f"DRC prediction average accuracy: {avg_accuracy:.3f}")

            # Tangibly adjust drc_predictor's internal parameters based on accuracy
            # For demonstration, assume drc_predictor has a 'model_weights' attribute (float)
            # In a real scenario, this would involve retraining or fine-tuning the actual ML model.
            if hasattr(drc_predictor, 'model_weights'):
                # Increase weight if accuracy is good, decrease if poor
                if avg_accuracy > 0.8:  # Good accuracy
                    drc_predictor.model_weights = min(drc_predictor.model_weights + 0.05, 1.0)
                elif avg_accuracy < 0.5: # Poor accuracy
                    drc_predictor.model_weights = max(drc_predictor.model_weights - 0.05, 0.1)
                self.logger.info(f"DRC predictor model_weights adjusted to: {drc_predictor.model_weights:.3f}")
        else:
            self.logger.info("No DRC training data available for model update.")



class LearningLoopController:
    """
    Learning Loop Controller - orchestrates the entire learning process
    """
    
    def __init__(self):
        self.feedback_processor = SiliconFeedbackProcessor()
        self.model_updater = ModelUpdater(self.feedback_processor)
        self.logger = get_logger(f"{__name__}.learning_loop_controller")
    
    def process_new_silicon_data(self, design_data: Dict[str, Any]):
        """
        Process new silicon data and update models
        
        Args:
            design_data: Complete design data including silicon results
        """
        self.logger.info("Processing new silicon data through learning loop")
        
        design_id = design_data.get('design_id', 'unknown')
        
        # Process different types of silicon data
        bringup_results = design_data.get('bringup_results', {})
        if bringup_results:
            self.feedback_processor.process_silicon_bringup_data(design_id, bringup_results)
        
        yield_data = design_data.get('yield_data', {})
        if yield_data:
            field_failures = design_data.get('field_failures', {})
            self.feedback_processor.process_yield_data(design_id, yield_data, field_failures)
        
        performance_data = design_data.get('performance_data', {})
        if performance_data:
            self.feedback_processor.process_performance_drift_data(design_id, performance_data)
        
        self.logger.info(f"Silicon data for design {design_id} added to learning loop")
    
    def update_all_models(self, congestion_predictor: CongestionPredictor,
                         timing_analyzer: TimingAnalyzer,
                         drc_predictor: DRCPredictor,
                         agents: List[BaseAgent],
                         incremental_update: bool = True): # Added incremental_update parameter
        """
        Update all models with accumulated feedback
        
        Args:
            congestion_predictor: Congestion prediction model to update
            timing_analyzer: Timing analysis model to update
            drc_predictor: DRC prediction model to update
            agents: List of agents to update authority for
            incremental_update: If True, perform incremental updates; otherwise, simulate full retraining.
        """
        self.logger.info("Updating all models with accumulated feedback")
        
        if incremental_update:
            self.logger.info("Performing incremental model updates.")
            # Update individual models incrementally
            self.model_updater.update_congestion_predictor(congestion_predictor)
            self.model_updater.update_timing_analyzer(timing_analyzer)
            self.model_updater.update_drc_predictor(drc_predictor)
        else:
            self.logger.info("Performing full model retraining (simulated).")
            # Placeholder for a full retraining pipeline
            # In a real system, this would involve loading a larger dataset,
            # potentially re-initializing models, and running a full training cycle.
            self.logger.warning("Full model retraining logic not yet implemented. Simulating retraining for now.")
        
        # Update agent authority (this can happen regardless of incremental/full model update)
        self.model_updater.update_agent_authority(agents)
        
        self.logger.info("All models updated successfully")
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning process"""
        feedback_data = self.feedback_processor.load_historical_feedback()
        
        insights = {
            'total_designs_learned': len(feedback_data),
            'data_coverage': self._analyze_data_coverage(feedback_data),
            'model_improvement_trends': self._analyze_improvement_trends(feedback_data)
        }
        
        return insights
    
    def _analyze_data_coverage(self, feedback_data: List[Dict]) -> Dict[str, Any]:
        """Analyze coverage of feedback data"""
        coverage = {
            'timing_data': 0,
            'power_data': 0,
            'congestion_data': 0,
            'drc_data': 0,
            'yield_data': 0
        }
        
        for entry in feedback_data:
            bringup_results = entry.get('bringup_results', {})
            if bringup_results.get('actual_timing'):
                coverage['timing_data'] += 1
            if bringup_results.get('actual_power'):
                coverage['power_data'] += 1
            if bringup_results.get('actual_congestion'):
                coverage['congestion_data'] += 1
            if bringup_results.get('actual_drc'):
                coverage['drc_data'] += 1
            
            if entry.get('yield_data'):
                coverage['yield_data'] += 1
        
        return coverage
    
    def _analyze_improvement_trends(self, feedback_data: List[Dict]) -> Dict[str, List[float]]:
        """Analyze trends in model improvements"""
        timing_accuracy_trend = []
        power_accuracy_trend = []
        congestion_accuracy_trend = []
        drc_accuracy_trend = [] # New: tracking DRC accuracy trend

        # Sort feedback data by timestamp to ensure correct trend analysis
        feedback_data_sorted = sorted(feedback_data, key=lambda x: x.get('timestamp', ''))

        for entry in feedback_data_sorted:
            performance_metrics = entry.get('performance_metrics', {})
            
            if 'timing_accuracy' in performance_metrics:
                timing_accuracy_trend.append(performance_metrics['timing_accuracy'])
            if 'power_accuracy' in performance_metrics:
                power_accuracy_trend.append(performance_metrics['power_accuracy'])
            if 'congestion_accuracy' in performance_metrics:
                congestion_accuracy_trend.append(performance_metrics['congestion_accuracy'])
            # Assuming DRC accuracy is also calculated in _extract_performance_metrics
            if 'drc_accuracy' in performance_metrics:
                drc_accuracy_trend.append(performance_metrics['drc_accuracy'])

        return {
            'timing_accuracy_trend': timing_accuracy_trend,
            'power_accuracy_trend': power_accuracy_trend,
            'congestion_accuracy_trend': congestion_accuracy_trend,
            'drc_accuracy_trend': drc_accuracy_trend
        }


# Example usage function
def example_learning_loop():
    """Example of how to use the learning loop"""
    logger = get_logger(__name__)
    
    # Initialize the learning loop
    learning_controller = LearningLoopController()
    logger.info("Learning loop initialized")
    
    # Simulate some silicon data coming in
    sample_silicon_data = {
        'design_id': 'chip_design_001',
        'bringup_results': {
            'predicted_timing': {'wns': -0.2, 'tns': -1.5},
            'actual_timing': {'wns': 0.1, 'tns': -0.8},
            'predicted_power': {'total': 2.5, 'leakage': 0.3},
            'actual_power': {'total': 2.7, 'leakage': 0.35},
            'predicted_congestion': {'average': 0.65},
            'actual_congestion': {'average': 0.72},
            'predicted_drc': {'total_violations': 50},
            'actual_drc': {'total_violations': 75}
        },
        'yield_data': {
            'die_001': 0.95,
            'die_002': 0.89,
            'die_003': 0.92
        },
        'performance_data': {
            'frequency': [3.2, 3.18, 3.15, 3.12, 3.10],  # Over time/temperature
            'power': [2.7, 2.75, 2.8, 2.85, 2.9]
        }
    }
    
    # Process the silicon data
    learning_controller.process_new_silicon_data(sample_silicon_data)
    logger.info("Silicon data processed")
    
    # In a real system, you would then update the models
    # For this example, we'll just get insights
    insights = learning_controller.get_learning_insights()
    logger.info(f"Learning insights: {insights}")


if __name__ == "__main__":
    example_learning_loop()