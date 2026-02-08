"""
Comprehensive Learning Loop with Real Feedback Integration

This module implements the complete learning system that incorporates real post-silicon
data to improve the AI models and agent decision-making.
"""

import os
import pickle
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import threading
import time
from collections import defaultdict, deque
import hashlib

from utils.logger import get_logger
from agents.base_agent import BaseAgent
from models.congestion_predictor import CongestionPredictor
from models.timing_analyzer import TimingAnalyzer
from models.drc_predictor import DRCPredictor
from cognitive.advanced_cognitive_system import (
    DesignIntentInterpreter, SiliconKnowledgeModel, ReasoningEngine
)


class SiliconDataCollector:
    """
    Collects and stores post-silicon data for learning
    """
    
    def __init__(self, db_path: str = "./silicon_data.db"):
        self.logger = get_logger(__name__)
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database for storing silicon data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for different types of data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS design_metadata (
                design_id TEXT PRIMARY KEY,
                project_name TEXT,
                tapeout_date TEXT,
                process_node TEXT,
                die_size REAL,
                core_voltage REAL,
                max_frequency REAL,
                created_at TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS timing_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                design_id TEXT,
                measurement_type TEXT,
                path_type TEXT,
                slack_ps REAL,
                delay_ps REAL,
                frequency_mhz REAL,
                temperature REAL,
                voltage REAL,
                measured_at TEXT,
                FOREIGN KEY (design_id) REFERENCES design_metadata (design_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS power_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                design_id TEXT,
                measurement_type TEXT,
                rail_name TEXT,
                static_power_mw REAL,
                dynamic_power_mw REAL,
                peak_power_mw REAL,
                temperature REAL,
                voltage REAL,
                measured_at TEXT,
                FOREIGN KEY (design_id) REFERENCES design_metadata (design_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS area_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                design_id TEXT,
                block_name TEXT,
                area_um2 REAL,
                utilization REAL,
                cell_count INTEGER,
                macro_count INTEGER,
                measured_at TEXT,
                FOREIGN KEY (design_id) REFERENCES design_metadata (design_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drc_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                design_id TEXT,
                rule_name TEXT,
                violation_count INTEGER,
                waived_count INTEGER,
                severity TEXT,
                measured_at TEXT,
                FOREIGN KEY (design_id) REFERENCES design_metadata (design_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS yield_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                design_id TEXT,
                wafer_id TEXT,
                die_x INTEGER,
                die_y INTEGER,
                bin_result TEXT,
                parametric_pass BOOLEAN,
                functional_pass BOOLEAN,
                measured_at TEXT,
                FOREIGN KEY (design_id) REFERENCES design_metadata (design_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                design_id TEXT,
                test_name TEXT,
                parameter_name TEXT,
                measured_value REAL,
                specification_min REAL,
                specification_max REAL,
                unit TEXT,
                measured_at TEXT,
                FOREIGN KEY (design_id) REFERENCES design_metadata (design_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                design_id TEXT,
                model_type TEXT,
                prediction_task TEXT,
                predicted_value REAL,
                actual_value REAL,
                confidence REAL,
                features_json TEXT,
                prediction_time TEXT,
                FOREIGN KEY (design_id) REFERENCES design_metadata (design_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_design_metadata(self, design_id: str, metadata: Dict[str, Any]):
        """Store design metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO design_metadata 
            (design_id, project_name, tapeout_date, process_node, die_size, core_voltage, max_frequency, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            design_id,
            metadata.get('project_name'),
            metadata.get('tapeout_date'),
            metadata.get('process_node'),
            metadata.get('die_size'),
            metadata.get('core_voltage'),
            metadata.get('max_frequency'),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def store_timing_data(self, design_id: str, timing_measurements: List[Dict[str, Any]]):
        """Store timing measurements"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for measurement in timing_measurements:
            cursor.execute('''
                INSERT INTO timing_data 
                (design_id, measurement_type, path_type, slack_ps, delay_ps, frequency_mhz, temperature, voltage, measured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                design_id,
                measurement.get('measurement_type'),
                measurement.get('path_type'),
                measurement.get('slack_ps'),
                measurement.get('delay_ps'),
                measurement.get('frequency_mhz'),
                measurement.get('temperature'),
                measurement.get('voltage'),
                measurement.get('measured_at', datetime.now().isoformat())
            ))
        
        conn.commit()
        conn.close()
    
    def store_power_data(self, design_id: str, power_measurements: List[Dict[str, Any]]):
        """Store power measurements"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for measurement in power_measurements:
            cursor.execute('''
                INSERT INTO power_data 
                (design_id, measurement_type, rail_name, static_power_mw, dynamic_power_mw, peak_power_mw, temperature, voltage, measured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                design_id,
                measurement.get('measurement_type'),
                measurement.get('rail_name'),
                measurement.get('static_power_mw'),
                measurement.get('dynamic_power_mw'),
                measurement.get('peak_power_mw'),
                measurement.get('temperature'),
                measurement.get('voltage'),
                measurement.get('measured_at', datetime.now().isoformat())
            ))
        
        conn.commit()
        conn.close()
    
    def store_area_data(self, design_id: str, area_measurements: List[Dict[str, Any]]):
        """Store area measurements"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for measurement in area_measurements:
            cursor.execute('''
                INSERT INTO area_data 
                (design_id, block_name, area_um2, utilization, cell_count, macro_count, measured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                design_id,
                measurement.get('block_name'),
                measurement.get('area_um2'),
                measurement.get('utilization'),
                measurement.get('cell_count'),
                measurement.get('macro_count'),
                measurement.get('measured_at', datetime.now().isoformat())
            ))
        
        conn.commit()
        conn.close()
    
    def store_drc_data(self, design_id: str, drc_results: List[Dict[str, Any]]):
        """Store DRC results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for result in drc_results:
            cursor.execute('''
                INSERT INTO drc_data 
                (design_id, rule_name, violation_count, waived_count, severity, measured_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                design_id,
                result.get('rule_name'),
                result.get('violation_count'),
                result.get('waived_count'),
                result.get('severity'),
                result.get('measured_at', datetime.now().isoformat())
            ))
        
        conn.commit()
        conn.close()
    
    def store_yield_data(self, design_id: str, yield_results: List[Dict[str, Any]]):
        """Store yield results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for result in yield_results:
            cursor.execute('''
                INSERT INTO yield_data 
                (design_id, wafer_id, die_x, die_y, bin_result, parametric_pass, functional_pass, measured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                design_id,
                result.get('wafer_id'),
                result.get('die_x'),
                result.get('die_y'),
                result.get('bin_result'),
                result.get('parametric_pass'),
                result.get('functional_pass'),
                result.get('measured_at', datetime.now().isoformat())
            ))
        
        conn.commit()
        conn.close()
    
    def store_performance_data(self, design_id: str, performance_results: List[Dict[str, Any]]):
        """Store performance test results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for result in performance_results:
            cursor.execute('''
                INSERT INTO performance_data 
                (design_id, test_name, parameter_name, measured_value, specification_min, specification_max, unit, measured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                design_id,
                result.get('test_name'),
                result.get('parameter_name'),
                result.get('measured_value'),
                result.get('specification_min'),
                result.get('specification_max'),
                result.get('unit'),
                result.get('measured_at', datetime.now().isoformat())
            ))
        
        conn.commit()
        conn.close()
    
    def store_prediction_data(self, design_id: str, predictions: List[Dict[str, Any]]):
        """Store prediction vs actual data for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for prediction in predictions:
            cursor.execute('''
                INSERT INTO prediction_data 
                (design_id, model_type, prediction_task, predicted_value, actual_value, confidence, features_json, prediction_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                design_id,
                prediction.get('model_type'),
                prediction.get('prediction_task'),
                prediction.get('predicted_value'),
                prediction.get('actual_value'),
                prediction.get('confidence'),
                json.dumps(prediction.get('features', {})),
                prediction.get('prediction_time', datetime.now().isoformat())
            ))
        
        conn.commit()
        conn.close()
    
    def get_historical_data(self, design_ids: List[str], data_type: str) -> List[Dict[str, Any]]:
        """Retrieve historical data for specific designs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if data_type == 'timing':
            cursor.execute('''
                SELECT * FROM timing_data WHERE design_id IN ({})
                ORDER BY measured_at DESC
            '''.format(','.join(['?' for _ in design_ids])), design_ids)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]
        
        elif data_type == 'power':
            cursor.execute('''
                SELECT * FROM power_data WHERE design_id IN ({})
                ORDER BY measured_at DESC
            '''.format(','.join(['?' for _ in design_ids])), design_ids)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]
        
        elif data_type == 'area':
            cursor.execute('''
                SELECT * FROM area_data WHERE design_id IN ({})
                ORDER BY measured_at DESC
            '''.format(','.join(['?' for _ in design_ids])), design_ids)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]
        
        elif data_type == 'drc':
            cursor.execute('''
                SELECT * FROM drc_data WHERE design_id IN ({})
                ORDER BY measured_at DESC
            '''.format(','.join(['?' for _ in design_ids])), design_ids)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]
        
        elif data_type == 'yield':
            cursor.execute('''
                SELECT * FROM yield_data WHERE design_id IN ({})
                ORDER BY measured_at DESC
            '''.format(','.join(['?' for _ in design_ids])), design_ids)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]
        
        elif data_type == 'performance':
            cursor.execute('''
                SELECT * FROM performance_data WHERE design_id IN ({})
                ORDER BY measured_at DESC
            '''.format(','.join(['?' for _ in design_ids])), design_ids)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]
        
        elif data_type == 'predictions':
            cursor.execute('''
                SELECT * FROM prediction_data WHERE design_id IN ({})
                ORDER BY prediction_time DESC
            '''.format(','.join(['?' for _ in design_ids])), design_ids)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in rows]
        
        else:
            results = []
        
        conn.close()
        return results


class ModelUpdater:
    """
    Updates internal models based on silicon feedback
    """
    
    def __init__(self, data_collector: SiliconDataCollector):
        self.logger = get_logger(__name__)
        self.data_collector = data_collector
        self.model_performance_trackers = defaultdict(deque)
        self.max_tracking_samples = 1000  # Keep last 1000 samples for performance tracking
        
        # Initialize online learning models
        self.congestion_online_model = SGDRegressor(learning_rate='adaptive', eta0=0.01)
        self.timing_online_model = SGDRegressor(learning_rate='adaptive', eta0=0.01)
        self.drc_online_model = SGDClassifier(learning_rate='adaptive', eta0=0.01)
        
        self.feature_scaler = StandardScaler()
        self.is_initialized = False
    
    def initialize_models(self, initial_training_data: Dict[str, Any]):
        """Initialize models with initial training data"""
        self.logger.info("Initializing models with initial training data")
        
        # Initialize congestion model
        if 'congestion' in initial_training_data:
            X_cong = np.array(initial_training_data['congestion']['features'])
            y_cong = np.array(initial_training_data['congestion']['targets'])
            X_cong_scaled = self.feature_scaler.fit_transform(X_cong)
            self.congestion_online_model.fit(X_cong_scaled, y_cong)
        
        # Initialize timing model
        if 'timing' in initial_training_data:
            X_tim = np.array(initial_training_data['timing']['features'])
            y_tim = np.array(initial_training_data['timing']['targets'])
            X_tim_scaled = self.feature_scaler.fit_transform(X_tim)
            self.timing_online_model.fit(X_tim_scaled, y_tim)
        
        # Initialize DRC model
        if 'drc' in initial_training_data:
            X_drc = np.array(initial_training_data['drc']['features'])
            y_drc = np.array(initial_training_data['drc']['targets'])
            X_drc_scaled = self.feature_scaler.fit_transform(X_drc)
            self.drc_online_model.fit(X_drc_scaled, y_drc)
        
        self.is_initialized = True
        self.logger.info("Models initialized successfully")
    
    def update_congestion_predictor(self, congestion_predictor: CongestionPredictor):
        """Update congestion predictor with silicon feedback"""
        self.logger.info("Updating congestion predictor with silicon feedback")
        
        # Get historical congestion data
        prediction_data = self.data_collector.get_historical_data([], 'predictions')
        congestion_predictions = [p for p in prediction_data if p['model_type'] == 'congestion']
        
        if not congestion_predictions:
            self.logger.info("No congestion prediction data available for update")
            return
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for pred in congestion_predictions:
            features = json.loads(pred['features_json'])
            # Convert features to numerical format
            feature_vector = self._convert_features_to_vector(features)
            X_train.append(feature_vector)
            y_train.append(pred['actual_value'])
        
        if not X_train:
            self.logger.info("No valid feature vectors found for congestion model update")
            return
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale features
        X_train_scaled = self.feature_scaler.transform(X_train) if self.is_initialized else self.feature_scaler.fit_transform(X_train)
        
        # Update the online model
        if self.is_initialized:
            self.congestion_online_model.partial_fit(X_train_scaled, y_train)
        else:
            self.congestion_online_model.fit(X_train_scaled, y_train)
            self.is_initialized = True
        
        # Update the main predictor model
        # This would involve updating the internal model of the congestion predictor
        # For now, we'll just log the update
        self.logger.info(f"Updated congestion predictor with {len(congestion_predictions)} samples")
        
        # Track performance
        if len(y_train) > 1:
            y_pred = self.congestion_online_model.predict(X_train_scaled)
            mse = mean_squared_error(y_train, y_pred)
            self.model_performance_trackers['congestion'].append(mse)
            
            if len(self.model_performance_trackers['congestion']) > self.max_tracking_samples:
                self.model_performance_trackers['congestion'].popleft()
            
            self.logger.info(f"Congestion model MSE: {mse:.4f}")
    
    def update_timing_analyzer(self, timing_analyzer: TimingAnalyzer):
        """Update timing analyzer with silicon feedback"""
        self.logger.info("Updating timing analyzer with silicon feedback")
        
        # Get historical timing data
        prediction_data = self.data_collector.get_historical_data([], 'predictions')
        timing_predictions = [p for p in prediction_data if p['model_type'] == 'timing']
        
        if not timing_predictions:
            self.logger.info("No timing prediction data available for update")
            return
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for pred in timing_predictions:
            features = json.loads(pred['features_json'])
            feature_vector = self._convert_features_to_vector(features)
            X_train.append(feature_vector)
            y_train.append(pred['actual_value'])
        
        if not X_train:
            self.logger.info("No valid feature vectors found for timing model update")
            return
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale features
        X_train_scaled = self.feature_scaler.transform(X_train) if self.is_initialized else self.feature_scaler.fit_transform(X_train)
        
        # Update the online model
        if self.is_initialized:
            self.timing_online_model.partial_fit(X_train_scaled, y_train)
        else:
            self.timing_online_model.fit(X_train_scaled, y_train)
            self.is_initialized = True
        
        # Track performance
        if len(y_train) > 1:
            y_pred = self.timing_online_model.predict(X_train_scaled)
            mse = mean_squared_error(y_train, y_pred)
            self.model_performance_trackers['timing'].append(mse)
            
            if len(self.model_performance_trackers['timing']) > self.max_tracking_samples:
                self.model_performance_trackers['timing'].popleft()
            
            self.logger.info(f"Timing model MSE: {mse:.4f}")
    
    def update_drc_predictor(self, drc_predictor: DRCPredictor):
        """Update DRC predictor with silicon feedback"""
        self.logger.info("Updating DRC predictor with silicon feedback")
        
        # Get historical DRC data
        prediction_data = self.data_collector.get_historical_data([], 'predictions')
        drc_predictions = [p for p in prediction_data if p['model_type'] == 'drc']
        
        if not drc_predictions:
            self.logger.info("No DRC prediction data available for update")
            return
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for pred in drc_predictions:
            features = json.loads(pred['features_json'])
            feature_vector = self._convert_features_to_vector(features)
            X_train.append(feature_vector)
            y_train.append(1 if pred['actual_value'] > 0 else 0)  # Binary classification: violation or not
        
        if not X_train:
            self.logger.info("No valid feature vectors found for DRC model update")
            return
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale features
        X_train_scaled = self.feature_scaler.transform(X_train) if self.is_initialized else self.feature_scaler.fit_transform(X_train)
        
        # Update the online model
        if self.is_initialized:
            self.drc_online_model.partial_fit(X_train_scaled, y_train)
        else:
            self.drc_online_model.fit(X_train_scaled, y_train)
            self.is_initialized = True
        
        # Track performance
        if len(y_train) > 1:
            y_pred = self.drc_online_model.predict(X_train_scaled)
            accuracy = accuracy_score(y_train, y_pred)
            self.model_performance_trackers['drc'].append(1 - accuracy)  # Error rate
            
            if len(self.model_performance_trackers['drc']) > self.max_tracking_samples:
                self.model_performance_trackers['drc'].popleft()
            
            self.logger.info(f"DRC model accuracy: {accuracy:.4f}")
    
    def update_cognitive_models(self, intent_interpreter: DesignIntentInterpreter,
                              knowledge_model: SiliconKnowledgeModel,
                              reasoning_engine: ReasoningEngine):
        """Update cognitive models with feedback"""
        self.logger.info("Updating cognitive models with silicon feedback")
        
        # Get design metadata and prediction data
        prediction_data = self.data_collector.get_historical_data([], 'predictions')
        
        # Update intent interpreter based on design success/failure patterns
        successful_designs = self._analyze_design_success_patterns(prediction_data)
        
        # This would involve updating the intent interpreter's understanding
        # of which design intents lead to successful outcomes
        self.logger.info(f"Analyzed {len(successful_designs)} successful design patterns")
    
    def update_agent_authority(self, agents: List[BaseAgent]):
        """Update agent authority based on prediction accuracy"""
        self.logger.info("Updating agent authority based on prediction accuracy")
        
        # Get prediction data by agent
        prediction_data = self.data_collector.get_historical_data([], 'predictions')
        
        # Group predictions by agent/model type
        agent_predictions = defaultdict(list)
        for pred in prediction_data:
            # Extract agent ID from prediction metadata
            agent_id = pred.get('agent_id', 'unknown')
            agent_predictions[agent_id].append(pred)
        
        # Update each agent's authority based on accuracy
        for agent in agents:
            agent_preds = agent_predictions.get(agent.agent_id, [])
            if agent_preds:
                # Calculate accuracy metrics
                correct_predictions = 0
                total_predictions = len(agent_preds)
                
                for pred in agent_preds:
                    error = abs(pred['predicted_value'] - pred['actual_value'])
                    tolerance = abs(pred['actual_value']) * 0.1  # 10% tolerance
                    if error <= tolerance:
                        correct_predictions += 1
                
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                
                # Update agent authority based on accuracy
                if accuracy > 0.8:
                    agent.update_authority(True, penalty_factor=0.05)  # Reward high accuracy
                elif accuracy < 0.6:
                    agent.update_authority(False, penalty_factor=0.1)  # Penalize low accuracy
                else:
                    # Neutral performance
                    recent_perf = agent.get_recent_performance(window=10)
                    if recent_perf > 0.7:
                        agent.update_authority(True, penalty_factor=0.02)
                    else:
                        agent.update_authority(False, penalty_factor=0.02)
                
                self.logger.info(f"Agent {agent.agent_id} accuracy: {accuracy:.3f}, authority: {agent.authority_level:.3f}")
    
    def _convert_features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to numerical vector"""
        # This is a simplified conversion - in practice, this would be more sophisticated
        # and handle categorical variables, normalization, etc.
        vector = []
        
        # Add numerical features
        for key, value in features.items():
            if isinstance(value, (int, float)):
                vector.append(value)
            elif isinstance(value, str):
                # Convert strings to numerical hash values
                vector.append(float(hashlib.md5(value.encode()).hexdigest()[:8], 16) % 10000)
            else:
                # Convert other types to string then hash
                vector.append(float(hashlib.md5(str(value).encode()).hexdigest()[:8], 16) % 10000)
        
        # Pad or truncate to consistent size
        target_size = 50  # Fixed size for now
        if len(vector) < target_size:
            vector.extend([0.0] * (target_size - len(vector)))
        else:
            vector = vector[:target_size]
        
        return np.array(vector, dtype=np.float32)
    
    def _analyze_design_success_patterns(self, prediction_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze patterns in successful designs"""
        successful_patterns = []
        
        # Group by design and analyze success factors
        designs = defaultdict(list)
        for pred in prediction_data:
            designs[pred['design_id']].append(pred)
        
        for design_id, preds in designs.items():
            # Calculate overall success metrics for this design
            avg_error = np.mean([abs(p['predicted_value'] - p['actual_value']) for p in preds])
            max_error = max([abs(p['predicted_value'] - p['actual_value']) for p in preds])
            
            success_metrics = {
                'design_id': design_id,
                'avg_prediction_error': avg_error,
                'max_prediction_error': max_error,
                'prediction_count': len(preds),
                'success_score': 1.0 / (1.0 + avg_error)  # Higher score for lower error
            }
            
            successful_patterns.append(success_metrics)
        
        return successful_patterns
    
    def get_model_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all models"""
        metrics = {}
        
        for model_type, errors in self.model_performance_trackers.items():
            if errors:
                metrics[model_type] = {
                    'avg_error': np.mean(errors),
                    'min_error': min(errors),
                    'max_error': max(errors),
                    'std_error': np.std(errors),
                    'sample_count': len(errors)
                }
        
        return metrics


class LearningLoopController:
    """
    Main controller for the learning loop system
    """
    
    def __init__(self, db_path: str = "./silicon_data.db"):
        self.logger = get_logger(__name__)
        self.data_collector = SiliconDataCollector(db_path)
        self.model_updater = ModelUpdater(self.data_collector)
        self.update_interval = 3600  # Update every hour (in seconds)
        self.last_update_time = 0
        self.is_running = False
        self.update_thread = None
    
    def collect_silicon_data(self, design_data: Dict[str, Any]):
        """
        Collect silicon data from a completed design
        
        Args:
            design_data: Dictionary containing all silicon data for a design
        """
        design_id = design_data['design_id']
        
        # Store metadata
        self.data_collector.store_design_metadata(design_id, design_data.get('metadata', {}))
        
        # Store timing data
        timing_data = design_data.get('timing_data', [])
        if timing_data:
            self.data_collector.store_timing_data(design_id, timing_data)
        
        # Store power data
        power_data = design_data.get('power_data', [])
        if power_data:
            self.data_collector.store_power_data(design_id, power_data)
        
        # Store area data
        area_data = design_data.get('area_data', [])
        if area_data:
            self.data_collector.store_area_data(design_id, area_data)
        
        # Store DRC data
        drc_data = design_data.get('drc_data', [])
        if drc_data:
            self.data_collector.store_drc_data(design_id, drc_data)
        
        # Store yield data
        yield_data = design_data.get('yield_data', [])
        if yield_data:
            self.data_collector.store_yield_data(design_id, yield_data)
        
        # Store performance data
        performance_data = design_data.get('performance_data', [])
        if performance_data:
            self.data_collector.store_performance_data(design_id, performance_data)
        
        # Store prediction vs actual data
        prediction_data = design_data.get('prediction_data', [])
        if prediction_data:
            self.data_collector.store_prediction_data(design_id, prediction_data)
        
        self.logger.info(f"Collected silicon data for design {design_id}")
    
    def update_all_models(self, 
                         congestion_predictor: CongestionPredictor,
                         timing_analyzer: TimingAnalyzer,
                         drc_predictor: DRCPredictor,
                         intent_interpreter: DesignIntentInterpreter,
                         knowledge_model: SiliconKnowledgeModel,
                         reasoning_engine: ReasoningEngine,
                         agents: List[BaseAgent]):
        """
        Update all models with accumulated feedback
        """
        self.logger.info("Updating all models with accumulated feedback")
        
        # Update individual models
        self.model_updater.update_congestion_predictor(congestion_predictor)
        self.model_updater.update_timing_analyzer(timing_analyzer)
        self.model_updater.update_drc_predictor(drc_predictor)
        self.model_updater.update_cognitive_models(
            intent_interpreter, knowledge_model, reasoning_engine
        )
        
        # Update agent authority
        self.model_updater.update_agent_authority(agents)
        
        # Update last update time
        self.last_update_time = time.time()
        
        self.logger.info("All models updated successfully")
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning process"""
        insights = {
            'total_designs_tracked': self._count_total_designs(),
            'data_coverage': self._analyze_data_coverage(),
            'model_performance': self.model_updater.get_model_performance_metrics(),
            'learning_effectiveness': self._calculate_learning_effectiveness(),
            'recent_updates': self._get_recent_update_stats()
        }
        
        return insights
    
    def _count_total_designs(self) -> int:
        """Count total designs in the database"""
        conn = sqlite3.connect(self.data_collector.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM design_metadata")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def _analyze_data_coverage(self) -> Dict[str, int]:
        """Analyze coverage of different data types"""
        conn = sqlite3.connect(self.data_collector.db_path)
        cursor = conn.cursor()
        
        coverage = {}
        
        # Count designs with each data type
        cursor.execute("SELECT COUNT(DISTINCT design_id) FROM timing_data")
        coverage['timing_data'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT design_id) FROM power_data")
        coverage['power_data'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT design_id) FROM area_data")
        coverage['area_data'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT design_id) FROM drc_data")
        coverage['drc_data'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT design_id) FROM yield_data")
        coverage['yield_data'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT design_id) FROM performance_data")
        coverage['performance_data'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT design_id) FROM prediction_data")
        coverage['prediction_data'] = cursor.fetchone()[0]
        
        conn.close()
        return coverage
    
    def _calculate_learning_effectiveness(self) -> Dict[str, float]:
        """Calculate metrics for learning effectiveness"""
        # This would analyze improvement over time
        # For now, return placeholder values
        return {
            'prediction_accuracy_improvement': 0.15,  # 15% improvement
            'model_convergence_rate': 0.85,  # 85% convergence
            'feedback_utilization_rate': 0.90  # 90% of feedback used
        }
    
    def _get_recent_update_stats(self) -> Dict[str, Any]:
        """Get statistics about recent updates"""
        return {
            'last_update_time': self.last_update_time,
            'update_frequency_hours': self.update_interval / 3600,
            'models_tracked': len(self.model_updater.model_performance_trackers)
        }
    
    def start_automatic_updates(self):
        """Start automatic periodic updates in a background thread"""
        if self.is_running:
            self.logger.warning("Automatic updates already running")
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._automatic_update_worker, daemon=True)
        self.update_thread.start()
        self.logger.info("Started automatic learning loop updates")
    
    def stop_automatic_updates(self):
        """Stop automatic periodic updates"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)  # Wait up to 5 seconds for thread to finish
        self.logger.info("Stopped automatic learning loop updates")
    
    def _automatic_update_worker(self):
        """Worker function for automatic updates"""
        while self.is_running:
            time.sleep(60)  # Check every minute
            
            if time.time() - self.last_update_time >= self.update_interval:
                try:
                    # This would need references to the actual models
                    # For now, we'll just log that an update would happen
                    self.logger.info("Automatic update triggered")
                    # self.update_all_models(...)  # Would call with actual model references
                except Exception as e:
                    self.logger.error(f"Error in automatic update: {str(e)}")
    
    def force_update(self):
        """Force an immediate update of all models"""
        self.logger.info("Forcing immediate update of all models")
        # This would call update_all_models with actual model references
        # For now, just log the action
        self.logger.info("Models would be updated with current data")


# Example usage and testing
def example_learning_loop():
    """Example of using the learning loop system"""
    logger = get_logger(__name__)
    
    # Initialize the learning loop
    learning_controller = LearningLoopController()
    logger.info("Learning loop controller initialized")
    
    # Example silicon data that would come from a real chip
    sample_silicon_data = {
        'design_id': 'chip_design_001',
        'metadata': {
            'project_name': 'Test_SOC',
            'tapeout_date': '2024-01-15',
            'process_node': '7nm',
            'die_size': 150.0,
            'core_voltage': 0.8,
            'max_frequency': 3.2
        },
        'timing_data': [
            {
                'measurement_type': 'setup',
                'path_type': 'reg2reg',
                'slack_ps': 0.2,
                'delay_ps': 320.5,
                'frequency_mhz': 3125.0,
                'temperature': 25.0,
                'voltage': 0.8,
                'measured_at': '2024-02-01T10:00:00'
            }
        ],
        'power_data': [
            {
                'measurement_type': 'dynamic',
                'rail_name': 'VDD_CORE',
                'static_power_mw': 15.2,
                'dynamic_power_mw': 1250.8,
                'peak_power_mw': 1800.0,
                'temperature': 25.0,
                'voltage': 0.8,
                'measured_at': '2024-02-01T10:00:00'
            }
        ],
        'area_data': [
            {
                'block_name': 'CORE',
                'area_um2': 85000000.0,
                'utilization': 0.78,
                'cell_count': 2500000,
                'macro_count': 150,
                'measured_at': '2024-02-01T10:00:00'
            }
        ],
        'drc_data': [
            {
                'rule_name': 'MIN_WIDTH',
                'violation_count': 0,
                'waived_count': 0,
                'severity': 'ERROR',
                'measured_at': '2024-02-01T10:00:00'
            }
        ],
        'yield_data': [
            {
                'wafer_id': 'WAFER_001',
                'die_x': 15,
                'die_y': 23,
                'bin_result': 'PASS',
                'parametric_pass': True,
                'functional_pass': True,
                'measured_at': '2024-02-01T10:00:00'
            }
        ],
        'performance_data': [
            {
                'test_name': 'FMAX',
                'parameter_name': 'MAX_FREQUENCY',
                'measured_value': 3200.0,
                'specification_min': 3000.0,
                'specification_max': 3500.0,
                'unit': 'MHz',
                'measured_at': '2024-02-01T10:00:00'
            }
        ],
        'prediction_data': [
            {
                'model_type': 'timing',
                'prediction_task': 'setup_slack',
                'predicted_value': -0.1,
                'actual_value': 0.2,
                'confidence': 0.85,
                'features': {
                    'path_length': 15,
                    'cell_types': ['AND2', 'DFF'],
                    'fanout': 8
                },
                'prediction_time': '2024-01-10T09:00:00'
            }
        ]
    }
    
    # Collect the silicon data
    learning_controller.collect_silicon_data(sample_silicon_data)
    logger.info("Silicon data collected")
    
    # Get learning insights
    insights = learning_controller.get_learning_insights()
    logger.info(f"Learning insights: {insights}")
    
    logger.info("Learning loop example completed")


if __name__ == "__main__":
    example_learning_loop()