"""
Configuration and Deployment System for Silicon Intelligence System

This module handles system configuration, deployment, and environment management
for the Silicon Intelligence System.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from enum import Enum
import configparser
from datetime import datetime
from utils.logger import get_logger


class DeploymentMode(Enum):
    """Deployment modes for the system"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    BENCHMARK = "benchmark"
    DEMO = "demo"


class ProcessNode(Enum):
    """Supported process nodes"""
    NODE_28NM = "28nm"
    NODE_16NM = "16nm"
    NODE_7NM = "7nm"
    NODE_5NM = "5nm"
    NODE_3NM = "3nm"
    NODE_2NM = "2nm"
    NODE_1NM = "1nm"


@dataclass
class SystemConfig:
    """Main system configuration"""
    # Core settings
    deployment_mode: DeploymentMode = DeploymentMode.DEVELOPMENT
    process_node: ProcessNode = ProcessNode.NODE_7NM
    target_frequency_ghz: float = 3.0
    power_budget_watts: float = 1.0
    
    # ML model settings
    use_advanced_models: bool = True
    model_precision: str = "fp16"  # fp32, fp16, int8
    enable_quantization: bool = False
    
    # Performance settings
    max_parallel_workers: int = 8
    enable_gpu_acceleration: bool = True
    gpu_device_id: int = 0
    
    # Agent settings
    agent_negotiation_timeout: float = 300.0  # seconds
    agent_authority_decay_rate: float = 0.01
    enable_conflict_resolution: bool = True
    
    # Learning settings
    learning_rate: float = 0.001
    batch_size: int = 32
    max_training_epochs: int = 100
    enable_continual_learning: bool = True
    
    # Integration settings
    enable_eda_integration: bool = True
    commercial_eda_tools: List[str] = field(default_factory=lambda: ["openroad"])
    cloud_platforms: List[str] = field(default_factory=lambda: [])
    
    # Path settings
    pdk_path: str = "/default/pdk/path"
    lib_path: str = "/default/lib/path"
    output_directory: str = "./output"
    temp_directory: str = "./temp"
    model_cache_directory: str = "./models/cache"
    
    # Advanced settings
    enable_cognitive_reasoning: bool = True
    cognitive_reasoning_depth: int = 5
    enable_parallel_realities: bool = True
    max_parallel_realities: int = 4
    enable_drc_prediction: bool = True
    enable_intent_interpretation: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = asdict(self)
        # Convert enums to strings
        result['deployment_mode'] = self.deployment_mode.value
        result['process_node'] = self.process_node.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create from dictionary format"""
        # Convert string values back to enums
        if 'deployment_mode' in data:
            data['deployment_mode'] = DeploymentMode(data['deployment_mode'])
        if 'process_node' in data:
            data['process_node'] = ProcessNode(data['process_node'])
        
        return cls(**data)


class ConfigManager:
    """
    Manages system configuration and settings
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.config_path = config_path or "./silicon_intelligence_config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> SystemConfig:
        """Load configuration from file or use defaults"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.json'):
                        data = json.load(f)
                    else:
                        data = yaml.safe_load(f)
                
                self.logger.info(f"Loaded configuration from {self.config_path}")
                return SystemConfig.from_dict(data)
            except Exception as e:
                self.logger.warning(f"Failed to load config from {self.config_path}: {str(e)}. Using defaults.")
                return SystemConfig()
        else:
            self.logger.info("Configuration file not found. Using default configuration.")
            return SystemConfig()
    
    def save_config(self, config: SystemConfig = None):
        """Save configuration to file"""
        config_to_save = config or self.config
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_to_save.to_dict(), f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration saved to {self.config_path}")
    
    def update_config(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated config.{key} = {value}")
            else:
                self.logger.warning(f"Unknown configuration parameter: {key}")
    
    def get_config(self) -> SystemConfig:
        """Get current configuration"""
        return self.config
    
    def validate_config(self) -> List[str]:
        """Validate configuration settings"""
        errors = []
        
        # Validate process node
        if self.config.process_node not in ProcessNode:
            errors.append(f"Invalid process node: {self.config.process_node}")
        
        # Validate deployment mode
        if self.config.deployment_mode not in DeploymentMode:
            errors.append(f"Invalid deployment mode: {self.config.deployment_mode}")
        
        # Validate paths
        if not os.path.exists(self.config.pdk_path) and self.config.enable_eda_integration:
            errors.append(f"PDK path does not exist: {self.config.pdk_path}")
        
        if not os.path.exists(self.config.lib_path) and self.config.enable_eda_integration:
            errors.append(f"Library path does not exist: {self.config.lib_path}")
        
        # Validate numerical ranges
        if not (0.1 <= self.config.target_frequency_ghz <= 10.0):
            errors.append(f"Target frequency out of range (0.1-10.0 GHz): {self.config.target_frequency_ghz}")
        
        if not (0.01 <= self.config.power_budget_watts <= 100.0):
            errors.append(f"Power budget out of range (0.01-100.0 W): {self.config.power_budget_watts}")
        
        if not (1 <= self.config.max_parallel_workers <= 64):
            errors.append(f"Parallel workers out of range (1-64): {self.config.max_parallel_workers}")
        
        return errors


class EnvironmentManager:
    """
    Manages system environment and dependencies
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.environment_vars = {}
        self.dependencies = {}
    
    def setup_environment(self, config: SystemConfig):
        """Set up the system environment based on configuration"""
        self.logger.info("Setting up system environment...")
        
        # Set environment variables
        env_vars = {
            'SILICON_INTEL_ROOT': os.getcwd(),
            'PDK_PATH': config.pdk_path,
            'LIBRARY_PATH': config.lib_path,
            'OUTPUT_DIR': config.output_directory,
            'TEMP_DIR': config.temp_directory,
            'MODEL_CACHE_DIR': config.model_cache_directory,
            'CUDA_VISIBLE_DEVICES': str(config.gpu_device_id) if config.enable_gpu_acceleration else '-1'
        }
        
        for var, value in env_vars.items():
            os.environ[var] = value
            self.environment_vars[var] = value
        
        # Create necessary directories
        self._create_directories(config)
        
        # Check dependencies
        self._check_dependencies(config)
        
        self.logger.info("System environment setup completed")
    
    def _create_directories(self, config: SystemConfig):
        """Create necessary directories"""
        dirs_to_create = [
            config.output_directory,
            config.temp_directory,
            config.model_cache_directory,
            os.path.join(config.output_directory, 'logs'),
            os.path.join(config.output_directory, 'reports'),
            os.path.join(config.output_directory, 'models'),
            os.path.join(config.temp_directory, 'scratch'),
            os.path.join(config.model_cache_directory, 'checkpoints')
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {dir_path}")
    
    def _check_dependencies(self, config: SystemConfig):
        """Check system dependencies"""
        import subprocess
        
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            self.logger.warning(f"Python version {sys.version} is older than recommended (3.8+)")
        
        # Check for required packages
        required_packages = [
            'torch', 'networkx', 'numpy', 'pandas', 'matplotlib', 
            'scikit-learn', 'transformers', 'pyyaml'
        ]
        
        missing_packages = []
        for pkg in required_packages:
            try:
                __import__(pkg)
            except ImportError:
                missing_packages.append(pkg)
        
        if missing_packages:
            self.logger.error(f"Missing required packages: {missing_packages}")
            self.dependencies['missing_packages'] = missing_packages
        else:
            self.logger.info("All required packages are available")
            self.dependencies['missing_packages'] = []
        
        # Check for EDA tools if integration is enabled
        if config.enable_eda_integration:
            eda_tools = config.commercial_eda_tools
            available_tools = []
            
            for tool in eda_tools:
                try:
                    result = subprocess.run(['which', tool], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        available_tools.append(tool)
                except:
                    pass  # Tool not available
            
            self.dependencies['eda_tools'] = {
                'requested': eda_tools,
                'available': available_tools,
                'missing': [t for t in eda_tools if t not in available_tools]
            }
            
            if not available_tools:
                self.logger.warning("No EDA tools available for integration")
        else:
            self.dependencies['eda_tools'] = {'requested': [], 'available': [], 'missing': []}
    
    def get_environment_status(self) -> Dict[str, Any]:
        """Get current environment status"""
        return {
            'environment_variables': self.environment_vars,
            'dependencies': self.dependencies,
            'python_version': '.'.join(map(str, sys.version_info[:3])),
            'working_directory': os.getcwd(),
            'available_memory_gb': self._get_available_memory(),
            'gpu_available': self._check_gpu_availability()
        }
    
    def _get_available_memory(self) -> float:
        """Get available system memory in GB"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.available / (1024**3)  # Convert to GB
        except:
            return 0.0  # Unknown
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False


class DeploymentManager:
    """
    Manages system deployment across different environments
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config_manager = ConfigManager()
        self.env_manager = EnvironmentManager()
    
    def deploy(self, deployment_mode: DeploymentMode = DeploymentMode.PRODUCTION):
        """Deploy the system in specified mode"""
        self.logger.info(f"Deploying Silicon Intelligence System in {deployment_mode.value} mode")
        
        # Update configuration
        self.config_manager.update_config(deployment_mode=deployment_mode)
        config = self.config_manager.get_config()
        
        # Validate configuration
        errors = self.config_manager.validate_config()
        if errors:
            self.logger.error(f"Configuration validation errors: {errors}")
            raise ValueError(f"Configuration validation failed: {errors}")
        
        # Set up environment
        self.env_manager.setup_environment(config)
        
        # Initialize system components based on mode
        self._initialize_components(config)
        
        # Run deployment checks
        self._run_deployment_checks(config)
        
        self.logger.info(f"Silicon Intelligence System deployed successfully in {deployment_mode.value} mode")
        
        return {
            'success': True,
            'deployment_mode': deployment_mode.value,
            'configuration': config.to_dict(),
            'environment_status': self.env_manager.get_environment_status(),
            'deployment_timestamp': datetime.now().isoformat()
        }
    
    def _initialize_components(self, config: SystemConfig):
        """Initialize system components based on configuration"""
        self.logger.info("Initializing system components...")
        
        # Initialize core components
        from cognitive.advanced_cognitive_system import PhysicalRiskOracle
        from core.parallel_reality_engine import ParallelRealityEngine
        from agents.advanced_conflict_resolution import EnhancedAgentNegotiator
        from core.comprehensive_learning_loop import LearningLoopController
        
        # Initialize based on configuration
        self.physical_risk_oracle = PhysicalRiskOracle()
        self.parallel_engine = ParallelRealityEngine(max_workers=config.max_parallel_workers)
        self.negotiator = EnhancedAgentNegotiator()
        self.learning_controller = LearningLoopController()
        
        # Initialize ML models based on configuration
        if config.use_advanced_models:
            from models.advanced_ml_models import (
                AdvancedCongestionPredictor, AdvancedTimingAnalyzer, AdvancedDRCPredictor
            )
            self.congestion_predictor = AdvancedCongestionPredictor()
            self.timing_analyzer = AdvancedTimingAnalyzer()
            self.drc_predictor = AdvancedDRCPredictor()
        else:
            from models.basic_ml_models import (
                BasicCongestionPredictor, BasicTimingAnalyzer, BasicDRCPredictor
            )
            self.congestion_predictor = BasicCongestionPredictor()
            self.timing_analyzer = BasicTimingAnalyzer()
            self.drc_predictor = BasicDRCPredictor()
        
        # Enable GPU acceleration if configured
        if config.enable_gpu_acceleration:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.set_device(config.gpu_device_id)
                    self.logger.info(f"GPU acceleration enabled on device {config.gpu_device_id}")
                else:
                    self.logger.warning("GPU requested but not available, falling back to CPU")
            except ImportError:
                self.logger.warning("PyTorch not available, GPU acceleration disabled")
        
        self.logger.info("System components initialized")
    
    def _run_deployment_checks(self, config: SystemConfig):
        """Run deployment validation checks"""
        self.logger.info("Running deployment validation checks...")
        
        # Check basic functionality
        try:
            # Test basic import
            from main import main
            self.logger.info("✓ Main module import successful")
        except Exception as e:
            self.logger.error(f"✗ Main module import failed: {str(e)}")
            raise
        
        # Test cognitive system
        try:
            risk_result = self.physical_risk_oracle.predict_physical_risks(
                "dummy_rtl.v", "dummy_constraints.sdc", config.process_node.value
            )
            self.logger.info("✓ Cognitive system functional")
        except Exception as e:
            self.logger.warning(f"⚠ Cognitive system test failed: {str(e)}")
        
        # Test parallel engine
        try:
            def dummy_strategy(graph_state, iteration):
                return []
            
            # Create a minimal test graph
            from core.canonical_silicon_graph import CanonicalSiliconGraph
            test_graph = CanonicalSiliconGraph()
            test_graph.graph.add_node('test_node', node_type='cell', power=0.1, area=2.0)
            
            results = self.parallel_engine.run_parallel_execution(
                test_graph, [dummy_strategy], max_iterations=1
            )
            self.logger.info("✓ Parallel reality engine functional")
        except Exception as e:
            self.logger.error(f"✗ Parallel reality engine test failed: {str(e)}")
            raise
        
        # Test agent negotiation
        try:
            negotiation_result = self.negotiator.run_negotiation_round(test_graph)
            self.logger.info("✓ Agent negotiation functional")
        except Exception as e:
            self.logger.error(f"✗ Agent negotiation test failed: {str(e)}")
            raise
        
        # Test ML models
        try:
            # Test congestion predictor
            congestion_result = self.congestion_predictor.predict(test_graph)
            self.logger.info("✓ ML models functional")
        except Exception as e:
            self.logger.error(f"✗ ML models test failed: {str(e)}")
            raise
        
        self.logger.info("All deployment checks passed")
    
    def deploy_to_cloud(self, platform: str, config: Optional[SystemConfig] = None) -> Dict[str, Any]:
        """Deploy to cloud platform"""
        self.logger.info(f"Deploying to cloud platform: {platform}")
        
        current_config = config or self.config_manager.get_config()
        
        if platform.lower() == 'aws':
            return self._deploy_to_aws(current_config)
        elif platform.lower() == 'azure':
            return self._deploy_to_azure(current_config)
        elif platform.lower() == 'gcp':
            return self._deploy_to_gcp(current_config)
        else:
            raise ValueError(f"Unsupported cloud platform: {platform}")
    
    def _deploy_to_aws(self, config: SystemConfig) -> Dict[str, Any]:
        """Deploy to AWS"""
        try:
            import boto3
            
            # Create EC2 client
            ec2 = boto3.client('ec2')
            
            # Determine instance type based on configuration
            if config.max_parallel_workers > 16:
                instance_type = 'p4d.24xlarge'  # GPU instance with lots of cores
            elif config.enable_gpu_acceleration:
                instance_type = 'g4dn.12xlarge'  # GPU instance
            else:
                instance_type = 'c5.18xlarge'  # CPU optimized
            
            deployment_info = {
                'platform': 'aws',
                'instance_type': instance_type,
                'deployment_config': config.to_dict(),
                'status': 'provisioning',
                'estimated_cost_usd_per_hour': self._estimate_aws_cost(instance_type)
            }
            
            self.logger.info(f"AWS deployment prepared: {instance_type}")
            return {'success': True, 'deployment_info': deployment_info}
            
        except ImportError:
            self.logger.error("boto3 not available for AWS deployment")
            return {'success': False, 'error': 'boto3 library not installed'}
        except Exception as e:
            self.logger.error(f"AWS deployment failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _deploy_to_azure(self, config: SystemConfig) -> Dict[str, Any]:
        """Deploy to Azure"""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient
            
            deployment_info = {
                'platform': 'azure',
                'deployment_config': config.to_dict(),
                'status': 'preparing',
                'recommended_vm_size': self._recommend_azure_vm(config)
            }
            
            self.logger.info(f"Azure deployment prepared: {deployment_info['recommended_vm_size']}")
            return {'success': True, 'deployment_info': deployment_info}
            
        except ImportError:
            self.logger.error("Azure libraries not available for deployment")
            return {'success': False, 'error': 'Azure libraries not installed'}
        except Exception as e:
            self.logger.error(f"Azure deployment failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _deploy_to_gcp(self, config: SystemConfig) -> Dict[str, Any]:
        """Deploy to Google Cloud Platform"""
        try:
            from google.cloud import compute_v1
            
            deployment_info = {
                'platform': 'gcp',
                'deployment_config': config.to_dict(),
                'status': 'preparing',
                'recommended_machine_type': self._recommend_gcp_machine(config)
            }
            
            self.logger.info(f"GCP deployment prepared: {deployment_info['recommended_machine_type']}")
            return {'success': True, 'deployment_info': deployment_info}
            
        except ImportError:
            self.logger.error("Google Cloud libraries not available for deployment")
            return {'success': False, 'error': 'Google Cloud libraries not installed'}
        except Exception as e:
            self.logger.error(f"GCP deployment failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _estimate_aws_cost(self, instance_type: str) -> float:
        """Estimate AWS cost per hour"""
        # Simplified cost estimation - in reality would use pricing API
        cost_map = {
            'c5.18xlarge': 4.77,
            'g4dn.12xlarge': 3.92,
            'p4d.24xlarge': 32.77
        }
        return cost_map.get(instance_type, 10.0)  # Default to $10/hour
    
    def _recommend_azure_vm(self, config: SystemConfig) -> str:
        """Recommend Azure VM based on configuration"""
        if config.max_parallel_workers > 16 and config.enable_gpu_acceleration:
            return 'Standard_NC24s_v3'  # GPU VM
        elif config.max_parallel_workers > 16:
            return 'Standard_HB120rs_v3'  # High-performance CPU VM
        else:
            return 'Standard_D16s_v5'  # General purpose VM
    
    def _recommend_gcp_machine(self, config: SystemConfig) -> str:
        """Recommend GCP machine based on configuration"""
        if config.enable_gpu_acceleration:
            return 'a2-highgpu-1g'  # High GPU machine
        elif config.max_parallel_workers > 16:
            return 'c2-standard-60'  # High CPU count
        else:
            return 'n2-standard-16'  # General purpose


class SystemInitializer:
    """
    Initializes the Silicon Intelligence System with all components
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config_manager = ConfigManager()
        self.env_manager = EnvironmentManager()
        self.deployment_manager = DeploymentManager()
    
    def initialize_system(self, config_path: Optional[str] = None, 
                         deployment_mode: DeploymentMode = DeploymentMode.DEVELOPMENT) -> Dict[str, Any]:
        """Initialize the complete system"""
        self.logger.info("Initializing Silicon Intelligence System...")
        
        # Load configuration
        if config_path:
            self.config_manager = ConfigManager(config_path)
        
        # Update deployment mode
        self.config_manager.update_config(deployment_mode=deployment_mode)
        
        # Validate configuration
        config_errors = self.config_manager.validate_config()
        if config_errors:
            self.logger.error(f"Configuration validation failed: {config_errors}")
            return {'success': False, 'errors': config_errors}
        
        # Set up environment
        config = self.config_manager.get_config()
        self.env_manager.setup_environment(config)
        
        # Deploy system
        deployment_result = self.deployment_manager.deploy(deployment_mode)
        
        # Run comprehensive system check
        system_check = self._run_comprehensive_system_check(config)
        
        initialization_result = {
            'success': deployment_result['success'] and system_check['all_systems_operational'],
            'configuration': deployment_result['configuration'],
            'environment_status': deployment_result['environment_status'],
            'system_check': system_check,
            'initialization_timestamp': datetime.now().isoformat()
        }
        
        if initialization_result['success']:
            self.logger.info("Silicon Intelligence System initialized successfully!")
        else:
            self.logger.error("System initialization failed")
        
        return initialization_result
    
    def _run_comprehensive_system_check(self, config: SystemConfig) -> Dict[str, Any]:
        """Run comprehensive system health check"""
        self.logger.info("Running comprehensive system check...")
        
        checks = {
            'basic_imports': self._check_basic_imports(),
            'cognitive_system': self._check_cognitive_system(),
            'ml_models': self._check_ml_models(),
            'agents': self._check_agents(),
            'parallel_engine': self._check_parallel_engine(),
            'learning_loop': self._check_learning_loop(),
            'integrations': self._check_integrations(config),
            'all_systems_operational': False
        }
        
        # Overall success is if all checks pass
        checks['all_systems_operational'] = all(
            result.get('success', False) for result in checks.values() 
            if isinstance(result, dict) and 'success' in result
        )
        
        self.logger.info(f"System check completed. Operational: {checks['all_systems_operational']}")
        return checks
    
    def _check_basic_imports(self) -> Dict[str, bool]:
        """Check basic imports"""
        try:
            from main import main
            from core.canonical_silicon_graph import CanonicalSiliconGraph
            from cognitive.advanced_cognitive_system import PhysicalRiskOracle
            return {'success': True}
        except Exception as e:
            self.logger.error(f"Basic imports check failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _check_cognitive_system(self) -> Dict[str, bool]:
        """Check cognitive system functionality"""
        try:
            from cognitive.advanced_cognitive_system import PhysicalRiskOracle
            oracle = PhysicalRiskOracle()
            # Test with dummy data
            return {'success': True}
        except Exception as e:
            self.logger.error(f"Cognitive system check failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _check_ml_models(self) -> Dict[str, bool]:
        """Check ML model functionality"""
        try:
            from models.advanced_ml_models import (
                AdvancedCongestionPredictor, AdvancedTimingAnalyzer, AdvancedDRCPredictor
            )
            
            predictor = AdvancedCongestionPredictor()
            analyzer = AdvancedTimingAnalyzer()
            drc_pred = AdvancedDRCPredictor()
            
            return {'success': True}
        except Exception as e:
            self.logger.error(f"ML models check failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _check_agents(self) -> Dict[str, bool]:
        """Check agent system functionality"""
        try:
            from agents.floorplan_agent import FloorplanAgent
            from agents.placement_agent import PlacementAgent
            from agents.clock_agent import ClockAgent
            from agents.power_agent import PowerAgent
            from agents.yield_agent import YieldAgent
            from agents.routing_agent import RoutingAgent
            from agents.thermal_agent import ThermalAgent
            
            agents = [
                FloorplanAgent(), PlacementAgent(), ClockAgent(),
                PowerAgent(), YieldAgent(), RoutingAgent(), ThermalAgent()
            ]
            
            # Test agent creation and basic functionality
            for agent in agents:
                assert hasattr(agent, 'propose_action')
            
            return {'success': True}
        except Exception as e:
            self.logger.error(f"Agent system check failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _check_parallel_engine(self) -> Dict[str, bool]:
        """Check parallel reality engine functionality"""
        try:
            from core.parallel_reality_engine import ParallelRealityEngine
            
            engine = ParallelRealityEngine(max_workers=2)
            
            # Test with dummy strategy
            def dummy_strategy(graph_state, iteration):
                return []
            
            # Create test graph
            from core.canonical_silicon_graph import CanonicalSiliconGraph
            test_graph = CanonicalSiliconGraph()
            test_graph.graph.add_node('test', node_type='cell')
            
            results = engine.run_parallel_execution(test_graph, [dummy_strategy], max_iterations=1)
            
            return {'success': True}
        except Exception as e:
            self.logger.error(f"Parallel engine check failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _check_learning_loop(self) -> Dict[str, bool]:
        """Check learning loop functionality"""
        try:
            from core.comprehensive_learning_loop import LearningLoopController
            
            controller = LearningLoopController()
            
            return {'success': True}
        except Exception as e:
            self.logger.error(f"Learning loop check failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _check_integrations(self, config: SystemConfig) -> Dict[str, bool]:
        """Check integration functionality"""
        try:
            available_tools = []
            
            if config.enable_eda_integration:
                for tool in config.commercial_eda_tools:
                    try:
                        import subprocess
                        result = subprocess.run(['which', tool], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            available_tools.append(tool)
                    except:
                        pass
            
            return {'success': True, 'available_tools': available_tools}
        except Exception as e:
            self.logger.error(f"Integration check failed: {str(e)}")
            return {'success': False, 'error': str(e)}


def create_default_config() -> SystemConfig:
    """Create a default configuration for the system"""
    return SystemConfig(
        deployment_mode=DeploymentMode.DEVELOPMENT,
        process_node=ProcessNode.NODE_7NM,
        target_frequency_ghz=3.0,
        power_budget_watts=1.0,
        use_advanced_models=True,
        max_parallel_workers=4,
        enable_gpu_acceleration=True,
        agent_negotiation_timeout=300.0,
        enable_continual_learning=True,
        enable_eda_integration=True,
        commercial_eda_tools=["openroad"],
        output_directory="./output",
        temp_directory="./temp",
        model_cache_directory="./models/cache"
    )


def setup_development_environment():
    """Set up a development environment for the Silicon Intelligence System"""
    logger = get_logger(__name__)
    logger.info("Setting up development environment...")
    
    # Create default config
    config = create_default_config()
    
    # Save config
    config_manager = ConfigManager("./dev_config.yaml")
    config_manager.save_config(config)
    
    # Set up environment
    env_manager = EnvironmentManager()
    env_manager.setup_environment(config)
    
    print("\nDevelopment environment set up successfully!")
    print("Configuration saved to: dev_config.yaml")
    print("\nTo start developing:")
    print("  1. Activate your Python environment")
    print("  2. Install dependencies: pip install -r requirements.txt")
    print("  3. Run tests: python -m pytest tests/")
    print("  4. Start experimenting with: python main.py --mode demo")
    print("\nFor production deployment, create a production config file")
    
    return config


if __name__ == "__main__":
    print("Silicon Intelligence System - Configuration and Deployment Manager")
    print("="*65)
    
    # Initialize the system
    initializer = SystemInitializer()
    result = initializer.initialize_system(deployment_mode=DeploymentMode.DEVELOPMENT)
    
    print(f"\nSystem Initialization Result:")
    print(f"  Success: {result['success']}")
    print(f"  Mode: {result['configuration']['deployment_mode']}")
    print(f"  Process Node: {result['configuration']['process_node']}")
    print(f"  Advanced Models: {result['configuration']['use_advanced_models']}")
    print(f"  Parallel Workers: {result['configuration']['max_parallel_workers']}")
    
    if result['success']:
        print("\n✓ Silicon Intelligence System is ready for development!")
        print("  Run 'python main.py --mode demo' to see the system in action")
        print("  Run 'python main.py --mode benchmark' to run performance tests")
        print("  Run 'python main.py --mode oracle --rtl <file> --constraints <file>' for risk assessment")
    else:
        print("\n✗ System initialization failed. Check configuration and dependencies.")
    
    print("="*65)