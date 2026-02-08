#!/usr/bin/env python3
"""
Autonomous Design Optimizer
Uses predictions and insights to automatically optimize designs
"""

import copy
from typing import Dict, List, Any, Tuple
from enum import Enum

from physical_design_intelligence import PhysicalDesignIntelligence
from synthetic_design_generator import DesignSpec
from core.rtl_transformer import RTLTransformer
import tempfile
import os
import pyverilog.vparser.ast as vast # Import vast directly

# RL-EDA Imports
try:
    from networks.rl_environment import DesignOptimizationEnv
    from networks.policy_agent import PolicyAgent
    from networks.credit_assignment import CreditAssignmentLogger
    from core.design_memory import DesignMemory
    from networks.credit_assignment import CreditAssignmentLogger
    from core.design_memory import DesignMemory
except ImportError:
    print("Warning: RL-EDA components not fully available. Some features will be disabled.")

try:
    import silicon_intelligence_cpp as sic
    HAS_CPP_KERNELS = hasattr(sic, 'OptimizationKernels')
except ImportError:
    HAS_CPP_KERNELS = False


class OptimizationStrategy(Enum):
    """Types of optimization strategies"""
    PIPELINE_CRITICAL_PATHS = "pipeline_critical_paths"
    CLUSTER_CONGESTED_AREAS = "cluster_congested_areas"
    REDUCE_REGISTER_COUNT = "reduce_register_count"
    OPTIMIZE_FANOUT = "optimize_fanout"
    BALANCE_AREA_POWER = "balance_area_power"
    BALANCE_AREA_POWER = "balance_area_power"
    CLOCK_GATING = "clock_gating"
    GLOBAL_PLACEMENT = "global_placement"


class AutonomousOptimizer:
    """
    Autonomous optimization system that uses predictions and insights
    to automatically improve designs
    """
    
    def __init__(self):
        self.design_intelligence = PhysicalDesignIntelligence()
        self.rtl_transformer = RTLTransformer()
        self.optimization_strategies = {
            OptimizationStrategy.PIPELINE_CRITICAL_PATHS: self._apply_pipelining,
            OptimizationStrategy.CLUSTER_CONGESTED_AREAS: self._apply_clustering,
            OptimizationStrategy.REDUCE_REGISTER_COUNT: self._reduce_registers,
            OptimizationStrategy.OPTIMIZE_FANOUT: self._optimize_fanout,
            OptimizationStrategy.BALANCE_AREA_POWER: self._balance_area_power,
            OptimizationStrategy.OPTIMIZE_FANOUT: self._optimize_fanout,
            OptimizationStrategy.BALANCE_AREA_POWER: self._balance_area_power,
            OptimizationStrategy.CLOCK_GATING: self._apply_clock_gating,
            OptimizationStrategy.GLOBAL_PLACEMENT: self._apply_global_placement
        }
    
    def analyze_and_optimize(self, rtl_content: str, design_name: str = "unnamed") -> Dict[str, Any]:
        """Analyze design and suggest/implement optimizations"""
        
        # Initial analysis
        initial_analysis = self.design_intelligence.analyze_design(rtl_content, design_name)
        
        # Identify optimization opportunities
        optimization_opportunities = self._identify_optimizations(initial_analysis)
        
        results = {
            'initial_analysis': initial_analysis,
            'optimization_opportunities': optimization_opportunities,
            'optimizations_applied': [],
            'final_analysis': None
        }
        
        # Apply optimizations
        optimized_rtl = rtl_content
        for strategy, params in optimization_opportunities:
            optimized_rtl = self._apply_strategy(optimized_rtl, strategy, params)
            results['optimizations_applied'].append({
                'strategy': strategy.value,
                'params': params,
                'applied': True
            })
        
        # Re-analyze optimized design
        if optimized_rtl != rtl_content:
            final_analysis = self.design_intelligence.analyze_design(optimized_rtl, f"optimized_{design_name}")
            results['final_analysis'] = final_analysis
            results['improvement'] = self._calculate_improvement(
                initial_analysis, final_analysis
            )
        else:
            results['final_analysis'] = initial_analysis
            results['improvement'] = {}
        
        return results
    
    def _identify_optimizations(self, analysis: Dict[str, Any]) -> List[Tuple[OptimizationStrategy, Dict]]:
        """Identify potential optimizations based on analysis"""
        
        opportunities = []
        
        # Check for timing violations
        timing_slack = analysis['openroad_results']['placement']['timing_slack_ps']
        if timing_slack < 0:
            # Critical path needs pipelining
            opportunities.append((
                OptimizationStrategy.PIPELINE_CRITICAL_PATHS,
                {'slack_required': abs(timing_slack)}
            ))
        
        # Check for congestion
        congestion_map = analysis['openroad_results']['placement']['congestion_map']
        if congestion_map and any(c['congestion_level'] > 0.8 for c in congestion_map):
            opportunities.append((
                OptimizationStrategy.CLUSTER_CONGESTED_AREAS,
                {'high_congestion_threshold': 0.8}
            ))
        
        # Check for high register count relative to combinational logic
        phys_stats = analysis['physical_ir_stats']
        reg_count = phys_stats.get('node_types', {}).get('register', 0)
        comb_count = phys_stats.get('node_types', {}).get('combinational', 0)
        
        if reg_count > 0 and comb_count > 0:
            reg_to_comb_ratio = reg_count / comb_count
            if reg_to_comb_ratio > 2.0:  # Too many registers
                opportunities.append((
                    OptimizationStrategy.REDUCE_REGISTER_COUNT,
                    {'ratio_threshold': 2.0, 'current_ratio': reg_to_comb_ratio}
                ))
        
        # Check for high average fanout
        if phys_stats.get('avg_fanout', 0) > 5.0:
            opportunities.append((
                OptimizationStrategy.OPTIMIZE_FANOUT,
                {'fanout_threshold': 5.0, 'current_avg': phys_stats['avg_fanout']}
            ))
        
        # Check for area vs power balance
        area = analysis['openroad_results']['overall_ppa']['area_um2']
        power = analysis['openroad_results']['overall_ppa']['power_mw']
        
        if area > 1000 and power > 1.0:  # Large and power-hungry
            opportunities.append((
                OptimizationStrategy.BALANCE_AREA_POWER,
                {'area_threshold': 1000, 'power_threshold': 1.0}
            ))

        # Check for C++ Optimization Kernels availability
        if HAS_CPP_KERNELS:
            opportunities.append((
                OptimizationStrategy.GLOBAL_PLACEMENT,
                {'iterations': 100, 'threads': 4}
            ))
        
        return opportunities
    
    def _apply_strategy(self, rtl_content: str, strategy: OptimizationStrategy, params: Dict) -> str:
        """Apply a specific optimization strategy to RTL"""
        
        strategy_func = self.optimization_strategies.get(strategy)
        if strategy_func:
            return strategy_func(rtl_content, params)
        else:
            print(f"Unknown strategy: {strategy}")
            return rtl_content
    
    def _apply_pipelining(self, rtl_content: str, params: Dict) -> str:
        """Apply pipelining to critical paths using AST transformation"""
        print(f"Applying AST-based pipelining for parameters: {params}")
        
        # 1. Write RTL to temp file for the transformer
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as tmp:
            tmp.write(rtl_content)
            tmp_path = tmp.name
        
        try:
            # 2. Parse and Transform
            ast = self.rtl_transformer.parse_rtl(tmp_path)
            
            # Dynamically find the top module and a suitable target signal
            module_name = None
            if ast.description.definitions:
                for item in ast.description.definitions:
                    if isinstance(item, vast.ModuleDef):
                        module_name = item.name
                        break
            
            if not module_name:
                print(f"  Warning: No module found in AST for pipelining. Returning original.")
                return rtl_content

            # Try to find a suitable signal to pipeline. Heuristic: take the first wide wire/reg.
            target_signal = None
            # Iterate through the items of the dynamically found module
            target_module_def = None
            for item in ast.description.definitions:
                if isinstance(item, vast.ModuleDef) and item.name == module_name:
                    target_module_def = item
                    break

            if target_module_def:
                for item in target_module_def.items:
                    if isinstance(item, vast.Decl):
                        for decl_item in item.list:
                            if isinstance(decl_item, (vast.Wire, vast.Reg)):
                                # Heuristic: Prioritize signals with width for pipelining
                                if hasattr(decl_item, 'width') and decl_item.width is not None and \
                                    isinstance(decl_item.width, vast.Width):
                                    target_signal = decl_item.name
                                    break
                            # Fallback: if no wide signal, take the first simple wire/reg
                            elif isinstance(decl_item, (vast.Wire, vast.Reg)):
                                if not target_signal: # Only set if no wide signal found yet
                                    target_signal = decl_item.name
                        if target_signal: # Found a target, break outer loop
                            break

            if not target_signal:
                print(f"  Warning: No suitable signal found in module '{module_name}' for pipelining. Returning original.")
                return rtl_content
                
            print(f"  Dynamically selected module '{module_name}' and signal '{target_signal}' for pipelining.")

            try:
                ast, reg_name = self.rtl_transformer.add_pipeline_stage(ast, module_name, target_signal)
                
                # Robustly update all sinks of the original signal to use the new pipe reg
                ast = self.rtl_transformer.update_signal_sinks(ast, module_name, target_signal, reg_name)
                
                # 3. Generate new RTL
                optimized_rtl = self.rtl_transformer.generate_verilog(ast)
                return optimized_rtl
            except Exception as transform_err:
                print(f"AST Pipelining Transformation failed on '{module_name}.{target_signal}': {transform_err}. Returning original.")
                return rtl_content
                
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _apply_clock_gating(self, rtl_content: str, params: Dict) -> str:
        """Apply clock gating to reduce dynamic power using AST transformation"""
        print(f"Applying AST-based clock gating for parameters: {params}")
        
        # 1. Write RTL to temp file for the transformer
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as tmp:
            tmp.write(rtl_content)
            tmp_path = tmp.name
        
        try:
            # 2. Parse and Transform
            ast = self.rtl_transformer.parse_rtl(tmp_path)
            
            target_signal = params.get('target_signal', 'reg_data')
            module_name = params.get('module_name', 'top_module')
            enable_signal = params.get('enable_signal', 'en')
            
            try:
                ast, gated_clk = self.rtl_transformer.insert_clock_gate(
                    ast, module_name, target_signal, enable_signal
                )
                
                # 3. Generate new RTL
                optimized_rtl = self.rtl_transformer.generate_verilog(ast)
                return optimized_rtl
            except Exception as transform_err:
                print(f"AST Power Transformation failed: {transform_err}. Returning original.")
                return rtl_content
                
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def _apply_clustering(self, rtl_content: str, params: Dict) -> str:
        """Apply clustering using AST-based logic merging to reduce congestion"""
        print(f"Applying AST-based clustering/merging for parameters: {params}")
        
        # 1. Write RTL to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as tmp:
            tmp.write(rtl_content)
            tmp_path = tmp.name
            
        try:
            # 2. Parse and Transform
            ast = self.rtl_transformer.parse_rtl(tmp_path)
            
            module_name = params.get('module_name', 'top_module')
            
            try:
                ast, merged = self.rtl_transformer.apply_logic_merging(ast, module_name)
                print(f"  Merged {len(merged)} redundant signals: {merged}")
                
                # 3. Generate new RTL
                optimized_rtl = self.rtl_transformer.generate_verilog(ast)
                return optimized_rtl
            except Exception as transform_err:
                print(f"AST Clustering/Merging Transformation failed: {transform_err}. Returning original.")
                return rtl_content
                
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def _reduce_registers(self, rtl_content: str, params: Dict) -> str:
        """Attempt to reduce register count by removing back-to-back registers with no logic"""
        print(f"Applying register reduction for parameters: {params}")
        
        # This is a heuristic optimization. In a real flow, this would use 
        # retiming analysis. Here we do a simple regex-based 'back-to-back' removal.
        lines = rtl_content.split('\n')
        new_lines = []
        reg_map = {} # old_reg -> next_reg
        
        # Look for: always @(posedge clk) next_reg <= old_reg;
        for i, line in enumerate(lines):
            if '<=' in line:
                parts = line.split('<=')
                lhs = parts[0].strip()
                rhs = parts[1].strip().rstrip(';')
                if lhs.endswith('_pipe_reg') and rhs.endswith('_pipe_reg'):
                    print(f"  Found redundant register chain: {rhs} -> {lhs}. Collapsing.")
                    reg_map[lhs] = rhs
                    continue
            new_lines.append(line)
            
        # Replace usages of collapsed registers
        final_content = '\n'.join(new_lines)
        for lhs, rhs in reg_map.items():
            final_content = final_content.replace(lhs, rhs)
            
        return final_content
    
    def _optimize_fanout(self, rtl_content: str, params: Dict) -> str:
        """Optimize high fanout nets using AST-based buffering"""
        print(f"Applying AST-based fanout optimization for parameters: {params}")
        
        # 1. Write RTL to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as tmp:
            tmp.write(rtl_content)
            tmp_path = tmp.name
            
        try:
            # 2. Parse and Transform
            ast = self.rtl_transformer.parse_rtl(tmp_path)
            
            target_signal = params.get('target_signal', 'clk') # Default to clk usually
            module_name = params.get('module_name', 'top_module')
            # Determine split degree based on fanout. Simple heuristic: 1 buffer per 4 loads.
            degree = int(params.get('current_avg', 8.0) / 4.0)
            degree = max(2, min(degree, 8)) # Clamp between 2 and 8
            
            try:
                ast, buffers = self.rtl_transformer.apply_fanout_buffering(
                    ast, module_name, target_signal, degree
                )
                print(f"  Buffered {target_signal} into {len(buffers)} branches: {buffers}")
                
                # 3. Generate new RTL
                optimized_rtl = self.rtl_transformer.generate_verilog(ast)
                return optimized_rtl
            except Exception as transform_err:
                print(f"AST Fanout Transformation failed: {transform_err}. Returning original.")
                return rtl_content
                
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def _balance_area_power(self, rtl_content: str, params: Dict) -> str:
        """Balance area and power using input isolation on combinational logic"""
        print(f"Applying AST-based input isolation for parameters: {params}")
        
        # 1. Write RTL to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as tmp:
            tmp.write(rtl_content)
            tmp_path = tmp.name
            
        try:
            # 2. Parse and Transform
            ast = self.rtl_transformer.parse_rtl(tmp_path)
            
            # Identify target signals. If not provided, we look for 'data' related inputs
            module_name = params.get('module_name', 'top_module')
            target_signals = params.get('targets', ['data_in']) 
            enable_signal = params.get('enable_signal', 'en')
            
            try:
                ast, isolated = self.rtl_transformer.apply_input_isolation(
                    ast, module_name, target_signals, enable_signal
                )
                print(f"  Isolated signals {target_signals} using control {enable_signal}")
                
                # 3. Generate new RTL
                optimized_rtl = self.rtl_transformer.generate_verilog(ast)
                return optimized_rtl
            except Exception as transform_err:
                print(f"AST Isolation Transformation failed: {transform_err}. Returning original.")
                return rtl_content
                
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    
    def _apply_global_placement(self, rtl_content: str, params: Dict) -> str:
        """Apply Global Placement using C++ Optimization Kernels steered by RL"""
        if not HAS_CPP_KERNELS:
            print("C++ Optimization Kernels not available. Skipping Global Placement.")
            return rtl_content

        print(f"Applying [RL-Steered] C++ Global Placement with parameters: {params}")
        
        # 1. Create a C++ Graph Engine instance
        engine = sic.GraphEngine()
        # In a real flow, this would be populated from the physical analysis
        # For demo, we add a few virtual nodes to optimize
        for i in range(10):
            name = f"u{i}"
            engine.add_node(name, sic.NodeAttributes())
            if i > 0:
                engine.add_edge(f"u{i-1}", name, sic.EdgeAttributes())
        
        # 2. Steer with RL if in Advanced mode
        iterations = params.get('iterations', 100)
        if hasattr(self, 'rl_agent'):
            print("  Steering placement using RL Agent...")
            # RL agent selects strategy based on design 'wisdom'
            # (In a full implementation, we'd pass a real state vector)
            state = [0.5] * 7 # Placeholder state
            action_idx, _ = self.rl_agent.select_action(state)
            if action_idx % 2 == 0:
                iterations = int(iterations * 1.5)
                print(f"  RL Decision: Increase placement effort. Iterations -> {iterations}")
        
        # 3. Configure Placement
        config = sic.PlacementConfig()
        config.iterations = iterations
        config.threads = params.get('threads', 4)
        config.area_width = 1000.0
        config.area_height = 1000.0
        
        # 4. Run Optimization
        try:
            optimizer = sic.OptimizationKernels(engine)
            optimizer.run_global_placement(config)
            
            # Phase 5: Explicit Force-Directed refinement
            optimizer.apply_forces(50.0, 50.0) 
            
            hpwl = optimizer.calculate_hpwl()
            print(f"  Placement complete. Final HPWL: {hpwl}")
            
            return rtl_content
        except Exception as e:
            print(f"Global Placement failed: {e}")
            return rtl_content

    def _calculate_improvement(self, initial_analysis: Dict, final_analysis: Dict) -> Dict:
        """Calculate improvement metrics"""
        
        initial_ppa = initial_analysis['openroad_results']['overall_ppa']
        final_ppa = final_analysis['openroad_results']['overall_ppa']
        
        improvement = {
            'area_improvement_pct': ((initial_ppa['area_um2'] - final_ppa['area_um2']) / initial_ppa['area_um2']) * 100 if initial_ppa['area_um2'] > 0 else 0,
            'power_improvement_pct': ((initial_ppa['power_mw'] - final_ppa['power_mw']) / initial_ppa['power_mw']) * 100 if initial_ppa['power_mw'] > 0 else 0,
            'timing_improvement_pct': ((initial_ppa['timing_ns'] - final_ppa['timing_ns']) / initial_ppa['timing_ns']) * 100 if initial_ppa['timing_ns'] > 0 else 0,
            'drc_improvement': initial_analysis['openroad_results']['routing']['drc_violations'] - final_analysis['openroad_results']['routing']['drc_violations']
        }
        
        return improvement


from ml_prediction_models import DesignPPAPredictor

class AdvancedAutonomousOptimizer(AutonomousOptimizer):
    """
    Advanced version that uses ML predictions to guide optimizations
    """
    
    def __init__(self):
        super().__init__()
        self.predictor = DesignPPAPredictor()
        # Try to load pretrained models if available
        try:
            if os.path.exists("design_ppa_predictor.pkl"):
                self.predictor.load_model("design_ppa_predictor.pkl")
                print("Loaded pretrained PPA predictor models")
            else:
                print("No pretrained models found, initializing fresh predictor")
        except Exception as e:
            print(f"Failed to load predictor models: {e}")
            
        # RL Initialization
        self.rl_agent = PolicyAgent(state_dim=7, action_dim=6)
        self.ca_logger = CreditAssignmentLogger()
        self.design_memory = DesignMemory()

    def run_rl_optimization_loop(self, rtl_content: str, design_name: str = "industrial") -> Dict[str, Any]:
        """
        Runs a complete Reinforcement Learning episode to discover optimal transformation sequences.
        This provides 'Design Wisdom' by learning from credit assignment.
        """
        print(f"\n[RL-EDA] Starting Optimization Episode for '{design_name}'...")
        
        # 1. Initialize Environment
        env = DesignOptimizationEnv(rtl_content, self, design_name)
        state = env.reset()
        
        episode_reward = 0
        steps_taken = []
        
        # 2. Iterate Optimization Steps
        for step in range(env.max_steps):
            # Select action via Policy Agent
            action_idx, log_prob = self.rl_agent.select_action(state)
            
            # Step inside environment
            next_state, reward, done, info = env.step(action_idx)
            
            # Record causal impact (Credit Assignment)
            self.ca_logger.log_step(
                info.get('strategy', 'unknown'),
                env.last_analysis['openroad_results']['overall_ppa'],
                env.history[-1]['openroad_results']['overall_ppa'] if len(env.history) > 1 else env.last_analysis['openroad_results']['overall_ppa'],
                reward
            )
            
            # Save to experience replay
            self.rl_agent.save_to_memory(state, log_prob, reward, next_state, done)
            
            # Persist in Design Memory
            self.design_memory.remember_optimization(
                env.current_rtl, 
                info.get('strategy', 'unknown'),
                env.last_analysis['openroad_results']['overall_ppa']
            )
            
            state = next_state
            episode_reward += reward
            steps_taken.append(info.get('strategy'))
            
            if done:
                break
        
        # 3. Learn from Design Experience
        print(f"[RL-EDA] Episode Complete. Total Reward: {episode_reward:.3f}")
        self.rl_agent.learn()
        
        # 4. Finalize
        improvement = self._calculate_improvement(env.baseline_analysis, env.last_analysis)
        self.ca_logger.finalize_episode(design_name, improvement)
        
        return {
            'design_name': design_name,
            'initial_ppa': env.baseline_analysis['openroad_results']['overall_ppa'],
            'final_ppa': env.last_analysis['openroad_results']['overall_ppa'],
            'improvement': improvement,
            'sequence': steps_taken,
            'reward': episode_reward
        }
            
    def predict_optimization_impact(self, rtl_content: str, strategy: OptimizationStrategy, params: Dict) -> Dict[str, float]:
        """Predict the impact of applying an optimization using ML models"""
        
        # 1. Analyze baseline (if not already cached/passed, currently we analyze from scratch for simplicity)
        # In a production system, we'd pass the initial analysis object to avoid re-work
        baseline_analysis = self.design_intelligence.analyze_design(rtl_content, "baseline")
        
        # 2. Get baseline predictions
        baseline_features = baseline_analysis.get('physical_ir_stats', {}) # Using stats as features proxy
        # The predictor expects features as a dictionary. 
        # Ideally we'd use system.get_learning_dataset logic to extract consistent features
        # For now, we rely on the predictor's robustness or lack thereof if features missing
        
        # Note: If predictor is not trained, these will be random/default
        if self.predictor.is_trained:
            baseline_preds = self.predictor.predict(baseline_features)
        else:
            # Fallback to heuristic if model not trained
            return self._heuristic_prediction(strategy, params)

        # 3. Apply optimization to get candidate RTL
        candidate_rtl = self._apply_strategy(rtl_content, strategy, params)
        
        # 4. Analyze candidate
        candidate_analysis = self.design_intelligence.analyze_design(candidate_rtl, "candidate")
        candidate_features = candidate_analysis.get('physical_ir_stats', {})
        
        # 5. Get candidate predictions
        candidate_preds = self.predictor.predict(candidate_features)
        
        # 6. Calculate deltas (Impact = Candidate - Baseline)
        # For area/power/timing, lower is usually better, but 'improvement' usually means reduction.
        # Here we return % change. Negative means reduction (good for area/power).
        
        impact = {}
        confidence = 0.8 # ML models usually have higher confidence than simple heuristics
        
        for metric in ['area', 'power', 'timing']:
            base_val = baseline_preds.get(metric, 0.0)
            cand_val = candidate_preds.get(metric, 0.0)
            
            if base_val > 0:
                pct_change = ((cand_val - base_val) / base_val) * 100
            else:
                pct_change = 0.0
                
            impact[f'predicted_{metric}_change_pct'] = pct_change
            
        impact['confidence'] = confidence
        
        # Add specific metric improvements if relevant
        impact['predicted_timing_improvement'] = max(0, baseline_preds.get('timing', 0) - candidate_preds.get('timing', 0))
        
        return impact

    def _heuristic_prediction(self, strategy: OptimizationStrategy, params: Dict) -> Dict[str, float]:
        """Fallback heuristic prediction when ML models are not available"""
        if strategy == OptimizationStrategy.PIPELINE_CRITICAL_PATHS:
            return {
                'predicted_area_change_pct': 2.0, # Registers add area
                'predicted_power_change_pct': 1.0, 
                'predicted_timing_improvement': 0.15,
                'confidence': 0.6
            }
        elif strategy == OptimizationStrategy.CLOCK_GATING:
            return {
                'predicted_area_change_pct': 1.0, # Gating logic adds area
                'predicted_power_change_pct': -15.0, # Big power win
                'predicted_timing_improvement': -0.05, # Slight timing penalty
                'confidence': 0.7
            }
        
        # Default fallback
        return {
            'predicted_area_change_pct': 0.0,
            'predicted_power_change_pct': 0.0,
            'predicted_timing_improvement': 0.0,
            'confidence': 0.1
        }
    
    def optimize_with_prediction_guidance(self, rtl_content: str, design_name: str = "unnamed") -> Dict[str, Any]:
        """Optimize using ML predictions to guide optimization selection"""
        
        # Analyze initial design
        initial_analysis = self.design_intelligence.analyze_design(rtl_content, design_name)
        
        # Identify possible optimizations
        opportunities = self._identify_optimizations(initial_analysis)
        
        # Predict impact of each optimization
        scored_opportunities = []
        for strategy, params in opportunities:
            # Note: We pass rtl_content, but for sequential optimizations we should pass the *current* state.
            # For simplicity in this demo, we evaluate each against the base state independently.
            predicted_impact = self.predict_optimization_impact(rtl_content, strategy, params)
            score = self._score_optimization(predicted_impact)
            scored_opportunities.append((strategy, params, predicted_impact, score))
        
        # Sort by score (higher is better)
        scored_opportunities.sort(key=lambda x: x[3], reverse=True)
        
        # Apply optimizations in order of predicted benefit
        optimized_rtl = rtl_content
        applied_optimizations = []
        
        for strategy, params, predicted_impact, score in scored_opportunities:
            if score > 0.05:  # Only apply if predicted positive impact (threshold 0.05)
                # Apply to the CUMULATIVE rtl
                new_rtl = self._apply_strategy(optimized_rtl, strategy, params)
                if new_rtl != optimized_rtl:
                    optimized_rtl = new_rtl
                    applied_optimizations.append({
                        'strategy': strategy.value,
                        'params': params,
                        'predicted_impact': predicted_impact,
                        'score': score
                    })
        
        # Final analysis
        final_analysis = self.design_intelligence.analyze_design(optimized_rtl, f"optimized_{design_name}")
        
        results = {
            'initial_analysis': initial_analysis,
            'predicted_optimizations': [
                {'strategy': s.value, 'params': p, 'impact': i, 'score': sc} 
                for s, p, i, sc in scored_opportunities
            ],
            'applied_optimizations': applied_optimizations,
            'final_analysis': final_analysis,
            'improvement': self._calculate_improvement(initial_analysis, final_analysis)
        }
        
        return results
    
    def _score_optimization(self, predicted_impact: Dict[str, float]) -> float:
        """Score an optimization based on predicted impact"""
        # Weight different metrics (this would be learned in practice)
        # Negative change in Area/Power is GOOD. Positive timing improvement is GOOD.
        
        area_weight = 0.4
        power_weight = 0.3
        timing_weight = 0.3
        
        area_change = predicted_impact.get('predicted_area_change_pct', 0)
        power_change = predicted_impact.get('predicted_power_change_pct', 0)
        timing_improvement = predicted_impact.get('predicted_timing_improvement', 0)
        
        # Score calculation: 
        # -Area% means improvement (if negative) -> add (-area * weight)
        # -Power% means improvement (if negative) -> add (-power * weight)
        # Timing improvement is absolute value in logic above? No, it's improvement.
        
        score = (
            area_weight * (-area_change) +
            power_weight * (-power_change) +
            timing_weight * (timing_improvement * 100) # Scale timing to be comparable to %
        )
        
        # Adjust for confidence
        confidence = predicted_impact.get('confidence', 0.5)
        score *= confidence
        
        return score


def test_autonomous_optimizer():
    """Test the autonomous optimizer"""
    print("Testing Autonomous Optimizer...")
    
    optimizer = AdvancedAutonomousOptimizer()
    
    # Test with a simple design
    test_rtl = '''
module test_adder (
    input [7:0] a,
    input [7:0] b,
    output [8:0] sum
);
    assign sum = a + b;
endmodule
    '''
    
    print("Original RTL:")
    print(test_rtl)
    
    # Apply optimization
    results = optimizer.optimize_with_prediction_guidance(test_rtl, "test_adder")
    
    print(f"\nInitial Area: {results['initial_analysis']['openroad_results']['overall_ppa']['area_um2']:.2f}")
    print(f"Final Area: {results['final_analysis']['openroad_results']['overall_ppa']['area_um2']:.2f}")
    print(f"Improvement: {results['improvement']['area_improvement_pct']:.2f}%")
    
    print(f"\nApplied {len(results['applied_optimizations'])} optimizations:")
    for opt in results['applied_optimizations']:
        print(f"  - {opt['strategy']}: Score={opt['score']:.3f}")
    
    return optimizer


if __name__ == "__main__":
    optimizer = test_autonomous_optimizer()