# silicon_intelligence/validation/validation_pipeline.py

from typing import Dict, Any, List
from silicon_intelligence.validation.ground_truth_generator import GroundTruthGenerator


class ValidationPipeline:
    def __init__(self):
        self.ground_truth_generator = GroundTruthGenerator()
    
    def run_validation_cycle(self, design_names: List[str]) -> Dict[str, Any]:
        """Run a complete validation cycle"""
        print(f"Starting validation cycle for {len(design_names)} designs")
        
        # Generate ground truth for all designs
        ground_truths = self.ground_truth_generator.batch_generate_ground_truth(design_names)
        
        if not ground_truths:
            print("No ground truths generated, exiting validation cycle")
            return {}
        
        # Validate predictions
        validation_results = self.ground_truth_generator.validate_predictions(ground_truths)
        
        # Generate insights
        insights = self._generate_insights(ground_truths, validation_results)
        
        result = {
            'ground_truths': ground_truths,
            'validation_results': validation_results,
            'insights': insights,
            'design_count': len(design_names),
            'successful_validations': len(ground_truths)
        }
        
        print(f"Completed validation cycle for {len(ground_truths)} designs")
        return result
    
    def _generate_insights(self, ground_truths: List[Dict], validation_results: Dict) -> Dict[str, Any]:
        """Generate insights from validation results"""
        insights = {
            'total_designs': len(ground_truths),
            'accuracy_summary': validation_results,
            'best_predictions': self._find_best_predictions(ground_truths),
            'worst_predictions': self._find_worst_predictions(ground_truths),
            'recommendations': self._generate_recommendations(validation_results)
        }
        
        return insights
    
    def _find_best_predictions(self, ground_truths: List[Dict]) -> List[Dict]:
        """Find designs with best prediction accuracy"""
        # For now, just return the first few
        return ground_truths[:min(3, len(ground_truths))]
    
    def _find_worst_predictions(self, ground_truths: List[Dict]) -> List[Dict]:
        """Find designs with worst prediction accuracy"""
        # For now, just return the first few
        return ground_truths[:min(3, len(ground_truths))]
    
    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Check if accuracy is low for any metrics
        for metric in ['area', 'power', 'timing', 'drc']:
            accuracy_key = f'{metric}_accuracy'
            if accuracy_key in validation_results:
                accuracy = validation_results[accuracy_key]
                if accuracy < 0.7:  # Less than 70% accuracy
                    recommendations.append(
                        f"Model accuracy for {metric} is low ({accuracy:.2f}), "
                        f"consider retraining with more {metric}-specific data"
                    )
        
        if not recommendations:
            recommendations.append("All prediction accuracies are acceptable (>70%)")
        
        return recommendations