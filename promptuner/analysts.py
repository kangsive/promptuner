"""
Analyst implementations for the prompt optimization framework.

This module provides concrete implementations of analysts for analyzing
evaluation results and providing feedback for prompt improvement.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import Counter

from .base import Analyst, Generator

logger = logging.getLogger(__name__)


class LLMAnalyst(Analyst):
    """
    Analyst that uses an LLM to analyze evaluation results and provide feedback.
    
    This analyst examines low-scoring samples, identifies patterns, and provides
    structured feedback for improving prompts.
    """
    
    def __init__(self, generator: Generator, low_score_threshold: float = 0.3):
        """
        Initialize LLM analyst.
        
        Args:
            generator: Text generator for analysis.
            low_score_threshold: Threshold below which samples are considered low-scoring.
        """
        self.generator = generator
        self.low_score_threshold = low_score_threshold
    
    def run(self, inputs: List[str], references: List[str], 
            generated: List[str], scores: List[float], 
            overall_score: float, current_prompt: str,
            ancestor_verified_hypotheses: Optional[List[str]] = None,
            ancestor_false_hypotheses: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze evaluation results using LLM.
        
        Args:
            inputs: List of input prompts.
            references: List of reference/expected outputs.
            generated: List of generated outputs.
            scores: List of individual sample scores.
            overall_score: Overall evaluation score.
            current_prompt: The prompt that generated these outputs.
            ancestor_verified_hypotheses: List of hypotheses verified by ancestors.
            ancestor_false_hypotheses: List of hypotheses falsified by ancestors.
            
        Returns:
            Analysis results with feedback for prompt improvement.
        """
        logger.info(f"Analyzing evaluation results (overall score: {overall_score:.4f})")
        
        # Identify low-scoring samples
        low_score_samples = self._identify_low_score_samples(
            inputs, references, generated, scores
        )
        
        if not low_score_samples:
            return {
                'summary': f"Performance is good (score: {overall_score:.4f}). No significant issues found.",
                'low_score_analysis': "No low-scoring samples identified.",
                'patterns': [],
                'improvement_suggestions': [],
                'hypothesis': "The prompt is performing well on the given dataset.",
                'new_hypothesis': None
            }
        
        # Generate analysis using LLM
        analysis_prompt = self._create_analysis_prompt(
            current_prompt, low_score_samples, overall_score,
            ancestor_verified_hypotheses, ancestor_false_hypotheses
        )
        
        try:
            analysis_response = self.generator.run(analysis_prompt)
            analysis_result = self._parse_analysis_response(analysis_response)
            
            # Add statistical information
            analysis_result['statistics'] = self._calculate_statistics(scores)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error during LLM analysis: {e}")
            return self._create_fallback_analysis(low_score_samples, overall_score)
    
    def _identify_low_score_samples(self, inputs: List[str], references: List[str], 
                                   generated: List[str], scores: List[float]) -> List[Dict[str, Any]]:
        """Identify samples with low scores."""
        low_score_samples = []
        
        for i, score in enumerate(scores):
            if score < self.low_score_threshold:
                low_score_samples.append({
                    'index': i,
                    'input': inputs[i],
                    'reference': references[i],
                    'generated': generated[i],
                    'score': score
                })
        
        # Sort by score (lowest first)
        low_score_samples.sort(key=lambda x: x['score'])
        
        # Limit to top 10 worst samples to avoid overwhelming the LLM
        return low_score_samples[:10]
    
    def _create_analysis_prompt(self, current_prompt: str, 
                               low_score_samples: List[Dict[str, Any]], 
                               overall_score: float,
                               ancestor_verified_hypotheses: Optional[List[str]] = None,
                               ancestor_false_hypotheses: Optional[List[str]] = None) -> str:
        """Create prompt for analyzing evaluation results."""
        samples_text = "\n".join([
            f"Sample {i+1} (Score: {sample['score']:.3f}):\n"
            f"  Input: {sample['input']}\n"
            f"  Expected: {sample['reference']}\n"
            f"  Generated: {sample['generated']}\n"
            for i, sample in enumerate(low_score_samples[:5])  # Show top 5 worst
        ])
        
        # Add ancestor hypothesis context
        hypothesis_context = ""
        if ancestor_verified_hypotheses:
            hypothesis_context += f"\nVERIFIED HYPOTHESES FROM ANCESTORS:\n"
            hypothesis_context += "\n".join([f"- {hyp}" for hyp in ancestor_verified_hypotheses])
        
        if ancestor_false_hypotheses:
            hypothesis_context += f"\nFALSE HYPOTHESES FROM ANCESTORS (AVOID THESE):\n"
            hypothesis_context += "\n".join([f"- {hyp}" for hyp in ancestor_false_hypotheses])
        
        analysis_prompt = f"""
You are an expert prompt engineer analyzing why a prompt is performing poorly. 

CURRENT PROMPT:
{current_prompt}

OVERALL SCORE: {overall_score:.4f}

LOW-SCORING SAMPLES:
{samples_text}
{hypothesis_context}

Please analyze these results and provide structured feedback:

1. SUMMARY: A brief summary of the overall performance and main issues.

2. LOW_SCORE_ANALYSIS: Detailed analysis of why these samples scored low. What patterns do you see in the failures?

3. PATTERNS: List specific patterns in the problematic data (e.g., certain types of inputs, output formats, content issues).

4. IMPROVEMENT_SUGGESTIONS: Specific, actionable suggestions for improving the prompt. Consider the verified hypotheses and avoid the false ones.

5. HYPOTHESIS: Your hypothesis about the root cause of poor performance and how to address it.

6. NEW_HYPOTHESIS: A new specific hypothesis for children prompts to test and verify. This should be different from ancestors' hypotheses and address the current issues.

Please provide clear, actionable feedback that can be used to improve the prompt. Focus on specific issues you can identify from the examples.

Format your response as:
SUMMARY: [your summary]
LOW_SCORE_ANALYSIS: [your analysis]
PATTERNS: [bullet points of patterns]
IMPROVEMENT_SUGGESTIONS: [bullet points of suggestions]
HYPOTHESIS: [your hypothesis]
NEW_HYPOTHESIS: [your new hypothesis for children to test]
"""
        
        return analysis_prompt.strip()
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM analysis response."""
        sections = {
            'summary': '',
            'low_score_analysis': '',
            'patterns': [],
            'improvement_suggestions': [],
            'hypothesis': '',
            'new_hypothesis': None
        }
        
        lines = response.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            if line.startswith('SUMMARY:'):
                current_section = 'summary'
                sections['summary'] = line.replace('SUMMARY:', '').strip()
            elif line.startswith('LOW_SCORE_ANALYSIS:'):
                current_section = 'low_score_analysis'
                sections['low_score_analysis'] = line.replace('LOW_SCORE_ANALYSIS:', '').strip()
            elif line.startswith('PATTERNS:'):
                current_section = 'patterns'
                continue
            elif line.startswith('IMPROVEMENT_SUGGESTIONS:'):
                current_section = 'improvement_suggestions'
                continue
            elif line.startswith('HYPOTHESIS:'):
                current_section = 'hypothesis'
                sections['hypothesis'] = line.replace('HYPOTHESIS:', '').strip()
            elif line.startswith('NEW_HYPOTHESIS:'):
                current_section = 'new_hypothesis'
                sections['new_hypothesis'] = line.replace('NEW_HYPOTHESIS:', '').strip()
            elif current_section == 'patterns' and line.startswith(('-', '•', '*')):
                sections['patterns'].append(line[1:].strip())
            elif current_section == 'improvement_suggestions' and line.startswith(('-', '•', '*')):
                sections['improvement_suggestions'].append(line[1:].strip())
            elif current_section in sections and isinstance(sections[current_section], str):
                sections[current_section] += ' ' + line
        
        return sections
    
    def _calculate_statistics(self, scores: List[float]) -> Dict[str, float]:
        """Calculate statistical information about scores."""
        if not scores:
            return {}
        
        scores_array = np.array(scores)
        return {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
            'low_score_count': int(np.sum(scores_array < self.low_score_threshold)),
            'low_score_percentage': float(np.mean(scores_array < self.low_score_threshold) * 100)
        }
    
    def _create_fallback_analysis(self, low_score_samples: List[Dict[str, Any]], 
                                 overall_score: float) -> Dict[str, Any]:
        """Create fallback analysis when LLM analysis fails."""
        return {
            'summary': f"Performance is low (score: {overall_score:.4f}). {len(low_score_samples)} samples scored below threshold.",
            'low_score_analysis': f"Found {len(low_score_samples)} samples with scores below {self.low_score_threshold}.",
            'patterns': ["Unable to analyze patterns due to LLM error"],
            'improvement_suggestions': ["Review low-scoring samples manually", "Consider adjusting prompt structure"],
            'hypothesis': "Analysis failed - manual review needed.",
            'new_hypothesis': "Manual review needed to generate hypothesis.",
            'statistics': self._calculate_statistics([s['score'] for s in low_score_samples])
        }


class RuleBasedAnalyst(Analyst):
    """
    Rule-based analyst that uses predefined rules to analyze evaluation results.
    
    This analyst provides basic analysis without requiring an LLM.
    """
    
    def __init__(self, low_score_threshold: float = 0.3):
        """
        Initialize rule-based analyst.
        
        Args:
            low_score_threshold: Threshold below which samples are considered low-scoring.
        """
        self.low_score_threshold = low_score_threshold
    
    def run(self, inputs: List[str], references: List[str], 
            generated: List[str], scores: List[float], 
            overall_score: float, current_prompt: str,
            ancestor_verified_hypotheses: Optional[List[str]] = None,
            ancestor_false_hypotheses: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze evaluation results using rule-based approach.
        
        Args:
            inputs: List of input prompts.
            references: List of reference/expected outputs.
            generated: List of generated outputs.
            scores: List of individual sample scores.
            overall_score: Overall evaluation score.
            current_prompt: The prompt that generated these outputs.
            ancestor_verified_hypotheses: List of hypotheses verified by ancestors.
            ancestor_false_hypotheses: List of hypotheses falsified by ancestors.
            
        Returns:
            Analysis results with feedback for prompt improvement.
        """
        logger.info(f"Analyzing evaluation results using rules (overall score: {overall_score:.4f})")
        
        # Calculate statistics
        statistics = self._calculate_statistics(scores)
        
        # Identify low-scoring samples
        low_score_indices = [i for i, score in enumerate(scores) if score < self.low_score_threshold]
        
        # Analyze patterns
        patterns = self._analyze_patterns(inputs, references, generated, low_score_indices)
        
        # Generate suggestions considering ancestor hypotheses
        suggestions = self._generate_suggestions(patterns, statistics, 
                                               ancestor_verified_hypotheses, 
                                               ancestor_false_hypotheses)
        
        # Create summary
        summary = self._create_summary(overall_score, statistics, len(low_score_indices))
        
        # Generate new hypothesis
        new_hypothesis = self._generate_new_hypothesis(patterns, statistics, 
                                                     ancestor_verified_hypotheses, 
                                                     ancestor_false_hypotheses)
        
        return {
            'summary': summary,
            'low_score_analysis': self._create_low_score_analysis(low_score_indices, statistics),
            'patterns': patterns,
            'improvement_suggestions': suggestions,
            'hypothesis': self._create_hypothesis(patterns, statistics),
            'new_hypothesis': new_hypothesis,
            'statistics': statistics
        }
    
    def _calculate_statistics(self, scores: List[float]) -> Dict[str, float]:
        """Calculate statistical information about scores."""
        if not scores:
            return {}
        
        scores_array = np.array(scores)
        return {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
            'low_score_count': int(np.sum(scores_array < self.low_score_threshold)),
            'low_score_percentage': float(np.mean(scores_array < self.low_score_threshold) * 100)
        }
    
    def _analyze_patterns(self, inputs: List[str], references: List[str], 
                         generated: List[str], low_score_indices: List[int]) -> List[str]:
        """Analyze patterns in low-scoring samples."""
        patterns = []
        
        if not low_score_indices:
            return patterns
        
        # Analyze input lengths
        low_score_input_lengths = [len(inputs[i]) for i in low_score_indices]
        all_input_lengths = [len(inp) for inp in inputs]
        
        if np.mean(low_score_input_lengths) > np.mean(all_input_lengths) * 1.2:
            patterns.append("Long inputs tend to have lower scores")
        elif np.mean(low_score_input_lengths) < np.mean(all_input_lengths) * 0.8:
            patterns.append("Short inputs tend to have lower scores")
        
        # Analyze output lengths
        low_score_output_lengths = [len(generated[i]) for i in low_score_indices]
        all_output_lengths = [len(gen) for gen in generated]
        
        if np.mean(low_score_output_lengths) > np.mean(all_output_lengths) * 1.2:
            patterns.append("Generated outputs are too long for low-scoring samples")
        elif np.mean(low_score_output_lengths) < np.mean(all_output_lengths) * 0.8:
            patterns.append("Generated outputs are too short for low-scoring samples")
        
        # Analyze common words in low-scoring samples
        low_score_words = []
        for i in low_score_indices:
            low_score_words.extend(inputs[i].lower().split())
        
        if low_score_words:
            word_counts = Counter(low_score_words)
            common_words = [word for word, count in word_counts.most_common(5) if count > 1]
            if common_words:
                patterns.append(f"Common words in low-scoring inputs: {', '.join(common_words)}")
        
        return patterns
    
    def _generate_suggestions(self, patterns: List[str], statistics: Dict[str, float],
                             ancestor_verified_hypotheses: Optional[List[str]] = None,
                             ancestor_false_hypotheses: Optional[List[str]] = None) -> List[str]:
        """Generate improvement suggestions based on patterns."""
        suggestions = []
        
        if statistics.get('low_score_percentage', 0) > 30:
            suggestions.append("Consider revising the prompt structure as many samples are underperforming")
        
        if statistics.get('std', 0) > 0.3:
            suggestions.append("High score variance suggests inconsistent performance - add more specific instructions")
        
        for pattern in patterns:
            if "long" in pattern.lower() and "input" in pattern.lower():
                suggestions.append("Add instructions for handling long inputs")
            elif "short" in pattern.lower() and "input" in pattern.lower():
                suggestions.append("Add instructions for handling short inputs")
            elif "too long" in pattern.lower() and "output" in pattern.lower():
                suggestions.append("Add length constraints to the prompt")
            elif "too short" in pattern.lower() and "output" in pattern.lower():
                suggestions.append("Encourage more detailed responses in the prompt")
        
        # Add suggestions based on verified hypotheses
        if ancestor_verified_hypotheses:
            suggestions.append("Continue building on proven strategies from ancestor hypotheses")
        
        # Avoid suggestions that conflict with false hypotheses
        if ancestor_false_hypotheses:
            suggestions.append("Avoid approaches that have been proven ineffective by ancestor nodes")
        
        if not suggestions:
            suggestions.append("Consider adding more specific examples or constraints to the prompt")
        
        return suggestions
    
    def _create_summary(self, overall_score: float, statistics: Dict[str, float], 
                       low_score_count: int) -> str:
        """Create summary of analysis."""
        performance_level = "good" if overall_score > 0.7 else "moderate" if overall_score > 0.4 else "poor"
        
        return (f"Overall performance is {performance_level} (score: {overall_score:.4f}). "
                f"{low_score_count} out of {statistics.get('low_score_count', 0) + (len(statistics) - low_score_count if statistics else 0)} samples scored below threshold.")
    
    def _create_low_score_analysis(self, low_score_indices: List[int], 
                                  statistics: Dict[str, float]) -> str:
        """Create analysis of low-scoring samples."""
        if not low_score_indices:
            return "No samples scored below the threshold."
        
        return (f"{len(low_score_indices)} samples scored below {self.low_score_threshold}. "
                f"This represents {statistics.get('low_score_percentage', 0):.1f}% of all samples.")
    
    def _create_hypothesis(self, patterns: List[str], statistics: Dict[str, float]) -> str:
        """Create hypothesis about performance issues."""
        if not patterns:
            return "No clear patterns identified in the data."
        
        return (f"Performance issues may be related to: {', '.join(patterns[:3])}. "
                f"Consider addressing these patterns to improve overall performance.") 
    
    def _generate_new_hypothesis(self, patterns: List[str], statistics: Dict[str, float],
                                ancestor_verified_hypotheses: Optional[List[str]] = None,
                                ancestor_false_hypotheses: Optional[List[str]] = None) -> Optional[str]:
        """Generate a new hypothesis for children to test."""
        if not patterns and statistics.get('low_score_percentage', 0) < 20:
            return None  # Only return None if there are no patterns AND performance is good
        
        # Create a specific hypothesis based on the most significant pattern
        primary_pattern = patterns[0] if patterns else None
        
        if primary_pattern:
            if "long" in primary_pattern.lower() and "input" in primary_pattern.lower():
                hypothesis = "Adding specific instructions for handling long inputs will improve performance on complex samples"
            elif "short" in primary_pattern.lower() and "input" in primary_pattern.lower():
                hypothesis = "Adding more detailed prompting will improve performance on simple samples"
            elif "too long" in primary_pattern.lower() and "output" in primary_pattern.lower():
                hypothesis = "Adding length constraints will improve output quality and relevance"
            elif "too short" in primary_pattern.lower() and "output" in primary_pattern.lower():
                hypothesis = "Encouraging more detailed responses will improve output completeness"
            else:
                hypothesis = f"Addressing the pattern '{primary_pattern}' will improve overall performance"
        else:
            # Generate a general hypothesis based on performance statistics
            low_score_percentage = statistics.get('low_score_percentage', 0)
            if low_score_percentage > 30:
                hypothesis = "Adding more specific instructions and examples will improve consistency and accuracy"
            else:
                hypothesis = "Refining prompt structure and clarity will enhance overall performance"
        
        # Ensure the hypothesis doesn't conflict with ancestor false hypotheses
        if ancestor_false_hypotheses:
            for false_hyp in ancestor_false_hypotheses:
                if false_hyp.lower() in hypothesis.lower():
                    return None  # Avoid generating conflicting hypothesis
        
        return hypothesis 