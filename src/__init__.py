"""HybridFraudDetection: fraud rules with a machine learning layer on top.

This package holds one class per step of the fraud check:
DataLoader (with DataQualityChecker, DataProfiler and SplitChecker),
InvestigationLabeler, FeatureBuilder, RuleEngine, RiskModel, DecisionPolicy,
Evaluator and FraudPipeline, which runs them in order.
"""
