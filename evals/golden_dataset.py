"""Hand-authored golden Q&A cases for the ML/deep-learning RAG corpus."""
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    question: str
    golden_answer: str
    source_files: Tuple[str, ...]
    source_sections: Tuple[str, ...]
    category: str
    answerable: bool

GOLDEN_CASES = [
    GoldenCase('ml01', 'What is the difference between an ML algorithm and a model?', 'An algorithm is the procedure used to fit a model; the model is the learned mapping produced by that procedure.', ('machine_learning_foundations.md',), ('What machine learning does',), 'lookup', True),
    GoldenCase('ml02', 'What is supervised learning?', 'Learning from examples containing inputs and known target values.', ('machine_learning_foundations.md',), ('Main learning settings',), 'lookup', True),
    GoldenCase('ml03', 'How do classification and regression differ?', 'Classification predicts categories; regression predicts numerical values.', ('machine_learning_foundations.md',), ('Main learning settings',), 'comparison', True),
    GoldenCase('ml04', 'What is a validation set used for?', 'Selecting hyperparameters, thresholds, and model variants.', ('machine_learning_foundations.md',), ('The dataset split',), 'lookup', True),
    GoldenCase('ml05', 'Why should a test set not be repeatedly used for tuning?', 'Repeated tuning leaks test information into model selection and makes scores optimistic.', ('machine_learning_foundations.md',), ('The dataset split',), 'reasoning', True),
    GoldenCase('ml06', 'What is data leakage? Give one example.', 'Using information unavailable at prediction time or derived from evaluation targets; for example, fitting a scaler on all data before splitting.', ('machine_learning_foundations.md',), ('Data leakage',), 'lookup', True),
    GoldenCase('ml07', 'What does standardization do?', "It subtracts a feature's training mean and divides by its training standard deviation.", ('machine_learning_foundations.md',), ('Preprocessing and feature representation',), 'lookup', True),
    GoldenCase('ml08', 'Why can integer encoding mislead a model for nominal categories?', 'The model may interpret arbitrary integer codes as meaningful order or distance.', ('machine_learning_foundations.md',), ('Preprocessing and feature representation',), 'reasoning', True),
    GoldenCase('ml09', 'How does k-NN regression predict?', 'It commonly averages target values from nearby training examples.', ('machine_learning_foundations.md',), ('Common algorithms',), 'lookup', True),
    GoldenCase('ml10', 'How do L1 and L2 regularization differ?', 'L1 penalizes absolute coefficients and can set some to zero; L2 penalizes squared coefficients and shrinks them.', ('machine_learning_foundations.md',), ('Regularization',), 'comparison', True),
    GoldenCase('ml11', 'Why can accuracy mislead on imbalanced data?', 'A model can favor the majority class and still have high accuracy while missing rare positives.', ('machine_learning_foundations.md',), ('Evaluation metrics', 'Imbalanced classification'), 'multi_section', True),
    GoldenCase('ml12', 'When can test R-squared be negative?', 'When predictions have greater squared error than the baseline predicting the target mean.', ('machine_learning_foundations.md',), ('Evaluation metrics',), 'lookup', True),
    GoldenCase('ml13', 'What is the purpose of cross-validation?', 'To evaluate across multiple folds and obtain a more stable estimate than one arbitrary split.', ('machine_learning_foundations.md',), ('Cross-validation and model selection',), 'lookup', True),
    GoldenCase('ml14', 'Where should resampling for class imbalance occur?', 'Inside the training process only, never before the train-test split.', ('machine_learning_foundations.md',), ('Imbalanced classification',), 'lookup', True),
    GoldenCase('ml15', 'Why establish a baseline model?', 'It provides a simple reference to determine whether a complex model adds useful performance.', ('machine_learning_foundations.md',), ('Baselines before complex models',), 'reasoning', True),
    GoldenCase('dl01', 'What is the equation for a dense neural-network layer?', 'h = phi(Wx + b), where x is input, W weights, b bias, and phi the activation.', ('deep_learning_systems.md',), ('What makes a model deep?',), 'lookup', True),
    GoldenCase('dl02', 'Why are nonlinear activations important in deep networks?', 'Without nonlinear activations, stacked linear transformations collapse to a single linear transformation.', ('deep_learning_systems.md',), ('What makes a model deep?',), 'reasoning', True),
    GoldenCase('dl03', 'What does backpropagation compute?', 'It applies the chain rule to calculate loss gradients with respect to parameters.', ('deep_learning_systems.md',), ('Forward pass, loss, and backpropagation',), 'lookup', True),
    GoldenCase('dl04', 'What is an epoch?', 'One complete pass through the training dataset.', ('deep_learning_systems.md',), ('Forward pass, loss, and backpropagation',), 'lookup', True),
    GoldenCase('dl05', 'How do ReLU and sigmoid differ?', 'ReLU returns max(0,x); sigmoid maps values to (0,1) and can saturate.', ('deep_learning_systems.md',), ('Activation functions',), 'comparison', True),
    GoldenCase('dl06', 'When is binary cross-entropy commonly used?', 'For binary classification with binary targets and predicted probabilities.', ('deep_learning_systems.md',), ('Loss functions',), 'lookup', True),
    GoldenCase('dl07', 'What does Adam do?', 'It tracks moving estimates of first and second gradient moments to adapt parameter-wise step sizes.', ('deep_learning_systems.md',), ('Gradient descent and optimizers',), 'lookup', True),
    GoldenCase('dl08', 'What is a convolutional filter?', 'A learned local pattern detector that produces a feature map.', ('deep_learning_systems.md',), ('Convolutional neural networks',), 'lookup', True),
    GoldenCase('dl09', 'What is a CNN receptive field?', 'The region of the input that can influence a particular feature.', ('deep_learning_systems.md',), ('Convolutional neural networks',), 'lookup', True),
    GoldenCase('dl10', 'Why is decoder self-attention masked?', 'To prevent a target position from accessing future target tokens during autoregressive generation.', ('deep_learning_systems.md',), ('Sequence models and attention',), 'lookup', True),
    GoldenCase('dl11', 'Where do queries, keys, and values come from in encoder-decoder attention?', 'Queries come from the decoder; keys and values come from encoder representations.', ('deep_learning_systems.md',), ('Sequence models and attention',), 'lookup', True),
    GoldenCase('dl12', 'Why do Transformers use positional information?', 'Self-attention alone does not inherently encode sequence order.', ('deep_learning_systems.md',), ('Embeddings and positional information',), 'reasoning', True),
    GoldenCase('dl13', 'What does a residual connection do?', 'It adds a block input to its transformed output, providing a direct path for information and gradients.', ('deep_learning_systems.md',), ('Normalization and residual connections',), 'lookup', True),
    GoldenCase('dl14', 'What is transfer learning?', 'Reusing a model trained on a related task or large dataset and adapting it to a new task.', ('deep_learning_systems.md',), ('Transfer learning and fine-tuning',), 'lookup', True),
    GoldenCase('dl15', 'Why keep near-duplicate images in the same split?', 'To prevent train-test leakage that can make evaluation overly optimistic.', ('deep_learning_systems.md',), ('Overfitting and generalization',), 'reasoning', True),
    GoldenCase('fe01', 'Why define prediction time before feature engineering?', 'To ensure features only use information available at inference time and avoid future leakage.', ('feature_engineering_and_diagnostics.txt',), ('Start with the prediction contract',), 'reasoning', True),
    GoldenCase('fe02', 'How does standardization differ from row-wise normalization?', 'Standardization scales each feature using training mean and standard deviation; row normalization scales each vector to a norm.', ('feature_engineering_and_diagnostics.txt',), ('Numerical features',), 'comparison', True),
    GoldenCase('fe03', 'What is target encoding and its risk?', 'It represents categories using target-derived statistics; careless fitting can leak labels.', ('feature_engineering_and_diagnostics.txt',), ('Categorical features',), 'lookup', True),
    GoldenCase('fe04', 'Why must rolling features use only past records?', 'Using future records leaks information unavailable at the prediction timestamp.', ('feature_engineering_and_diagnostics.txt',), ('Date, time, and group features',), 'reasoning', True),
    GoldenCase('fe05', 'What is multicollinearity?', 'Predictors contain overlapping linear information, which can make regression coefficients unstable.', ('feature_engineering_and_diagnostics.txt',), ('Multicollinearity',), 'lookup', True),
    GoldenCase('fe06', 'Should a correlated feature always be removed?', 'No; the decision depends on the objective and validation. Correlation may affect coefficient interpretation without harming prediction.', ('feature_engineering_and_diagnostics.txt',), ('Multicollinearity',), 'reasoning', True),
    GoldenCase('fe07', 'What should be checked before removing an outlier?', 'Whether it is an error, a valid rare case, or an important event.', ('feature_engineering_and_diagnostics.txt',), ('Outliers and robustness',), 'lookup', True),
    GoldenCase('fe08', 'Why evaluate subgroup performance?', 'Aggregate metrics can conceal systematic failures in a smaller segment.', ('feature_engineering_and_diagnostics.txt',), ('Model diagnostics',), 'reasoning', True),
    GoldenCase('fe09', 'What should an ML experiment record?', 'Dataset/split, preprocessing, model settings, random seed, software versions, selection rule, test results, and limitations.', ('feature_engineering_and_diagnostics.txt',), ('Experiment tracking',), 'lookup', True),
    GoldenCase('fe10', 'Why define handling for unknown categories?', 'Inference data may contain categories absent from training, so their handling must be explicit.', ('feature_engineering_and_diagnostics.txt',), ('Categorical features',), 'reasoning', True),
    GoldenCase('ev01', 'What does precision measure?', 'The fraction of predicted positives that are truly positive.', ('model_evaluation_guide.html',), ('Classification metrics',), 'lookup', True),
    GoldenCase('ev02', 'When can PR-AUC be more informative than ROC-AUC?', 'When positive cases are rare, precision-recall behavior can be more informative.', ('model_evaluation_guide.html',), ('Classification metrics',), 'comparison', True),
    GoldenCase('ev03', 'How do MAE and MSE differ?', 'MAE averages absolute errors; MSE squares errors and penalizes large misses more strongly.', ('model_evaluation_guide.html',), ('Regression metrics',), 'comparison', True),
    GoldenCase('ev04', 'What is probability calibration?', 'Whether predicted probabilities correspond to observed event frequencies.', ('model_evaluation_guide.html',), ('Threshold selection and calibration',), 'lookup', True),
    GoldenCase('ev05', 'Why is a 0.5 threshold not always optimal?', 'False-positive and false-negative costs differ by task, so threshold selection should use validation data and operational needs.', ('model_evaluation_guide.html',), ('Threshold selection and calibration',), 'reasoning', True),
    GoldenCase('ev06', 'What should a responsible evaluation report include?', 'Split strategy, target, metrics, baseline, subgroup results, errors, model/preprocessing versions, and limitations.', ('model_evaluation_guide.html',), ('Reporting results responsibly',), 'lookup', True),
    GoldenCase('mh01', 'A fraud dataset has 1% positives. What evaluation approach is appropriate?', 'Use precision, recall, F1 and PR-AUC; avoid relying on accuracy, resample only in training, and choose thresholds on validation data.', ('machine_learning_foundations.md', 'model_evaluation_guide.html'), ('Imbalanced classification', 'Classification metrics', 'Threshold selection and calibration'), 'multi_hop', True),
    GoldenCase('mh02', 'How can a CNN defect-image evaluation leak information?', 'Near-duplicate images or images from the same source can cross splits; group related images together and reserve test data for final evaluation.', ('deep_learning_systems.md', 'model_evaluation_guide.html'), ('Overfitting and generalization', 'Why the split matters'), 'multi_hop', True),
    GoldenCase('mh03', 'How should rolling sensor features be evaluated for future prediction?', 'Use only past records to build rolling features, split chronologically, and fit preprocessing on training data only.', ('feature_engineering_and_diagnostics.txt', 'machine_learning_foundations.md'), ('Date, time, and group features', 'The dataset split', 'Data leakage'), 'multi_hop', True),
    GoldenCase('mh04', 'A neural classifier has high accuracy but poor recall on a rare class. What should be checked?', 'Inspect per-class metrics and the confusion matrix; consider training-only resampling or class weights and choose thresholds using validation costs.', ('deep_learning_systems.md', 'machine_learning_foundations.md', 'model_evaluation_guide.html'), ('Evaluation beyond one score', 'Imbalanced classification', 'Threshold selection and calibration'), 'multi_hop', True),
    GoldenCase('mh05', 'Why can regression coefficients be unstable while predictions are acceptable?', 'Multicollinearity can destabilize individual coefficients; inspect redundancy and consider regularization validated against the prediction objective.', ('feature_engineering_and_diagnostics.txt', 'machine_learning_foundations.md'), ('Multicollinearity', 'Regularization'), 'multi_hop', True),
    GoldenCase('mh06', 'How should validation and test sets be used when selecting a deep-learning checkpoint?', 'Use validation data to select checkpoints; keep the test set untouched for final evaluation.', ('deep_learning_systems.md', 'machine_learning_foundations.md'), ('Forward pass, loss, and backpropagation', 'The dataset split'), 'multi_hop', True),
    GoldenCase('mh07', 'Why save preprocessing settings with a trained model?', 'Inference must repeat the same transformations, and recorded settings make experiments reproducible.', ('machine_learning_foundations.md', 'feature_engineering_and_diagnostics.txt'), ('A reproducible ML workflow', 'Experiment tracking'), 'multi_hop', True),
    GoldenCase('mh08', 'How do ranking quality and calibration differ?', 'Ranking concerns ordering examples; calibration concerns whether predicted probabilities match observed frequencies.', ('machine_learning_foundations.md', 'model_evaluation_guide.html'), ('Evaluation metrics', 'Threshold selection and calibration'), 'multi_hop', True),
    GoldenCase('ua01', "What was the exact validation accuracy of the user's latest private CNN run?", 'The corpus does not contain that private training-run result.', (), (), 'unanswerable', False),
    GoldenCase('ua02', "What is today's exact NVIDIA stock price?", 'The corpus contains no current market-price data.', (), (), 'unanswerable', False),
    GoldenCase('ua03', 'Which GPU and VRAM capacity did the user use in their latest experiment?', "The corpus does not specify the user's GPU model or VRAM capacity.", (), (), 'unanswerable', False),
    GoldenCase('ua04', "What is the exact count of defect images in the user's current private dataset?", "The corpus does not establish the exact count for the user's current dataset.", (), (), 'unanswerable', False),
    GoldenCase('ua05', 'Which optimizer won an experiment not described in these documents?', 'The corpus does not describe that experiment, so its winning optimizer cannot be identified.', (), (), 'unanswerable', False),
    GoldenCase('ua06', "What is the live inference latency of the user's deployed RAG service?", 'The corpus contains no live latency measurement.', (), (), 'unanswerable', False),
    GoldenCase('am01', 'Is normalization better than standardization?', 'There is no universal winner; standardization scales features, while normalization may scale each vector. The choice depends on data and model.', ('feature_engineering_and_diagnostics.txt', 'machine_learning_foundations.md'), ('Numerical features', 'Preprocessing and feature representation'), 'ambiguous', True),
    GoldenCase('am02', 'Which ML model is best?', 'There is no universal best model; compare candidates against a baseline using a split and metric appropriate to the task.', ('machine_learning_foundations.md',), ('Baselines before complex models', 'Cross-validation and model selection'), 'ambiguous', True),
    GoldenCase('am03', 'Should I use a Transformer?', 'The corpus cannot give a universal yes/no; suitability depends on task, data, compute, latency, and validation performance.', ('deep_learning_systems.md',), ('Sequence models and attention', 'Transfer learning and fine-tuning'), 'ambiguous', True),
    GoldenCase('am04', 'Is a high F1 score enough?', 'Not necessarily; inspect task-specific metrics, error examples, subgroup performance, and the costs of different mistakes.', ('machine_learning_foundations.md', 'model_evaluation_guide.html'), ('Evaluation metrics', 'Error analysis'), 'ambiguous', True),
    GoldenCase('am05', 'Should outliers always be removed?', 'No; first determine whether an outlier is erroneous, valid but rare, or operationally important.', ('feature_engineering_and_diagnostics.txt',), 'Outliers and robustness', 'ambiguous', True),
]

def validate_golden_cases():
    ids = [c.case_id for c in GOLDEN_CASES]
    assert len(ids) == len(set(ids)), "Duplicate IDs"
    assert len(GOLDEN_CASES) >= 50, "Need at least 50 cases"
    assert all(c.question.strip() and c.golden_answer.strip() for c in GOLDEN_CASES)

if __name__ == "__main__":
    validate_golden_cases()
    from collections import Counter
    print("Cases:", len(GOLDEN_CASES))
    print("Categories:", dict(Counter(c.category for c in GOLDEN_CASES)))
    print("Answerable:", sum(c.answerable for c in GOLDEN_CASES))
    print("Unanswerable:", sum(not c.answerable for c in GOLDEN_CASES))
