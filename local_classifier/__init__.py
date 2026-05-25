"""local_classifier — KLUE-RoBERTa-base distill + cascade router.

5-class comment classifier (PRODUCT_OPINION / VIDEO_REACTION / CHATTER /
QUESTION / OFF_TOPIC) trained on GPT-4.1 teacher labels exported from
the operations DB. Inference output is compatible with
`comment_filtering_agent.classifiers.classifier_interface.ClassificationResult`.
"""
