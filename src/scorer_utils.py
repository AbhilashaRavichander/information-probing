from sklearn.metrics import precision_recall_fscore_support


def get_prfs(x, binary_preds):
    """
    Get precision, recall, fscore, support for a given binary prediction
    Args:
        x (list): List of binary predictions
        binary_preds (list): List of binary predictions
    Returns:
        dict: Dictionary containing precision, recall, fscore, support
    """

    precision_beta, recall_beta, fscore_beta, support_beta = precision_recall_fscore_support(
        x, binary_preds, beta=0.1, pos_label=1)

    return {"precision": round(precision_beta[1]*100.0,2), "recall": round(recall_beta[1]*100.0,2), "fbeta": round(fscore_beta[1]*100.0,2), "support": support_beta}
