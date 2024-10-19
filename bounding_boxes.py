import numpy as np


def render_to_list(y_pred, class_mapping):
    bboxes = []
    labels = []
    labels_encoded = []

    for i in range(len(y_pred["boxes"][0])):
        if y_pred["boxes"][0][i][0] == -1:
            continue
        bboxes.append(y_pred["boxes"][0][i])
        labels.append(y_pred["classes"][0][i])

    for label in labels:
        labels_encoded.append(class_mapping[label])

    return bboxes, labels, labels_encoded
