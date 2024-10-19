# Loads best model from freeze for output predictions API.

import tensorflow as tf

import keras_cv
from keras_cv import visualization
from bounding_boxes import render_to_list

bbxf = "xyxy"


def get_predictions(
    image, model, confidence, iou, class_mapping, render_img=False, rescale_boxes=False
):
    print("Confidence: ", confidence)
    print("IOU: ", iou)

    model.prediction_decoder = keras_cv.layers.NonMaxSuppression(
        bounding_box_format=bbxf,
        from_logits=True,
        iou_threshold=iou,
        confidence_threshold=confidence,
    )

    predictions = model.predict(image)
    if not predictions:
        print("No predictions")
        return
    y_pred = predictions.copy()
    # convert to numpy arrays
    if isinstance(y_pred["boxes"], tf.Tensor):
        y_pred["boxes"] = y_pred["boxes"].numpy()
    if isinstance(y_pred["classes"], tf.Tensor):
        y_pred["classes"] = y_pred["classes"].numpy()
    if render_img is True:
        images_pred = visualization.draw_bounding_boxes(
            image,
            y_pred,
            bounding_box_format=bbxf,
            color=(255, 255, 59),
            class_mapping=class_mapping,
            font_scale=1,
        )
        image = images_pred[0]
    bboxes, labels, labels_encoded = render_to_list(y_pred, class_mapping)
    if rescale_boxes:
        for i, box in enumerate(bboxes):
            bboxes[i] = [box[0] * 640, box[1] * 640, box[2] * 640, box[3] * 640]
    print(bboxes)
    print(labels)
    print(labels_encoded)
    return image, bboxes, labels, labels_encoded
