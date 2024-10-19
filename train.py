import os
from re import T
import tensorflow as tf
from tensorflow import keras
import keras_cv
from keras_cv import bounding_box
import datetime
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

# globals
gl_labels = "datasets/voc-xml"
bbxf = "xyxy"


# Load Image
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


# Load Dataset into tf bbx format
def load_dataset(image_path, classes, bbox):
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {"boxes": bbox, "classes": classes}
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}


# # Convert dict to tuple
# def dict_to_tuple(inputs):
#     return inputs["images"], inputs["bounding_boxes"]
def dict_to_tuple(inputs):
    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32
    )

# Parse XML trees of dataset
def parse_annotation(xml_file, class_mapping):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(gl_labels, image_name)

    boxes = []
    classes = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        cls = cls.lower()
        if cls == "check box":
            cls = "checkbox"
        if cls == "radio button":
            cls = "radio"
        classes.append(cls)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])

    class_ids = [
        list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
        for cls in classes
    ]
    return image_path, boxes, class_ids


# Sort data into image paths, bounding boxes and classes
def sort_data(class_mapping):
    print("Sorting data")
    # Get all XML file paths in path_annot and sort them
    xml_files = sorted(
        [
            os.path.join(gl_labels, file_name)
            for file_name in os.listdir(gl_labels)
            if file_name.endswith(".xml")
        ]
    )

    # Get all JPEG image file paths in path_images and sort them
    jpg_files = sorted(
        [
            os.path.join(gl_labels, file_name)
            for file_name in os.listdir(gl_labels)
            if file_name.endswith(".jpg")
        ]
    )

    print(f"Total XML files: {len(xml_files)}" f"\nTotal JPEG files: {len(jpg_files)}")

    image_paths = []
    bbox = []
    classes = []
    for xml_file in tqdm(xml_files):
        image_path, boxes, class_ids = parse_annotation(xml_file, class_mapping)
        image_paths.append(image_path)
        bbox.append(boxes)
        classes.append(class_ids)

    return image_paths, bbox, classes


# Generate train/val data from image paths, bounding boxes and classes
def tf_data_gen(image_paths, bbox, classes, split):
    print("Generating data")
    files = tf.ragged.constant(image_paths)
    labels = tf.ragged.constant(classes)
    bboxes = tf.ragged.constant(bbox)

    dataset = tf.data.Dataset.from_tensor_slices((files, labels, bboxes))

    # Split the dataset into train and validation sets
    train_size = int(split * len(dataset))
    train_data = dataset.take(train_size)
    val_data = dataset.skip(train_size)

    print(len(dataset))
    print("Train Size: ", len(train_data))
    print("Validation Size: ", len(val_data))

    return train_data, val_data


# Data augmentation
def augment_data(train_ds, val_ds):
    print("Augmenting data")
    augmenters = keras.Sequential(
        layers=[
            keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format=bbxf),
            keras_cv.layers.JitteredResize(
                target_size=(640, 640),
                scale_factor=(0.75, 1.3),
                bounding_box_format=bbxf,
            ),
        ]
    )

    train_ds = train_ds.map(augmenters, num_parallel_calls=tf.data.AUTOTUNE)
    resizing = keras_cv.layers.Resizing(
        width=640,
        height=640,
        bounding_box_format=bbxf,
        pad_to_aspect_ratio=True,
    )
    val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


# Train model
def train(
    class_mapping,
    backbone="yolo_v8_xs_backbone_coco",
    lr=0.001,
    num_epochs=10,
    split=0.7,
    patience=10,
    batch_size=4,
    weights=None,
):
    print("Training model")
    print("Class Mapping: ", class_mapping)
    print("Backbone: ", backbone)
    print("Learning Rate: ", lr)
    print("Number of Epochs: ", num_epochs)
    print("Split: ", split)
    print("Patience: ", patience)
    print("Batch Size: ", batch_size)

    image_paths, bbox, classes = sort_data(class_mapping=class_mapping)
    train_data, val_data = tf_data_gen(image_paths, bbox, classes, split=split)
    print(train_data)

    train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(batch_size * 4)
    train_ds = train_ds.ragged_batch(batch_size, drop_remainder=True)
    val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.ragged_batch(batch_size, drop_remainder=True)

    train_ds, val_ds = augment_data(train_ds, val_ds)

    print("Backbone: ", backbone)

    backbone = keras_cv.models.YOLOV8Backbone.from_preset(backbone)

    num_classes = len(class_mapping)
    print("Number of Classes: ", num_classes)

    model = keras_cv.models.YOLOV8Detector(
        num_classes=len(class_mapping),
        bounding_box_format=bbxf,
        backbone=backbone,
        fpn_depth=2,
    )

    # including a global_clipnorm is extremely important in object detection tasks
    optimizer = keras.optimizers.SGD(
        learning_rate=lr, momentum=0.9, global_clipnorm=10.0
    )

    model.compile(
        optimizer=optimizer,
        classification_loss="binary_crossentropy",
        box_loss="ciou",
    )

    dt = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    # with training time
    csvlogger = keras.callbacks.CSVLogger(
        "histories/sketch2code_training_" + dt + ".log",
        append=True,
    )

    modelcheckpoint = keras.callbacks.ModelCheckpoint(
        "histories/sketch2code_history_" + dt + ".weights.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
    )

    reducelronplateau = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=5, verbose=1, min_lr=1e-7
    )

    pycoco = keras_cv.callbacks.PyCOCOCallback(
        val_ds, bounding_box_format=bbxf, cache=True
    )

    tensorboard = keras.callbacks.TensorBoard(
        log_dir="logs/sketch2code_" + dt, histogram_freq=1
    )

    callbacks = [
        pycoco,
        callback,
        csvlogger,
        modelcheckpoint,
        reducelronplateau,
        tensorboard,
    ]

    if weights is not None:
        print("Loading weights: " + weights)
        model.load_weights(weights)

    model.fit(
        # Run for 10-35~ epochs to achieve good scores.
        train_ds,
        epochs=num_epochs,
        callbacks=[callbacks],
        validation_data=val_ds,
    )
    results = model.evaluate(val_ds)
    print("Results: ", results)
    return model, dt


def save_model(
    path,
    model: tf.keras.models.Model,
    time=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
):
    if path is not None:
        model.save(path)
    else:
        model.save("models/sketch2code_model_" + time + ".keras")
