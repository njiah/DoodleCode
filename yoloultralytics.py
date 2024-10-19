from ultralytics import YOLO

def train_ultralytics(
        dataset,
        model="yolov8s.pt",
        epochs=10,
        imgsz=640,
        save_dir="histories/ultralytics",
        ):
    model = YOLO("yolov8s.pt")
    # Run for M1/M2, MPS for M1 neural engine
    # model.train(
    #     device="mps",
    #     data=dataset,
    #     epochs=epochs,
    #     batch=-1,
    #     imgsz=imgsz,
    #     save_dir=save_dir,
    # )
    model.train(
        data=dataset,
        epochs=epochs,
        batch=2,
        imgsz=imgsz,
        save_dir=save_dir,
    )


def predict_ultralytics(
    gl_class_mapping,
    img,
    model_path="models/ultralytics/model_1/weights/best.pt",
    conf=0.5,
    iou=0.7,
):
    model = YOLO(model_path)
    results = model.predict(img,
                            save=True,
                            conf=conf,
                            iou=iou)

    boxes = []
    classes = []
    classes_encoded = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_map = int(b.cls)
            # Re map the class
            cls = gl_class_mapping[cls_map]
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)
            classes_encoded.append(cls_map)

    return boxes, classes, classes_encoded
