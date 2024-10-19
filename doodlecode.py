from predictions import get_predictions
from image_utils import load_image, reshape_image
from model_utils import load_model
from train import train, save_model
from yoloultralytics import train_ultralytics, predict_ultralytics
import argparse


# Default global labels
gl_lbls = [
    "button",
    "checkbox",
    "container",
    "icon-button",
    "image",
    "input",
    "label",
    "link",
    "number-input",
    "radio",
    "search",
    "select",
    "slider",
    "table",
    "text",
    "textarea",
    "textbox",
    "toggle",
    "pagination",
    "paragraph",
    "carousel",
    "heading",
]
# Default global class mapping
gl_class_mapping = dict(zip(range(len(gl_lbls)), gl_lbls))


class DoodleCode:
    def __init__(self, class_mapping=None, model_loc=None):
        if class_mapping is None:
            self.class_mapping = gl_class_mapping
        else:
            self.class_mapping = class_mapping

        self.model = model_loc

    def predict(
        self,
        image_path,
        confidence=0.5,
        iou=0.7,
        render=True,
        output=True,
        rescale=False,
        ultralytics=False,
    ):
        if not ultralytics:
            model = load_model(model_loc=self.model)
            image, dimens = load_image(image_path)
            pred, bboxes, labels, labels_encoded = get_predictions(
                image,
                model,
                confidence,
                iou,
                self.class_mapping,
                render_img=render,
                rescale_boxes=rescale,
            )
            if output is True:
                image = reshape_image(pred, dimens, output=True)

            print(bboxes, labels, labels_encoded)
            return image, bboxes, labels, labels_encoded, dimens
        else:
            image = image_path
            boxes, classes, encoded = predict_ultralytics(
                gl_class_mapping, image, conf=confidence, iou=iou
            )
            image, dimens = load_image(image_path)
            print(boxes, encoded, classes)
            return image, boxes, encoded, classes, dimens

    def train_model(
        self,
        backbone="yolo_v8_xs_backbone_coco",
        lr=1e-2,
        split=0.7,
        patience=10,
        epochs=10,
        batch_size=4,
        path=None,
        weights=None,
        ultralytics=False,
    ):
        if not ultralytics:
            model, dt = train(
                gl_class_mapping,
                backbone=backbone,
                lr=lr,
                num_epochs=epochs,
                split=split,
                patience=patience,
                batch_size=batch_size,
                weights=weights,
            )
            save_model(model=model, path=path, time=dt)
        else:
            train_ultralytics(
                dataset="datasets/yolo-v8/data.yaml",
                model="yolov8n.pt",
                epochs=epochs,
                imgsz=640,
                save_dir="histories/ultralytics",
            )

    def command_line(self):
        parser = argparse.ArgumentParser(description="Sketch2Code")

        subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

        # Subparser for the 'train' subcommand
        train_parser = subparsers.add_parser(
            "train", help="Train the Doodlecode model."
        )
        train_parser.add_argument(
            "--backbone",
            help="Backbone to use for model. Default is yolo_v8_xs_backbone_coco.",
            default="yolo_v8_xs_backbone_coco",
            type=str,
        )
        train_parser.add_argument(
            "--lr",
            help="Learning rate. Default is 0.005 (SGD Optimizer). Specify as a float.",
            default=1e-2,
            type=float,
        )
        train_parser.add_argument(
            "--split",
            help="Train-validation split ratio. Default is 70/30 (0.7). Specify as a float, 0-1.",
            default=0.7,
            type=float,
        )
        train_parser.add_argument(
            "--patience",
            help="Patience for early stopping. Default is 5.",
            default=5,
            type=int,
        )
        train_parser.add_argument(
            "--epochs", help="Number of epochs. Default is 10.", default=10, type=int
        )
        train_parser.add_argument(
            "--batch_size", help="Batch size. Default is 4.", default=4, type=int
        )
        train_parser.add_argument(
            "--path",
            help="Path to save model data. Default saves the model to models directory.",
            default=None,
            type=str,
        )
        train_parser.add_argument(
            "--weights",
            help="Path to weights to load. Default is None.",
            default=None,
            type=str,
        )
        train_parser.add_argument(
            "--ultralytics",
            help="Use Ultralytics for training. Default is False.",
            required=False,
            default=False,
            type=bool,
        )

        # Subparser for the 'visualize' subcommand
        visualize_parser = subparsers.add_parser(
            "visualize", help="Visualize predictions"
        )
        visualize_parser.add_argument(
            "--image", help="Image path to predict.", required=True, type=str
        )
        visualize_parser.add_argument(
            "--confidence",
            help="Confidence threshold. Default is 0.55.",
            required=False,
            default=0.55,
            type=float,
        )
        visualize_parser.add_argument(
            "--iou",
            help="IOU threshold. Default is 0.3.",
            required=False,
            default=0.3,
            type=float,
        )
        visualize_parser.add_argument(
            "--ultralytics",
            help="Use Ultralytics for predictions. Default is False.",
            required=False,
            default=False,
            type=bool,
        )
        args = parser.parse_args()

        if args.subcommand == "train":
            self.train_model(
                backbone=args.backbone,
                lr=args.lr,
                split=args.split,
                patience=args.patience,
                epochs=args.epochs,
                batch_size=args.batch_size,
                path=args.path,
                weights=args.weights,
                ultralytics=args.ultralytics,
            )
        elif args.subcommand == "visualize":
            self.predict(args.image, args.confidence, args.iou, ultralytics=args.ultralytics)
        else:
            print("Invalid subcommand. Please specify either 'train' or 'visualize'.")


if __name__ == "__main__":
    doodlecode = DoodleCode()
    doodlecode.command_line()
