import tensorflow as tf
import datetime


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # get original image dimensions
    original_dims = image.shape
    image = tf.image.resize(image, (640, 640))
    # image = image / 255.0
    image = tf.expand_dims(image, 0)
    return image, original_dims


def reshape_image(image, original_dims, output=True):
    print("Reshaping image")
    image = tf.image.resize(image, (original_dims[0], original_dims[1]))
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image)
    print("Image reshaped")
    # save with current timestamp as name
    if output:
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        tf.io.write_file("outputs/output" + time + ".jpg", image)
    return image
