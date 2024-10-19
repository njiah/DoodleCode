import os
import glob
import tensorflow as tf


# if not load model from freeze folder, load latest model.
def load_model(model_loc=None):
    if model_loc:
        model = tf.keras.models.load_model(model_loc)
        return model
    list_of_files = glob.glob("models/*.keras")
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    model = tf.keras.models.load_model(latest_file)
    return model
