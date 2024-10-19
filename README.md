# Doodlecode
## Final year project for AAI 2023/2024

### Project Description
Doodlecode is a YoloV8 Keras & PyTorch (Ultralytics) implementation for object detection in sketches of websites, converting them into HTML code using a multi-agent system. The object detection model is trained on a custom dataset of sketches of websites, and the multi-agent system is responsible for converting the detected objects into HTML code. The project is implemented in Python and uses the Django framework for the web application.

Doodlecode can be used as a Python module or script.

### Python Module
```python
import doodlecode
doodle = doodlecode.Doodlecode()

# To predict
doodle.predict('path/to/sketch.jpg', iou=0.7, confidence=0.5, ultralytics=True/False) # switch between Keras and Ultralytics

# To train
doodle.train_model(ultralytics=True/False) # switch between Keras and Ultralytics
```

See more options in main doodlecode.py file.

### Python Script
```bash
python doodlecode.py --predict path/to/sketch.jpg --iou 0.7 --confidence 0.5 (--ultralytics True) # switch between Keras and Ultralytics

python doodlecode.py --train # training with Ultralytics is not supported
```

See more options in main doodlecode.py file.

## Credits
- [Ultralytics YoloV8](https://docs.ultralytics.com/)
- [Keras & KerasCV](https://keras.io/)
- Jago Gardiner
- Jiah Linn
- Amro Hassan Mahmoud
- Joseph Cauvy-Foster
- Tair Akhmetov
