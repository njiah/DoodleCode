import logging
import random
from numpy import asarray, uint8, float32
from skimage.metrics import structural_similarity as ssim
import cv2
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import base64

logging.basicConfig(level=logging.DEBUG, filename='environment_log.txt', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def html_render(html_content):
    driver = None
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("window-size=1200x600")
        driver = webdriver.Chrome(options=chrome_options)

        encoded_html = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
        data_uri = f"data:text/html;base64,{encoded_html}"
        driver.get(data_uri)

        image_stream = BytesIO()
        image_stream.write(driver.get_screenshot_as_png())
        image_stream.seek(0)

        return image_stream
    except Exception as e:
        logging.error(f"Failed to render HTML: {e}")
        return None
    finally:
        if driver:
            driver.quit()

def one_hot_encoder(label_index, total_labels):
    one_hot_vector = [0] * total_labels
    one_hot_vector[label_index] = 1
    return one_hot_vector

def construct_state_vector(html_elements, gl_lbls):
    state_vector = []
    for element in html_elements:
        label_index = gl_lbls.index(element['type'])
        one_hot_encoded = one_hot_encoder(label_index, len(gl_lbls))
        bbox = [element['style'].get('left', 0), element['style'].get('top', 0),
                element['style'].get('width', 0), element['style'].get('height', 0)]
        element_vector = one_hot_encoded + bbox
        state_vector.extend(element_vector)
    return state_vector

def prepare_image(image_input):
    try:
        if image_input is None:
            logging.error("No image stream provided.")
            return None

        image_input.seek(0)
        file_bytes = asarray(bytearray(image_input.read()), dtype=uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Failed to decode image from stream.")

        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    except Exception as e:
        logging.error(f"Error preparing image: {e}")
        return None

def prepare_image_with_path(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Failed to load image at " + image_path)

        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    except Exception as e:
        logging.error(f"Error preparing image from path {image_path}: {e}")
        return None

def calculate_ssim(image1, image2):
    try:
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

        if image1.dtype != float32:
            image1 = image1.astype(float32) / 255.0
        if image2.dtype != float32:
            image2 = image2.astype(float32) / 255.0

        score, diff_image = ssim(image1, image2, full=True, data_range=image1.max() - image1.min())
        logging.info(f"SSIM Score: {score}")
        return score
    except Exception as e:
        logging.error(f"Error calculating SSIM: {e}")
        return None

class HTMLDesignerEnv:
    def __init__(self, html_templates, sketch_path, bboxes, labels):
        self.html_templates = html_templates
        self.sketch_path = sketch_path
        self.bboxes = bboxes
        self.labels = labels
        self.html_elements = []
        self.current_ssim = 0
        self.gl_lbls = [
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
        self.state_size = 78

    def reset(self):
        self.html_elements = self.convert_bboxes_to_elements(self.bboxes, self.labels)
        initial_html = self.render_html()
        initial_image_stream = html_render(initial_html)
        initial_ssim = self.compute_ssim(initial_image_stream, prepare_image_with_path(self.sketch_path)) or 0
        self.current_ssim = initial_ssim
        return construct_state_vector(self.html_elements, self.gl_lbls)

    def label_to_type(self, label):
        try:
            label_index = int(label)
            return self.gl_lbls[label_index]
        except ValueError:
            raise ValueError(f"Label '{label}' is not a valid index.")
        except IndexError:
            raise ValueError(f"Label index {label_index} is out of range for gl_lbls.")

    def convert_bboxes_to_elements(self, bboxes, labels):
        elements = []
        for bbox, label in zip(bboxes, labels):
            element_type = self.label_to_type(label)
            element = {
                'type': element_type,
                'style': {
                    'left': bbox[0],
                    'top': bbox[1],
                    'width': bbox[2] - bbox[0],
                    'height': bbox[3] - bbox[1]
                },
                'id': len(elements)
            }
            elements.append(element)
        return elements

    def step(self, action):
        previous_ssim = self.current_ssim
        state_vector, new_ssim = self.apply_action_and_render(action)

        reward = self.calculate_reward(action, previous_ssim, new_ssim)

        self.current_ssim = new_ssim

        return state_vector, reward, new_ssim

    def calculate_reward(self, action, previous_ssim, new_ssim):
        reward = 0
        ssim_improvement = new_ssim - previous_ssim
        if ssim_improvement > 0:
            reward += ssim_improvement * 100

        if ssim_improvement <= 0:
            reward -= 5

        if action.action_type == 'add' and ssim_improvement < 0.01:
            reward -= 10

        return reward

    def apply_action_and_render(self, action):
        self.apply_action(action)
        rendered_image_stream = self.render_html()

        if rendered_image_stream is not None:
            rendered_image = prepare_image(rendered_image_stream)
        else:
            logging.error("Rendered image stream is None.")
            rendered_image = None
        sketch_image = prepare_image_with_path(self.sketch_path)
        if sketch_image is None:
            logging.error("Failed to load or process sketch image from path.")

        new_ssim = self.compute_ssim(rendered_image, sketch_image)

        state_vector = construct_state_vector(self.html_elements, self.gl_lbls)

        return state_vector, new_ssim

    def apply_action(self, action):
        if action.action_type == 'add':
            self.add_element(action.element_type, action.style)
        elif action.action_type == 'modify':
            self.modify_element(action.element_id, action.style)
        elif action.action_type == 'remove':
            self.remove_element(action.element_id)
        elif action.action_type == 'rearrange':
            if hasattr(action, 'new_position') and action.new_position is not None:
                self.rearrange_element(action.element_id, action.new_position)
            else:
                logging.error("Rearrange action missing 'new_position'")

    def add_element(self, element_type, style, attributes=None):
        new_element = {
            'type': element_type,
            'style': style,
            'attributes': attributes if attributes else {},
            'id': len(self.html_elements)
        }
        self.html_elements.append(new_element)

    def modify_element(self, element_id, new_style):
        for element in self.html_elements:
            if element['id'] == element_id:
                element['style'] = new_style
                break

    def remove_element(self, element_id):
        self.html_elements = [el for el in self.html_elements if el['id'] != element_id]

    def rearrange_element(self, element_id, new_position):
        pass

    def compute_ssim(self, rendered_image_stream, sketch_image_path):
        rendered_image = prepare_image(rendered_image_stream)
        sketch_image = prepare_image_with_path(sketch_image_path)
        if rendered_image is None or sketch_image is None:
            logging.error("One or both images failed to load or process.")
            return 0

        try:
            if rendered_image.shape != sketch_image.shape:
                sketch_image = cv2.resize(sketch_image, (rendered_image.shape[1], rendered_image.shape[0]))
            rendered_image = rendered_image.astype(float32) / 255.0
            sketch_image = sketch_image.astype(float32) / 255.0

            score, _ = ssim(rendered_image, sketch_image, full=True)
            logging.info(f"SSIM Score: {score}")
            return score
        except Exception as e:
            logging.error(f"Error calculating SSIM: {e}")
            return 0

    def render_html(self):
        html_content = "<html><body>"
        for element in self.html_elements:
            attributes_string = ""
            if 'attributes' in element and element['attributes']:
                attributes_string = ' '.join(f'{key}="{value}"' for key, value in element['attributes'].items())

            if element['type'] in self.html_templates:
                element_html = self.html_templates[element['type']].format(style=element['style'], attributes=attributes_string)
            else:
                element_html = f"<div {attributes_string} style='{element['style']}'>Unknown Element</div>"

            html_content += element_html
        html_content += "</body></html>"
        return html_render(html_content)
