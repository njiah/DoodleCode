import os
import numpy as np
import json
import environment_code
import agent_code
#from predict import predict_and_show

def load_model(model_path):
    """
    Load a TensorFlow model from the specified path.
    
    Args:
    model_path (str): Path to the TensorFlow model (.h5 file).
    
    Returns:
    TensorFlow model: Loaded model.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully from:", model_path)
        return model
    except Exception as e:
        print(f"Failed to load model from {model_path}. Error: {e}")
        return None


html_templates = {
    "button": '<button {attributes} style="{style}">Button</button>',
    "checkbox": '<label {attributes} style="{style}"><input type="checkbox"> Checkbox</label>',
    "container": '<div {attributes} style="{style}">Container Content</div>',
    "icon-button": '<button {attributes} style="{style}"><img src="icon.png" alt="icon" style="width: 20px; height: 20px;"> Icon Button</button>',
    "image": '<img  {attributes} src="image.jpg" alt="Description" style="{style}">',
    "input": '<input {attributes} type="text" placeholder="Enter text" style="{style}">',
    "label": '<label {attributes} for="inputExample" style="{style}">Label:</label>',
    "link": '<a href="http://example.com" style="{style}">Visit Example</a>',
    "number-input": '<input type="number" placeholder="Enter number" style="{style}">',
    "radio": '<label style="{style}"><input type="radio" name="radioExample"> Radio Button</label>',
    "search": '<input type="search" placeholder="Search here" style="{style}">',
    "select": '<select style="{style}"><option value="option1">Option 1</option><option value="option2">Option 2</option></select>',
    "slider": '<input type="range" min="1" max="100" value="50" style="{style}">',
    "table": '<table style="{style}"><tr><th>Header 1</th><th>Header 2</th></tr><tr><td>Data 1</td><td>Data 2</td></tr></table>',
    "text": '<span style="{style}">Some text here</span>',
    "textarea": '<textarea placeholder="Enter multi-line text" style="{style}"></textarea>',
    "textbox": '<input type="text" placeholder="Enter text" style="{style}">',
    "toggle": '<label style="{style}"><input type="checkbox"> Toggle</label>',
    "pagination": '<div style="{style}"><a href="#">&laquo;</a> <a href="#">1</a> <a href="#">2</a> <a href="#">&raquo;</a></div>',
    "paragraph": '<p style="{style}">A paragraph of text.</p>',
    "carousel": '<div style="{style}"><img src="slide1.jpg" alt="Slide 1"> <img src="slide2.jpg" alt="Slide 2"> <img src="slide3.jpg" alt="Slide 3"></div>',
    "heading": '<h1 style="{style}">Heading Text</h1>'
}

# Load JSON data
with open('compiled_predictions.json', 'r') as file:
    json_data = json.load(file)

# Directory containing sketches
sketches_dir = '/Users/ahmetair/uni/y3/desd/doodlecode/datasets/yolo-v8/train/images'

def train_on_sketches(json_data, sketches_dir, num_episodes=1000):
    for sketch_filename, sketch_info in json_data.items():
        sketch_path = os.path.join(sketches_dir, sketch_filename + '.jpg')
        if os.path.isfile(sketch_path):
            bboxes = sketch_info['bboxes']
            labels = sketch_info['labels']

            env = environment_code.HTMLDesignerEnv(html_templates, sketch_path, bboxes, labels)
            agent = agent_code.DQNAgent(env.state_size, len(env.gl_lbls), env)

            agent.current_labels = labels  # Update agent with current labels

            for e in range(num_episodes):
                state = env.reset()
                state = np.reshape(state, [1, env.state_size])
                total_reward = 0
                done = False

                while not done:
                    actions = agent.act()  # Get actions based on current labels
                    for action in actions:
                        next_state, reward, done = env.step(action)
                        next_state = np.reshape(next_state, [1, env.state_size])
                        agent.remember(state, action, reward, next_state, done)
                        state = next_state
                        total_reward += reward

                    if len(agent.memory) > 32:  # Batch size assumption
                        agent.replay(32)

                print(f"Episode: {e + 1}, Total Reward: {total_reward}")

            agent.save(f'model_checkpoint_after_{num_episodes}_episodes')



            # Optionally save the model after training

def predict(sketch_path):
    model = load_model()
    predict_and_show(model, sketch_path)

# Example prediction call
#predict("/Users/ahmetair/uni/y3/desd/doodlecode/datasets/yolo-v8/train/images/59_jpeg.rf.9ab6d86478d22d5d113ce144108c5fe8.jpg")

train_on_sketches(json_data, "/Users/ahmetair/uni/y3/desd/doodlecode/datasets/yolo-v8/train/images/")
