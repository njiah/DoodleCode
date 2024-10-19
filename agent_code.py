import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque

logging.basicConfig(level=logging.DEBUG, filename='agent_log.txt', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Action:
    def __init__(self, action_type, element_id=None, element_type=None, style=None, new_position=None):
        self.action_type = action_type
        self.element_id = element_id
        self.element_type = element_type
        self.style = style
        self.new_position = new_position

    def __repr__(self):
        return (f"Action(type={self.action_type}, element_id={self.element_id}, "
                f"element_type={self.element_type}, style={self.style}, new_position={self.new_position})")

class ActionSpace:
    def __init__(self, step_sizes):
        self.step_sizes = step_sizes
        self.actions = {
            'add_element': self.add_element,
            'remove_element': self.remove_element,
            'increase_width': lambda element, step: self.modify_dimension(element, 'width', step),
            'decrease_width': lambda element, step: self.modify_dimension(element, 'width', -step),
            'increase_height': lambda element, step: self.modify_dimension(element, 'height', step),
            'decrease_height': lambda element, step: self.modify_dimension(element, 'height', -step),
            'move_up': lambda element, step: self.modify_position(element, 'top', -step),
            'move_down': lambda element, step: self.modify_position(element, 'top', step),
            'move_left': lambda element, step: self.modify_position(element, 'left', -step),
            'move_right': lambda element, step: self.modify_position(element, 'left', step)
        }

    def modify_dimension(self, element, dimension, delta):
        if dimension in element['style']:
            element['style'][dimension] += delta
            element['style'][dimension] = max(0, element['style'][dimension])

    def modify_position(self, element, position, delta):
        if position in element['style']:
            element['style'][position] += delta

    def add_element(self, element_type, style):
        return {'type': element_type, 'style': style}

    def remove_element(self, element_id, elements):
        return [el for el in elements if el['id'] != element_id]

class DQNAgent:
    def __init__(self, state_size, action_size, environment):
        self.state_size = state_size
        self.action_size = action_size
        self.environment = environment
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.current_labels = [
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

    def _build_model(self):
        model = tf.keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(48, activation='relu'),
            layers.Dropout(0.25),
            layers.Dense(48, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action_index, reward, next_state, done):
        self.memory.append((state, action_index, reward, next_state, done))
    
    def act(self):
        actions = []
        for label in self.current_labels:
            element_type = self.environment.gl_lbls[label]  # Get element type from label index
            style = {'width': '100px', 'height': '50px'}  # Assuming generic style
            actions.append(Action(action_type='add_element', element_type=element_type, style=style))
        return actions

    def index_to_action(self, index):
        action_types = ['add_element', 'remove_element', 'modify_dimension', 'modify_position']
        action_type = action_types[index % len(action_types)]

        if action_type == 'add_element':
            element_type = 'button'
            style = {'width': 100, 'height': 50}
            return Action(action_type='add_element', element_type=element_type, style=style)
        elif action_type == 'remove_element':
            # Assuming access to the environment's current elements list
            element_id = random.choice([e['id'] for e in self.environment.html_elements])
            return Action(action_type='remove_element', element_id=element_id)
        elif action_type == 'modify_dimension':
            element_id = random.choice([e['id'] for e in self.environment.html_elements])
            dimension_change = {'width': random.randint(-10, 10)}
            return Action(action_type='modify_dimension', element_id=element_id, style=dimension_change)
        elif action_type == 'modify_position':
            element_id = random.choice([e['id'] for e in self.environment.html_elements])
            position_change = {'top': random.randint(-5, 5), 'left': random.randint(-5, 5)}
            return Action(action_type='modify_position', element_id=element_id, style=position_change)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action_index, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action_index] = target  # Use the action index directly
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name + '.h5')
        with open(name + '_params.json', 'r') as f:
            params = json.load(f)
            self.epsilon = params['epsilon']
        logging.info(f"Model loaded from {name}.h5 and params from {name}_params.json")

    def save(self, name):
        self.model.save_weights(name + '.h5')
        with open(name + '_params.json', 'w') as f:
            json.dump({'epsilon': self.epsilon}, f)
        logging.info(f"Model saved to {name}.h5 and params to {name}_params.json")
