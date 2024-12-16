import torch
import numpy as np


# Function used to create the diffuse one-hot label
def create_diffuse_one_hot(labels, num_classes=10, diffuse_value=0.1):
    diffuse_one_hot = np.full((labels.size(0), num_classes), diffuse_value)
    return torch.tensor(diffuse_one_hot, dtype=torch.float32)


# Function used to set DoC (degree of confusion)
def set_DoC(single_label, num_classes, num_confused, device, includes_true):
    true_class = single_label
    classes = list(range(10))
    classes.remove(true_class)
    target_label = torch.zeros(1, num_classes, device=device)
    loops = num_confused  # assert this case and redefine if the case
    if includes_true:
        target_label[0, true_class] = 1 / num_confused
        loops = num_confused - 1
    for i in range(loops):
        random_class = np.random.choice(classes)
        classes.remove(random_class)
        target_label[0, random_class] = 1 / num_confused

    return target_label



