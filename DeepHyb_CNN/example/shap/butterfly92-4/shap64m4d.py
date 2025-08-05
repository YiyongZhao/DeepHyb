### Part 1: Explain GED Prediction
#### 1.1 Load Model and Generate SHAP values for GED prediction
# Model architecture - please note that the forward function is only for the first task (target1 prediction)
# load data function: load raw pattern counts (not ratios)
import numpy as np
import torch
import json
import os
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from tqdm import tqdm
import pandas as pd
import time

start_time = time.time()

suffix = "64m4d"

# Using GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(
    f"#################### Your GPU usage is {torch.cuda.is_available()}! ########################\n")

#####################################################################################################
# Model Architecture
########################

# Parameters
input_dim = 256
num_classes_1 = 2
num_classes_3 = 64

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(64 * (input_dim // 8), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3_1 = nn.Linear(64, num_classes_1)
        self.fc3_3 = nn.Linear(64, num_classes_3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, 2)
        
        x = x.view(-1, 64 * (input_dim // 8))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x1 = self.fc3_1(x)
        x2 = self.fc3_3(x)
        
        return x1#, x2
        # CrossEntropyLoss expect raw logits as input
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
#######################################################################################################
# Function for loading data - use pattern counts
###########################
def load_data_from_json(folder_path):
    X = []
    X_labels_1 = []
    X_labels_2 = []
    X_labels_3 = []
    X_labels_4 = []

    y = []
    target2 = []
    file_names = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    onedim = np.concatenate([

                    np.array(data['labels_4']).flatten(), # horizontal kmer 4 groups
                ])

                    if len(data['labels_1']) == 15:
                        X.append(onedim)
                        X_labels_1.append(data['labels_1'])  # 1x15
                        X_labels_2.append(data['labels_2'])  # 1x256
                        X_labels_3.append(data['labels_3'])  
                        X_labels_4.append(data['labels_4']) 

                        target1_mapping = {2: 1, 8: 0}
                        y.append(target1_mapping.get(data['target1'], -1))  # Default to -1 for unexpected values

                        target2.append(data['target2'])  

                        file_names.append(filename.replace('.json', ''))
                    else:
                        print(f"Warning: Inconsistent shape in file {filename}, skipped.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {filename}: {e}")
            except KeyError as e:
                print(f"Key error in file {filename}: {e}")
            except Exception as e:
                print(f"An error occurred in file {filename}: {e}")

    return (np.array(X_labels_1, dtype=np.float32), #1
            np.array(X_labels_2, dtype=np.float32), #2
            np.array(X_labels_3, dtype=np.float32), #3
            np.array(X_labels_4, dtype=np.float32), #4
            np.array(y, dtype=np.int32), #5
            np.array(target2, dtype=np.int32), #6
            file_names, #7
            np.array(X, dtype=np.float32)) #8

# Load saved model weights
################################################################################
import torch

train_folder = "../../butterfly92-4/train"
test_folder = "../../butterfly92-4/test"
load_model_path = f"../../butterfly92-4/model/1225_concate_CNN_model_{suffix}.pth" # model trained use pattern counts, replace it using your own model path


model = CNNModel().to(device)
model.load_state_dict(torch.load(load_model_path, weights_only=True, map_location=torch.device(device)))
model.eval()
# Load data
###########################################################

_, _, _, _, y_train_a, target2_train_a, _, X_train = load_data_from_json(train_folder)
_, _, _, _, y_test_a, target2_test, file_names_b, X_test = load_data_from_json(test_folder)

# convert input to torch tensors
X_train_tensor = torch.tensor(X_train).to(device).unsqueeze(1)
X_test_tensor = torch.tensor(X_test).to(device).unsqueeze(1)

# Define background data (in tensor form, randomly choose 17547 instances)
torch.manual_seed(42)
background_size = 17547
indices = torch.randperm(X_train_tensor.size(0))
background_train = X_train_tensor[indices[:background_size]]

# Select test samples for explanation
#torch.manual_seed(42)
test_samples = 2460 # for this dataset, there are 3824 test samples
#indices2 = torch.randperm(X_test_tensor.size(0))
test_data = X_test_tensor[:test_samples]#[indices2[:test_samples]]

# Obtain index for the target1 predictions
with torch.no_grad():
    target1_pred = model(test_data)
    target1_prob = F.softmax(target1_pred, dim = 1)
    _, target1_prediction = torch.max(target1_prob, 1)
    target1_index = target1_prediction.cpu().numpy()

print("Target1 Index shape:", target1_index.shape)
print(target1_index[:100])
accuracy_score(y_test_a, target1_index)
# confusion matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# initialize SHAP - Target1 explainer (GradientExplainer)
# calculate shap values, slow for complex model and many instances
# use GradientExplainer since DeepExplainer results in an Assertion Error when check_additivity=True

# initialize SHAP - Target1 explainer (GradientExplainer)
# calculate shap values, slow for complex model and many instances
# use GradientExplainer since DeepExplainer results in an Assertion Error when check_additivity=True

######################################
import shap

explainer = shap.GradientExplainer(model, background_train)


shap_values = explainer.shap_values(test_data)
print("shap_values type", type(shap_values))
print("shap_values shape", shap_values.shape)

import pickle
os.makedirs("./saved_shap_values", exist_ok= True)

with open(f'./saved_shap_values/butt92-4_target1_{suffix}.pkl', 'wb') as f:
    pickle.dump(shap_values, f)
#######################################



import numpy as np
import torch
import json
import os
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from tqdm import tqdm
import pandas as pd
import time

start_time = time.time()

# Using GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(
    f"#################### Your GPU usage is {torch.cuda.is_available()}! ########################\n")

#####################################################################################################
# Model Architecture
########################

# Parameters
input_dim = 256
num_classes_1 = 2
num_classes_3 = 64

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(64 * (input_dim // 8), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3_1 = nn.Linear(64, num_classes_1)
        self.fc3_3 = nn.Linear(64, num_classes_3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, 2)
        
        x = x.view(-1, 64 * (input_dim // 8))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x1 = self.fc3_1(x)
        x2 = self.fc3_3(x)
        
        return x2 #x1, x2
        # CrossEntropyLoss expect raw logits as input
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
#######################################################################################################
# Function for loading data - use pattern ratios
###########################
def load_data_from_json(folder_path):
    X = []
    X_labels_1 = []
    X_labels_2 = []
    X_labels_3 = []
    X_labels_4 = []

    y = []
    target2 = []
    file_names = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    onedim = np.concatenate([

                    np.array(data['labels_4']).flatten(), # horizontal kmer 4 groups
                ])

                    if len(data['labels_1']) == 15:
                        X.append(onedim)
                        X_labels_1.append(data['labels_1'])  # 1x15
                        X_labels_2.append(data['labels_2'])  # 1x256
                        X_labels_3.append(data['labels_3'])  
                        X_labels_4.append(data['labels_4']) 

                        target1_mapping = {2: 1, 8: 0}
                        y.append(target1_mapping.get(data['target1'], -1))  # Default to -1 for unexpected values

                        target2.append(data['target2'])  

                        file_names.append(filename.replace('.json', ''))
                    else:
                        print(f"Warning: Inconsistent shape in file {filename}, skipped.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {filename}: {e}")
            except KeyError as e:
                print(f"Key error in file {filename}: {e}")
            except Exception as e:
                print(f"An error occurred in file {filename}: {e}")

    return (np.array(X_labels_1, dtype=np.float32), #1
            np.array(X_labels_2, dtype=np.float32), #2
            np.array(X_labels_3, dtype=np.float32), #3
            np.array(X_labels_4, dtype=np.float32), #4
            np.array(y, dtype=np.int32), #5
            np.array(target2, dtype=np.int32), #6
            file_names, #7
            np.array(X, dtype=np.float32)) #8

# Load saved model weights
################################################################################
import torch

train_folder = "../../butterfly92-4/train"
test_folder = "../../butterfly92-4/test"
load_model_path = f"../../butterfly92-4/model/1225_concate_CNN_model_{suffix}.pth" # model trained use pattern counts, replace it using your own model path


model = CNNModel().to(device)
model.load_state_dict(torch.load(load_model_path, weights_only=True, map_location=torch.device(device)))
model.eval()
# Load data
###########################################################

_, _, _, _, y_train_a, target2_train_a, _, X_train = load_data_from_json(train_folder)
_, _, _, _, y_test_a, target2_test, file_names_b, X_test = load_data_from_json(test_folder)

# convert input to torch tensors
X_train_tensor = torch.tensor(X_train).to(device).unsqueeze(1)
X_test_tensor = torch.tensor(X_test).to(device).unsqueeze(1)

# Define background data (in tensor form, randomly choose 5000 instances)
torch.manual_seed(42)
background_size = 17547
indices = torch.randperm(X_train_tensor.size(0))
background_train = X_train_tensor[indices[:background_size]]

# Select test samples for explanation
#torch.manual_seed(42)
test_samples = 2460 # for this dataset, there are 3824 test samples
#indices2 = torch.randperm(X_test_tensor.size(0))
test_data = X_test_tensor[:test_samples]#[indices2[:test_samples]]

# Obtain index for the target2 predictions
with torch.no_grad():
    target2_pred = model(test_data)
    target2_prob = F.softmax(target2_pred, dim = 1)
    _, target2_prediction = torch.max(target2_prob, 1)
    target2_index = target2_prediction.cpu().numpy()

print("Target2 Index shape:", target2_index.shape)
print(target2_index[:100])
accuracy_score(target2_test, target2_index)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix




# initialize SHAP - target2 explainer (GradientExplainer)
# calculate shap values, slow for complex model and many instances
# use GradientExplainer since DeepExplainer results in an Assertion Error when check_additivity=True

import shap

explainer = shap.GradientExplainer(model, background_train)

# Compute SHAP values
shap_values = explainer.shap_values(test_data)
print("shap_values type", type(shap_values))
print("shap_values shape", shap_values.shape)
import pickle

# Save to a file
with open(f'./saved_shap_values/butt92-4_target2_{suffix}.pkl', 'wb') as f:
    pickle.dump(shap_values, f)


