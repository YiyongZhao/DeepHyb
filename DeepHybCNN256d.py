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

train_folder = './train'
test_folder = './test'
suffix = "256d"

epochs = 300000
batch_size = 512
maxloss = 0.001
lr = 0.001

# Parameters
input_dim = 256
num_classes_1 = 2
num_classes_3 = 64

model_save_path = f"./model/1225_concate_CNN_model_{suffix}.pth"
loss_save_path = f"./loss_history/1226_concate_CNN_model_{suffix}_avg_loss.json"

os.makedirs("./model", exist_ok= True)
os.makedirs("./loss_history", exist_ok= True)

# Print parameters
print("=" * 40)
print("MODEL PARAMETERS".center(40))
print("=" * 40)
print(f"{'Train Folder:':<20} {train_folder}")
print(f"{'Test Folder:':<20} {test_folder}")
print(f"{'Epochs:':<20} {epochs}")
print(f"{'Batch Size:':<20} {batch_size}")
print(f"{'Maximum Loss:':<20} {maxloss}")
print(f"{'Learning Rate:':<20} {lr}")
print(f"{'Model Save Path:':<20} {model_save_path}")
print(f"{'Loss Save Path:':<20} {loss_save_path}")
print("=" * 40)

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
        
        return x1, x2
        # CrossEntropyLoss expect raw logits as input
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss

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
                    # np.array(data['labels_1']), # Hyde 15
                    # np.array(data['labels_2']), # 256
                    # np.array(data['labels_3']).flatten(), # vertical kmer 5 groups
                    # np.array(data['labels_4']).flatten(), # horizontal kmer 4 groups
                    np.array(data['labels_2']), # 256
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

    return (np.array(X_labels_1, dtype=np.float32), 
            np.array(X_labels_2, dtype=np.float32), 
            np.array(X_labels_3, dtype=np.float32), 
            np.array(X_labels_4, dtype=np.float32), 
            np.array(y, dtype=np.int32), 
            np.array(target2, dtype=np.int32), 
            file_names, 
            np.array(X, dtype=np.float32)) 


_, _, _, _, y_train_a, target2_train_a, _, X_train = load_data_from_json(train_folder)
_, _, _, _, y_test_a, target2_test, file_names_b, X_test = load_data_from_json(test_folder)


X_train_tensor = torch.tensor(X_train).unsqueeze(1)
y_train_tensor = torch.tensor(y_train_a, dtype=torch.long)
target2_train_tensor = torch.tensor(target2_train_a, dtype=torch.long)

X_test_tensor = torch.tensor(X_test).unsqueeze(1)
y_test_tensor = torch.tensor(y_test_a, dtype=torch.long)
target2_test_tensor = torch.tensor(target2_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor, target2_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    
class MultiTargetLoss(nn.Module):
    def __init__(self):
        super(MultiTargetLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, target1_pred, target2_pred, target1_true, target2_true):
        loss_target1 = self.cross_entropy(target1_pred, target1_true)
        loss_target2 = self.cross_entropy(target2_pred, target2_true)
        total_loss = loss_target1 + loss_target2
        return total_loss

model = CNNModel().to(device)
criterion = MultiTargetLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss_history = []

patience = 15000  
min_val_loss = float('inf')  
early_stopping_counter = 0  

for epoch in range(epochs):
    model.train()
    total_loss = 0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for X_batch, y_batch, target2_batch in train_loader:
            optimizer.zero_grad()

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            target2_batch = target2_batch.to(device)

            target1_pred, target2_pred = model(X_batch)
            loss = criterion(target1_pred, target2_pred, y_batch, target2_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    average_loss = total_loss / len(train_loader)
    loss_history.append(average_loss)
    print(f"Epoch {epoch+1} finished with average loss: {average_loss}, total loss: {total_loss}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch, target2_batch in train_loader:  
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            target2_batch = target2_batch.to(device)

            target1_pred, target2_pred = model(X_batch)
            loss = criterion(target1_pred, target2_pred, y_batch, target2_batch)
            val_loss += loss.item()

    val_loss /= len(train_loader)  
    print(f"Validation loss: {val_loss}")

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        early_stopping_counter = 0  
        torch.save(model.state_dict(), model_save_path)  
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break  

with open(loss_save_path, 'w') as f:
    json.dump(loss_history, f)

def evaluate(model, y_true, target2_true, file_names, X):
    model.eval()
    predictions = []
    target2_predictions = []

    with torch.no_grad():
        X_tensor = torch.tensor(X).unsqueeze(1).to(device)
        target1_pred, target2_pred= model(X_tensor)

        target1_prob = F.softmax(target1_pred, dim = 1)
        _, target1_prediction = torch.max(target1_prob, 1)

        target2_prob = F.softmax(target2_pred, dim = 1)
        _, target2_prediction = torch.max(target2_prob, 1)

        predictions.extend(target1_prediction.cpu().numpy())
        target2_predictions.extend(target2_prediction.cpu().numpy())

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, predictions)

    target2_accuracy = accuracy_score(target2_true, target2_predictions)
    cm_2 = confusion_matrix(target2_true, target2_predictions)

    return (predictions, y_true, accuracy, precision,
            recall, f1, cm, cm_2, target2_predictions,
            target2_true, target2_accuracy)

predictions, actuals, accuracy, precision, recall, f1, cm, cm_2, target2_predictions, target2_actuals, target2_accuracy= evaluate(model, y_test_a, target2_test, file_names_b, X_test)
unique_target2 = np.unique(list(target2_actuals) + list(target2_predictions))


cm1_df = pd.DataFrame(cm, index=["no-Hybrid", "Hybrid"], columns=["Predicted no-Hybrid", "Predicted Hybrid"])
cm2_df = pd.DataFrame(cm_2, index = unique_target2,
                           columns= unique_target2)
print(f'\nAccuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('\nTarget1 Confusion Matrix:')
print(cm1_df)


print(f'\nTarget 2 Accuracy: {target2_accuracy}')
print("\nTarget2 Confusion Matrix")
print(cm2_df)


end_time = time.time()
duration = end_time - start_time

print(f"\nModel Training and Testing time: {duration}")
