import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torchvision import models
import os
from PIL import Image
import matplotlib.pyplot as plt

# ShuffleNet modelini yükle
shufflenet = models.shufflenet_v2_x0_5(pretrained=True)
num_ftrs_shufflenet = shufflenet.fc.in_features
shufflenet.fc = nn.Identity()  # ShuffleNet'in çıkış katmanını kaldır

# SqueezeNet modelini yükle
squeezenet = models.squeezenet1_0(pretrained=True)
squeezenet.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))  # SqueezeNet'in çıkış katmanını değiştir

# Modeli birleştirme işlemi
class CombinedModel(nn.Module):
    def __init__(self, shufflenet, squeezenet):
        super(CombinedModel, self).__init__()
        self.shufflenet = shufflenet
        self.squeezenet = squeezenet
        self.fc = nn.Linear(num_ftrs_shufflenet + 1, 1)  # ShuffleNet ve SqueezeNet çıktılarına ek olarak sınıflandırma için yeni bir çıkış katmanı
        
    def forward(self, x):
        shufflenet_output = self.shufflenet(x)
        squeezenet_output = self.squeezenet(x)
        squeezenet_output = squeezenet_output.view(squeezenet_output.size(0), -1)  # SqueezeNet çıktısını flatten
        combined_output = torch.cat((shufflenet_output, squeezenet_output), dim=1)  # ShuffleNet ve SqueezeNet çıktılarını birleştir
        return self.fc(combined_output)

# Yeni birleştirilmiş modeli oluştur
combined_model = CombinedModel(shufflenet, squeezenet)

# Modeli GPU'ya taşı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
combined_model = combined_model.to(device)

transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Veri artırma işlemi için transform tanımla
"""
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
"""

# Veri yükleyicilerini tanımla
dataset = datasets.ImageFolder(root='dataset/data5_h/train/', transform=transform)

# Dataseti %80 train, %10 validation ve %10 test olarak böl
train_size = int(0.8 * len(dataset))
val_test_size = len(dataset) - train_size
val_size = test_size = val_test_size // 2
train_dataset, val_test_dataset = random_split(dataset, [train_size, val_test_size])
val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Loss fonksiyonu ve optimizer tanımla
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001)

# Eğitim ve doğrulama fonksiyonlarını tanımla
def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).squeeze().long()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return running_loss / len(train_loader.dataset), correct / total

def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.float().view(-1, 1))
            running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).squeeze().long()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(val_loader.dataset), correct / total

def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.float().view(-1, 1))
            running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).squeeze().long()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(test_loader.dataset), correct / total

# Eğitim, doğrulama ve test kayıp ve doğruluk değerlerini saklamak için listeler
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
test_losses = []
test_accuracies = []

# Modeli eğit
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train(combined_model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(combined_model, val_loader, criterion)
    test_loss, test_acc = test(combined_model, test_loader, criterion)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
#%%
# Grafik çizme
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')



plt.show()
#%%

output_dir = "predicted_images_shufflenetAndSqueznet_mix"
os.makedirs(output_dir, exist_ok=True)

combined_model.eval()  # Modeli değerlendirme moduna geçir
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = combined_model(images)
        predictions = torch.sigmoid(outputs)

        # Tahminleri ve etiketleri uygun hale getir
        predicted_labels = (predictions > 0.5).long().squeeze()
        if predicted_labels.dim() == 0:  # Skaler ise yeniden boyutlandır
            predicted_labels = predicted_labels.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        for i, img in enumerate(images):
            pred_label = predicted_labels[i].item()
            true_label = labels[i].item()

            img = img.cpu().numpy().transpose((1, 2, 0))
            img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            img = img.clip(0, 1)

            label = "real" if pred_label == 1 else "fake"
            true_label_str = "real" if true_label == 1 else "fake"

            plt.imshow(img)
            plt.text(10, 30, f"Prediction: {label}", bbox=dict(facecolor='white', alpha=0.5))
            plt.text(10, 40, f"True Label: {true_label_str}", bbox=dict(facecolor='white', alpha=0.5))
            plt.text(10, 50, f"Train Accuracy: {train_accuracies[-1]:.4f}, Train Loss: {train_losses[-1]:.4f}", bbox=dict(facecolor='white', alpha=0.5))
            plt.text(10, 60, f"Validation Accuracy: {val_accuracies[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}", bbox=dict(facecolor='white', alpha=0.5))
            plt.axis('off')

            output_path = os.path.join(output_dir, f"predicted_{label}_{true_label_str}_{i}.png")  # Dosya adına indeks ekle
            plt.savefig(output_path)
            plt.close()
            
#%%
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)

plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 3, 2)

plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title(' Accuracy over Epochs')
plt.show()

#%%

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
