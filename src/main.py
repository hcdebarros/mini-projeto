from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN, device
from dataset import trainloader, testloader

model = SimpleCNN().to(device)
def evaluate(loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            _, preds = torch.max(out, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total



if __name__ == "__main__":

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter('runs/fashion_cnn_demo')
    
    epochs = 3
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        total, correct = 0, 0
        for i, (imgs, labels) in enumerate(trainloader, 1):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(out, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            if i % 100 == 0:
                print(f'Batch {i} loss {running_loss/i:.4f}')
        train_acc = correct/total
        val_acc = evaluate(testloader)
        avg_loss = running_loss / len(trainloader)
        print(f'Epoch {epoch}: loss={avg_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}')
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
    writer.close()