import torch
from torch.utils.data import DataLoader
from src.model import build_model

def train_model(train_loader, val_loader, num_epochs=10, lr=0.001, model_save_path='leukemia_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=3).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")
    
    # Save the model after training
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return model
