"""
Funciones de entrenamiento y validación del modelo multimodal.

Incluye:
- Entrenamiento con validación
- Early stopping
- Guardado de mejor modelo
- Tracking de métricas

"""

import torch
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                epochs=10, patience=5, device='cuda'):
    """
    Entrena el modelo con validación y early stopping.
    
    Args:
        model (nn.Module): Modelo a entrenar
        train_loader (DataLoader): DataLoader de entrenamiento
        val_loader (DataLoader): DataLoader de validación
        criterion: Función de pérdida
        optimizer: Optimizador
        scheduler: Learning rate scheduler
        epochs (int): Número máximo de épocas
        patience (int): Épocas sin mejora antes de detener
        device (str): Dispositivo ('cuda', 'mps', 'cpu')
        
    Returns:
        dict: Historia del entrenamiento con losses y accuracies
    """
    best_val_acc = 0
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # ========== ENTRENAMIENTO ==========
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, tabs, targets in train_loader:
            images = images.to(device)
            tabs = tabs.to(device)
            targets = targets.to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(images, tabs)
            loss = criterion(outputs, targets)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Métricas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        
        # ========== VALIDACIÓN ==========
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, tabs, targets in val_loader:
                images = images.to(device)
                tabs = tabs.to(device)
                targets = targets.to(device)
                
                outputs = model(images, tabs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        history['val_acc'].append(val_acc)
        
        # ========== EARLY STOPPING Y GUARDADO ==========
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Época {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | '
                  f'Train: {train_acc:.2f}% | Val: {val_acc:.2f}% ✓ MEJOR')
        else:
            epochs_no_improve += 1
            print(f'Época {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | '
                  f'Train: {train_acc:.2f}% | Val: {val_acc:.2f}% '
                  f'(sin mejora: {epochs_no_improve}/{patience})')
        
        # Early Stopping
        if epochs_no_improve >= patience:
            print(f'\n⚠️ Early Stopping activado después de {epoch+1} épocas '
                  f'(sin mejora por {patience} épocas)')
            break
        
        # Actualizar learning rate
        scheduler.step(avg_loss)
    
    return history


def plot_training_history(history):
    """
    Visualiza la evolución de las métricas durante el entrenamiento.
    
    Args:
        history (dict): Diccionario con 'train_loss', 'train_acc', 'val_acc'
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Evolución de Loss durante Entrenamiento')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Evolución de Accuracy durante Entrenamiento')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Resumen numérico
    print("\n" + "="*60)
    print("RESUMEN DEL ENTRENAMIENTO")
    print("="*60)
    print(f"Mejor validación accuracy: {max(history['val_acc']):.2f}%")
    print(f"Mejor época: {history['val_acc'].index(max(history['val_acc'])) + 1}")
    print(f"Train accuracy final: {history['train_acc'][-1]:.2f}%")
    print(f"Val accuracy final: {history['val_acc'][-1]:.2f}%")
    print(f"Diferencia (overfitting): {history['train_acc'][-1] - history['val_acc'][-1]:.2f}%")
