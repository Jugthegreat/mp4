import numpy as np
from torch import nn
import torch
from vision_transformer import vit_b_32, ViT_B_32_Weights
from tqdm import tqdm

def get_encoder(name):
    if name == 'vit_b_32':
        torch.hub.set_dir("model")
        model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    return model

class ViTPrompt(nn.Module):
    def __init__(self, n_classes, encoder_name, prompt_len=10):
        super(ViTPrompt, self).__init__()
        
        self.vit_b = [get_encoder(encoder_name)]
        
        # Freeze the ViT backbone
        for param in self.vit_b[0].parameters():
            param.requires_grad = False
        
        # Replace the classifier head
        self.vit_b[0].heads.head = nn.Linear(768, n_classes)
        
        # Set the number of prompts
        self.prompt_len = prompt_len
        self.num_layers = len(self.vit_b[0].encoder.layers)
        self.hidden_dim = 768  # ViT hidden dimension
        
        # Initialize the prompts
        v = np.sqrt(6. / float(self.hidden_dim + self.hidden_dim))
        self.prompts = nn.Parameter(torch.FloatTensor(1, self.num_layers, self.prompt_len, self.hidden_dim).uniform_(-v, v))
    
    def to(self, device):
        super(ViTPrompt, self).to(device)
        self.vit_b[0] = self.vit_b[0].to(device)
    
    def forward(self, x):
        # Pass prompts to the ViT encoder
        out = self.vit_b[0](x, prompts=self.prompts)
        return out

def test(test_loader, model, device):
    model.eval()
    total_loss, correct, n = 0., 0., 0

    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        correct += (y_hat.argmax(dim=1) == y).float().mean().item()
        loss = nn.CrossEntropyLoss()(y_hat, y)
        total_loss += loss.item()
        n += 1
    accuracy = correct / n
    loss = total_loss / n
    return loss, accuracy

def inference(test_loader, model, device, result_path):
    """Generate predicted labels for the test set."""
    model.eval()

    predictions = []
    with torch.no_grad():
        for x, _ in tqdm(test_loader):
            x = x.to(device)
            y_hat = model(x)
            pred = y_hat.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
    
    with open(result_path, "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Predictions saved to {result_path}")

class Trainer():
    def __init__(self, model, train_loader, val_loader, writer,
                 optimizer, lr, wd, momentum, 
                 scheduler, epochs, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = epochs
        self.device = device
        self.writer = writer
        
        self.model.to(self.device)

        if optimizer == 'sgd':
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = torch.optim.SGD(trainable_params, lr=lr, weight_decay=wd, momentum=momentum)
            
        if scheduler == 'multi_step':
            self.lr_schedule = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[60, 80], gamma=0.1)

    def train_epoch(self):
        self.model.train()
        total_loss, correct, n = 0., 0., 0
        
        for x, y in tqdm(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            loss = nn.CrossEntropyLoss()(y_hat, y)
            total_loss += loss.item()
            correct += (y_hat.argmax(dim=1) == y).float().mean().item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            n += 1
        return total_loss / n, correct / n
    
    def val_epoch(self):
        self.model.eval()
        total_loss, correct, n = 0., 0., 0

        with torch.no_grad():
            for x, y in tqdm(self.val_loader):
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                correct += (y_hat.argmax(dim=1) == y).float().mean().item()
                loss = nn.CrossEntropyLoss()(y_hat, y)
                total_loss += loss.item()
                n += 1
        accuracy = correct / n
        loss = total_loss / n
        return loss, accuracy

    def train(self, model_file_name, best_val_acc=-np.inf):
        best_epoch = np.NaN
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.val_epoch()
            self.writer.add_scalar('lr', self.lr_schedule.get_last_lr()[0], epoch)
            self.writer.add_scalar('val_acc', val_acc, epoch)
            self.writer.add_scalar('val_loss', val_loss, epoch)
            self.writer.add_scalar('train_acc', train_acc, epoch)
            self.writer.add_scalar('train_loss', train_loss, epoch)
            pbar.set_description("Epoch {}: val acc: {:.4f}, train acc: {:.4f}".format(epoch, val_acc, train_acc), refresh=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(self.model.state_dict(), model_file_name)
            self.lr_schedule.step()
        
        return best_val_acc, best_epoch
