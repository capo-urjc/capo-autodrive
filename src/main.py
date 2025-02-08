import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb  # Importamos wandb

from src.dataloaders.transformations import ImageNormalization, RecordsTransform
from src.dataloaders.dataset import AutodriveDataset
from src.models.model import Model

# Inicialización de wandb
wandb.init(project="autodrive-training", config={
    "learning_rate": 1e-4,
    "epochs": 500,
    "batch_size": 4,
    "train_percentage": 0.8,
    "seq_len": 5
})
config = wandb.config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csv_file = "src/dataloaders/csv/config_folders.csv"

# Definir transformaciones
transforms = transforms.Compose([
    ImageNormalization(),
    RecordsTransform()
])

# Crear el dataset y dividirlo en entrenamiento y validación
dataset = AutodriveDataset(csv_file, seq_len=config.seq_len, transform=transforms, sensors=['rgb_f', 'records'])
train_percentage = config.train_percentage
train_ds, valid_ds = torch.utils.data.random_split(dataset, [int(len(dataset) * train_percentage), len(dataset) - int(len(dataset) * train_percentage)])

train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, pin_memory=True, drop_last=True)
valid_dl = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, pin_memory=True)

# Configurar modelo, función de pérdida y optimizador
loss_fn = nn.L1Loss()
model = Model(latent_features=256).to(device)
opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
# opt = torch.optim.Adam(model.dynamic_predictor.parameters(), lr=config.learning_rate)

# Función para calcular la pérdida en validación
def validate(model, valid_dl, loss_fn):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for b in valid_dl:
            imseq_batch = b['rgb_f'].to(device)
            control_batch = b['records'].to(device)
            y_true = control_batch[..., 0:4].float()
            y_pred = model(imseq_batch)
            valid_loss += loss_fn(y_pred, y_true).item()
    model.train()
    return valid_loss / len(valid_dl)

# Entrenamiento con validación periódica
n_epochs = config.epochs

for e in range(n_epochs):
    print(f"Epoch {e + 1}/{n_epochs}")

    n_batches = len(train_dl)
    mae_loss = 0
    epoch_loss = 0
    for i, b in enumerate(train_dl):
        opt.zero_grad()

        imseq_batch = b['rgb_f'].to(device)
        control_batch = b['control'].to(device)
        y_true = control_batch[..., 0:4].float()

        y_pred = model(imseq_batch)

        loss = loss_fn(y_pred, y_true)

        loss.backward()
        opt.step()

        epoch_loss += loss
        mae_loss += torch.abs(y_true - y_pred).sum((0,1))

        # Registrar pérdida en wandb
        wandb.log({"train_loss":     epoch_loss.item()/(i+1)})
        wandb.log({"break_mae":     mae_loss[0].item()/(i+1)})
        wandb.log({"reverse_mae":   mae_loss[1].item()/(i+1)})
        wandb.log({"steer_mae":     mae_loss[2].item()/(i+1)})
        wandb.log({"throttle_mae":  mae_loss[3].item()/(i+1)})
    # # Registrar pérdida en wandb
    # wandb.log({"train_loss":    epoch_loss.item()/n_batches})
    # wandb.log({"break_mae":     mae_loss[0].item()/n_batches})
    # wandb.log({"reverse_mae":   mae_loss[1].item()/n_batches})
    # wandb.log({"steer_mae":     mae_loss[2].item()/n_batches})
    # wandb.log({"throttle_mae":  mae_loss[3].item()/n_batches})


    if e % 10 == 0:
        # Validación al final de la época
        valid_loss = validate(model, valid_dl, loss_fn)
        print(f"End of Epoch {e + 1}, Validation Loss: {valid_loss:.4f}")
        wandb.log({"epoch_valid_loss": valid_loss})

# Guardar el modelo en wandb
torch.save(model.state_dict(), "model.pth")
wandb.save("model.pth")

