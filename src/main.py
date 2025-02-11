import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb  # Importamos wandb

from dataloaders.transformations import ImageNormalization, RecordsTransform
from dataloaders.dataset import AutodriveDataset
from models.model import Model
from tqdm import tqdm

# Inicialización de wandb
wandb.init(project="autodrive-training", config={
    "learning_rate": 0.5e-4,
    "epochs": 20,
    "batch_size": 8,
    "seq_len": 12,
    "csv_file": "src/config/train_routes.csv"
})
config = wandb.config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definir transformaciones
transform = transforms.Compose([
    ImageNormalization(), # Esta solo se aplica si el dataset usa imágenes use_encoded_images=True
    RecordsTransform(),
])

# dataset = AutodriveDataset(config.csv_file, seq_len=10, transform=transform, sensors=['rgb_f', 'rgb_lf', 'rgb_rf', 'rgb_lb', 'rgb_rb', 'rgb_b', 'records'], use_encoded_images=False)
train_ds = AutodriveDataset(config.csv_file, subset='train', seq_len=config.seq_len, transform=transform, sensors=['rgb_f', 'records'], use_encoded_images=True)
valid_ds = AutodriveDataset(config.csv_file, subset='test',  seq_len=config.seq_len, transform=transform, sensors=['rgb_f', 'records'], use_encoded_images=True)

train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=32)
valid_dl = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=8)

# Configurar modelo, función de pérdida y optimizador
loss_fn = nn.L1Loss()
model = Model(latent_features=256, config=config).to(device)
opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Función para calcular la pérdida en validación
def validate(model, valid_dl, loss_fn, epoch):
    model.eval()
    valid_loss = 0
    euclidean_dist = 0
    with torch.no_grad():
        for i, b in tqdm(enumerate(valid_dl), total=len(valid_dl)):
            imseq_batch = b['rgb_f'].to(device)
            kps_batch = b['kps'].to(device)
            y_true = kps_batch
            y_pred = model(imseq_batch)
            valid_loss += loss_fn(y_pred, y_true).item()

            euclidean_dist += (y_true - y_pred).square().sum(dim=-1).sqrt().mean(dim=0)

        wandb.log({"valid_loss": valid_loss / len(valid_dl), "epoch": epoch})
        for s in range(config.seq_len):
            wandb.log({f"valid_euclidean_dist_kp_{s+1}": euclidean_dist[s].item() / len(valid_dl), "epoch": epoch})

    return valid_loss / len(valid_dl)

# Entrenamiento con validación periódica
n_epochs = config.epochs

for e in range(n_epochs):
    print(f"Epoch {e + 1}/{n_epochs}")

    n_batches = len(train_dl)
    euclidean_dist = 0
    epoch_loss = 0

    model.train()

    for i, b in tqdm(enumerate(train_dl), total=len(train_dl)):
        opt.zero_grad()

        imseq_batch = b['rgb_f'].to(device)
        kps_batch = b['kps'].to(device)
        y_true = kps_batch

        y_pred = model(imseq_batch)

        loss = loss_fn(y_pred, y_true)

        loss.backward()
        opt.step()

        epoch_loss += loss.item()
        euclidean_dist += (y_true - y_pred).square().sum(dim=-1).sqrt().mean(dim=0)

        if i % 100 == 0 and i != 0:
            wandb.log({"train_loss": epoch_loss / (i+1), "epoch": e}, step=i+e*len(train_dl))
            for s in range(config.seq_len):
                wandb.log({f"train_euclidean_dist_kp_{s+1}": euclidean_dist[s].item() / (i+1), "epoch": e}, step=i+e*len(train_dl))

    if e % 10 == 0:
        valid_loss = validate(model, valid_dl, loss_fn, e)
        print(f"End of Epoch {e + 1}, Validation Loss: {valid_loss:.4f}")
        wandb.log({"epoch_valid_loss": valid_loss, "epoch": e}, step=e)

# Guardar el modelo en wandb
torch.save(model.state_dict(), "model.pth")
wandb.save("model.pth")