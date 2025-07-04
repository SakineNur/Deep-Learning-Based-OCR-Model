import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn import CTCLoss
from dataset_iiit import IIIT5KDataset, iiit_collate_fn
from model1 import CRNN
from config1 import train_config as config

transform = transforms.Compose([
    transforms.RandomRotation(degrees=2),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]
    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size).to(device)
    target_lengths = torch.flatten(target_lengths).to(device)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5)
    optimizer.step()
    return loss.item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = IIIT5KDataset(
        img_root=config['image_dir'],
        label_file=config['label_file'],
        img_height=config['img_height'],
        img_width=config['img_width'],
        transform = transform
    )
    loader = DataLoader(dataset, batch_size=config['train_batch_size'], shuffle=True,
                        num_workers=config['cpu_workers'], collate_fn=iiit_collate_fn)

    num_class = len(IIIT5KDataset.CHAR2LABEL) + 1
    crnn = CRNN(1, config['img_height'], config['img_width'], num_class,
                config['map_to_seq_hidden'], config['rnn_hidden'], config['leaky_relu'])

    if config['reload_checkpoint']:
        crnn.load_state_dict(torch.load(config['reload_checkpoint'], map_location=device))
    crnn.to(device)

    optimizer = optim.RMSprop(crnn.parameters(), lr=config['lr'])
    criterion = CTCLoss(reduction='sum', zero_infinity=True).to(device)

    i = 1
    for epoch in range(1, config['epochs'] + 1):
        total_loss = 0
        for batch in loader:
            loss = train_batch(crnn, batch, optimizer, criterion, device)
            total_loss += loss
            if i % config['show_interval'] == 0:
                print(f'[{i}] loss: {loss:.4f}')
            if i % config['save_interval'] == 0:
                os.makedirs(config['checkpoints_dir'], exist_ok=True)
                path = os.path.join(config['checkpoints_dir'], f'crnn_{i:06}_loss{loss:.4f}.pt')
                torch.save(crnn.state_dict(), path)
            i += 1
        print(f'Epoch {epoch} average loss: {total_loss / len(loader):.4f}')

if __name__ == '__main__':
    main()
