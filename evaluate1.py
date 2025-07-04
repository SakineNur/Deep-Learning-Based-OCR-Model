import torch
from torch.utils.data import DataLoader
from model1 import CRNN
from config1 import train_config as config
from dataset_iiit import IIIT5KDataset, iiit_collate_fn
from ctc_decoder1 import greedy_decode

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = IIIT5KDataset(
        img_root=config['image_dir'],
        label_file=config['label_file'],
        img_height=config['img_height'],
        img_width=config['img_width']
    )
    print(f"Total dataset size: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=iiit_collate_fn)

    num_class = len(IIIT5KDataset.CHAR2LABEL) + 1
    crnn = CRNN(1, config['img_height'], config['img_width'], num_class,
                config['map_to_seq_hidden'], config['rnn_hidden'], config['leaky_relu'])

    crnn.load_state_dict(torch.load(config['reload_checkpoint'], map_location=device))
    crnn.eval()
    crnn.to(device)

    total = 0
    correct = 0

    for batch in loader:
        print(f"Processing item {total + 1}")
        images, targets, _ = [x.to(device) for x in batch]
        with torch.no_grad():
            logits = crnn(images)
        preds = greedy_decode(logits, IIIT5KDataset.LABEL2CHAR)
        truths = [''.join([IIIT5KDataset.LABEL2CHAR.get(t.item(), '') for t in targets])]

        print(f'Truth: {truths[0]} | Pred: {preds[0]}')
        total += 1
        if preds[0] == truths[0]:
            correct += 1

    print(f'Accuracy: {correct}/{total} = {correct / total:.2%}')

if __name__ == '__main__':
    evaluate()
