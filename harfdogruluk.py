import torch
from PIL import Image
import numpy as np
from model1 import CRNN
from ctc_decoder1 import greedy_decode
from config1 import train_config as config
from dataset_iiit import IIIT5KDataset
import os

def predict_image(image_path, crnn, device):
    image = Image.open(image_path).convert('L')
    image = image.resize((config['img_width'], config['img_height']))
    image = np.array(image).astype(np.float32)
    image = (image / 127.5) - 1.0
    image = image.reshape((1, 1, config['img_height'], config['img_width']))
    image = torch.FloatTensor(image).to(device)

    with torch.no_grad():
        logits = crnn(image)
        preds = greedy_decode(logits, IIIT5KDataset.LABEL2CHAR)
        return preds[0]

def char_level_accuracy(pred, true):
    """Tahmin ve gerçek değer arasında harf bazlı doğruluğu hesaplar."""
    min_len = min(len(pred), len(true))
    match_count = sum(1 for i in range(min_len) if pred[i] == true[i])
    return match_count / max(len(true), 1)  # Bölüm sıfır olmasın diye

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_class = len(IIIT5KDataset.CHAR2LABEL) + 1

    crnn = CRNN(1, config['img_height'], config['img_width'], num_class,
                config['map_to_seq_hidden'], config['rnn_hidden'], config['leaky_relu'])
    crnn.load_state_dict(torch.load(config['reload_checkpoint'], map_location=device))
    crnn.eval()
    crnn.to(device)

    total = 0
    total_char_acc = 0.0

    with open('C:/Users/Asus/Desktop/pythonProjectICDAR13/archive (1)/IIIT5K-Word_V3.0/IIIT5K/test.txt', 'r') as f:
        lines = f.readlines()

        base_path = 'C:/Users/Asus/Desktop/pythonProjectICDAR13/archive (1)/IIIT5K-Word_V3.0/IIIT5K/'

        for line in lines:
            img_path, true_label = line.strip().split(' ')
            img_full_path = os.path.join(base_path, img_path)
            pred_text = predict_image(img_full_path, crnn, device)

            true_label_clean = true_label.strip()
            pred_text_clean = pred_text.strip()

            acc = char_level_accuracy(pred_text_clean, true_label_clean)
            total_char_acc += acc
            total += 1

            print(f'{img_path} | Doğru: {true_label_clean} | Tahmin: {pred_text_clean} | Harf Doğruluğu: {acc:.2f}')

        print(f'\n✅ Toplam Görsel: {total}, Ortalama Harf Bazlı Doğruluk: {(total_char_acc / total):.2%}')
