import torch
from PIL import Image
import numpy as np
from model1 import CRNN
from ctc_decoder1 import greedy_decode
from config1 import train_config as config
from dataset_iiit import IIIT5KDataset

def predict_image(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = Image.open(image_path).convert('L')
    image = image.resize((config['img_width'], config['img_height']))
    image = np.array(image).astype(np.float32)
    image = (image / 127.5) - 1.0
    image = image.reshape((1, 1, config['img_height'], config['img_width']))
    image = torch.FloatTensor(image).to(device)

    num_class = len(IIIT5KDataset.CHAR2LABEL) + 1
    crnn = CRNN(1, config['img_height'], config['img_width'], num_class,
                config['map_to_seq_hidden'], config['rnn_hidden'], config['leaky_relu'])

    crnn.load_state_dict(torch.load(config['reload_checkpoint'], map_location=device))
    crnn.eval()
    crnn.to(device)

    with torch.no_grad():
        logits = crnn(image)
        preds = greedy_decode(logits, IIIT5KDataset.LABEL2CHAR)
        print(f'Predicted text: {preds[0]}')
        return preds[0]  # 👈 BUNU EKLE 🔥

if __name__ == '__main__':
    test_img = "C:/Users/Asus/Desktop/pythonProjectICDAR13/gorsel0.png"

    predict_image(test_img)
