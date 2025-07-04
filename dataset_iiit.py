import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class IIIT5KDataset(Dataset):
    #KULLANILACAK KARAKTER KÜMESİ BELİRLENİR.
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    #Her karakteri benzersiz bir etiket numarası (label ID) ile eşleştirir.
    # i + 1 ile başlar çünkü 0 genellikle boşluk karakteri (blank - CTC'de) için kullanılır.
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    # Sayıdan karaktere dönüşüm sağlar (tahmin sonucunu karaktere çevirmek için).
    LABEL2CHAR = {v: k for k, v in CHAR2LABEL.items()}

    def __init__(self, img_root, label_file, img_height=32, img_width=100, transform=None):
        self.img_root = img_root #Görsellerin bulunduğu klasör yolu
        self.label_file = label_file
        self.img_height = img_height #Her görselin yeniden boyutlandırılacağı sabit boyutlar
        self.img_width = img_width
        self.samples = self._load_samples() #Etiket dosyasındaki tüm (görsel yolu, etiket) çiftlerini yükler.
        self.transform = transform #Veri artırma (transform.Compose) uygulanacaksa alınan parametre,Veri artırma fonksiyonlarını tutar

    #label_file dosyasını satır satır okur.
    #Her satırda görsel adı ve etiket varsa:
        #Görselin tam yolu img_path oluşturulur.
        #Dosya sisteminde varsa tuple olarak listeye eklenir.
    def _load_samples(self):
        samples = []
        with open(self.label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name, label = parts[0], ' '.join(parts[1:])
                    img_path = os.path.join(self.img_root, img_name)
                    if os.path.exists(img_path):
                        samples.append((img_path, label))
        return samples

    #Veri kümesindeki örnek sayısını döndürür. len(dataset) çağrılarında kullanılır.
    def __len__(self):
        return len(self.samples)

    #PyTorch DataLoader içinde her batch oluşturulurken bu fonksiyon çağrılır. Belirli bir indeksteki görüntüyü
    # ve metin etiketini işler.
    def __getitem__(self, index):
        path, text = self.samples[index] #İndekse karşılık gelen görselin yolu ve etiketini alır.
        image = Image.open(path).convert('L') #Görseli PIL.Image olarak açar ve grayscale moda çevirir (RGB → tek kanal).
        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR) #Görseli 100x32 boyutuna getirir. BILINEAR interpolasyon kaliteyi korur.

        if self.transform:
            image = self.transform(image)  #transform tanımlıysa RandomRotation, ToTensor, Normalize, vs. uygulanır.
        else:
            image = np.array(image).astype(np.float32) #Görsel NumPy array'e dönüştürülür.
            image = (image / 127.5) - 1.0 #Piksel değerleri normalize edilir: [0,255] → [-1,1]
            image = image.reshape((1, self.img_height, self.img_width)) #PyTorch’un beklediği şekil olan [1, 32, 100] formatına getirilir.
            image = torch.FloatTensor(image) #Tensöre çevrilir (FloatTensor).

        target = [self.CHAR2LABEL.get(c, 0) for c in text] #Metindeki her karakter, sayısal ID’ye dönüştürülür.
        #get(..., 0) # Bilinmeyen karakter varsa 0 verir (CTC boşluk olabilir).
        target = torch.LongTensor(target) #Etiketler ve uzunluğu tensor formatına getirilir.
        target_length = torch.LongTensor([len(target)])

        return image, target, target_length #DataLoader bu üç öğeyi batch içinde alır.


def iiit_collate_fn(batch):
    images, targets, target_lengths = zip(*batch) #Batch olarak alınan örnekleri birleştirir.
    images = torch.stack(images, 0) #Batch içindeki tüm image, target, target_length değerleri ayrı ayrı gruplanır.
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0) #Her örneğin karakter sayısı ayrı ayrı tutulur.
    return images, targets, target_lengths
