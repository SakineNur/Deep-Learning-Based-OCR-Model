def greedy_decode(logits, label2char): #Her zaman adımı için her sınıfın olasılığı var
    logits = logits.detach().cpu().numpy() #PyTorch tensörü, GPU’da olabilir. CPU’ya alınır ve NumPy dizisine çevrilir.
    preds = logits.argmax(2) #Her zaman adımı (her sütun) için en yüksek olasılığa sahip sınıf indeksi alınır.
    pred_texts = []
    for pred in preds.transpose(1, 0):
        prev = -1
        text = ''
        for p in pred: #Her örnek (batch'teki) ayrı ayrı işlenir.
            if p != prev and p != 0:
                text += label2char.get(p, '')
            prev = p
        pred_texts.append(text)
    return pred_texts #Geriye tüm örnekler için tahmin edilen metinleri içeren bir liste döner.

''' p: CTC algoritmasından gelen sınıf indexi (örneğin 3 → ‘c’)

prev: Önceki tahmin edilen sınıf, tekrarı engellemek için kullanılır.

p != prev: Aynı harfin tekrarını engeller (l-l-l-o-o gibi).

p != 0: CTC'de 0 boşluk karakteri (blank) olarak kullanılır, bu nedenle dahil edilmez.

label2char.get(p, ''): Sayıdan harfe dönüşüm yapılır.'''