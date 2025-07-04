train_config = {
    'epochs': 70,
    'train_batch_size': 8, #Her batch (mini yığın) 8 örnek içerir. Aynı anda 8 görsel modele verilir
    'lr': 0.001, #Öğrenme oranı (learning rate): Modelin ağırlıkları her adımda ne kadar değiştirilecek?
    'show_interval': 10, #Her 10 adımda (iterasyonda) bir eğitim kaybı (loss) çıktısı ekrana yazdırılır.
    'save_interval': 100, #Her 100 adımda bir modelin ağırlıkları dosyaya (.pt) olarak kaydedilir.
    'cpu_workers': 2, #DataLoader için kullanılacak CPU iş parçacığı (thread) sayısıdır.
    #Eğitim öncesi daha önce kaydedilmiş bir modelden (checkpoint) ağırlıklar yüklenir.
    'reload_checkpoint': 'C:/Users/Asus/Desktop/pythonProjectICDAR13/crnn_017500_loss0.1492son.pt',

    'img_width': 100,
    'img_height': 32,
    'image_dir': 'C:/Users/Asus/Desktop/pythonProjectICDAR13/archive (1)/IIIT5K-Word_V3.0/IIIT5K/train',
    'label_file': 'C:/Users/Asus/Desktop/pythonProjectICDAR13/archive (1)/IIIT5K-Word_V3.0/IIIT5K/train.txt',

    'checkpoints_dir': './checkpoints/', #Eğitim sırasında oluşturulan .pt uzantılı model dosyaları buraya kaydedilir.
    'map_to_seq_hidden': 64,#CNN'den gelen öznitelikler, sıralı veri için 64 boyutlu bir yapıya dönüştürülür (mapping layer).
     #Bu değer, RNN'e giriş olarak verilen özellik boyutunu belirler.
    'rnn_hidden': 256, #RNN (BiLSTM) katmanlarının her yönü için gizli katman boyutu.
     #BiLSTM olduğu için aslında 256 * 2 = 512 boyutlu bir çıktı oluşur.
    'leaky_relu': False
}

