import torch.nn as nn

#Bu sınıf bir katman gibi çalışan çift yönlü LSTM + Linear katmanı temsil eder. CRNN'nin RNN kısmını oluşturur.
class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut) #Sonuç: Her konum için 2 yönde toplam 2 * nHidden boyutlu çıktı verir.
        ##BiLSTM çıktısını daha küçük (veya uygun) boyuta gömmek için kullanılan tam bağlantılı katmandır.

    #LSTM zaman serisi üzerinde çalışır ve T x B x F formatında çıktı verir.
    #Bu çıktı, karakter olasılıklarını içeren tensöre dönüştürülür.
    def forward(self, input):
        recurrent, _ = self.rnn(input)
        output = self.embedding(recurrent)
        return output

#Bu sınıf tüm CRNN modelini temsil eder. CNN + RNN yapısı içerir.
class CRNN(nn.Module):
    def __init__(self, nc, imgH, imgW, nclass, map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__()
        #Bu listeler CNN katmanlarındaki Conv2d yapılarına parametre olarak verilecek değerlerdir.
        ks = [3, 3, 3, 3, 3, 3, 2] # kernel sizes
        ps = [1, 1, 1, 1, 1, 1, 0]  # padding
        ss = [1, 1, 1, 1, 1, 1, 1]  # stride
        nm = [64, 128, 256, 256, 512, 512, 512] # output channel sizes

        cnn = nn.Sequential() #Bütün CNN katmanları bu Sequential içine eklenir.
        def conv_relu(i, batchNormalization=False): #Belirli i indeksli Conv2d katmanı,ReLU veya LeakyReLU aktivasyonu ekler.
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(True))

        #Toplamda 7 adet convolutional katman tanımlanır ve 4 adet pooling uygulanır.
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(2, 2)) #(2,2): Klasik aşağı örnekleme.
        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(2, 2))
        conv_relu(2, True)
        conv_relu(3)
        cnn.add_module('pooling2', nn.MaxPool2d((2, 1), (2, 1))) #(2,1): Sadece dikey eksende küçültme (genişlik bilgisi korunur — metin için önemlidir).
        conv_relu(4, True)
        conv_relu(5)
        cnn.add_module('pooling3', nn.MaxPool2d((2, 1), (2, 1)))
        conv_relu(6, True)

        self.cnn = cnn
        # RNN Yapısı: 2 adet BiLSTM katmanı üst üste bağlanmıştır.
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, rnn_hidden, rnn_hidden), #İlk katman 512 → 256*2 → 256 dönüşümünü sağlar.
            BidirectionalLSTM(rnn_hidden, rnn_hidden, nclass) #İkinci katman karakter sayısı kadar sınıfa çıkış verir (nclass).
        )

    def forward(self, input):
        conv = self.cnn(input) #Giriş görseli CNN’den geçirilir
        b, c, h, w = conv.size()
        assert h == 1 #CNN sonrasında yükseklik = 1 olduğu varsayılır.
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1) #Bu LSTM’nin beklediği biçimdir → time_step, batch, features
        output = self.rnn(conv) #Son çıktı: [seq_len, batch_size, nclass] — CTC loss için uygun formattır.
        return output

#▶Toplam CNN katmanı:
#7 Conv + 3 BN + 7 Activation + 4 Pooling = 21 katman
# Toplam RNN katmanı:
#2 BiLSTM + 2 Linear = 4 katman  Toplamda 25 katman