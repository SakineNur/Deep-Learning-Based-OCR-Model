# Deep-Learning-Based-OCR-Model
Bu projede, derin öğrenme tabanlı CRNN (Convolutional Recurrent Neural Network) mimarisi kullanılarak bir OCR (Optik Karakter Tanıma) modeli geliştirilmiştir. Görüntü işleme ve yapay zeka tekniklerinin bir araya getirildiği bu sistemde, kelime içeren görsellerden metinlerin otomatik olarak tanınması hedeflenmiştir. Proje Python programlama dili ile geliştirilmiştir.

Modelin eğitimi ve testi için yaygın olarak kullanılan IIIT5K-Word veri seti kullanılmıştır. Bu veri seti toplam 5.000 adet kelime görseli içermektedir. Bunların 2.000'i eğitim, 3.000'i test verisi olarak ayrılmıştır. Eğitilen CRNN modeli, eğitim verisi üzerinde %95 doğruluk oranına ulaşmıştır. Test verisi üzerinde ise harf bazlı doğruluk oranı %70 civarındadır.

Bu proje, OCR sistemlerinin temel bileşenleri olan CNN (özellik çıkarımı) ve RNN (sıralı veri analizi) yapılarını birleştirerek, görsellerden sıralı metin çıkarımı gerçekleştirmektedir. Proje aynı zamanda karakterlerin zaman içerisindeki ilişkisini modellemek için CTC (Connectionist Temporal Classification) loss fonksiyonu kullanmaktadır.

Veri Seti

IIIT5K-Word veri setini aşağıdaki bağlantıdan indirebilirsiniz:

[Google Drive üzerinden indir](https://drive.google.com/file/d/1F9edF6jdpv4om-zEbi0Fx8SkcEIxbLmj/view?usp=drive_link)
