# Büyük Verilerde Finansal Piyasa Tahmini: Derin Öğrenme

Özetçe — Derin Öğrenme, büyük miktarda etiketsiz/denetlenmemiş veriden öğrenmeyi başarılı bir şekilde gerçekleştirebildiğinden, büyük verilerden anlamlı gösterimler ve desenler çıkartmayı cazip hale getiriyor. En basit tanımıyla derin öğrenme, makine öğrenmesi yöntemlerinin büyük verilere uygulanması olarak ifade edilmektedir. Bu çalışmada finansal tahmin ve sınıflama problemlerinde derin öğrenme hiyerarşik modellerinin nasıl kullanılabileceği araştırılmıştır. Finansal tahmin sorunları - menkul kıymetleri tasarlama ve fiyatlandırma, portföy oluşturma, risk yönetimi gibi konular - genellikle karmaşık veri etkileşimlerine sahip büyük veri setlerini içerdiğinden tam ekonomik model oluşturmak şu an için zor veya imkansızdır. Bu problemlere derin öğrenme yöntemleri uygulandığında, finanstaki standart metotlardan daha faydalı sonuçlar alınabilir. Bilhassa, derin öğrenme, en azından şu an için, mevcut herhangi bir finansal ekonomik teori için görünmez olan veri etkileşimlerini algılayabilir ve bunlardan istifade edebilir. 

Anahtar Kelimeler — Derin Öğrenme, Makine Öğrenmesi, Büyük Veriler, Yapay Zeka, Finans, Piyasa Tahmini.



## Borsa Hisse Senetleri Yerleştirme

En yaygın kullanılan kelime yerleştirme algoritması word2vec tir. Bu çalışmada word2vec algoritmasında olduğu gibi kelimeler yerine borsadaki hisse senetlerinin verisi yerleştirilmiştir. İlk sütun tarih olup o tarihe karşılık gelen hisse senetlerinin Açılış Fiyatı, En Yüksek Fiyat, En Düşük Fiyat, Kapanış Fiyatı (OHLC) ve Volüm değerleri sisteme girdi olarak kullanılmıştır. 

### 1)	Veri Setinin Oluşturulması: 
Veri seti olarak S&P 500 hisse senetlerinin 18.11.1999 ile 09.08.2013 tarihleri arasındaki günlük değerleri kullanılmıştır. Örnek veri setinin yapısı Tablo 1 de gösterilmiştir.

![tablo 1](https://user-images.githubusercontent.com/29254495/29993032-0a803e60-8fb3-11e7-90c1-24d1e5b02d1c.PNG)


Finansal Piyasa Tahmin Vektörünün oluşturulması için her hisse senedinin Açılış, En Yüksek, En Düşük, Kapanış fiyatları arasındaki ilişkinin ortaya koyulabilmesi için işleme tabi tutulup farklı değerler elde edilmiştir. Uygulanan işlem adımları şu şekilde tanımlanabilmektedir: Arasındaki ilişkinin belirlenmesini istenen sütuna ilk önce log getiri (log return) uygulanır, elde edilen sonuca da Z-Puanı (z score) uygulanır. Elde edilen bu yeni değerler de yeni vektöre eklenir. Matematiksel olarak yapılan işlemi gösterebilmek için örnek olarak Kapanış ve Açılış fiyatlarına uygulanan işlem altta gösterilmiştir. 

```
c_2_o: (kapanış fiyatı ile açılış fiyat arasındaki ilişki);

logGetiri = log(Kapanış Fiyat / Açılış Fiyat); 

zPuan = (logGetiri – Ortalama)/StandartSapma;

c_2_o = zPuan;
```

Fiyatlara uygulanan işlemlerin Python dilinde kodlaması altta gösterilmiştir. Aynı şekilde bir tek hisseye uygulandığında oluşan veri yapısı Tablo 2’de listelenmiştir. 

![tablo 2](https://user-images.githubusercontent.com/29254495/29993801-06503da6-8fca-11e7-9c6a-d807594f3394.PNG)

```
ret = lambda x,y: log(y/x) #Log getiri 
zscore = lambda x:(x -x.mean())/x.std() # zpuan

Res['c_2_o'] = zscore(ret(D.o,D.c))
Res['h_2_o'] = zscore(ret(D.o,D.h))
Res['l_2_o'] = zscore(ret(D.o,D.l))
Res['c_2_h'] = zscore(ret(D.h,D.c))
Res['h_2_l'] = zscore(ret(D.h,D.l))

Res['c1_c0'] = ret(D.c,D.c.shift(-1)).fillna(0) #Bir sonraki günün getirisi 

Res['vol'] = zscore(D.v)
```

  Veri setinde bulunan tüm hisse senetlerine üsteki yöntem uygulandığında Finansal Piyasa Vektörünü elde edilmektedir. Bu çalışmadaki Finansal Piyasa Vektörü 3900 satır ve 2328 sütundan oluşmaktadır [3900 x 2328]. Bunlardan 3300 ü eğitim 600 ü ise test için kullanılmıştır. Tablo 3’te veri setini oluşturan tüm endeks hisselerine işlem uygulandıktan sonra oluşan Finansal Piyasa Vektörünün yapısı gösterilmiştir.

  Oluşturulan Finansal Piyasa Vektör matrisi; Giriş Verisi ve Hedeflenen Veri olarak ikiye ayrılmıştır. Giriş Verisi sistemi eğitmek için kullanılmıştır. Hedeflenen Veri de çıktı olarak tahmin edilmesi istenen verilerden oluşmaktadır.

  Giriş Verisi olarak her hisse senedinin Açılış Fiyatı, En Yüksek Fiyat, En Düşük Fiyatı ve Kapanış Fiyatı arasındaki ilişkiyi ortaya koyan sırayla “c_2_h” - (kapanış fiyatı ile en yüksek fiyat) “c_2_o” - (kapanış fiyatı ile açılış fiyat arasındaki ilişki) “h_2_l” -  (en yüksek ile en düşük fiyat) “h_2_o”- (en yüksek fiyat ile açılış fiyatı)  “l_2_o” -  (en düşük fiyat ile açılış fiyatı)  ve volüm kullanılmıştır. 


Hedeflenen Veri olarak günlük getirinin tutulduğu “c1_c0” dan oluşmaktadır.

Örnek olarak Apple hisse senedi için sistemde oluşturulan verileri ele aldığımızda Giriş verisi olarak “aapl_c_2_h”, “aapl_c_2_o”, “aapl_h_2_l”, “aapl_h_2_o”, “aapl_l_2_o” ve “aapl_vol” olup Hedeflenen Veri olarak da “aapl_c1_c0” olarak kullanılmıştır. 

Sistemi eğitmek için girdi olarak kullanılan finansal piyasa vektör matrisi Tablo 4’te gösterilmiştir.

Sistem çıktısı olarak tahmin edilmeye çalışılan finansal piyasa vektörü Tablo 5’te gösterilmiştir.

Sistem çıktısı olarak tahmin edilmeye çalışılan finansal piyasa vektörü oluşturulduktan sonra her günlük toplam getiri hesaplanmıştır. Hesaplanan günlük toplam getiri de üç farklı sınıfa yerleştirilmiştir. Toplam getirinin hesaplanması ve sınıflandırılmasında kullanılan Python kodu alta eklenmiştir.

```
#Sınıflandırma kodu, 
(1 -> Alış) (-1 -> Satış) (0 -> İşlem yok)

def labeler(x):
    if x>0.004:
        return 1
    if x<-0.004:
        return -1
    else:
        return 0


# Sınıflandırma
Labeled = pd.DataFrame()
Labeled['return'] = TotalReturn


# Getirilerin 3 sınıfa ayrılması
Labeled['class'] = TotalReturn.apply(labeler,1)


# Getirilerin 11 sınıfa ayrılması
Labeled['multi_class'] = pd.qcut(TotalReturn,11,labels=range(11))
```

İlk sınıflandırmada getiriler 3 sınıfa yerleştirilmiştir. % 0,4 ten fazla getiri olan günler +1 % -0,4 ten fazla zarar eden günler -1, bunların dışında kalan günler de getiri olmadığını kabul edip 0 ile sınıflandırılmıştır.

```
1    1582
0    1259
-1    1059
```

İkinci sınıflandırmada ise getiriler 11 sınıfa ayrılmıştır. Bu 11 sınıftan 5 i pozitif getiri, 5 i negatif getiri, 1 i de getiri olmayan günleri ifade etmek için kullanılmıştır.

Eğer tahminde mükemmel bir sistem yapabilsek elde edilebilecek maksimum getiri Şekil 3’teki grafikte gösterilmiştir. Getiri potansiyelini ortaya koyabilmek için böyle bir gösterime gidilmiştir.

![sekil 3](https://user-images.githubusercontent.com/29254495/29993901-110e05aa-8fcc-11e7-9641-a10b03c6f396.PNG)


### 2)	Lojistik Regresyona Göre Sistemin Eğitilmesiı: 

Lojistik Regresyon için Phyton kütüphanelerinden sklearn kütüphanesi kullanılmıştır. Model olarak da LogisticRegression(C=1e5) modeli kullanılmıştır.
```
from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5)
```
Sistemin eğitilmesi için 3900 kayıttan 3300 ü kullanılmıştır. 600 kayıt da test için kullanılmıştır. Sistemin eğitilmesi
```
res = logreg.fit(X,Y)
```
komutu ile gerçekleştirilmiştir. X parametresi olarak Giriş verilerini Y parametresi olarak da sınıflandırılmış getiri seçilmiştir. Buna göre test işleminde günlük getirinin hangi sınıfa ait olduğu tespit edilmeye çalışılmıştır. İşlem sonucu elde edilen getiri Şekil 4’te gösterilmiştir.

![sekil 4](https://user-images.githubusercontent.com/29254495/29993927-ba6ebaf4-8fcc-11e7-87be-8da025d801f0.PNG)

Şekil 4’te x ekseni işlem tarihini y ekseni de getiriyi ifade etmektedir. Grafikte kümülatif getiri gösterilmiştir. Görüldüğü üzere Lojistik Regresyon a göre yapılan tahmin pek de başarılı olduğu söyleyemeyiz. Sonlara doğru performansı düzlese de Toplam Endeks getirisiyle neredeyse aynı getiriyi vermektedir. Lojistik Regresyonun 3 sınıfa göre gruplanmış confusion matrix i altta gösterilmiştir.

```
[[90 36 77]
[73 35 92]
[91 30 76]]
```

### 3)	Yapay Sinir Ağlara (NN - Neural Network) Göre Sistemin Eğitilmesi

Yapay Sinir Ağlara Göre Sistemin Eğitilmesi için TensorFlow Kütüphanesi kullanılmıştır. Yapay Sinir Ağ iki katmandan oluşmuştur. Sinir ağın ilk katman boyutu 1000 ikinci katman boyutu 250, dropout 0.2, batch boyutu 50 olarak tanımlanmıştır.
```
import tensorflow as tf
from  tensorflow.contrib.learn.python.learn.estimators.dnn  import DNNClassifier
from tensorflow.contrib.layers import real_valued_column
```
Sistemin eğitilmesi için 3900 kayıttan 3300 ü kullanılmıştır. 600 kayıt da test için kullanılmıştır. Sistemin eğitilmesi 
```
train = (InputDF[:-test_size].values,Labeled.tf_class[:-test_size].values)

val = (InputDF[-test_size:].values,Labeled.tf_class[-test_size:].values)

NUM_TRAIN_BATCHES = int(len(train[0])/BATCH_SIZE) -- 3300/50 = 66
NUM_VAL_BATCHES = int(len(val[1])/BATCH_SIZE) -- 600/50 = 12

self.logits = tf.contrib.layers.fully_connected
```
komutu ile gerçekleştirilmiştir. TRAIN parametresi olarak giriş verilerini VAL parametresi olarak da sınıflandırılmış getiri seçilmiştir. Buna göre test işleminde günlük getirinin hangi sınıfa ait olduğu tespit edilmeye çalışılmıştır.

İşlem sonucu elde edilen çıkıtı Şekil 5’teki grafikte gösterilmiştir.

![sekil 5](https://user-images.githubusercontent.com/29254495/29994120-aee06878-8fd0-11e7-920a-a3932dcfb46b.PNG)


Şekil 5’te x ekseni işlem tarihini y ekseni de getiriyi ifade etmektedir. Grafikte kümülatif getiri gösterilmiştir. Görüldüğü üzere Yapay Sinir Ağlarına göre yapılan tahmin modeli hem Lojistik Regresyon a göre hem de Toplam Endeks getirisine göre çok daha iyi sonuç vermiştir. Yapay sinir ağların çıktısı 3 sınıfa göre gruplanmış confusion matrix i altta gösterilmiştir.

```
[[87 35 81]
 [60 46 94]
 [82 26 89]]
```

### 4)	Tekrarlayan Sinir Ağlara (RNN - Recurrent Neural Network) Göre Sistemin Eğitilmesi: 

Tekrarlayan Sinir Ağlara Göre Sistemin Eğitilmesi için TensorFlow Kütüphanesi kullanılmıştır. Tekrarlayan sinir ağı iki katmandan oluşmuştur. Sinir ağın ilk katmanı boyutu 1000 ikinci katman boyutu 250, RNN gizli boyutu 100 batch boyutu 50 olarak tanımlanmıştır. Sistemin eğitilmesi için 3900 kayıttan 3300 ü kullanılmıştır. 600 kayıt da test için kullanılmıştır. Sistemin eğitilmesi

def makeGRUCells():
            base_cell = tf.contrib.rnn.GRUCell(num_units=RNN_HIDDEN_SIZE,) 
            layered_cell = tf.contrib.rnn.MultiRNNCell([base_cell] * NUM_LAYERS,state_is_tuple=False) 
            attn_cell =tf.contrib.rnn.AttentionCellWrapper(cell=layered_cell,attn_length=ATTN_LENGTH,state_is_tuple=False)
            return attn_cell
                
        self.gru_cell = makeGRUCells()


komutu ile gerçekleştirilmiştir. İşlem sonucu elde edilen çıkıtı alttaki grafikte gösterilmiştir.

![sekil 6](https://user-images.githubusercontent.com/29254495/29994136-ffe9d66e-8fd0-11e7-9130-85a35c04b612.PNG)

Şekil 6’da x ekseni işlem tarihini y ekseni de getiriyi ifade etmektedir. Grafikte kümülatif getiri gösterilmiştir. Görüldüğü üzere Tekrarlayan Sinir Ağlara göre yapılan tahmin hem Yapay Sinir Ağlarına göre hem Lojistik Regresyon a göre hem de Toplam Endeks getirisine göre çok daha iyi sonuç vermiştir. Tekrarlayan sinir ağların çıktısı 3 sınıfa göre gruplanmış confusion matrix i altta gösterilmiştir.

```
[[101  33  69]
 [ 82  34  84]
 [ 85  32  80]]
```

Şekil 5 ve Şekil 6'da görüldüğü üzere yöntemler arasındaki fark gün geçtikçe artmaktadır, bunun nedeni yapa sinir ağlarının doğru tahmin başarısı sürdüğünden kümülatif getirinin artmasına neden olmaktadır.

# SONUÇ

Derin öğrenme, tahmin performansını optimize etmek için büyük veri setlerini kullanabilmeye olanak tanıyan genel bir çerçeve sunar. Bu nedenle, derin öğrenme çerçeveleri, finans alanında birçok soruna (hem pratik hem de teorik olarak) uygundur. Bu çalışmada, finansal tahmin ve sınıflandırma problemleri için derin öğrenme hiyerarşik karar modelleri nasıl kullanılabileceği üzerinde durulmuştur. Gösterdiğimiz gibi, derin öğrenme, klasik uygulamalarda öngörülen performansı - bazen çarpıcı olarak - geliştirme potansiyeline sahiptir. Bölüm 3'teki finansal piyasa tahmin vektörü konusundaki örneğimiz, derin öğrenme modellerini finansta uygulamak için yalnızca bir yol sunmaktadır. Derin öğrenme yöntemlerinin çok farklı finans problemine uygulanabilmektedir. 

Aynı zamanda, derin öğrenme, finans alanında mevcut düşünceye, özellikle piyasa etkinliği kavramı da dâhil olmak üzere önemli zorluklar ortaya koymaktadır.

Verilerdeki karmaşık doğrusal olmayanlıkları modelleyebildiği için, derin öğrenme varlıkları keyfi küçük fiyatlandırma hataları içinde fiyatlandırabilir. Bu, pazarların bilgi açısından verimli olduğunu veya piyasadaki verimlilik testlerinin yeniden yapılması gerektiği anlamına mı gelecektir? Genel olarak, var olan aksiyomatik temellerden inşa edilen teorik modellerin derin öğrenme modellerinin öngörülü performansıyla rekabet edebilmesi pek mümkün değildir. Bunun finansın geleceği için ne anlama geldiğini görmek gerekiyor.

Aynı şekilde, derin öğrenme modelleri, tahminin her şeyden önemli olduğunu parametresini ele aldığımızda, finans uygulamasında daha büyük ve daha fazla etki yaratacağını öngörebilmekteyiz.

# GELECEK ÇALIŞMALAR

Borsa İstanbul verilerini kullanıp derin öğrenme yöntemlerinden DNN ve LSTM i uygulayarak yapılabilecek çalışmaları şu şekilde listeleyebiliriz:
•	BIST 100 deki her hisse senedinin bir sonraki fiyatını tahmin etmek.
•	Borsa İstanbul’daki XU100 ve XU30 endekslerini, endeksleri oluşturan hisseleri kullanarak n dakika sonraki fiyatını tahmin etmek.
•	Hangi hissenin %x den fazla yukarı ya da aşağı yönlü hareket yapacağını tahmin etmek.
•	Bir sonraki n dakika içinde hangi hisselerin % x'den fazla yükselteceğini tahmin etmek.
•	Bir sonraki n dakika içinde hangi hisselerin % 2x'den fazla yükselteceğini / azalacağını ve bu süre içinde % x'den fazla azalmayacağını / yükselmeyeceğini tahmin etmek.
•	Hangi n dakika içinde XU100 endeksi % 2x'den fazla yükselteceğini / azalacağını ve bu süre içinde % x'den fazla azalmayacağını / yükselmeyeceğini tahmin etmek.


