# Büyük Verilerde Finansal Piyasa Tahmini: Derin Öğrenme

Özetçe — Derin Öğrenme, büyük miktarda etiketsiz/denetlenmemiş veriden öğrenmeyi başarılı bir şekilde gerçekleştirebildiğinden, büyük verilerden anlamlı gösterimler ve desenler çıkartmayı cazip hale getiriyor. En basit tanımıyla derin öğrenme, makine öğrenmesi yöntemlerinin büyük verilere uygulanması olarak ifade edilmektedir. Bu çalışmada finansal tahmin ve sınıflama problemlerinde derin öğrenme hiyerarşik modellerinin nasıl kullanılabileceği araştırılmıştır. Finansal tahmin sorunları - menkul kıymetleri tasarlama ve fiyatlandırma, portföy oluşturma, risk yönetimi gibi konular - genellikle karmaşık veri etkileşimlerine sahip büyük veri setlerini içerdiğinden tam ekonomik model oluşturmak şu an için zor veya imkansızdır. Bu problemlere derin öğrenme yöntemleri uygulandığında, finanstaki standart metotlardan daha faydalı sonuçlar alınabilir. Bilhassa, derin öğrenme, en azından şu an için, mevcut herhangi bir finansal ekonomik teori için görünmez olan veri etkileşimlerini algılayabilir ve bunlardan istifade edebilir. 

Anahtar Kelimeler — Derin Öğrenme, Makine Öğrenmesi, Büyük Veriler, Yapay Zeka, Finans, Piyasa Tahmini.



## Borsa Hisse Senetleri Yerleştirme

En yaygın kullanılan kelime yerleştirme algoritması word2vec tir. Bu çalışmada word2vec algoritmasında olduğu gibi kelimeler yerine borsadaki hisse senetlerinin verisi yerleştirilmiştir. İlk sütun tarih olup o tarihe karşılık gelen hisse senetlerinin Açılış Fiyatı, En Yüksek Fiyat, En Düşük Fiyat, Kapanış Fiyatı (OHLC) ve Volüm değerleri sisteme girdi olarak kullanılmıştır. 

### 1)	Veri Setinin Oluşturulması: 
Veri seti olarak S&P 500 hisse senetlerinin 18.11.1999 ile 09.08.2013 tarihleri arasındaki günlük değerleri kullanılmıştır. Örnek veri setinin yapısı Tablo 1 de gösterilmiştir.

![tablo 1](https://user-images.githubusercontent.com/29254495/29993032-0a803e60-8fb3-11e7-90c1-24d1e5b02d1c.PNG)


Finansal Piyasa Tahmin Vektörünün oluşturulması için her hisse senedinin Açılış, En Yüksek, En Düşük, Kapanış fiyatları arasındaki ilişkinin ortaya koyulabilmesi için işleme tabi tutulup farklı değerler elde edilmiştir. Uygulanan işlem adımları şu şekilde tanımlanabilmektedir: Arasındaki ilişkinin belirlenmesini istenen sütuna ilk önce log getiri (log return) uygulanır, elde edilen sonuca da Z-Puanı (z score) uygulanır. Elde edilen bu yeni değerler de yeni vektöre eklenir. Matematiksel olarak yapılan işlemi gösterebilmek için örnek olarak Kapanış ve Açılış fiyatlarına uygulanan işlem altta gösterilmiştir. 



Tarih	Açılış	En Yüksek	En Düşük	Kapanış	Volüm
20130802	46.06	46.51	46.00	46.36	2.1834E+06
20130805	46.26	46.40	45.80	45.87	1.7933E+06
20130806	45.93	46.21	45.61	46.05	2.0693E+06
20130807	45.88	46.46	45.62	46.30	1.4912E+06
20130808	46.49	46.55	45.64	45.86	1.5091E+06
20130809	45.85	46.46	45.81	46.24	1.7409E+06

Fiyatlara uygulanan işlemlerin Python dilinde kodlaması altta gösterilmiştir. Aynı şekilde bir tek hisseye uygulandığında oluşan veri yapısı Tablo 2’de listelenmiştir. 

![tablo 2](https://user-images.githubusercontent.com/29254495/29993801-06503da6-8fca-11e7-9c6a-d807594f3394.PNG)

ret = lambda x,y: log(y/x) #Log getiri 
zscore = lambda x:(x -x.mean())/x.std() # zpuan

Res['c_2_o'] = zscore(ret(D.o,D.c))
Res['h_2_o'] = zscore(ret(D.o,D.h))
Res['l_2_o'] = zscore(ret(D.o,D.l))
Res['c_2_h'] = zscore(ret(D.h,D.c))
Res['h_2_l'] = zscore(ret(D.h,D.l))

Res['c1_c0'] = ret(D.c,D.c.shift(-1)).fillna(0) #Bir sonraki günün getirisi 

Res['vol'] = zscore(D.v)


  Veri setinde bulunan tüm hisse senetlerine üsteki yöntem uygulandığında Finansal Piyasa Vektörünü elde edilmektedir. Bu çalışmadaki Finansal Piyasa Vektörü 3900 satır ve 2328 sütundan oluşmaktadır [3900 x 2328]. Bunlardan 3300 ü eğitim 600 ü ise test için kullanılmıştır. Tablo 3’te veri setini oluşturan tüm endeks hisselerine işlem uygulandıktan sonra oluşan Finansal Piyasa Vektörünün yapısı gösterilmiştir.

  Oluşturulan Finansal Piyasa Vektör matrisi; Giriş Verisi ve Hedeflenen Veri olarak ikiye ayrılmıştır. Giriş Verisi sistemi eğitmek için kullanılmıştır. Hedeflenen Veri de çıktı olarak tahmin edilmesi istenen verilerden oluşmaktadır.

  Giriş Verisi olarak her hisse senedinin Açılış Fiyatı, En Yüksek Fiyat, En Düşük Fiyatı ve Kapanış Fiyatı arasındaki ilişkiyi ortaya koyan sırayla “c_2_h” - (kapanış fiyatı ile en yüksek fiyat) “c_2_o” - (kapanış fiyatı ile açılış fiyat arasındaki ilişki) “h_2_l” -  (en yüksek ile en düşük fiyat) “h_2_o”- (en yüksek fiyat ile açılış fiyatı)  “l_2_o” -  (en düşük fiyat ile açılış fiyatı)  ve volüm kullanılmıştır. 


Hedeflenen Veri olarak günlük getirinin tutulduğu “c1_c0” dan oluşmaktadır.

Örnek olarak Apple hisse senedi için sistemde oluşturulan verileri ele aldığımızda Giriş verisi olarak “aapl_c_2_h”, “aapl_c_2_o”, “aapl_h_2_l”, “aapl_h_2_o”, “aapl_l_2_o” ve “aapl_vol” olup Hedeflenen Veri olarak da “aapl_c1_c0” olarak kullanılmıştır. 

Sistemi eğitmek için girdi olarak kullanılan finansal piyasa vektör matrisi Tablo 4’te gösterilmiştir.

Sistem çıktısı olarak tahmin edilmeye çalışılan finansal piyasa vektörü Tablo 5’te gösterilmiştir.

Sistem çıktısı olarak tahmin edilmeye çalışılan finansal piyasa vektörü oluşturulduktan sonra her günlük toplam getiri hesaplanmıştır. Hesaplanan günlük toplam getiri de üç farklı sınıfa yerleştirilmiştir. Toplam getirinin hesaplanması ve sınıflandırılmasında kullanılan Python kodu alta eklenmiştir.

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

İlk sınıflandırmada getiriler 3 sınıfa yerleştirilmiştir. % 0,4 ten fazla getiri olan günler +1 % -0,4 ten fazla zarar eden günler -1, bunların dışında kalan günler de getiri olmadığını kabul edip 0 ile sınıflandırılmıştır.

1    1582
0    1259
-1    1059

İkinci sınıflandırmada ise getiriler 11 sınıfa ayrılmıştır. Bu 11 sınıftan 5 i pozitif getiri, 5 i negatif getiri, 1 i de getiri olmayan günleri ifade etmek için kullanılmıştır.

Eğer tahminde mükemmel bir sistem yapabilsek elde edilebilecek maksimum getiri Şekil 3’teki grafikte gösterilmiştir. Getiri potansiyelini ortaya koyabilmek için böyle bir gösterime gidilmiştir.



![test](https://user-images.githubusercontent.com/29254495/29874405-5859908e-8d9f-11e7-8ce7-f797fb2b44dd.PNG)
