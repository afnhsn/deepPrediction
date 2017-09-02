# Büyük Verilerde Finansal Piyasa Tahmini: Derin Öğrenme

Özetçe — Derin Öğrenme, büyük miktarda etiketsiz/denetlenmemiş veriden öğrenmeyi başarılı bir şekilde gerçekleştirebildiğinden, büyük verilerden anlamlı gösterimler ve desenler çıkartmayı cazip hale getiriyor. En basit tanımıyla derin öğrenme, makine öğrenmesi yöntemlerinin büyük verilere uygulanması olarak ifade edilmektedir. Bu çalışmada finansal tahmin ve sınıflama problemlerinde derin öğrenme hiyerarşik modellerinin nasıl kullanılabileceği araştırılmıştır. Finansal tahmin sorunları - menkul kıymetleri tasarlama ve fiyatlandırma, portföy oluşturma, risk yönetimi gibi konular - genellikle karmaşık veri etkileşimlerine sahip büyük veri setlerini içerdiğinden tam ekonomik model oluşturmak şu an için zor veya imkansızdır. Bu problemlere derin öğrenme yöntemleri uygulandığında, finanstaki standart metotlardan daha faydalı sonuçlar alınabilir. Bilhassa, derin öğrenme, en azından şu an için, mevcut herhangi bir finansal ekonomik teori için görünmez olan veri etkileşimlerini algılayabilir ve bunlardan istifade edebilir. 

Anahtar Kelimeler — Derin Öğrenme, Makine Öğrenmesi, Büyük Veriler, Yapay Zeka, Finans, Piyasa Tahmini.



## Borsa Hisse Senetleri Yerleştirme

En yaygın kullanılan kelime yerleştirme algoritması word2vec tir. Bu çalışmada word2vec algoritmasında olduğu gibi kelimeler yerine borsadaki hisse senetlerinin verisi yerleştirilmiştir. İlk sütun tarih olup o tarihe karşılık gelen hisse senetlerinin Açılış Fiyatı, En Yüksek Fiyat, En Düşük Fiyat, Kapanış Fiyatı (OHLC) ve Volüm değerleri sisteme girdi olarak kullanılmıştır. 

### 1)	Veri Setinin Oluşturulması: 
Veri seti olarak S&P 500 hisse senetlerinin 18.11.1999 ile 09.08.2013 tarihleri arasındaki günlük değerleri kullanılmıştır. Örnek veri setinin yapısı Tablo 1 de gösterilmiştir.

Finansal Piyasa Tahmin Vektörünün oluşturulması için her hisse senedinin Açılış, En Yüksek, En Düşük, Kapanış fiyatları arasındaki ilişkinin ortaya koyulabilmesi için işleme tabi tutulup farklı değerler elde edilmiştir. Uygulanan işlem adımları şu şekilde tanımlanabilmektedir: Arasındaki ilişkinin belirlenmesini istenen sütuna ilk önce log getiri (log return) uygulanır, elde edilen sonuca da Z-Puanı (z score) uygulanır. Elde edilen bu yeni değerler de yeni vektöre eklenir. Matematiksel olarak yapılan işlemi gösterebilmek için örnek olarak Kapanış ve Açılış fiyatlarına uygulanan işlem altta gösterilmiştir. 


![test](https://user-images.githubusercontent.com/29254495/29874405-5859908e-8d9f-11e7-8ce7-f797fb2b44dd.PNG)
