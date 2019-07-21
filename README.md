# Botnet-Detection-with-Machine-Learning
Iscx-Bot-2014 Dataset

https://www.unb.ca/cic/datasets/botnet.html
1. adım : "dataset" klasörünün içerisinde training.csv ve test.csv dosyalarını açar.
2. adım : Datasetinin içerisinde botnet olan ip adreslerinin yaptığı işlemleri işaretler.
3. adım : İlk 2 adıma göre "flowdata.pickle" ve "flowdatatest.pickle" dosyalarını oluşturur.
4. adım : Çeşitli makine öğrenmesi algoritmaları ile "flowdata.pickle" dosyası ile eğitilir ve "flowdatatest.pickle" dosyasında botnet tahmini yapar.

Dipnot : İlk 3 adımı sağlayan "dataset_load.py"
