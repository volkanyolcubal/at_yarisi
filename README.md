🏇 At Yarışı Tahmin ve Veri İşleme Sistemi (at_yarisi)
Bu depo; geçmiş at yarışı verilerini analiz eden, özellik mühendisliği (feature engineering) uygulayan ve özel tahmin/sıralama modelleri koşturan yerel bir Python otomasyon projesidir.

🚀 Proje İçeriği ve Modüller
sonyeni.py: Ham veri setlerini temizleyen, NaN değerleri ayıklayan ve makine öğrenmesi algoritmaları için optimize edilmiş özellik satırlarına dönüştüren ana veri işleme motoru.

yaris_tahmin_v4.py: Üretilen özellik matrislerini kullanarak model çıktıları üreten ve güncel tahmin süreçlerini yöneten ana script.

v4_tahmin_takip.csv: Tahmin sonuçlarının ve performans süreçlerinin kayıt altına alındığı takip tablosu.

yaris_ranker_v4.log: Sistem loglarının tutulduğu kayıt dosyası.

🛠️ Kurulum ve Çalıştırma
Projeyi kendi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

Depoyu klonlayın:

Bash
git clone https://github.com/volkanyolcubal/at_yarisi.git
cd at_yarisi
Gerekli Python bağımlılıklarını yükleyin:

Bash
pip install -r requirements.txt
Veri işleme ve tahmin döngüsünü başlatın:

Bash
python sonyeni.py
python yaris_tahmin_v4.py
Not: Büyük hacimli yerel model checkpoint ve ham geçmiş dosyaları (.pkl) dosya boyutu limitleri nedeniyle yerel .gitignore kapsamında hariç tutulmuştur.
