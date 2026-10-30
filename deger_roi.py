# ============================================================================
# DEĞER ADAYLARI ROI (DENEME)
#
# Ana kod her koşu için "değer adayı" işaretler: AGF'siz temel modelin ilk 3'ünde
# olup AGF sırası 4 ya da daha geride olan atlar. Bu araç, o atların her birine
# 1 birim ganyan oynasaydık ne olacağını hesaplar ve aynı koşularda AGF
# favorisine oynamayla kıyaslar.
#
# Ana koda dokunmaz. roi_takip.py ile aynı klasörde çalıştır:
#   python deger_roi.py
#
# UYARI: Yüksek ganyanlı atlarda sonuç çok oynaktır; tek bir 30 katlık galibiyet
# tabloyu tek başına değiştirir. Yüzlerce bahis olmadan sonuca güvenme.
# Bu bir yatırım tavsiyesi değildir.
# ============================================================================

import os

import pandas as pd

from roi_takip import (
    tarih_indeksi, kazananlari_oku, SONUC_KLASORU, TAKIP_DOSYASI
)

AZ_VERI_BAHIS = 200


def main():
    takip = pd.read_csv(TAKIP_DOSYASI, dtype=str)

    print(f"\n  Takip dosyası: {os.path.abspath(TAKIP_DOSYASI)} ({len(takip)} satır)")
    print(f"  Durum dağılımı: {takip['Durum'].value_counts().to_dict()}")

    if "DegerAdaylari" not in takip.columns:
        raise SystemExit(
            "  Takip dosyasında DegerAdaylari sütunu yok: güncel ana kod bu "
            "klasörde hiç çalışmamış. Güncel yaris_tahmin_v4.py ile tahmin üret."
        )

    dolu = (
        takip["DegerAdaylari"].notna()
        & (takip["DegerAdaylari"].str.strip() != "")
    )
    cozuldu = takip["Durum"] == "cozuldu"

    print(
        f"  DegerAdaylari dolu satır: {int(dolu.sum())} "
        f"(bunlardan çözülmüş: {int((dolu & cozuldu).sum())})"
    )

    cozulen = takip[dolu & cozuldu].copy()

    if cozulen.empty:
        if int(dolu.sum()) == 0:
            raise SystemExit(
                "  Değer adayı kaydı yok: güncel kodla henüz tahmin üretilmemiş "
                "ya da hiçbir koşuda aday çıkmamış."
            )

        raise SystemExit(
            "  Değer adayları kaydedilmiş ama henüz sonuçlanmamış. Yarış "
            "sonuç CSV'lerini sonuclar/ klasörüne ekle ve ANA KODU çalıştır "
            "(Durum'u 'cozuldu' yapan o), sonra bunu tekrar çalıştır."
        )

    indeks = tarih_indeksi()
    kazananlar = {}

    for tarih in sorted(cozulen["Tarih"].dropna().unique()):
        for dosya in indeks.get(tarih, []):
            sehir, sonuclar = kazananlari_oku(dosya)

            for kosu, (at, ganyan) in sonuclar.items():
                kazananlar[(tarih, sehir, kosu)] = (at, ganyan)

    bahis = isabet = kosu_sayisi = eslesmeyen = 0
    net = 0.0
    fav_n = fav_isabet = 0
    fav_net = 0.0

    for _, r in cozulen.iterrows():
        anahtar = (r["Tarih"], r["Sehir"], int(r["Kosu"]))
        bilgi = kazananlar.get(anahtar)

        if bilgi is None or pd.isna(bilgi[1]):
            eslesmeyen += 1
            continue

        kazanan, ganyan = bilgi
        kosu_sayisi += 1

        for at in str(r["DegerAdaylari"]).split(";"):
            if not at:
                continue

            bahis += 1

            if at == kazanan:
                isabet += 1
                net += ganyan - 1.0
            else:
                net -= 1.0

        # Karşılaştırma: aynı koşularda AGF favorisine 1 birim
        fav_n += 1

        if r["AGFFavori"] == kazanan:
            fav_isabet += 1
            fav_net += ganyan - 1.0
        else:
            fav_net -= 1.0

    print(f"\n{'=' * 64}")
    print("  💎 DEĞER ADAYLARI ROI (deneme, her adaya 1 birim ganyan)")
    print(f"{'=' * 64}")
    print(
        f"  Koşu: {kosu_sayisi} | bahis: {bahis} | eşleşmeyen koşu: {eslesmeyen}"
    )

    if bahis == 0:
        print("  Hesaplanacak bahis yok.")
        print(f"{'=' * 64}\n")
        return

    print(
        f"\n  Değer adayları : isabet {isabet}/{bahis} "
        f"(%{100 * isabet / bahis:.1f}) | net {net:+.2f} birim | "
        f"bahis başına {net / bahis:+.3f}"
    )
    print(
        f"  AGF favorisi   : isabet {fav_isabet}/{fav_n} "
        f"(%{100 * fav_isabet / fav_n:.1f}) | net {fav_net:+.2f} birim | "
        f"koşu başına {fav_net / fav_n:+.3f}"
    )
    print("  (iki satırın bahis sayısı farklı: adaylar koşu başına 0-3 at)")

    if bahis < AZ_VERI_BAHIS:
        print(
            f"\n  ⚠️ {bahis} bahis çok az (<{AZ_VERI_BAHIS}), "
            "sonuç şans eseri olabilir."
        )

    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
