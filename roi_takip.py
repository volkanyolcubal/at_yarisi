# ============================================================================
# ROI TAKİP - v4_tahmin_takip.csv'deki çözülmüş tahminlerin getirisi
#
# Soru: "Modelin 1. seçimine her koşuda 1 birim ganyan oynasaydık, AGF
# favorisine oynamaya göre kâr/zarar ne olurdu?"
#
# Ana koda dokunmaz. Sadece okur. Ana kodla aynı klasörde çalıştır:
#   python roi_takip.py
#
# Ganyan değeri sonuç CSV'sindeki 'Ganyan' sütunundan (kazanan atın 1 TL'ye
# ödediği tutar) alınır. Kâr = ganyan - 1, kayıp = -1.
#
# UYARI: Kısa dönemde kâr/zarar büyük ölçüde şanstır. Yüzlerce koşu olmadan
# sonuca güvenme. Bu bir yatırım tavsiyesi değildir.
# ============================================================================

import os
import re
import glob

import pandas as pd

try:
    from yaris_tahmin_v4 import (
        read_text_file, norm_key, normalize_column_name,
        canonical_horse_name, degree_to_seconds, parse_float
    )
except Exception as hata:
    raise SystemExit(
        "yaris_tahmin_v4.py ile aynı klasörde çalıştırmalısın.\n"
        f"İçe aktarma hatası: {hata}"
    )

TAKIP_DOSYASI = "v4_tahmin_takip.csv"
SONUC_KLASORU = "sonuclar"
RE_KOSU = re.compile(r"^(\d+)\.\s*Kosu\s*:", re.IGNORECASE)
AZ_VERI_ESIGI = 100
AZ_VERI_SEHIR = 20  # şehir kırılımı için minimum koşu sayısı


def kazananlari_oku(dosya):
    """Bir sonuç CSV'sinden (şehir, {koşu: (kazanan_at, ganyan)}) döner."""
    icerik = read_text_file(dosya)

    if not icerik:
        return "", {}

    satirlar = icerik.splitlines()
    sehir = norm_key(satirlar[0].split(";")[0].replace("\ufeff", ""))

    baslik = None
    kosu = 0
    en_iyi = {}

    for satir in satirlar:
        s = satir.strip()

        if not s:
            continue

        n = normalize_column_name(s)

        if "ATNO" in n and "ATISMI" in n and "JOKEY" in n:
            baslik = [normalize_column_name(c) for c in s.split(";")]
            continue

        eslesme = RE_KOSU.match(s)

        if eslesme:
            kosu = int(eslesme.group(1))
            continue

        parcalar = s.split(";")

        if baslik is None or kosu == 0 or len(parcalar) < 5:
            continue

        if not re.match(
            r"^\d+(\s*-\s*[A-Za-z]+|[A-Za-z]{1,3})?$",
            parcalar[0].strip()
        ):
            continue

        def al(ad):
            if ad in baslik and baslik.index(ad) < len(parcalar):
                return parcalar[baslik.index(ad)].strip()
            return ""

        sure = degree_to_seconds(al("DERECE"))

        if pd.isna(sure):
            continue

        ganyan = parse_float(al("GANYAN"), float("nan"))
        at = canonical_horse_name(al("ATISMI"))

        if kosu not in en_iyi or sure < en_iyi[kosu][0]:
            en_iyi[kosu] = (sure, at, ganyan)

    return sehir, {k: (v[1], v[2]) for k, v in en_iyi.items()}


RE_TARIH = re.compile(r"(\d{2})/(\d{2})/(\d{4})")


def tarih_indeksi():
    """Sonuç CSV'lerinin tarihini dosyanın İLK SATIRLARINDAN okur."""
    indeks = {}

    for dosya in glob.glob(os.path.join(SONUC_KLASORU, "*.csv")):
        try:
            with open(dosya, "rb") as f:
                bas = f.read(4096).decode("utf-8", errors="ignore")
        except Exception:
            continue

        eslesme = None

        for satir in bas.splitlines()[:10]:
            eslesme = RE_TARIH.search(satir)

            if eslesme:
                break

        if not eslesme:
            continue

        gun, ay, yil = eslesme.groups()

        try:
            tarih = pd.Timestamp(year=int(yil), month=int(ay), day=int(gun))
        except Exception:
            continue

        indeks.setdefault(tarih.strftime("%Y-%m-%d"), []).append(dosya)

    return indeks


def getiri_hesapla(satirlar, secim_sutunu):
    """(koşu sayısı, isabet, toplam kâr) döner. Ganyanı bilinmeyenler atlanır."""
    n = isabet = 0
    kar = 0.0

    for _, r in satirlar.iterrows():
        if pd.isna(r["_ganyan"]):
            continue

        n += 1

        if r[secim_sutunu] == r["_kazanan"]:
            isabet += 1
            kar += r["_ganyan"] - 1.0
        else:
            kar -= 1.0

    return n, isabet, kar


def yaz(ad, n, isabet, kar):
    if n == 0:
        print(f"     {ad:<18} veri yok")
        return

    print(
        f"     {ad:<18} isabet {isabet}/{n} (%{100 * isabet / n:.1f}) | "
        f"net {kar:+.2f} birim | koşu başına {kar / n:+.3f}"
    )


def model2_secim(takip):
    """
    ModelTop3 sütunundan modelin 2. seçimini çıkarır.
    ModelTop3 formatı: "AT1;AT2;AT3" (HorseKey'ler).
    """
    def ikinci(deger):
        if not deger:
            return ""
        parcalar = str(deger).split(";")
        return parcalar[1] if len(parcalar) > 1 else ""

    return takip["ModelTop3"].apply(ikinci)


def main():
    if not os.path.exists(TAKIP_DOSYASI):
        raise SystemExit(f"{TAKIP_DOSYASI} bulunamadı.")

    takip = pd.read_csv(TAKIP_DOSYASI, dtype=str)
    cozulen = takip[takip["Durum"] == "cozuldu"].copy()

    if cozulen.empty:
        raise SystemExit("Çözülmüş tahmin yok.")

    indeks = tarih_indeksi()
    toplam_dosya = len(glob.glob(os.path.join(SONUC_KLASORU, "*.csv")))

    print(
        f"\n  '{os.path.abspath(SONUC_KLASORU)}' içinde {toplam_dosya} CSV, "
        f"{sum(len(v) for v in indeks.values())} tanesinin tarihi okundu."
    )

    kazananlar = {}
    sehirler_gun = {}

    for tarih in sorted(cozulen["Tarih"].dropna().unique()):
        for dosya in indeks.get(tarih, []):
            sehir, sonuclar = kazananlari_oku(dosya)
            sehirler_gun.setdefault(tarih, set()).add(sehir)

            for kosu, (at, ganyan) in sonuclar.items():
                kazananlar[(tarih, sehir, kosu)] = (at, ganyan)

    kazanan_at, ganyan_listesi = [], []

    for _, r in cozulen.iterrows():
        anahtar = (r["Tarih"], r["Sehir"], int(r["Kosu"]))
        at, ganyan = kazananlar.get(anahtar, (None, float("nan")))
        kazanan_at.append(at)
        ganyan_listesi.append(ganyan)

    cozulen["_kazanan"] = kazanan_at
    cozulen["_ganyan"] = ganyan_listesi
    cozulen["_model2"] = model2_secim(cozulen)

    bulunamayan = int(pd.isna(cozulen["_ganyan"]).sum())
    kullanilan = cozulen[cozulen["_ganyan"].notna()]

    print(f"\n{'=' * 64}")
    print("  💰 ROI TAKİP (her koşuya 1 birim ganyan)")
    print(f"{'=' * 64}")
    print(
        f"  Çözülmüş tahmin: {len(cozulen)} | ganyanı bulunan: {len(kullanilan)} "
        f"| bulunamayan: {bulunamayan}"
    )

    if bulunamayan:
        for _, r in cozulen[pd.isna(cozulen["_ganyan"])].head(3).iterrows():
            gun_sehirler = sorted(sehirler_gun.get(r["Tarih"], [])) or "o güne ait dosya yok"
            print(
                f"  örnek eşleşmeyen: {r['Tarih']} {r['Sehir']} K{r['Kosu']} "
                f"| bulunan şehirler: {gun_sehirler}"
            )

    # --- TÜM KOŞULAR -------------------------------------------------------
    print("\n  TÜM KOŞULAR")
    yaz("Model 1. seçim", *getiri_hesapla(kullanilan, "ModelFavori"))
    yaz("Model 2. seçim", *getiri_hesapla(kullanilan, "_model2"))
    yaz("AGF favorisi", *getiri_hesapla(kullanilan, "AGFFavori"))

    # --- MODEL ≠ AGF -------------------------------------------------------
    farkli = kullanilan[kullanilan["ModelFavori"] != kullanilan["AGFFavori"]]

    print(f"\n  SADECE MODEL ≠ AGF OLAN KOŞULAR ({len(farkli)} koşu)")
    yaz("Model 1. seçim", *getiri_hesapla(farkli, "ModelFavori"))
    yaz("Model 2. seçim", *getiri_hesapla(farkli, "_model2"))
    yaz("AGF favorisi", *getiri_hesapla(farkli, "AGFFavori"))

    # --- ŞEHİR KIRILIMI ---------------------------------------------------
    print(f"\n  ŞEHİR KIRILIMI (en az {AZ_VERI_SEHIR} koşu)")

    for sehir in sorted(kullanilan["Sehir"].dropna().unique()):
        alt = kullanilan[kullanilan["Sehir"] == sehir]

        if len(alt) < AZ_VERI_SEHIR:
            continue

        n, isa, kar = getiri_hesapla(alt, "ModelFavori")
        na, isaa, kara = getiri_hesapla(alt, "AGFFavori")

        print(
            f"     {sehir:<12} n={n:<4} "
            f"Model %{100 * isa / n:4.1f} | AGF %{100 * isaa / na:4.1f} | "
            f"Model net {kar / n:+.3f} | AGF net {kara / na:+.3f}"
        )

    # --- UYARI -------------------------------------------------------------
    if len(kullanilan) < AZ_VERI_ESIGI:
        print(
            f"\n  ⚠️ {len(kullanilan)} koşu çok az (<{AZ_VERI_ESIGI}), "
            "kâr/zarar şans eseri olabilir."
        )

    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
