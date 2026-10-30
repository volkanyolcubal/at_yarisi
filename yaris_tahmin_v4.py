# ============================================================================
# YARIŞ TAHMİN SİSTEMİ v4.4
# XGBoost Race Ranking
# - CSV okuma: Her 100 dosyada checkpoint
# - Feature üretimi: Her 25 yarış gününde checkpoint
# - Kapatılırsa kaldığı yerden devam eder
# - Tarih filtrelemede groupby kullanır
# - Aynı günün sonuçlarını birbirine sızdırmaz
#
# v4.3 -> v4.4 DEĞİŞİKLİĞİ:
#   Eküri (aynı sahip/antrenörden birden fazla at aynı koşuda) bilgisi
#   artık feature olarak kullanılıyor. CSV'lerde
#   "[(1)AT_ADI,(7)AT_ADI2] eküridir." şeklinde geçen satır ayrıştırılıp
#   ilgili atlara IsEkuru=1 ve EkuriOrtakSayisi (kaçlı eküri) işaretleniyor.
# ============================================================================

import os
import re
import pickle
import hashlib
import logging
import warnings
import shutil
import gc
from datetime import datetime
from collections import deque

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from xgboost import XGBRanker
except ImportError:
    raise SystemExit(
        "\nXGBoost bulunamadı.\n"
        "Terminalde çalıştır:\n"
        "pip install xgboost\n"
    )

# ============================================================================
# AYARLAR
# ============================================================================

PROGRAM_DIR = "program"
RESULT_DIR = "sonuclar"

# CSV checkpoint
RAW_HISTORY_CACHE = "v4_raw_history_checkpoint.pkl"

# Feature checkpoint
FEATURE_PROGRESS_CACHE = "v4_feature_progress.pkl"
FEATURE_CHUNK_DIR = "v4_feature_chunks"
FINAL_FEATURE_CACHE = "v4_training_features.pkl"

# Model cache
MODEL_DIR = "v4_models"

TOP_N = 8

# 'Değer adayı' (DENEME): AGF'siz temel modelin ilk DEGER_TOP'unda olup AGF
# sırası DEGER_AGF_SIRA veya daha geride olan atlar.
DEGER_TOP = 3
DEGER_AGF_SIRA = 4

# Her bu kadar sonuç CSV okununca kayıt yapılır.
CHECKPOINT_EVERY_FILES = 100

# Her bu kadar yarış günü işlendiğinde feature checkpoint kaydedilir.
FEATURE_CHECKPOINT_DAYS = 25

# program/ içinde çok CSV varsa sadece en yeni dosyayı analiz eder.
PREDICT_ONLY_LATEST_PROGRAM = True

# İdman (antrenman) verisi - hava.txt gibi TEK dosya, her gün üzerine
# eklenir (her satırın kendi tarihi zaten var, dosya adına gerek yok).
IDMAN_DOSYA = "idman.txt"
IDMAN_CACHE = "v4_idman_history.pkl"
IDMAN_CACHE_SCHEMA = "v4_idman_schema_v2"

# Jokey / Antrenör kariyer istatistikleri - statik anlık görüntü (tarih
# yok), tek dosya, isim eşleşmesiyle katılır.
ANTRENOR_DOSYA = "antrenor.txt"
JOKEY_DOSYA = "jokey.txt"
KARIYER_CACHE = "v4_kariyer_istatistik.pkl"
KARIYER_CACHE_SCHEMA = "v4_kariyer_schema_v1"

# Hava durumu - tek dosya, her gün üzerine ekleniyor (idman gibi silinmiyor,
# biriktirilerek modelin zamanla öğrenmesi sağlanıyor).
HAVA_DOSYA = "hava.txt"
HAVA_CACHE = "v4_hava_history.pkl"
HAVA_CACHE_SCHEMA = "v4_hava_schema_v1"
HAVA_ALAN_SAYISI = 10  # Rapor,Şehir,Hipodrom,Tarih,Saat,Sıcaklık,Nem,Basınç,Gökyüzü,Rüzgar

RAW_CACHE_SCHEMA = "v4_raw_checkpoint_schema_v3"
FEATURE_PROGRESS_SCHEMA = "v4_feature_checkpoint_schema_v2"

# DİKKAT: eküri feature'ı eklendiği için sürüm artırıldı - eski
# feature/model cache'i otomatik geçersiz sayılıp yeniden üretilecek.
# RAW_CACHE_SCHEMA da v3 yapıldı çünkü ham geçmiş verisine yeni sütunlar
# (IsEkuru, EkuriOrtakSayisi) eklendi; eski ham cache'te bu sütunlar yok.
# FEATURE_VERSION ham cache şemasına bağlıdır: ham geçmiş yeniden okunduğunda
# (RAW_CACHE_SCHEMA değişince) feature cache'i ve model cache'i de otomatik
# geçersiz olur. Böylece eski veriden üretilmiş feature/model sessizce
# kullanılmaya devam edemez.
FEATURE_VERSION = "v4_8_ekuri_" + RAW_CACHE_SCHEMA

# ============================================================================
# LOGLAMA
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)-8s - %(message)s",
    handlers=[
        logging.FileHandler("yaris_ranker_v4.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# REGEX / SABİTLER
# ============================================================================

TR_TABLE = str.maketrans(
    "İışŞçÇüÜöÖğĞ",
    "Iissccuuoogg"
)

ENCODINGS = [
    "utf-8",
    "utf-8-sig",
    "cp1254",
    "iso-8859-9",
    "latin1"
]

RE_RACE = re.compile(r"^(\d+)\.\s*Kosu\s*:", re.IGNORECASE)
RE_DISTANCE = re.compile(r"(\d{3,4})\s*m", re.IGNORECASE)
RE_DATE = re.compile(r"(\d{2})/(\d{2})/(\d{4})")
RE_DATE_FILE_1 = re.compile(r"(\d{8})")
RE_DATE_FILE_2 = re.compile(r"(\d{2})[.\-_](\d{2})[.\-_](\d{4})")

# v4.4: Eküri satırı - "[(1)SANCAKALAN,(7)DEDEBATUR] eküridir." formatında.
RE_EKURI = re.compile(r"\[(.*?)\]\s*ek[üuÜU]ridir", re.IGNORECASE)
RE_EKURI_AT = re.compile(r"\(\d+\)\s*([^,\]]+)")

EQUIPMENT_WORDS = {
    "KG", "K", "DB", "SK", "SKG", "GKR", "BB",
    "OG", "YP", "T", "DS", "AP"
}

# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def norm_key(value):
    if value is None:
        return ""

    text = str(value).translate(TR_TABLE).upper()
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_column_name(value):
    return norm_key(value).replace(" ", "")


def canonical_horse_name(horse_name):
    """
    KING OF TIME KG DB SK -> KING OF TIME
    BABA DOÇKA K SKG SK   -> BABA DOÇKA
    """
    if horse_name is None:
        return ""

    text = re.sub(r"\([^)]*\)", " ", str(horse_name))
    tokens = norm_key(text).split()

    while tokens and tokens[-1] in EQUIPMENT_WORDS:
        tokens.pop()

    return " ".join(tokens)


def normalize_surface(value):
    text = str(value).lower()

    if "sentetik" in text or "synthetic" in text:
        return "sentetik"

    if "çim" in text or "cim" in text or "grass" in text:
        return "cim"

    if "kum" in text or "dirt" in text:
        return "kum"

    return "bilinmiyor"


def parse_float(value, default=0.0):
    try:
        return float(str(value).strip().replace(",", "."))
    except Exception:
        return default


def parse_agf(value):
    """%25.60(1) -> 25.60"""
    try:
        match = re.search(r"(\d+[.,]\d+|\d+)", str(value))

        if match:
            return float(match.group(1).replace(",", "."))

        return 0.0
    except Exception:
        return 0.0


def parse_weight(value):
    """
    54 +2.00 -> 56.0
    56,5 -> 56.5
    """
    try:
        text = str(value).strip().replace(",", ".")

        match = re.match(r"(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)", text)
        if match:
            return float(match.group(1)) + float(match.group(2))

        match = re.match(r"(\d+\.?\d*)", text)
        if match:
            return float(match.group(1))

        return 58.0
    except Exception:
        return 58.0


def parse_start_box(value):
    """7 - DS -> 7 | 8DS -> 8"""
    try:
        match = re.match(r"^(\d+)", str(value).strip())
        return int(match.group(1)) if match else 5
    except Exception:
        return 5


def parse_age(value):
    """3y d e -> 3"""
    try:
        match = re.search(r"(\d+)", str(value))
        return int(match.group(1)) if match else 0
    except Exception:
        return 0


def degree_to_seconds(value):
    """
    1:15.44 -> 75.44
    1.15.44 -> 75.44
    """
    try:
        text = str(value).strip()

        match = re.match(r"^(\d{1,2})[:.](\d{2})\.(\d{2})$", text)

        if match:
            minute = int(match.group(1))
            second = int(match.group(2))
            centi = int(match.group(3))

            return minute * 60 + second + centi / 100.0

        return np.nan
    except Exception:
        return np.nan


def distance_group(distance):
    try:
        distance = float(distance)

        if distance <= 1200:
            return "sprint"
        elif distance <= 1400:
            return "kisa"
        elif distance <= 1600:
            return "orta"
        elif distance <= 2000:
            return "uzun"

        return "cok_uzun"

    except Exception:
        return "orta"


def race_flags(race_description):
    text = norm_key(race_description)

    return {
        "IsMaiden": int("MAIDEN" in text),
        "IsHandicap": int("HANDIKAP" in text),
        "IsGroup": int(bool(re.search(r"\bG\s*[123]\b", text))),
        "IsArab": int("ARAP" in text),
        "IsAge2": int("2 YAS" in text),
        "IsAge3": int("3 YAS" in text)
    }


def source_signature(file_path):
    """Dosyanın değişip değişmediğini kontrol eder."""
    try:
        stat = os.stat(file_path)
        return f"{stat.st_size}:{stat.st_mtime_ns}"
    except Exception:
        return ""


def file_fingerprint(files):
    """Tüm sonuç klasörünün parmak izi."""
    hasher = hashlib.md5()

    for file_path in sorted(files):
        try:
            stat = os.stat(file_path)

            hasher.update(os.path.basename(file_path).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
        except Exception:
            continue

    return hasher.hexdigest()


def extract_date_from_filename(file_path):
    file_name = os.path.basename(file_path)

    match = RE_DATE_FILE_1.search(file_name)
    if match:
        try:
            return pd.to_datetime(
                match.group(1),
                format="%Y%m%d"
            ).normalize()
        except Exception:
            pass

    match = RE_DATE_FILE_2.search(file_name)
    if match:
        try:
            day, month, year = match.groups()

            return pd.Timestamp(
                year=int(year),
                month=int(month),
                day=int(day)
            )
        except Exception:
            pass

    return pd.NaT


def safe_pickle_write(data, file_path):
    """Geçici dosyaya yazar, sonra güvenli şekilde asıl dosyaya taşır."""
    temp_path = file_path + ".tmp"

    try:
        pd.to_pickle(data, temp_path)
        os.replace(temp_path, file_path)
        return True

    except Exception as error:
        logger.error(f"🚫 Dosya kaydedilemedi: {file_path} | {error}")

        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

        return False


# ============================================================================
# CSV OKUMA
# ============================================================================

def read_text_file(file_path):
    for encoding in ENCODINGS:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()

        except UnicodeDecodeError:
            continue
        except Exception:
            continue

    return None


def parse_racing_file(file_path, is_result=True):
    """
    TJK noktalı virgüllü CSV dosyalarını okur.

    Sonuç:
    At No;At İsmi;...;Derece;Ganyan;Fark

    Program:
    At No;At İsmi;...;Son 6 Yarış;KGS;s20;EnİyiDerece
    """
    content = read_text_file(file_path)

    if not content:
        return pd.DataFrame()

    lines = content.splitlines()

    city = ""
    race_date = pd.NaT

    # Tarihi ilk 10 satırda ara.
    for line in lines[:10]:
        date_match = RE_DATE.search(line)

        if date_match:
            day, month, year = date_match.groups()

            try:
                race_date = pd.Timestamp(
                    year=int(year),
                    month=int(month),
                    day=int(day)
                )
                break
            except Exception:
                pass

    if pd.isna(race_date):
        race_date = extract_date_from_filename(file_path)

    # Şehir ilk satırdadır.
    if lines:
        first_parts = lines[0].split(";")

        if first_parts:
            city = norm_key(
                first_parts[0].replace("\ufeff", "").strip()
            )

    header_columns = None

    current_race_no = 0
    current_distance = 0.0
    current_surface = "bilinmiyor"
    current_description = ""

    records = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        normalized_line = normalize_column_name(stripped)

        # Header satırı.
        if (
            "ATNO" in normalized_line
            and "ATISMI" in normalized_line
            and "JOKEY" in normalized_line
        ):
            header_columns = [
                normalize_column_name(column)
                for column in stripped.split(";")
            ]
            continue

        # Koşu başlığı.
        race_match = RE_RACE.match(stripped)

        if race_match:
            current_race_no = int(race_match.group(1))
            current_description = stripped

            distance_match = RE_DISTANCE.search(stripped)

            if distance_match:
                current_distance = float(distance_match.group(1))

            current_surface = "bilinmiyor"

            for part in stripped.split(";"):
                detected = normalize_surface(part)

                if detected != "bilinmiyor":
                    current_surface = detected
                    break

            continue

        # v4.4: Eküri satırı - "[(1)SANCAKALAN,(7)DEDEBATUR] eküridir."
        # Bu satır atların LİSTESİNDEN SONRA geldiği için, records
        # listesinde zaten var olan, aynı koşuya (current_race_no) ait
        # ilgili atları geriye dönük işaretliyoruz.
        ekuri_match = RE_EKURI.search(stripped)

        if ekuri_match:
            ekuri_isimleri = RE_EKURI_AT.findall(ekuri_match.group(1))
            ekuri_keys = {
                canonical_horse_name(isim) for isim in ekuri_isimleri
            }
            ekuri_keys.discard("")

            if ekuri_keys and current_race_no > 0:
                for rec in records:
                    if (
                        rec.get("Kosu") == current_race_no
                        and rec.get("HorseKey") in ekuri_keys
                    ):
                        rec["IsEkuru"] = 1
                        rec["EkuriOrtakSayisi"] = len(ekuri_keys)

            continue

        # Bahis/ödül satırları.
        upper = norm_key(stripped)

        if upper.startswith((
            "IKRAMIYE",
            "GANYAN",
            "IKILI",
            "SIRALI",
            "CIFTE",
            "TABELA",
            "PLASE",
            "SABIT",
            "REKOR",
            "YETISTIRICI",
            "AT SAHIBI"
        )):
            continue

        parts = stripped.split(";")

        if len(parts) < 5:
            continue

        at_no_raw = parts[0].strip()
        at_name_raw = parts[1].strip() if len(parts) > 1 else ""

        # At satırı doğrulaması.
        if not re.match(
            r"^\d+(\s*-\s*[A-Za-z]+|[A-Za-z]{1,3})?$",
            at_no_raw
        ):
            continue

        if not at_name_raw or current_race_no <= 0:
            continue

        row_map = {}

        if header_columns:
            for index, column_name in enumerate(header_columns):
                row_map[column_name] = (
                    parts[index].strip()
                    if index < len(parts)
                    else ""
                )

        def get_value(names, fallback_index, default=""):
            for name in names:
                if name in row_map:
                    return row_map[name]

            if fallback_index < len(parts):
                return parts[fallback_index].strip()

            return default

        horse_key = canonical_horse_name(at_name_raw)

        if not horse_key:
            continue

        is_non_runner = "KOSMAZ" in norm_key(at_name_raw)

        age_raw = get_value(["YAS"], 2, "")
        weight_raw = get_value(["KILO", "SIKLET"], 5, "58")

        jockey_raw = get_value(
            ["JOKEYADI", "JOKEYAD", "JOKEY"],
            6,
            ""
        )

        trainer_raw = get_value(
            ["ANTRENORADI", "ANTRENORAD", "ANTRENOR"],
            8,
            ""
        )

        start_raw = get_value(["ST"], 9, "5")
        agf_raw = get_value(["AGF"], 10, "0")
        h_raw = get_value(["H", "HP", "HANDIKAP"], 11, "0")

        flags = race_flags(current_description)

        record = {
            "Tarih": race_date,
            "Sehir": city,
            "Kosu": current_race_no,
            "Mesafe": current_distance,
            "PistTipi": current_surface,
            "RaceDescription": current_description,
            "AtIsmiRaw": at_name_raw,
            "HorseKey": horse_key,
            "JokeyKey": norm_key(jockey_raw.replace(" AP", "")),
            "AntrenorKey": norm_key(trainer_raw),
            "JokeyAdi": jockey_raw,
            "AntrenorAdi": trainer_raw,
            "Yas": parse_age(age_raw),
            "Kilo": parse_weight(weight_raw),
            "St": parse_start_box(start_raw),
            "AGF": parse_agf(agf_raw),
            "H": parse_float(h_raw),
            "Kosmaz": is_non_runner,
            "IsMaiden": flags["IsMaiden"],
            "IsHandicap": flags["IsHandicap"],
            "IsGroup": flags["IsGroup"],
            "IsArab": flags["IsArab"],
            "IsAge2": flags["IsAge2"],
            "IsAge3": flags["IsAge3"],
            # v4.4: eküri varsayılanları - aşağıda RE_EKURI satırı
            # bulunursa bu değerler geriye dönük güncellenir.
            "IsEkuru": 0,
            "EkuriOrtakSayisi": 1,
            "SourceFile": os.path.basename(file_path)
        }

        if is_result:
            degree_raw = get_value(["DERECE"], 12, "")
            record["DereceSec"] = degree_to_seconds(degree_raw)

        else:
            record["Son6Yaris"] = get_value(
                ["SON6YARIS", "SON6YARISI", "SON6"],
                12,
                ""
            )

            record["KGS"] = parse_float(
                get_value(["KGS"], 13, "0")
            )

            record["s20"] = parse_float(
                get_value(["S20"], 14, "0")
            )

            record["EnIyiDerece"] = get_value(
                ["ENIYIDERECE", "ENIYID", "IYIDERECE"],
                15,
                ""
            )

        records.append(record)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    df = df[~df["Kosmaz"]].copy()
    df = df[df["HorseKey"].str.len() > 1].copy()

    df["RaceID"] = (
        df["Tarih"].astype(str)
        + "|"
        + df["Sehir"].astype(str)
        + "|"
        + df["Kosu"].astype(str)
        + "|"
        + df["SourceFile"].astype(str)
    )

    if not is_result:
        return df.reset_index(drop=True)

    df["DereceSec"] = pd.to_numeric(
        df["DereceSec"],
        errors="coerce"
    )

    valid = df[df["DereceSec"].notna()].copy()

    if valid.empty:
        return pd.DataFrame()

    valid["Siralama"] = (
        valid.groupby("RaceID")["DereceSec"]
        .rank(method="first", ascending=True)
        .astype(int)
    )

    valid["FieldSize"] = (
        valid.groupby("RaceID")["RaceID"]
        .transform("count")
    )

    valid["Kazandi"] = (valid["Siralama"] == 1).astype(int)
    valid["Top3"] = (valid["Siralama"] <= 3).astype(int)

    valid["FinishPct"] = np.where(
        valid["FieldSize"] > 1,
        (
            valid["FieldSize"] - valid["Siralama"]
        ) / (
            valid["FieldSize"] - 1
        ),
        0.5
    )

    valid["Speed"] = valid["Mesafe"] / valid["DereceSec"]

    median_speed = (
        valid.groupby("RaceID")["Speed"]
        .transform("median")
    )

    valid["SpeedRatio"] = np.where(
        median_speed > 0,
        valid["Speed"] / median_speed,
        1.0
    )

    valid["SpeedRatio"] = valid["SpeedRatio"].clip(0.60, 1.40)

    return valid.reset_index(drop=True)


# ============================================================================
# İDMAN (ANTRENMAN) VERİSİ OKUMA
# ============================================================================

IDMAN_MESAFE_SUTUNLARI = [
    "1400M", "1200M", "1000M", "800M", "600M", "400M", "200M"
]

# Web sayfasından kopyalarken bulaşan çöp satırlar - veri satırı DEĞİL,
# görürsek atlarız.
COP_SATIR_DESENLERI = [
    re.compile(r"^Daha Fazla Sonuç Göster$", re.IGNORECASE),
    re.compile(r"^Toplam\s+[\d.,]+\s+sonuçtan", re.IGNORECASE),
    re.compile(r"^Başa Dön$", re.IGNORECASE),
]


def cop_satiri_mi(satir):
    satir_temiz = satir.strip()
    return any(desen.match(satir_temiz) for desen in COP_SATIR_DESENLERI)


def parse_idman_dosyasi():
    """
    idman.txt - hava.txt gibi TEK dosya, her gün üzerine yeni satırlar
    eklenir (silinmez).

    ÖNEMLİ FORMAT NOTU: Veri satırları TAB ile ayrılmış, sabit 19 alanlık
    bir yapıda (web sayfasından kopyalanınca başlık satırı farklı - tek
    boşlukla - geliyor ve alan isimlerinin kendisi de boşluk içerdiği için
    ("İ. Tarihi" gibi) otomatik ayrıştırılamıyor). Bu yüzden başlık satırı
    tamamen atlanır, sabit/bilinen sütun sırası kullanılır:

    0=At Adı 1=Irk 2=Cins. 3=Yaş 4=1400m 5=1200m 6=1000m 7=800m 8=600m
    9=400m 10=200m 11=Durum 12=İ.Tarihi 13=İ.Hip. 14=P.Dur 15=Pist
    16=İ.Türü 17=İ.Jokeyi 18=Detay

    Web sayfasından kopyalarken bulaşan "Daha Fazla Sonuç Göster",
    "Toplam X sonuçtan..." "Başa Dön" gibi çöp satırlar ve tab içermeyen
    (başlık gibi) satırlar otomatik atlanır.
    """
    if not os.path.exists(IDMAN_DOSYA):
        return pd.DataFrame()

    content = read_text_file(IDMAN_DOSYA)

    if not content:
        return pd.DataFrame()

    lines = [line for line in content.splitlines() if line.strip()]

    if not lines:
        return pd.DataFrame()

    IDX_AT_ADI, IDX_TARIH, IDX_HIP = 0, 12, 13
    IDX_PIST, IDX_TUR, IDX_JOKEY, IDX_DURUM = 15, 16, 17, 11
    MESAFE_INDEX_MAP = {
        1400: 4, 1200: 5, 1000: 6, 800: 7, 600: 8, 400: 9, 200: 10
    }
    MIN_ALAN_SAYISI = 15  # başlık/çöp satırları bu kadar tab içermez

    records = []

    for line in lines:
        if cop_satiri_mi(line):
            continue

        parts = line.split("\t")

        if len(parts) < MIN_ALAN_SAYISI:
            continue  # başlık satırı ya da tanınmayan bir şey - atla

        at_adi_raw = parts[IDX_AT_ADI].strip()
        horse_key = canonical_horse_name(at_adi_raw)

        if not horse_key:
            continue

        tarih_raw = parts[IDX_TARIH].strip() if IDX_TARIH < len(parts) else ""
        date_match = RE_DATE.search(tarih_raw) or RE_DATE_FILE_2.search(tarih_raw)

        if not date_match:
            continue

        day, month, year = date_match.groups()

        try:
            idman_tarihi = pd.Timestamp(
                year=int(year), month=int(month), day=int(day)
            )
        except Exception:
            continue

        # Bariz hatalı/gelecek tarihleri (ör. yazım hatasıyla yanlış yıl)
        # ele: bugünden 3 günden fazla ileriyse muhtemelen veri hatasıdır.
        if idman_tarihi > pd.Timestamp.today() + pd.Timedelta(days=3):
            continue

        # Hangi mesafe sütununda değer varsa, o idmanın derecesidir.
        idman_mesafe = None
        idman_derece_sec = np.nan

        for mesafe, index in MESAFE_INDEX_MAP.items():
            if index < len(parts):
                deger = parts[index].strip()

                if deger:
                    saniye = degree_to_seconds(deger)

                    if not pd.isna(saniye):
                        idman_mesafe = mesafe
                        idman_derece_sec = saniye
                        break

        record = {
            "HorseKey": horse_key,
            "IdmanTarihi": idman_tarihi,
            "IdmanMesafe": idman_mesafe if idman_mesafe else 0,
            "IdmanDereceSec": idman_derece_sec,
            "IdmanPist": (
                normalize_surface(parts[IDX_PIST])
                if IDX_PIST < len(parts) else "bilinmiyor"
            ),
            "IdmanTuru": (
                norm_key(parts[IDX_TUR]) if IDX_TUR < len(parts) else ""
            ),
            "IdmanJokey": (
                norm_key(parts[IDX_JOKEY]) if IDX_JOKEY < len(parts) else ""
            ),
            "IdmanDurum": (
                norm_key(parts[IDX_DURUM]) if IDX_DURUM < len(parts) else ""
            ),
        }

        records.append(record)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)

def load_idman_history():
    """
    idman.txt küçük/orta boy bir dosya olduğu için (hava.txt gibi) ağır
    bir checkpoint sistemine gerek yok - dosya değişmediyse cache'ten döner.
    """
    if not os.path.exists(IDMAN_DOSYA):
        return pd.DataFrame()

    fingerprint = source_signature(IDMAN_DOSYA)

    if os.path.exists(IDMAN_CACHE):
        try:
            payload = pd.read_pickle(IDMAN_CACHE)

            if (
                isinstance(payload, dict)
                and payload.get("cache_schema") == IDMAN_CACHE_SCHEMA
                and payload.get("fingerprint") == fingerprint
            ):
                idman_df = payload.get("idman_history", pd.DataFrame())

                logger.info(
                    f"♻️ İdman cache yüklendi | {len(idman_df):,} kayıt"
                )

                return idman_df

        except Exception:
            pass

    idman_df = parse_idman_dosyasi()

    if idman_df.empty:
        logger.warning(
            "⚠️ idman.txt okunamadı veya geçerli kayıt bulunamadı, "
            "idman özellikleri nötr kullanılacak."
        )
        return idman_df

    idman_df = idman_df.drop_duplicates(
        subset=["HorseKey", "IdmanTarihi", "IdmanMesafe"],
        keep="last"
    )
    idman_df = idman_df.sort_values(
        ["HorseKey", "IdmanTarihi"]
    ).reset_index(drop=True)

    safe_pickle_write(
        {
            "cache_schema": IDMAN_CACHE_SCHEMA,
            "fingerprint": fingerprint,
            "idman_history": idman_df
        },
        IDMAN_CACHE
    )

    logger.info(f"✅ İdman geçmişi hazır | {len(idman_df):,} kayıt")

    return idman_df

def idman_ozellikleri_ekle(df, idman_history, tarih_sutunu="Tarih"):
    """
    Her satırdaki (HorseKey, Tarih) çiftine göre, o attan o tarihten
    ÖNCEKİ EN YAKIN idman kaydını bulur (pd.merge_asof ile - vektörize,
    hızlı, döngü yok). Sızıntı olmasın diye SADECE geçmişteki idmanlar
    kullanılır (allow_exact_matches=False -> aynı gün sayılmaz, o günden
    önceki son idman aranır).

    İdman verisi olmayan (henüz toplanmamış / eski) satırlarda nötr
    varsayılan değerler kullanılır - model bu durumda feature'ı "bilgi
    yok" olarak öğrenir, hata vermez.
    """
    if idman_history.empty:
        df["IdmanTazelikNorm"] = 0.0
        df["IdmanDereceNorm"] = 0.5
        df["IdmanPistUyumu"] = 0
        df["IdmanVarMi"] = 0
        return df

    sol = df[["HorseKey", tarih_sutunu]].copy()
    sol["_sira"] = np.arange(len(sol))
    sol = sol.sort_values(tarih_sutunu)

    sag = idman_history.sort_values("IdmanTarihi")

    birlesik = pd.merge_asof(
        sol,
        sag,
        left_on=tarih_sutunu,
        right_on="IdmanTarihi",
        by="HorseKey",
        direction="backward",
        allow_exact_matches=False
    )

    birlesik = birlesik.sort_values("_sira").reset_index(drop=True)

    gun_once = (birlesik[tarih_sutunu] - birlesik["IdmanTarihi"]).dt.days

    # Tazelik: idman ne kadar yakın zamanda ise 1'e o kadar yakın,
    # hiç idman yoksa / çok eskiyse 0.
    tazelik = 1.0 - (gun_once.fillna(9999) / 30.0).clip(0.0, 1.0)
    df["IdmanTazelikNorm"] = tazelik.to_numpy()

    df["IdmanVarMi"] = gun_once.notna().astype(int).to_numpy()

    # Derece ne kadar düşükse (hızlıysa) o kadar iyi - normalize edip
    # ters çeviriyoruz (yüksek = iyi) diğer feature'larla tutarlı olsun.
    derece = birlesik["IdmanDereceSec"]
    derece_norm = 1.0 - ((derece - 10.0) / 50.0).clip(0.0, 1.0)
    df["IdmanDereceNorm"] = derece_norm.fillna(0.5).to_numpy()

    pist_uyumu = (
        birlesik["IdmanPist"].fillna("bilinmiyor") == df["PistTipi"].values
    ).astype(int)
    df["IdmanPistUyumu"] = pist_uyumu.to_numpy()

    return df


# ============================================================================
# HAVA DURUMU VERİSİ
# ============================================================================

RE_RUZGAR_SAYI = re.compile(r"(\d+)")
RE_BIRIM_TEMIZLE = re.compile(r"(?i)km\s*/?\s*s\.?a?\.?|kmh|km|saat|sa\b")


def ruzgar_ayikla(ruzgar_raw):
    """
    Rüzgar metni çok farklı biçimlerde gelebiliyor:
    'Doğudan 4', 'Batıdan15km/sa', 'KUZEYDEN 6 KM/S', "Kuzey'den 10 km/sa",
    '21 km/sa B', 'GB 14', 'KD16', '3 KM SAAT', '0' gibi. Sayı her zaman
    rüzgar hızı, geri kalan harfler (varsa) yön bilgisidir - sırası
    (önce/sonra) fark etmez.
    """
    metin = (ruzgar_raw or "").strip()

    sayi_match = RE_RUZGAR_SAYI.search(metin)
    hiz = float(sayi_match.group(1)) if sayi_match else np.nan

    temiz = re.sub(r"\d+", " ", metin)
    temiz = RE_BIRIM_TEMIZLE.sub(" ", temiz)
    temiz = temiz.replace("'", " ").replace("%", " ")
    yon = norm_key(temiz)

    return hiz, yon


def parse_hava_dosyasi():
    """
    hava.txt - her satır TAB ile ayrılmış TEK bir hava raporu (10 alan):
    Rapor, Şehir, Hipodrom, Tarih, Saat, Sıcaklık, Nem, Basınç, Gökyüzü,
    Rüzgar. Başlık satırı sekmesiz (tek boşlukla) geldiği için otomatik
    atlanır (tab sayısı yetersiz olduğu için), sabit sütun sırası kullanılır.
    """
    if not os.path.exists(HAVA_DOSYA):
        return pd.DataFrame()

    content = read_text_file(HAVA_DOSYA)

    if not content:
        return pd.DataFrame()

    lines = [line for line in content.splitlines() if line.strip()]

    if not lines:
        return pd.DataFrame()

    IDX_SEHIR, IDX_HIPODROM, IDX_TARIH, IDX_SAAT = 1, 2, 3, 4
    IDX_SICAKLIK, IDX_NEM, IDX_BASINC = 5, 6, 7
    IDX_GOKYUZU, IDX_RUZGAR = 8, 9
    MIN_ALAN_SAYISI = 8

    kayitlar = []

    for line in lines:
        if cop_satiri_mi(line):
            continue

        parts = line.split("\t")

        if len(parts) < MIN_ALAN_SAYISI:
            continue  # başlık satırı ya da tanınmayan bir şey - atla

        tarih_raw = parts[IDX_TARIH].strip() if IDX_TARIH < len(parts) else ""
        date_match = RE_DATE.search(tarih_raw) or RE_DATE_FILE_2.search(tarih_raw)

        if not date_match:
            continue

        gun, ay, yil = date_match.groups()

        try:
            tarih = pd.Timestamp(year=int(yil), month=int(ay), day=int(gun))
        except Exception:
            continue

        ruzgar_hiz, ruzgar_yon = ruzgar_ayikla(
            parts[IDX_RUZGAR] if IDX_RUZGAR < len(parts) else ""
        )

        kayitlar.append({
            "Sehir": norm_key(parts[IDX_SEHIR]) if IDX_SEHIR < len(parts) else "",
            "Hipodrom": (
                norm_key(parts[IDX_HIPODROM]) if IDX_HIPODROM < len(parts) else ""
            ),
            "Tarih": tarih,
            "Saat": parts[IDX_SAAT].strip() if IDX_SAAT < len(parts) else "",
            "Sicaklik": parse_float(parts[IDX_SICAKLIK], np.nan) if IDX_SICAKLIK < len(parts) else np.nan,
            "Nem": parse_float(parts[IDX_NEM], np.nan) if IDX_NEM < len(parts) else np.nan,
            "Basinc": parse_float(parts[IDX_BASINC], np.nan) if IDX_BASINC < len(parts) else np.nan,
            "Gokyuzu": norm_key(parts[IDX_GOKYUZU]) if IDX_GOKYUZU < len(parts) else "",
            "RuzgarHiz": ruzgar_hiz,
            "RuzgarYon": ruzgar_yon,
        })

    if not kayitlar:
        return pd.DataFrame()

    return pd.DataFrame(kayitlar)

def load_hava_history():
    """
    hava.txt küçük bir dosya olduğu için (günde birkaç satır) ağır bir
    checkpoint sistemine gerek yok - dosya değişmediyse cache'ten döner.
    """
    if not os.path.exists(HAVA_DOSYA):
        return pd.DataFrame()

    fingerprint = source_signature(HAVA_DOSYA)

    if os.path.exists(HAVA_CACHE):
        try:
            payload = pd.read_pickle(HAVA_CACHE)

            if (
                isinstance(payload, dict)
                and payload.get("cache_schema") == HAVA_CACHE_SCHEMA
                and payload.get("fingerprint") == fingerprint
            ):
                hava_df = payload.get("hava_history", pd.DataFrame())

                logger.info(f"♻️ Hava cache yüklendi | {len(hava_df):,} kayıt")

                return hava_df

        except Exception:
            pass

    hava_df = parse_hava_dosyasi()

    if hava_df.empty:
        logger.warning("⚠️ hava.txt okunamadı veya boş, hava özellikleri nötr kullanılacak.")
        return hava_df

    hava_df = hava_df.drop_duplicates(
        subset=["Sehir", "Tarih", "Saat"], keep="last"
    )

    # Aynı şehir + aynı gün birden fazla rapor varsa (farklı saatlerde) ortalama al.
    hava_gunluk = hava_df.groupby(["Sehir", "Tarih"], as_index=False).agg({
        "Sicaklik": "mean",
        "Nem": "mean",
        "Basinc": "mean",
        "RuzgarHiz": "mean",
        "Gokyuzu": "first",
        "RuzgarYon": "first",
    })

    safe_pickle_write(
        {
            "cache_schema": HAVA_CACHE_SCHEMA,
            "fingerprint": fingerprint,
            "hava_history": hava_gunluk
        },
        HAVA_CACHE
    )

    logger.info(f"✅ Hava geçmişi hazır | {len(hava_gunluk):,} şehir-gün kaydı")

    return hava_gunluk


def hava_ozellikleri_ekle(df, hava_history, tarih_sutunu="Tarih"):
    """
    Şehir + Tarih eşleşmesiyle (at bazlı değil - hava tüm hipodromu
    aynı şekilde etkiler) hava özelliklerini ekler. Veri yoksa nötr
    varsayılanlar kullanılır, hata vermez.
    """
    if hava_history is None or hava_history.empty:
        df["SicaklikNorm"] = 0.5
        df["NemNorm"] = 0.5
        df["BasincNorm"] = 0.5
        df["RuzgarHizNorm"] = 0.0
        df["HavaAcikMi"] = 0
        df["HavaVarMi"] = 0
        return df

    sol = df[["Sehir", tarih_sutunu]].copy()
    sol["_sira"] = np.arange(len(sol))

    birlesik = sol.merge(
        hava_history,
        left_on=["Sehir", tarih_sutunu],
        right_on=["Sehir", "Tarih"],
        how="left",
        suffixes=("", "_hava")
    ).sort_values("_sira").reset_index(drop=True)

    df["SicaklikNorm"] = (
        (birlesik["Sicaklik"] / 45.0).clip(0.0, 1.0).fillna(0.5).to_numpy()
    )
    df["NemNorm"] = (
        (birlesik["Nem"] / 100.0).clip(0.0, 1.0).fillna(0.5).to_numpy()
    )
    df["BasincNorm"] = (
        ((birlesik["Basinc"] - 980.0) / 60.0).clip(0.0, 1.0).fillna(0.5).to_numpy()
    )
    df["RuzgarHizNorm"] = (
        (birlesik["RuzgarHiz"] / 40.0).clip(0.0, 1.0).fillna(0.0).to_numpy()
    )
    df["HavaAcikMi"] = (
        birlesik["Gokyuzu"].fillna("").str.contains("ACIK", na=False)
        | birlesik["Gokyuzu"].fillna("").str.contains("AZ BULUTLU", na=False)
    ).astype(int).to_numpy()
    df["HavaVarMi"] = birlesik["Sicaklik"].notna().astype(int).to_numpy()

    return df


# ============================================================================
# JOKEY / ANTRENÖR KARİYER İSTATİSTİKLERİ
# ============================================================================

def _isim_anahtari_temizle(isim):
    """
    Jokey/antrenör isimlerini HorseKey ile aynı normalize mantığıyla
    (norm_key) işler, ayrıca ' AP' (aprenti) ve '(Y0)' gibi ekleri temizler
    - JokeyKey/AntrenorKey ile birebir eşleşsin diye.
    """
    isim = re.sub(r"\(Y\d+\)", "", isim or "")
    isim = isim.replace(" AP", "")
    return norm_key(isim)


def parse_kariyer_dosyasi(dosya_yolu, isim_sutun_adi):
    """
    antrenor.txt / jokey.txt - tek satır başlık (boşluklu), sonrasında
    TAB ile ayrılmış satırlar: İsim, Koşu, 1., 2., 3., 4., 5., 1.%, 2.%,
    3.%, 4.%, 5.%. Web sayfası çöp satırları ('Daha Fazla Sonuç Göster'
    vb.) otomatik atlanır.
    """
    if not os.path.exists(dosya_yolu):
        return pd.DataFrame()

    content = read_text_file(dosya_yolu)

    if not content:
        return pd.DataFrame()

    lines = [line for line in content.splitlines() if line.strip()]

    IDX_ISIM, IDX_KOSU = 0, 1
    IDX_1, IDX_2, IDX_3, IDX_4, IDX_5 = 2, 3, 4, 5, 6
    IDX_P1, IDX_P2, IDX_P3, IDX_P4, IDX_P5 = 7, 8, 9, 10, 11
    MIN_ALAN_SAYISI = 12

    kayitlar = []

    for line in lines:
        if cop_satiri_mi(line):
            continue

        parts = line.split("\t")

        if len(parts) < MIN_ALAN_SAYISI:
            continue  # başlık satırı ya da tanınmayan bir şey - atla

        isim_raw = parts[IDX_ISIM].strip()
        anahtar = _isim_anahtari_temizle(isim_raw)

        if not anahtar:
            continue

        try:
            kosu = int(parse_float(parts[IDX_KOSU], 0))
            pct1 = parse_float(parts[IDX_P1], 0.0)
            pct2 = parse_float(parts[IDX_P2], 0.0)
            pct3 = parse_float(parts[IDX_P3], 0.0)
        except Exception:
            continue

        kayitlar.append({
            isim_sutun_adi: anahtar,
            "KariyerKosu": kosu,
            "KariyerBirinciPct": pct1 / 100.0,
            "KariyerTop3Pct": (pct1 + pct2 + pct3) / 100.0,
        })

    if not kayitlar:
        return pd.DataFrame()

    df = pd.DataFrame(kayitlar)
    # Aynı isim birden fazla kez geçerse (nadiren olabilir) en yüksek
    # koşu sayılı (en güvenilir) satırı tut.
    df = df.sort_values("KariyerKosu", ascending=False).drop_duplicates(
        subset=[isim_sutun_adi], keep="first"
    )

    return df.reset_index(drop=True)


def load_kariyer_istatistikleri():
    """
    antrenor.txt + jokey.txt statik anlık görüntü dosyaları - tarih yok,
    tek seferlik isim eşleşmesiyle katılır. Küçük dosyalar olduğu için
    basit fingerprint cache yeterli.
    """
    if not os.path.exists(ANTRENOR_DOSYA) and not os.path.exists(JOKEY_DOSYA):
        return pd.DataFrame(), pd.DataFrame()

    fingerprint = (
        source_signature(ANTRENOR_DOSYA) + "|" + source_signature(JOKEY_DOSYA)
    )

    if os.path.exists(KARIYER_CACHE):
        try:
            payload = pd.read_pickle(KARIYER_CACHE)

            if (
                isinstance(payload, dict)
                and payload.get("cache_schema") == KARIYER_CACHE_SCHEMA
                and payload.get("fingerprint") == fingerprint
            ):
                logger.info(
                    f"♻️ Kariyer istatistik cache yüklendi | "
                    f"{len(payload.get('antrenor', pd.DataFrame())):,} antrenör, "
                    f"{len(payload.get('jokey', pd.DataFrame())):,} jokey"
                )

                return payload.get("antrenor", pd.DataFrame()), payload.get("jokey", pd.DataFrame())

        except Exception:
            pass

    antrenor_df = parse_kariyer_dosyasi(ANTRENOR_DOSYA, "AntrenorKey")
    jokey_df = parse_kariyer_dosyasi(JOKEY_DOSYA, "JokeyKey")

    safe_pickle_write(
        {
            "cache_schema": KARIYER_CACHE_SCHEMA,
            "fingerprint": fingerprint,
            "antrenor": antrenor_df,
            "jokey": jokey_df,
        },
        KARIYER_CACHE
    )

    logger.info(
        f"✅ Kariyer istatistikleri hazır | "
        f"{len(antrenor_df):,} antrenör, {len(jokey_df):,} jokey"
    )

    return antrenor_df, jokey_df


def kariyer_ozellikleri_ekle(df, antrenor_df, jokey_df):
    """
    JokeyKey / AntrenorKey isim eşleşmesiyle kariyer istatistiklerini
    ekler. Not: bu statik bir anlık görüntü (tarihsiz), yani geçmiş
    eğitim satırlarında hafif bir 'ileriye bakış' payı var (kariyer
    toplamı o tarihten sonrasını da içerebilir) - ama jokey/antrenör
    genel yetkinliği zaman içinde nispeten stabil olduğu için pratikte
    kabul edilebilir bir yaklaşım. Veri yoksa nötr varsayılanlar kullanılır.
    """
    if antrenor_df is None or antrenor_df.empty:
        df["AntrenorKariyerBirinciPct"] = 0.10
        df["AntrenorKariyerTop3Pct"] = 0.30
        df["AntrenorKariyerVarMi"] = 0
    else:
        birlesik = df[["AntrenorKey"]].merge(
            antrenor_df, on="AntrenorKey", how="left"
        )
        df["AntrenorKariyerBirinciPct"] = (
            birlesik["KariyerBirinciPct"].fillna(0.10).to_numpy()
        )
        df["AntrenorKariyerTop3Pct"] = (
            birlesik["KariyerTop3Pct"].fillna(0.30).to_numpy()
        )
        df["AntrenorKariyerVarMi"] = (
            birlesik["KariyerKosu"].notna().astype(int).to_numpy()
        )

    if jokey_df is None or jokey_df.empty:
        df["JokeyKariyerBirinciPct"] = 0.10
        df["JokeyKariyerTop3Pct"] = 0.30
        df["JokeyKariyerVarMi"] = 0
    else:
        birlesik = df[["JokeyKey"]].merge(
            jokey_df, on="JokeyKey", how="left"
        )
        df["JokeyKariyerBirinciPct"] = (
            birlesik["KariyerBirinciPct"].fillna(0.10).to_numpy()
        )
        df["JokeyKariyerTop3Pct"] = (
            birlesik["KariyerTop3Pct"].fillna(0.30).to_numpy()
        )
        df["JokeyKariyerVarMi"] = (
            birlesik["KariyerKosu"].notna().astype(int).to_numpy()
        )

    return df


# ============================================================================
# RAW CSV CHECKPOINT
# ============================================================================

def load_raw_checkpoint():
    if not os.path.exists(RAW_HISTORY_CACHE):
        return pd.DataFrame(), {}

    try:
        payload = pd.read_pickle(RAW_HISTORY_CACHE)

        if not isinstance(payload, dict):
            return pd.DataFrame(), {}

        if payload.get("cache_schema") != RAW_CACHE_SCHEMA:
            logger.warning(
                "⚠️ Eski raw cache formatı bulundu (eküri sütunları yok). "
                "Tüm sonuç CSV'leri yeniden okunacak (checkpoint'li)."
            )
            return pd.DataFrame(), {}

        history = payload.get("history", pd.DataFrame())
        processed_files = payload.get("processed_files", {})

        if not isinstance(history, pd.DataFrame):
            history = pd.DataFrame()

        if not isinstance(processed_files, dict):
            processed_files = {}

        logger.info(
            f"♻️ Raw checkpoint yüklendi | "
            f"CSV: {len(processed_files):,} | "
            f"At kaydı: {len(history):,}"
        )

        return history, processed_files

    except Exception as error:
        logger.warning(f"⚠️ Raw checkpoint okunamadı: {error}")
        return pd.DataFrame(), {}


def save_raw_checkpoint(history, processed_files):
    payload = {
        "cache_schema": RAW_CACHE_SCHEMA,
        "feature_version": FEATURE_VERSION,
        "processed_files": processed_files,
        "history": history,
        "saved_at": datetime.now().isoformat()
    }

    if safe_pickle_write(payload, RAW_HISTORY_CACHE):
        logger.info(
            f"💾 RAW CHECKPOINT | "
            f"CSV: {len(processed_files):,} | "
            f"At kaydı: {len(history):,}"
        )


def normalize_history(history):
    if history.empty:
        return history

    history = history.copy()

    history["Tarih"] = pd.to_datetime(
        history["Tarih"],
        errors="coerce"
    )

    history = history[history["Tarih"].notna()].copy()
    history["Tarih"] = history["Tarih"].dt.normalize()

    duplicate_cols = [
        col
        for col in ["RaceID", "HorseKey", "Siralama"]
        if col in history.columns
    ]

    if duplicate_cols:
        history = history.drop_duplicates(
            subset=duplicate_cols,
            keep="last"
        )

    history = history.sort_values(
        ["Tarih", "RaceID", "Siralama"]
    ).reset_index(drop=True)

    return history


def load_or_build_raw_history(result_files):
    """
    Her 100 CSV sonunda v4_raw_history_checkpoint.pkl kaydeder.
    Kapatılırsa tekrar açıldığında kaldığı dosyadan devam eder.
    """
    logger.info(
        f"📂 Sonuç klasöründe toplam "
        f"{len(result_files):,} CSV bulundu."
    )

    history, processed_files = load_raw_checkpoint()

    current_files = {}

    for file_path in result_files:
        full_path = os.path.abspath(file_path)

        current_files[full_path] = {
            "path": file_path,
            "signature": source_signature(file_path),
            "name": os.path.basename(file_path)
        }

    # Değişen veya silinen dosyaların eski kayıtlarını kaldır.
    files_to_remove = set()

    for old_path in list(processed_files.keys()):
        if old_path not in current_files:
            files_to_remove.add(os.path.basename(old_path))
            processed_files.pop(old_path, None)

    for full_path, info in current_files.items():
        old_signature = processed_files.get(full_path)

        if (
            old_signature is not None
            and old_signature != info["signature"]
        ):
            files_to_remove.add(info["name"])
            processed_files.pop(full_path, None)

    if files_to_remove and not history.empty:
        logger.info(
            f"🔄 Değişmiş/silinmiş "
            f"{len(files_to_remove)} CSV kaydı temizleniyor..."
        )

        history = history[
            ~history["SourceFile"].isin(files_to_remove)
        ].copy()

    pending_files = []

    for full_path, info in current_files.items():
        if processed_files.get(full_path) != info["signature"]:
            pending_files.append((
                full_path,
                info["path"],
                info["signature"],
                info["name"]
            ))

    if not pending_files:
        history = normalize_history(history)
        save_raw_checkpoint(history, processed_files)

        logger.info("✅ Tüm sonuç CSV dosyaları zaten işlenmiş.")

        return history, file_fingerprint(result_files)

    logger.info(
        f"⏳ Kalan CSV: {len(pending_files):,} | "
        f"Her {CHECKPOINT_EVERY_FILES} dosyada kayıt."
    )

    pending_frames = []

    try:
        with tqdm(
            total=len(pending_files),
            desc="📂 CSV okunuyor",
            unit="dosya"
        ) as progress:

            for index, (
                full_path,
                file_path,
                signature,
                file_name
            ) in enumerate(pending_files, start=1):

                try:
                    df = parse_racing_file(
                        file_path,
                        is_result=True
                    )

                    if not df.empty:
                        pending_frames.append(df)

                except Exception as error:
                    logger.warning(
                        f"⚠️ Okunamadı: {file_name} | {error}"
                    )

                # Dosya boş olsa bile işlenmiş olarak kaydedilir.
                processed_files[full_path] = signature
                progress.update(1)

                # Her 100 dosyada raw checkpoint.
                if (
                    index % CHECKPOINT_EVERY_FILES == 0
                    or index == len(pending_files)
                ):
                    if pending_frames:
                        new_history = pd.concat(
                            pending_frames,
                            ignore_index=True
                        )

                        if history.empty:
                            history = new_history
                        else:
                            history = pd.concat(
                                [history, new_history],
                                ignore_index=True
                            )

                        pending_frames = []

                    save_raw_checkpoint(history, processed_files)

                    logger.info(
                        f"✅ CSV grup tamamlandı | "
                        f"İşlenen: {len(processed_files):,} / "
                        f"{len(result_files):,}"
                    )

    except KeyboardInterrupt:
        if pending_frames:
            new_history = pd.concat(
                pending_frames,
                ignore_index=True
            )

            if history.empty:
                history = new_history
            else:
                history = pd.concat(
                    [history, new_history],
                    ignore_index=True
                )

        save_raw_checkpoint(history, processed_files)

        logger.warning(
            "⚠️ CSV okuma durduruldu. "
            "Sonraki açılışta checkpoint'ten devam eder."
        )

        raise

    history = normalize_history(history)
    save_raw_checkpoint(history, processed_files)

    fingerprint = file_fingerprint(result_files)

    logger.info(
        f"✅ Sonuç geçmişi hazır | "
        f"{len(history):,} at kaydı | "
        f"{history['RaceID'].nunique():,} koşu | "
        f"{history['HorseKey'].nunique():,} farklı at"
    )

    return history, fingerprint


# ============================================================================
# İSTATİSTİK DURUMU
# ============================================================================

class HistoryState:
    """
    Tüm yapılar pickle ile kaydedilebilir.
    Feature üretimi yarıda kalırsa state kaydedilir.
    """

    def __init__(self):
        self.global_rec = [0, 0, 0, 0.0, 0.0]

        self.horse = {}
        self.horse_surface = {}
        self.horse_distance = {}
        self.horse_surface_distance = {}

        self.jockey = {}
        self.trainer = {}
        self.horse_jockey = {}
        self.jockey_trainer = {}

        self.recent_form = {}
        self.last_date = {}

    @staticmethod
    def new_record():
        # n, win, top3, finish_sum, speed_sum
        return [0, 0, 0, 0.0, 0.0]

    def add(self, bucket, key, win, top3, finish, speed):
        if not key:
            return

        if key not in bucket:
            bucket[key] = self.new_record()

        rec = bucket[key]

        rec[0] += 1
        rec[1] += int(win)
        rec[2] += int(top3)
        rec[3] += float(finish)
        rec[4] += float(speed)

    def global_priors(self):
        n = self.global_rec[0]

        if n <= 0:
            return {
                "win": 0.10,
                "top3": 0.30,
                "finish": 0.45,
                "speed": 1.0
            }

        return {
            "win": self.global_rec[1] / n,
            "top3": self.global_rec[2] / n,
            "finish": self.global_rec[3] / n,
            "speed": self.global_rec[4] / n
        }

    def metrics(self, bucket, key, alpha=8.0):
        """
        Az yarışlı atların 1-2 iyi sonuçla aşırı skor almasını önler.
        """
        priors = self.global_priors()
        rec = bucket.get(key)

        if rec is None or rec[0] == 0:
            return {
                "n": 0,
                "win": priors["win"],
                "top3": priors["top3"],
                "finish": priors["finish"],
                "speed": priors["speed"]
            }

        n, wins, top3_count, finish_sum, speed_sum = rec

        return {
            "n": n,
            "win": (
                wins + alpha * priors["win"]
            ) / (n + alpha),

            "top3": (
                top3_count + alpha * priors["top3"]
            ) / (n + alpha),

            "finish": (
                finish_sum + alpha * priors["finish"]
            ) / (n + alpha),

            "speed": (
                speed_sum + alpha * priors["speed"]
            ) / (n + alpha)
        }

    def update(self, row):
        horse = row.HorseKey
        surface = row.PistTipi
        dist_group = distance_group(row.Mesafe)

        jockey = row.JokeyKey
        trainer = row.AntrenorKey

        win = int(row.Kazandi)
        top3 = int(row.Top3)
        finish = float(row.FinishPct)
        speed = float(row.SpeedRatio)

        self.global_rec[0] += 1
        self.global_rec[1] += win
        self.global_rec[2] += top3
        self.global_rec[3] += finish
        self.global_rec[4] += speed

        self.add(self.horse, horse, win, top3, finish, speed)

        self.add(
            self.horse_surface,
            (horse, surface),
            win,
            top3,
            finish,
            speed
        )

        self.add(
            self.horse_distance,
            (horse, dist_group),
            win,
            top3,
            finish,
            speed
        )

        self.add(
            self.horse_surface_distance,
            (horse, surface, dist_group),
            win,
            top3,
            finish,
            speed
        )

        self.add(self.jockey, jockey, win, top3, finish, speed)
        self.add(self.trainer, trainer, win, top3, finish, speed)

        self.add(
            self.horse_jockey,
            (horse, jockey),
            win,
            top3,
            finish,
            speed
        )

        self.add(
            self.jockey_trainer,
            (jockey, trainer),
            win,
            top3,
            finish,
            speed
        )

        if horse not in self.recent_form:
            self.recent_form[horse] = deque(maxlen=6)

        self.recent_form[horse].append(finish)
        self.last_date[horse] = row.Tarih

    def to_payload(self):
        return {
            "global_rec": self.global_rec,
            "horse": self.horse,
            "horse_surface": self.horse_surface,
            "horse_distance": self.horse_distance,
            "horse_surface_distance": self.horse_surface_distance,
            "jockey": self.jockey,
            "trainer": self.trainer,
            "horse_jockey": self.horse_jockey,
            "jockey_trainer": self.jockey_trainer,
            "recent_form": {
                key: list(values)
                for key, values in self.recent_form.items()
            },
            "last_date": self.last_date
        }

    @classmethod
    def from_payload(cls, payload):
        state = cls()

        if not isinstance(payload, dict):
            return state

        state.global_rec = payload.get(
            "global_rec",
            [0, 0, 0, 0.0, 0.0]
        )

        state.horse = payload.get("horse", {})
        state.horse_surface = payload.get("horse_surface", {})
        state.horse_distance = payload.get("horse_distance", {})
        state.horse_surface_distance = payload.get(
            "horse_surface_distance",
            {}
        )

        state.jockey = payload.get("jockey", {})
        state.trainer = payload.get("trainer", {})
        state.horse_jockey = payload.get("horse_jockey", {})
        state.jockey_trainer = payload.get("jockey_trainer", {})

        saved_recent = payload.get("recent_form", {})

        state.recent_form = {
            key: deque(values, maxlen=6)
            for key, values in saved_recent.items()
        }

        state.last_date = payload.get("last_date", {})

        return state


# ============================================================================
# FEATURE LİSTESİ
# ============================================================================
#
# v4.4: "IsEkuru" ve "EkuriOrtakSayisiNorm" eklendi. CSV'lerdeki
# "[(1)AT_ADI,(7)AT_ADI2] eküridir." satırından çıkarılıyor - aynı
# sahip/antrenörden birden fazla at aynı koşuda olduğunda (eküri),
# ilgili atlar işaretleniyor. Bu bilgi geçmişte hiç kullanılmıyordu,
# atlanıyordu.
# ============================================================================

FEATURE_COLS = [
    "AGFNorm",
    "AGFOverExpected",
    "AgeNorm",
    "WeightNorm",
    "HandikapNorm",
    "DrawFrac",
    "FieldSizeNorm",

    "IsKum",
    "IsCim",
    "IsSentetik",
    "IsMaiden",
    "IsHandicap",
    "IsGroup",
    "IsArab",
    "IsAge2",
    "IsAge3",
    "IsEkuru",               # v4.4: YENİ
    "EkuriOrtakSayisiNorm",  # v4.4: YENİ

    "HorseStarts",
    "HorseWin",
    "HorseTop3",
    "HorseFinish",
    "HorseSpeed",

    "SurfaceStarts",
    "SurfaceWin",
    "SurfaceTop3",
    "SurfaceFinish",

    "DistanceStarts",
    "DistanceWin",
    "DistanceTop3",
    "DistanceFinish",

    "SurfaceDistanceStarts",
    "SurfaceDistanceWin",
    "SurfaceDistanceTop3",

    "Recent1",
    "Recent3",
    "Recent6",
    "DaysSinceLast",

    "JockeyStarts",
    "JockeyWin",
    "JockeyTop3",

    "TrainerStarts",
    "TrainerWin",
    "TrainerTop3",

    "HorseJockeyStarts",
    "HorseJockeyTop3",

    "JockeyTrainerStarts",
    "JockeyTrainerTop3",

    "IdmanTazelikNorm",
    "IdmanDereceNorm",
    "IdmanPistUyumu",
    "IdmanVarMi",

    "SicaklikNorm",
    "NemNorm",
    "BasincNorm",
    "RuzgarHizNorm",
    "HavaAcikMi",
    "HavaVarMi",

    "AntrenorKariyerBirinciPct",
    "AntrenorKariyerTop3Pct",
    "AntrenorKariyerVarMi",
    "JokeyKariyerBirinciPct",
    "JokeyKariyerTop3Pct",
    "JokeyKariyerVarMi"
]

# AGF'siz "temel görüş" modeli için feature listesi (piyasa bilgisi yok).
AGF_FEATURE_ADLARI = ("AGFNorm", "AGFOverExpected")
FEATURE_COLS_NOAGF = [c for c in FEATURE_COLS if c not in AGF_FEATURE_ADLARI]


def count_to_feature(count):
    return float(min(np.log1p(count) / 6.0, 1.0))


def recent_average(values, count, default):
    if not values:
        return default

    values = list(values)[-count:]

    if not values:
        return default

    return float(np.mean(values))


def make_feature_row(row, state, field_size, agf_total):
    horse = row.HorseKey
    jockey = row.JokeyKey
    trainer = row.AntrenorKey

    surface = row.PistTipi
    dist_group = distance_group(row.Mesafe)

    h = state.metrics(state.horse, horse)

    hs = state.metrics(
        state.horse_surface,
        (horse, surface)
    )

    hd = state.metrics(
        state.horse_distance,
        (horse, dist_group)
    )

    hsd = state.metrics(
        state.horse_surface_distance,
        (horse, surface, dist_group)
    )

    j = state.metrics(state.jockey, jockey)
    t = state.metrics(state.trainer, trainer)

    hj = state.metrics(
        state.horse_jockey,
        (horse, jockey)
    )

    jt = state.metrics(
        state.jockey_trainer,
        (jockey, trainer)
    )

    recent = state.recent_form.get(horse, deque(maxlen=6))

    recent1 = recent_average(recent, 1, h["finish"])
    recent3 = recent_average(recent, 3, h["finish"])
    recent6 = recent_average(recent, 6, h["finish"])

    last_date = state.last_date.get(horse)

    if last_date is None or pd.isna(last_date):
        days_norm = 0.50
    else:
        days = max(0, (row.Tarih - last_date).days)
        days_norm = min(days, 180) / 180.0

    if field_size > 1 and row.St > 0:
        draw_frac = (row.St - 1) / (field_size - 1)
    else:
        draw_frac = 0.50

    # --- AGFOverExpected -----------------------------------------------
    # field_size atlı bir koşuda "adil pay" 100/field_size'dır (herkes
    # eşit ilgi görseydi olması gereken AGF). AGFOverExpected, bu atın
    # gerçek AGF'sinin bu adil paya oranını ölçer - 1.0 = piyasa tam
    # nötr bakıyor, >1.0 = piyasa favorisi, <1.0 = piyasa gözden
    # çıkarmış. AGFNorm (mutlak AGF seviyesi) ile matematiksel olarak
    # kolineer DEĞİLDİR çünkü field_size'a göre normalize eder.
    if field_size > 0:
        expected_share = 100.0 / field_size
        agf_over_expected = float(row.AGF) / max(expected_share, 1e-6)
    else:
        agf_over_expected = 1.0

    # Aşırı uçlarda (çok kalabalık alanda tek at'a AGF yığılması gibi)
    # modelin bozulmaması için makul bir aralığa sıkıştırıyoruz, sonra
    # 0-1'e normalize ediyoruz (0=hiç ilgi yok, 1=alanın çok üzerinde
    # ilgi görüyor).
    agf_over_expected = min(max(agf_over_expected, 0.0), 6.0) / 6.0

    return {
        "AGFNorm": min(max(float(row.AGF) / 100.0, 0.0), 1.0),
        "AGFOverExpected": agf_over_expected,
        "AgeNorm": min(float(row.Yas) / 10.0, 1.0),
        "WeightNorm": min(max(float(row.Kilo) / 65.0, 0.0), 1.0),
        "HandikapNorm": min(max(float(row.H) / 120.0, 0.0), 1.0),
        "DrawFrac": min(max(draw_frac, 0.0), 1.0),
        "FieldSizeNorm": min(field_size / 20.0, 1.0),

        "IsKum": int(surface == "kum"),
        "IsCim": int(surface == "cim"),
        "IsSentetik": int(surface == "sentetik"),
        "IsMaiden": int(row.IsMaiden),
        "IsHandicap": int(row.IsHandicap),
        "IsGroup": int(row.IsGroup),
        "IsArab": int(row.IsArab),
        "IsAge2": int(row.IsAge2),
        "IsAge3": int(row.IsAge3),

        # v4.4: eküri feature'ları.
        "IsEkuru": int(getattr(row, "IsEkuru", 0)),
        "EkuriOrtakSayisiNorm": min(
            float(getattr(row, "EkuriOrtakSayisi", 1)) / 3.0, 1.0
        ),

        "HorseStarts": count_to_feature(h["n"]),
        "HorseWin": h["win"],
        "HorseTop3": h["top3"],
        "HorseFinish": h["finish"],
        "HorseSpeed": h["speed"],

        "SurfaceStarts": count_to_feature(hs["n"]),
        "SurfaceWin": hs["win"],
        "SurfaceTop3": hs["top3"],
        "SurfaceFinish": hs["finish"],

        "DistanceStarts": count_to_feature(hd["n"]),
        "DistanceWin": hd["win"],
        "DistanceTop3": hd["top3"],
        "DistanceFinish": hd["finish"],

        "SurfaceDistanceStarts": count_to_feature(hsd["n"]),
        "SurfaceDistanceWin": hsd["win"],
        "SurfaceDistanceTop3": hsd["top3"],

        "Recent1": recent1,
        "Recent3": recent3,
        "Recent6": recent6,
        "DaysSinceLast": days_norm,

        "JockeyStarts": count_to_feature(j["n"]),
        "JockeyWin": j["win"],
        "JockeyTop3": j["top3"],

        "TrainerStarts": count_to_feature(t["n"]),
        "TrainerWin": t["win"],
        "TrainerTop3": t["top3"],

        "IdmanTazelikNorm": float(getattr(row, "IdmanTazelikNorm", 0.0)),
        "IdmanDereceNorm": float(getattr(row, "IdmanDereceNorm", 0.5)),
        "IdmanPistUyumu": int(getattr(row, "IdmanPistUyumu", 0)),
        "IdmanVarMi": int(getattr(row, "IdmanVarMi", 0)),

        "SicaklikNorm": float(getattr(row, "SicaklikNorm", 0.5)),
        "NemNorm": float(getattr(row, "NemNorm", 0.5)),
        "BasincNorm": float(getattr(row, "BasincNorm", 0.5)),
        "RuzgarHizNorm": float(getattr(row, "RuzgarHizNorm", 0.0)),
        "HavaAcikMi": int(getattr(row, "HavaAcikMi", 0)),
        "HavaVarMi": int(getattr(row, "HavaVarMi", 0)),

        "AntrenorKariyerBirinciPct": float(getattr(row, "AntrenorKariyerBirinciPct", 0.10)),
        "AntrenorKariyerTop3Pct": float(getattr(row, "AntrenorKariyerTop3Pct", 0.30)),
        "AntrenorKariyerVarMi": int(getattr(row, "AntrenorKariyerVarMi", 0)),
        "JokeyKariyerBirinciPct": float(getattr(row, "JokeyKariyerBirinciPct", 0.10)),
        "JokeyKariyerTop3Pct": float(getattr(row, "JokeyKariyerTop3Pct", 0.30)),
        "JokeyKariyerVarMi": int(getattr(row, "JokeyKariyerVarMi", 0)),

        "HorseJockeyStarts": count_to_feature(hj["n"]),
        "HorseJockeyTop3": hj["top3"],

        "JockeyTrainerStarts": count_to_feature(jt["n"]),
        "JockeyTrainerTop3": jt["top3"]
    }


# ============================================================================
# FEATURE CHECKPOINT SİSTEMİ
# ============================================================================

def reset_feature_storage():
    """Eski/uyumsuz feature cache ve chunk dosyalarını temizler."""
    if os.path.exists(FEATURE_PROGRESS_CACHE):
        try:
            os.remove(FEATURE_PROGRESS_CACHE)
        except Exception:
            pass

    if os.path.exists(FINAL_FEATURE_CACHE):
        try:
            os.remove(FINAL_FEATURE_CACHE)
        except Exception:
            pass

    if os.path.isdir(FEATURE_CHUNK_DIR):
        try:
            shutil.rmtree(FEATURE_CHUNK_DIR)
        except Exception:
            pass

    os.makedirs(FEATURE_CHUNK_DIR, exist_ok=True)


def load_final_feature_cache(raw_fingerprint):
    if not os.path.exists(FINAL_FEATURE_CACHE):
        return None

    try:
        payload = pd.read_pickle(FINAL_FEATURE_CACHE)

        if not isinstance(payload, dict):
            return None

        if (
            payload.get("feature_version") != FEATURE_VERSION
            or payload.get("raw_fingerprint") != raw_fingerprint
        ):
            return None

        features = payload.get("features")

        if isinstance(features, pd.DataFrame) and not features.empty:
            logger.info(
                f"♻️ Final feature cache yüklendi | "
                f"{len(features):,} satır"
            )

            return features

    except Exception:
        return None

    return None


def load_feature_progress(raw_fingerprint):
    if not os.path.exists(FEATURE_PROGRESS_CACHE):
        return None

    try:
        payload = pd.read_pickle(FEATURE_PROGRESS_CACHE)

        if not isinstance(payload, dict):
            return None

        if payload.get("cache_schema") != FEATURE_PROGRESS_SCHEMA:
            return None

        if payload.get("feature_version") != FEATURE_VERSION:
            return None

        if payload.get("raw_fingerprint") != raw_fingerprint:
            return None

        chunk_files = payload.get("chunk_files", [])

        for chunk_file in chunk_files:
            chunk_path = os.path.join(
                FEATURE_CHUNK_DIR,
                chunk_file
            )

            if not os.path.exists(chunk_path):
                return None

        logger.info(
            f"♻️ Feature checkpoint yüklendi | "
            f"Tamamlanan gün: {payload.get('next_date_index', 0)} | "
            f"Chunk: {len(chunk_files)}"
        )

        return payload

    except Exception:
        return None


def save_feature_progress(
    raw_fingerprint,
    next_date_index,
    state,
    chunk_files
):
    payload = {
        "cache_schema": FEATURE_PROGRESS_SCHEMA,
        "feature_version": FEATURE_VERSION,
        "raw_fingerprint": raw_fingerprint,
        "next_date_index": next_date_index,
        "state": state.to_payload(),
        "chunk_files": chunk_files,
        "saved_at": datetime.now().isoformat()
    }

    if safe_pickle_write(payload, FEATURE_PROGRESS_CACHE):
        logger.info(
            f"💾 FEATURE CHECKPOINT | "
            f"Tamamlanan gün: {next_date_index} | "
            f"Chunk: {len(chunk_files)}"
        )


def save_feature_chunk(feature_rows, chunk_number):
    if not feature_rows:
        return None

    os.makedirs(FEATURE_CHUNK_DIR, exist_ok=True)

    chunk_name = f"feature_chunk_{chunk_number:05d}.pkl"
    chunk_path = os.path.join(FEATURE_CHUNK_DIR, chunk_name)

    feature_df = pd.DataFrame(feature_rows)

    for col in FEATURE_COLS:
        if col not in feature_df.columns:
            feature_df[col] = 0.0

    feature_df[FEATURE_COLS] = (
        feature_df[FEATURE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    if safe_pickle_write(feature_df, chunk_path):
        logger.info(
            f"📦 Feature chunk kaydedildi: "
            f"{chunk_name} | Satır: {len(feature_df):,}"
        )

        return chunk_name

    return None


def combine_feature_chunks(chunk_files):
    frames = []

    for chunk_file in chunk_files:
        chunk_path = os.path.join(FEATURE_CHUNK_DIR, chunk_file)

        try:
            frame = pd.read_pickle(chunk_path)

            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frames.append(frame)

        except Exception as error:
            logger.warning(
                f"⚠️ Feature chunk okunamadı: {chunk_file} | {error}"
            )

    if not frames:
        return pd.DataFrame()

    features = pd.concat(frames, ignore_index=True)

    for col in FEATURE_COLS:
        if col not in features.columns:
            features[col] = 0.0

    features[FEATURE_COLS] = (
        features[FEATURE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    features = features.sort_values(
        ["Tarih", "RaceID", "Siralama"]
    ).reset_index(drop=True)

    return features


def build_or_load_training_features(
    history, raw_fingerprint, idman_history=None, hava_history=None,
    antrenor_kariyer=None, jokey_kariyer=None
):
    """
    groupby("Tarih") kullanır.
    Her 25 yarış gününde feature checkpoint kaydeder.
    idman_history verilirse, her satıra o tarihten ÖNCEKİ en yakın idman
    kaydı pd.merge_asof ile eklenir (sızıntı yok, döngü yok - vektörize).
    hava_history verilirse, Şehir+Tarih eşleşmesiyle o günün hava
    özellikleri eklenir. antrenor_kariyer/jokey_kariyer verilirse isim
    eşleşmesiyle kariyer istatistikleri eklenir.
    """
    final_features = load_final_feature_cache(raw_fingerprint)

    if final_features is not None:
        return final_features

    progress = load_feature_progress(raw_fingerprint)

    if progress is None:
        logger.info(
            "🧠 Yeni feature üretimi başlıyor "
            "(groupby + checkpoint)."
        )

        reset_feature_storage()

        next_date_index = 0
        state = HistoryState()
        chunk_files = []

    else:
        next_date_index = int(
            progress.get("next_date_index", 0)
        )

        state = HistoryState.from_payload(
            progress.get("state", {})
        )

        chunk_files = progress.get("chunk_files", [])

    history = history.sort_values(
        ["Tarih", "RaceID", "Siralama"]
    ).reset_index(drop=True)

    if idman_history is not None and not idman_history.empty:
        history = idman_ozellikleri_ekle(
            history.copy(), idman_history, tarih_sutunu="Tarih"
        )
    else:
        history = history.copy()
        history["IdmanTazelikNorm"] = 0.0
        history["IdmanDereceNorm"] = 0.5
        history["IdmanPistUyumu"] = 0
        history["IdmanVarMi"] = 0

    history = hava_ozellikleri_ekle(history, hava_history, tarih_sutunu="Tarih")
    history = kariyer_ozellikleri_ekle(history, antrenor_kariyer, jokey_kariyer)

    # Doğrulama: eküri bilgisi eğitim verisine gerçekten girmiş mi?
    if "IsEkuru" in history.columns:
        ekuri_sayisi = int(pd.to_numeric(history["IsEkuru"], errors="coerce").fillna(0).sum())

        if ekuri_sayisi > 0:
            logger.info(
                f"🔗 Eğitim verisinde eküri işaretli at: "
                f"{ekuri_sayisi:,} / {len(history):,}"
            )
        else:
            logger.warning(
                "⚠️ Eğitim verisinde HİÇ eküri işaretli at yok - eküri "
                "feature'ı bu eğitimde etkisiz kalır (ham geçmiş eski "
                "formatta olabilir)."
            )
    else:
        logger.warning(
            "⚠️ Eğitim verisinde IsEkuru sütunu yok - eküri feature'ı etkisiz."
        )

    date_group_count = history["Tarih"].nunique()

    logger.info(
        f"📊 Feature üretimi | "
        f"Toplam gün: {date_group_count:,} | "
        f"Başlangıç günü: {next_date_index:,}"
    )

    # ÖNEMLİ PERFORMANS NOTU:
    # Bu döngü yüz binlerce küçük Python nesnesi oluşturuyor. Python'ın
    # çöp toplayıcısı (GC) nesne sayısı arttıkça giderek yavaşlar. Bu kod
    # döngüsel referans üretmediği için GC'yi geçici olarak kapatmak
    # güvenlidir ve genelde 2-5 kat hızlanma sağlar.
    gc_acikti_mi = gc.isenabled()
    gc.disable()

    current_rows = []
    completed_day_index = next_date_index

    try:
        date_iterator = history.groupby("Tarih", sort=True)

        with tqdm(
            total=date_group_count,
            initial=next_date_index,
            desc="📊 Feature üretiliyor",
            unit="gün"
        ) as progress_bar:

            for date_index, (_, day_df) in enumerate(date_iterator):
                # Daha önce checkpoint'te tamamlanan günleri atla.
                if date_index < next_date_index:
                    continue

                # Bu günün koşuları; önceki günlerin state'i kullanılır.
                for race_id, race_df in day_df.groupby(
                    "RaceID",
                    sort=False
                ):
                    field_size = len(race_df)
                    agf_total = float(race_df["AGF"].sum())

                    race_rows = list(
                        race_df.itertuples(index=False)
                    )

                    # Feature üret.
                    for row in race_rows:
                        feature = make_feature_row(
                            row=row,
                            state=state,
                            field_size=field_size,
                            agf_total=agf_total
                        )

                        feature.update({
                            "RaceID": race_id,
                            "Tarih": row.Tarih,
                            "Kazandi": int(row.Kazandi),
                            "Siralama": int(row.Siralama),
                            "HorseKey": row.HorseKey
                        })

                        current_rows.append(feature)

                # Günün sonuçlarını feature üretiminden sonra state'e ekle.
                for row in day_df.itertuples(index=False):
                    state.update(row)

                completed_day_index = date_index + 1
                progress_bar.update(1)

                # Her 25 günde bir feature chunk + checkpoint.
                if (
                    completed_day_index % FEATURE_CHECKPOINT_DAYS == 0
                    or completed_day_index == date_group_count
                ):
                    if current_rows:
                        chunk_name = save_feature_chunk(
                            current_rows,
                            len(chunk_files) + 1
                        )

                        if chunk_name:
                            chunk_files.append(chunk_name)

                        current_rows = []

                    save_feature_progress(
                        raw_fingerprint=raw_fingerprint,
                        next_date_index=completed_day_index,
                        state=state,
                        chunk_files=chunk_files
                    )

    except KeyboardInterrupt:
        gc.enable() if gc_acikti_mi else None

        if current_rows:
            chunk_name = save_feature_chunk(
                current_rows,
                len(chunk_files) + 1
            )

            if chunk_name:
                chunk_files.append(chunk_name)

        save_feature_progress(
            raw_fingerprint=raw_fingerprint,
            next_date_index=completed_day_index,
            state=state,
            chunk_files=chunk_files
        )

        logger.warning(
            "⚠️ Feature üretimi durduruldu. "
            "Sonraki açılışta checkpoint'ten devam eder."
        )

        raise

    finally:
        if gc_acikti_mi:
            gc.enable()

    # Tüm feature chunk dosyalarını tek tabloda birleştir.
    logger.info("📦 Feature chunk dosyaları birleştiriliyor...")

    features = combine_feature_chunks(chunk_files)

    if features.empty:
        raise RuntimeError(
            "Feature üretimi tamamlandı fakat feature verisi boş."
        )

    payload = {
        "feature_version": FEATURE_VERSION,
        "raw_fingerprint": raw_fingerprint,
        "features": features,
        "saved_at": datetime.now().isoformat()
    }

    safe_pickle_write(payload, FINAL_FEATURE_CACHE)

    logger.info(
        f"✅ Feature üretimi tamamlandı | "
        f"Satır: {len(features):,} | "
        f"Koşu: {features['RaceID'].nunique():,}"
    )

    # Final feature cache oluştu; geçici progress dosyaları temizlenebilir.
    try:
        if os.path.exists(FEATURE_PROGRESS_CACHE):
            os.remove(FEATURE_PROGRESS_CACHE)

        if os.path.isdir(FEATURE_CHUNK_DIR):
            shutil.rmtree(FEATURE_CHUNK_DIR)

    except Exception:
        pass

    return features


# ============================================================================
# XGBOOST RANKER
# ============================================================================

def build_rank_arrays(df, cols=None):
    cols = cols or FEATURE_COLS

    ordered = df.sort_values(
        ["Tarih", "RaceID", "Siralama"]
    ).reset_index(drop=True)

    groups = (
        ordered.groupby("RaceID", sort=False)
        .size()
        .to_numpy()
    )

    x_values = (
        ordered[cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
        .to_numpy()
    )

    y_values = ordered["Kazandi"].astype(np.float32).to_numpy()

    return ordered, x_values, y_values, groups


def create_ranker():
    return XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg@3",
        n_estimators=350,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )


def top_k_accuracy(df, scores, k=1):
    result = df[["RaceID", "Kazandi"]].copy()
    result["Score"] = scores

    total = 0
    success = 0

    for _, group in result.groupby("RaceID"):
        if group["Kazandi"].sum() == 0:
            continue

        total += 1

        top = group.sort_values(
            "Score",
            ascending=False
        ).head(k)

        if top["Kazandi"].sum() > 0:
            success += 1

    return success / total if total else 0.0


def log_feature_importance(model, top_n=20):
    """
    Her eğitimden sonra hangi feature'ların modeli domine ettiğini
    logluyoruz. Amaç: AGFNorm/AGFOverExpected gibi tekil bir feature'ın
    yeniden tüm skoru domine edip etmediğini erken fark edebilmek.
    """
    try:
        importances = pd.Series(
            model.feature_importances_,
            index=FEATURE_COLS
        ).sort_values(ascending=False)

        logger.info(f"📈 Feature importance (ilk {top_n}):")

        for name, value in importances.head(top_n).items():
            logger.info(f"     {name:<28} {value:.4f}")

        sira_listesi = list(importances.index)

        for ozel in ("IsEkuru", "EkuriOrtakSayisiNorm"):
            if ozel in importances.index:
                logger.info(
                    f"🔗 {ozel}: {importances[ozel]:.4f} "
                    f"(sıra {sira_listesi.index(ozel) + 1}/{len(sira_listesi)})"
                )

        top_share = importances.head(1).iloc[0] / max(importances.sum(), 1e-9)

        if top_share > 0.35:
            logger.warning(
                f"⚠️ Tek bir feature ('{importances.index[0]}') toplam "
                f"önemin %{top_share * 100:.0f}'ini taşıyor. Model tek bir "
                f"sinyale (ör. AGF) aşırı bağımlı olabilir - diğer "
                f"feature'ları / regularizasyonu gözden geçirin."
            )

    except Exception as error:
        logger.warning(f"⚠️ Feature importance loglanamadı: {error}")


def train_ranker(features, cutoff_date=None, cols=None, etiket=""):
    cols = cols or FEATURE_COLS
    temel_mi = bool(etiket)
    dogrulama_baslik = (
        f"📊 [{etiket}] doğrulama" if temel_mi else "📊 Kronolojik doğrulama"
    )
    train_data = features.copy()

    if cutoff_date is not None and not pd.isna(cutoff_date):
        train_data = train_data[
            train_data["Tarih"] < cutoff_date
        ].copy()

    valid_races = train_data.groupby("RaceID").size()
    valid_race_ids = valid_races[valid_races >= 2].index

    train_data = train_data[
        train_data["RaceID"].isin(valid_race_ids)
    ].copy()

    if train_data["RaceID"].nunique() < 50:
        raise RuntimeError(
            "Model eğitimi için en az 50 geçmiş koşu gerekli."
        )

    race_dates = (
        train_data.groupby("RaceID")["Tarih"]
        .min()
        .sort_values()
    )

    race_ids = race_dates.index.tolist()
    split_index = max(1, int(len(race_ids) * 0.80))

    train_ids = set(race_ids[:split_index])
    valid_ids = set(race_ids[split_index:])

    train_part = train_data[
        train_data["RaceID"].isin(train_ids)
    ].copy()

    valid_part = train_data[
        train_data["RaceID"].isin(valid_ids)
    ].copy()

    logger.info(
        f"🧠 Ranker eğitiliyor | "
        f"Eğitim koşusu: {train_part['RaceID'].nunique():,} | "
        f"Doğrulama koşusu: {valid_part['RaceID'].nunique():,}"
    )

    _, x_train, y_train, group_train = build_rank_arrays(train_part, cols)

    validation_model = create_ranker()

    validation_model.fit(
        x_train,
        y_train,
        group=group_train,
        verbose=False
    )

    if not valid_part.empty:
        valid_ordered, x_valid, _, _ = build_rank_arrays(valid_part, cols)
        valid_scores = validation_model.predict(x_valid)

        top1 = top_k_accuracy(valid_ordered, valid_scores, 1)
        top3 = top_k_accuracy(valid_ordered, valid_scores, 3)
        top5 = top_k_accuracy(valid_ordered, valid_scores, 5)

        logger.info(
            f"{dogrulama_baslik} | "
            f"Top-1: %{top1 * 100:.1f} | "
            f"Top-3: %{top3 * 100:.1f} | "
            f"Top-5: %{top5 * 100:.1f}"
        )

        # Karşılaştırma: sadece AGF'ye bakan "tahminci" (AGFNorm'a göre sırala)
        # aynı doğrulama koşularında ne yapıyor? Modelin piyasaya katkısı bu
        # farktır. Sadece log satırıdır, modeli etkilemez.
        agf_skor = valid_ordered["AGFNorm"].to_numpy()
        agf1 = top_k_accuracy(valid_ordered, agf_skor, 1)
        agf3 = top_k_accuracy(valid_ordered, agf_skor, 3)
        agf5 = top_k_accuracy(valid_ordered, agf_skor, 5)

        logger.info(
            f"📏 Sadece-AGF taban çizgisi{' - ' + etiket + ' modeli için' if temel_mi else ''} (aynı doğrulama koşuları) | "
            f"Top-1: %{agf1 * 100:.1f} | "
            f"Top-3: %{agf3 * 100:.1f} | "
            f"Top-5: %{agf5 * 100:.1f} | "
            f"Model farkı (Top-1/3/5): "
            f"{(top1 - agf1) * 100:+.1f} / "
            f"{(top3 - agf3) * 100:+.1f} / "
            f"{(top5 - agf5) * 100:+.1f} puan"
        )

    _, x_all, y_all, group_all = build_rank_arrays(train_data, cols)

    final_model = create_ranker()

    final_model.fit(
        x_all,
        y_all,
        group=group_all,
        verbose=False
    )

    if not temel_mi:
        log_feature_importance(final_model)

    return final_model


def model_paths(cutoff_date, etiket=""):
    if cutoff_date is None or pd.isna(cutoff_date):
        tag = "latest"
    else:
        tag = pd.Timestamp(cutoff_date).strftime("%Y%m%d")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(
        MODEL_DIR,
        f"ranker_{etiket + '_' if etiket else ''}{tag}.json"
    )

    meta_path = os.path.join(
        MODEL_DIR,
        f"ranker_{etiket + '_' if etiket else ''}{tag}.pkl"
    )

    return model_path, meta_path


def load_or_train_model(
    features,
    raw_fingerprint,
    cutoff_date,
    cols=None,
    etiket=""
):
    """
    Her tahmin tarihi için ayrı model cache'i tutulur.
    Aynı geçmiş program tekrar test edilirse yeniden eğitim yapmaz.
    """
    cols = cols or FEATURE_COLS
    model_path, meta_path = model_paths(cutoff_date, etiket)

    if os.path.exists(model_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "rb") as file:
                meta = pickle.load(file)

            if (
                meta.get("raw_fingerprint") == raw_fingerprint
                and meta.get("feature_version") == FEATURE_VERSION
                and meta.get("feature_columns") == cols
            ):
                model = create_ranker()
                model.load_model(model_path)

                logger.info(
                    f"♻️ Ranker modeli cache üzerinden yüklendi: "
                    f"{os.path.basename(model_path)}"
                )

                return model

        except Exception:
            pass

    model = train_ranker(
        features=features,
        cutoff_date=cutoff_date,
        cols=cols,
        etiket=etiket
    )

    model.save_model(model_path)

    with open(meta_path, "wb") as file:
        pickle.dump(
            {
                "raw_fingerprint": raw_fingerprint,
                "feature_version": FEATURE_VERSION,
                "feature_columns": cols,
                "cutoff_date": str(cutoff_date)
            },
            file
        )

    logger.info(
        f"💾 Ranker modeli kaydedildi: "
        f"{os.path.basename(model_path)}"
    )

    return model


# ============================================================================
# PROGRAM İÇİN GEÇMİŞ STATE
# ============================================================================

def build_state_before_date(history, cutoff_date):
    """
    groupby ile tarih sırasıyla ilerler.
    cutoff_date sonrası verileri görmez.
    """
    state = HistoryState()

    for current_date, day_df in history.groupby("Tarih", sort=True):
        if (
            cutoff_date is not None
            and not pd.isna(cutoff_date)
            and current_date >= cutoff_date
        ):
            break

        for row in day_df.itertuples(index=False):
            state.update(row)

    return state


# ============================================================================
# PROGRAM FEATURE / TAHMİN
# ============================================================================

def program_feature_table(program_df, state):
    feature_rows = []

    for _, race_df in program_df.groupby("Kosu", sort=True):
        field_size = len(race_df)
        agf_total = float(race_df["AGF"].sum())

        for row in race_df.itertuples(index=False):
            feature = make_feature_row(
                row=row,
                state=state,
                field_size=field_size,
                agf_total=agf_total
            )

            feature.update({
                "RaceID": (
                    f"PROGRAM|{row.Tarih}|"
                    f"{row.Sehir}|{row.Kosu}"
                ),
                "Kosu": row.Kosu,
                "AtIsmiRaw": row.AtIsmiRaw,
                "HorseKey": row.HorseKey,
                "JokeyAdi": row.JokeyAdi,
                "AntrenorAdi": row.AntrenorAdi,
                "AGF": row.AGF,
                "Kilo": row.Kilo,
                "St": row.St,
                "Son6Yaris": getattr(row, "Son6Yaris", ""),
                "IsMaiden": row.IsMaiden,
                "IsHandicap": row.IsHandicap,
                "IsGroup": row.IsGroup,
                "HorsePastStarts": state.metrics(
                    state.horse,
                    row.HorseKey
                )["n"]
            })

            feature_rows.append(feature)

    feature_df = pd.DataFrame(feature_rows)

    for col in FEATURE_COLS:
        if col not in feature_df.columns:
            feature_df[col] = 0.0

    feature_df[FEATURE_COLS] = (
        feature_df[FEATURE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return feature_df


def print_predictions(pred_df, program_date, city):
    print(f"\n{'=' * 88}")
    print(
        f"  🏇 {program_date.strftime('%d.%m.%Y')} - "
        f"{city} - RANKING TAHMİNLERİ v4.4"
    )
    print(f"{'=' * 88}")

    for race_no, group in pred_df.groupby("Kosu", sort=True):
        group = group.copy()

        surface = group["Surface"].iloc[0]
        distance = group["Distance"].iloc[0]

        ordered = group.sort_values(
            "ModelHamSkor",
            ascending=False
        ).copy()

        ordered["ModelSira"] = range(1, len(ordered) + 1)

        ordered["AGFSira"] = (
            ordered["AGF"]
            .rank(method="min", ascending=False)
            .astype(int)
        )

        top = ordered.head(TOP_N)

        unknown_count = int(
            (top["HorsePastStarts"] == 0).sum()
        )

        is_maiden = int(group["IsMaiden"].iloc[0])

        print(f"\n{'─' * 88}")
        print(
            f"  🏁 {race_no}. KOŞU │ "
            f"{distance:.0f}m │ {surface.upper()} │ "
            f"Alan: {len(ordered)} at"
        )
        print(f"{'─' * 88}")

        if is_maiden:
            print(
                "  ⚠️ Maiden koşu: geçmiş veri sınırlı olabilir."
            )

        if unknown_count >= 2:
            print(
                "  ⚠️ İlk adaylarda geçmişi az olan atlar var."
            )

        # v4.4: bu koşuda eküri varsa kısa bir bilgi satırı bas.
        if "IsEkuru" in ordered.columns and ordered["IsEkuru"].sum() > 0:
            ekuri_atlar = ordered[ordered["IsEkuru"] == 1]["AtIsmiRaw"].tolist()
            print("  🔗 Eküri: " + ", ".join(ekuri_atlar))

        print(
            f"\n  {'#':<3} {'At':<30} {'Skor':>7} "
            f"{'AGF':>7} {'AGF Sıra':>9} "
            f"{'Geçmiş':>8} {'Jokey':<18}"
        )

        for _, row in top.iterrows():
            ekuri_isareti = " 🔗" if row.get("IsEkuru", 0) == 1 else ""

            print(
                f"  {int(row['ModelSira']):<3} "
                f"{row['AtIsmiRaw'][:29]:<30} "
                f"{row['ModelSkor'] * 100:6.1f}% "
                f"%{row['AGF']:5.1f} "
                f"{int(row['AGFSira']):>8}. "
                f"{int(row['HorsePastStarts']):>6} "
                f"{row['JokeyAdi'][:18]:<18}"
                f"{ekuri_isareti}"
            )

            print(
                f"      Form:{row['Recent3']:.2f} | "
                f"Pist:{row['SurfaceTop3']:.2f} | "
                f"Mesafe:{row['DistanceTop3']:.2f} | "
                f"Jokey:{row['JockeyTop3']:.2f} | "
                f"Antrenör:{row['TrainerTop3']:.2f}"
            )

        if len(ordered) > TOP_N:
            print(f"\n  ... alanda {len(ordered) - TOP_N} at daha var.")

        first = ordered.iloc[0]
        second = ordered.iloc[1] if len(ordered) > 1 else None

        print("\n  🎯 MODEL SIRALAMASI:")

        print(
            f"     1. {first['AtIsmiRaw']} "
            f"(Skor:%{first['ModelSkor'] * 100:.1f} | "
            f"AGF:%{first['AGF']:.1f})"
        )

        if second is not None:
            print(
                f"     2. {second['AtIsmiRaw']} "
                f"(Skor:%{second['ModelSkor'] * 100:.1f} | "
                f"AGF:%{second['AGF']:.1f})"
            )

        disagreement = ordered[
            (ordered["ModelSira"] <= 3)
            & (ordered["AGFSira"] >= 4)
        ]

        if not disagreement.empty:
            print("\n  💡 Modelin piyasanın önüne koyduğu adaylar:")

            for _, row in disagreement.head(2).iterrows():
                print(
                    f"     - {row['AtIsmiRaw']} "
                    f"(Model sıra:{int(row['ModelSira'])}, "
                    f"AGF sıra:{int(row['AGFSira'])})"
                )
        else:
            print(
                "\n  ℹ️ Model sıralaması bu koşuda AGF sıralamasıyla "
                "büyük ölçüde örtüşüyor."
            )

        # DENEME: AGF'siz temel görüş
        if "TemelSira" in ordered.columns and ordered["TemelSira"].notna().any():
            temel_ilk3 = ordered.sort_values("TemelSira").head(3)
            print(
                "\n  🧠 Temel görüş (AGF'siz model, ilk 3): "
                + " | ".join(
                    f"{int(r['TemelSira'])}. {r['AtIsmiRaw'][:22]}"
                    for _, r in temel_ilk3.iterrows()
                )
            )

            deger = ordered[ordered["DegerAdayi"] == 1].sort_values("TemelSira")

            for _, r in deger.iterrows():
                print(
                    f"  💎 Değer adayı [deneme]: {r['AtIsmiRaw']} "
                    f"(Temel sıra:{int(r['TemelSira'])}, "
                    f"AGF sıra:{int(r['AGFSira'])}, AGF:%{r['AGF']:.1f})"
                )

    print(f"\n{'=' * 88}\n")


def predict_program(
    program_path,
    history,
    features,
    raw_fingerprint,
    idman_history=None,
    hava_history=None,
    antrenor_kariyer=None,
    jokey_kariyer=None
):
    program_df = parse_racing_file(
        program_path,
        is_result=False
    )

    if program_df.empty:
        logger.error(f"🚫 Program okunamadı: {program_path}")
        return

    program_df["Tarih"] = pd.to_datetime(
        program_df["Tarih"],
        errors="coerce"
    )

    if program_df["Tarih"].notna().any():
        program_date = program_df["Tarih"].dropna().iloc[0]
        cutoff_date = pd.Timestamp(program_date).normalize()
    else:
        program_date = pd.NaT
        cutoff_date = None

    if cutoff_date is not None:
        logger.info(
            f"🎯 Tahmin tarihi: "
            f"{cutoff_date.strftime('%d.%m.%Y')}"
        )

        logger.info(
            "🔒 Sadece tahmin tarihinden ÖNCEKİ sonuçlar kullanılacak."
        )

    else:
        logger.warning(
            "⚠️ Program tarihi bulunamadı. "
            "Tüm geçmiş kullanılacak."
        )

    # Program günü için de idman verisini aynı mantıkla ekle (cutoff'tan
    # önceki en yakın idman - sızıntı yok).
    if idman_history is not None and not idman_history.empty:
        program_df = idman_ozellikleri_ekle(
            program_df, idman_history, tarih_sutunu="Tarih"
        )
    else:
        program_df["IdmanTazelikNorm"] = 0.0
        program_df["IdmanDereceNorm"] = 0.5
        program_df["IdmanPistUyumu"] = 0
        program_df["IdmanVarMi"] = 0

    program_df = hava_ozellikleri_ekle(
        program_df, hava_history, tarih_sutunu="Tarih"
    )
    program_df = kariyer_ozellikleri_ekle(
        program_df, antrenor_kariyer, jokey_kariyer
    )

    model = load_or_train_model(
        features=features,
        raw_fingerprint=raw_fingerprint,
        cutoff_date=cutoff_date
    )

    state = build_state_before_date(
        history=history,
        cutoff_date=cutoff_date
    )

    feature_df = program_feature_table(
        program_df=program_df,
        state=state
    )

    x_predict = (
        feature_df[FEATURE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
        .to_numpy()
    )

    feature_df["ModelHamSkor"] = model.predict(x_predict)
    feature_df["ModelSkor"] = 0.5

    for _, index_list in feature_df.groupby("RaceID").groups.items():
        values = feature_df.loc[index_list, "ModelHamSkor"]

        min_value = values.min()
        max_value = values.max()

        if max_value > min_value:
            normalized = (
                values - min_value
            ) / (
                max_value - min_value
            )
        else:
            normalized = pd.Series(
                np.full(len(values), 0.5),
                index=values.index
            )

        feature_df.loc[index_list, "ModelSkor"] = normalized

    # --- DENEME: AGF'siz "temel görüş" modeli + değer adayları -------------
    # Ana sıralamayı DEĞİŞTİRMEZ. Hata verirse ana çıktı yine basılır.
    try:
        model_temel = load_or_train_model(
            features=features,
            raw_fingerprint=raw_fingerprint,
            cutoff_date=cutoff_date,
            cols=FEATURE_COLS_NOAGF,
            etiket="temel"
        )

        x_temel = (
            feature_df[FEATURE_COLS_NOAGF]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
            .to_numpy()
        )

        feature_df["TemelHamSkor"] = model_temel.predict(x_temel)

    except Exception as hata:
        logger.warning(f"⚠️ Temel görüş modeli çalışmadı: {hata}")
        feature_df["TemelHamSkor"] = np.nan

    feature_df["TemelSira"] = (
        feature_df.groupby("RaceID")["TemelHamSkor"]
        .rank(method="first", ascending=False)
    )
    feature_df["AGFSiraTemel"] = (
        feature_df.groupby("RaceID")["AGF"]
        .rank(method="min", ascending=False)
    )
    feature_df["DegerAdayi"] = (
        (feature_df["TemelSira"] <= DEGER_TOP)
        & (feature_df["AGFSiraTemel"] >= DEGER_AGF_SIRA)
    ).astype(int)

    feature_df["Surface"] = program_df["PistTipi"].values
    feature_df["Distance"] = program_df["Mesafe"].values
    feature_df["Sehir"] = program_df["Sehir"].values

    if not pd.isna(program_date):
        kaydet_tahmin_takibi(feature_df, program_date)

    city = (
        program_df["Sehir"].iloc[0]
        if "Sehir" in program_df.columns
        else "Bilinmeyen Yer"
    )

    if pd.isna(program_date):
        program_date = pd.Timestamp.today()

    print_predictions(
        pred_df=feature_df,
        program_date=program_date,
        city=city
    )


# ============================================================================
# TAHMİN TAKİP / BAŞARI RAPORU SİSTEMİ
# ============================================================================

TAHMIN_TAKIP_DOSYA = "v4_tahmin_takip.csv"

TAKIP_SUTUNLARI = [
    "TahminZamani", "Tarih", "Sehir", "Kosu",
    "ModelFavori", "ModelTop3", "AGFFavori",
    "Durum", "GercekKazanan", "Top1Isabet", "Top3Isabet", "AGFTop1Isabet",
    # Ölçüm iyileştirmesi: adil kıyas (AGF Top-3) ve segment kırılımı.
    # Bunlar SADECE raporlama içindir, tahmin hesabını etkilemez.
    "AGFTop3", "AGFTop3Isabet",
    "Handikap", "Maiden", "EkuriVar", "AlanSayisi",
    "DegerAdaylari"
]

SEGMENT_SUTUNLARI = ["Handikap", "Maiden", "EkuriVar", "AlanSayisi"]


def _takip_anahtari(tarih, sehir, kosu):
    return f"{pd.Timestamp(tarih).strftime('%Y-%m-%d')}|{sehir}|{int(kosu)}"


def _bos_mu(deger):
    """CSV'den okunan boş hücre NaN ya da '' gelebilir; ikisini de yakalar."""
    return pd.isna(deger) or str(deger).strip() == ""


def takip_segment_doldur(takip, history):
    """
    Eski takip satırlarında Handikap/Maiden/EkuriVar/AlanSayisi boşsa,
    sonuç geçmişinden (history) geriye dönük doldurur. Koşunun niteliği
    (handikap mı, maiden mi, eküri var mı) sonuçtan bağımsız olduğu için
    bu güvenli. Sadece 'cozuldu' satırlar doldurulur (sonuç history'de var).
    Takip DataFrame'i yerinde değişir; değiştiyse True döner.
    """
    for sutun in SEGMENT_SUTUNLARI:
        if sutun not in takip.columns:
            takip[sutun] = ""

    if takip.empty or history is None or history.empty:
        return False

    eksik = pd.Series(False, index=takip.index)

    for sutun in SEGMENT_SUTUNLARI:
        eksik |= takip[sutun].map(_bos_mu)

    eksik &= (takip["Durum"] == "cozuldu")

    if not eksik.any():
        return False

    agg = {"Tarih": "first", "Sehir": "first", "Kosu": "first"}

    for sutun in ("IsHandicap", "IsMaiden", "FieldSize", "IsEkuru"):
        if sutun in history.columns:
            agg[sutun] = "max"

    ozet = history.groupby("RaceID").agg(agg)
    ozet["_k"] = [
        _takip_anahtari(t, s, k)
        for t, s, k in zip(ozet["Tarih"], ozet["Sehir"], ozet["Kosu"])
    ]
    ozet = ozet.drop_duplicates("_k", keep="last").set_index("_k")

    degisti = False

    for index in takip.index[eksik]:
        anahtar = _takip_anahtari(
            takip.at[index, "Tarih"],
            takip.at[index, "Sehir"],
            takip.at[index, "Kosu"]
        )

        if anahtar not in ozet.index:
            continue

        r = ozet.loc[anahtar]

        degerler = {}

        if "IsHandicap" in ozet.columns:
            degerler["Handikap"] = int(r["IsHandicap"] > 0)
        if "IsMaiden" in ozet.columns:
            degerler["Maiden"] = int(r["IsMaiden"] > 0)
        if "IsEkuru" in ozet.columns:
            degerler["EkuriVar"] = int(r["IsEkuru"] > 0)
        if "FieldSize" in ozet.columns:
            degerler["AlanSayisi"] = int(r["FieldSize"])

        for sutun, deger in degerler.items():
            if _bos_mu(takip.at[index, sutun]):
                takip.at[index, sutun] = str(deger)
                degisti = True

    return degisti


def takip_dosyasi_yukle():
    if not os.path.exists(TAHMIN_TAKIP_DOSYA):
        return pd.DataFrame(columns=TAKIP_SUTUNLARI)

    try:
        df = pd.read_csv(TAHMIN_TAKIP_DOSYA, dtype=str)

        for col in TAKIP_SUTUNLARI:
            if col not in df.columns:
                df[col] = ""

        return df

    except Exception as error:
        logger.warning(f"⚠️ Takip dosyası okunamadı: {error}")
        return pd.DataFrame(columns=TAKIP_SUTUNLARI)


def takip_dosyasi_kaydet(df):
    temp_path = TAHMIN_TAKIP_DOSYA + ".tmp"

    try:
        df.to_csv(temp_path, index=False, encoding="utf-8-sig")
        os.replace(temp_path, TAHMIN_TAKIP_DOSYA)

    except Exception as error:
        logger.error(f"🚫 Takip dosyası kaydedilemedi: {error}")

        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def kaydet_tahmin_takibi(feature_df, program_date):
    """
    Her koşu için model favorisini + ilk 3'ünü ve AGF favorisini
    'beklemede' durumunda takip dosyasına yazar. Aynı yarış (Tarih,
    Şehir, Koşu) daha önce kaydedilmişse üzerine yazar (tekrar
    çalıştırma durumunda çift kayıt olmasın diye).
    """
    if "Sehir" not in feature_df.columns:
        return

    takip = takip_dosyasi_yukle()
    yeni_satirlar = []

    for kosu_no, group in feature_df.groupby("Kosu", sort=True):
        ordered = group.sort_values("ModelHamSkor", ascending=False)

        top3 = ordered.head(3)["HorseKey"].tolist()
        favori = top3[0] if top3 else ""

        agf_favori_satir = group.sort_values("AGF", ascending=False).head(1)
        agf_favori = (
            agf_favori_satir["HorseKey"].iloc[0]
            if not agf_favori_satir.empty
            else ""
        )

        agf_top3 = (
            group.sort_values("AGF", ascending=False)
            .head(3)["HorseKey"].tolist()
        )

        deger_adaylari = (
            group[group["DegerAdayi"] == 1]["HorseKey"].tolist()
            if "DegerAdayi" in group.columns else []
        )

        # Segment bilgisi (sadece raporlama için, tahmini etkilemez).
        handikap = (
            str(int(group["IsHandicap"].max() > 0))
            if "IsHandicap" in group.columns else ""
        )
        maiden = (
            str(int(group["IsMaiden"].max() > 0))
            if "IsMaiden" in group.columns else ""
        )
        ekuri_var = (
            str(int(group["IsEkuru"].max() > 0))
            if "IsEkuru" in group.columns else ""
        )

        sehir = group["Sehir"].iloc[0]
        anahtar = _takip_anahtari(program_date, sehir, kosu_no)

        # Bu yarış zaten ÇÖZÜLMÜŞSE (gerçek sonucu elde edilmişse) üzerine
        # tekrar 'beklemede' yazıp çözümü kaybetmeyelim - atla.
        if not takip.empty:
            eslesen = takip[
                takip.apply(
                    lambda satir: (
                        _takip_anahtari(
                            satir["Tarih"], satir["Sehir"], satir["Kosu"]
                        ) == anahtar
                    ) if satir["Tarih"] else False,
                    axis=1
                )
            ]

            if not eslesen.empty and (eslesen["Durum"] == "cozuldu").any():
                continue

        # Aynı anahtarla eski (henüz çözülmemiş) kayıt varsa üzerine yaz.
        if not takip.empty:
            mevcut_anahtarlar = takip.apply(
                lambda satir: _takip_anahtari(
                    satir["Tarih"], satir["Sehir"], satir["Kosu"]
                ) if satir["Tarih"] else "",
                axis=1
            )
            takip = takip[mevcut_anahtarlar != anahtar]

        yeni_satirlar.append({
            "TahminZamani": datetime.now().isoformat(timespec="seconds"),
            "Tarih": pd.Timestamp(program_date).strftime("%Y-%m-%d"),
            "Sehir": sehir,
            "Kosu": int(kosu_no),
            "ModelFavori": favori,
            "ModelTop3": ";".join(top3),
            "AGFFavori": agf_favori,
            "Durum": "beklemede",
            "GercekKazanan": "",
            "Top1Isabet": "",
            "Top3Isabet": "",
            "AGFTop1Isabet": "",
            "AGFTop3": ";".join(agf_top3),
            "AGFTop3Isabet": "",
            "Handikap": handikap,
            "Maiden": maiden,
            "EkuriVar": ekuri_var,
            "AlanSayisi": str(len(group)),
            "DegerAdaylari": ";".join(deger_adaylari)
        })

    if yeni_satirlar:
        takip = pd.concat(
            [takip, pd.DataFrame(yeni_satirlar)],
            ignore_index=True
        )
        takip_dosyasi_kaydet(takip)

        logger.info(
            f"📝 {len(yeni_satirlar)} koşu tahmini takip dosyasına eklendi "
            f"(sonuç geldiğinde otomatik değerlendirilecek)."
        )


def coz_bekleyen_tahminler(history):
    """
    'beklemede' durumundaki tahminleri günceldeki history ile karşılaştırır.
    Artık gerçek sonucu elde olanları çözer (Top1/Top3 isabet mi diye
    işaretler) ve dosyayı günceller.
    """
    takip = takip_dosyasi_yukle()

    # Eski satırlarda boş kalan segment bilgisini sonuç geçmişinden doldur.
    segment_degisti = takip_segment_doldur(takip, history)

    bekleyenler = takip[takip["Durum"] == "beklemede"]

    if bekleyenler.empty:
        if segment_degisti:
            takip_dosyasi_kaydet(takip)

        return takip

    kazananlar = (
        history[history["Siralama"] == 1]
        .set_index(
            history[history["Siralama"] == 1].apply(
                lambda satir: _takip_anahtari(
                    satir["Tarih"], satir["Sehir"], satir["Kosu"]
                ),
                axis=1
            )
        )["HorseKey"]
    )

    cozulen_sayisi = 0

    for index in bekleyenler.index:
        satir = takip.loc[index]
        anahtar = _takip_anahtari(satir["Tarih"], satir["Sehir"], satir["Kosu"])

        if anahtar not in kazananlar.index:
            continue

        gercek_kazanan = kazananlar.loc[anahtar]

        if isinstance(gercek_kazanan, pd.Series):
            gercek_kazanan = gercek_kazanan.iloc[0]

        model_top3 = str(satir["ModelTop3"]).split(";") if satir["ModelTop3"] else []

        takip.loc[index, "GercekKazanan"] = gercek_kazanan
        takip.loc[index, "Durum"] = "cozuldu"
        takip.loc[index, "Top1Isabet"] = str(
            int(satir["ModelFavori"] == gercek_kazanan)
        )
        takip.loc[index, "Top3Isabet"] = str(
            int(gercek_kazanan in model_top3)
        )
        takip.loc[index, "AGFTop1Isabet"] = str(
            int(satir["AGFFavori"] == gercek_kazanan)
        )

        # AGF Top-3: sadece tahmin anında AGF ilk 3'ü kaydedilmiş
        # satırlar için hesaplanır (eski satırlarda boş kalır).
        if not _bos_mu(satir.get("AGFTop3", "")):
            agf_top3 = str(satir["AGFTop3"]).split(";")
            takip.loc[index, "AGFTop3Isabet"] = str(
                int(gercek_kazanan in agf_top3)
            )

        cozulen_sayisi += 1

    # Yeni çözülen satırların segment bilgisi boşsa onu da doldur.
    if cozulen_sayisi:
        segment_degisti = takip_segment_doldur(takip, history) or segment_degisti

    if cozulen_sayisi or segment_degisti:
        takip_dosyasi_kaydet(takip)

    if cozulen_sayisi:
        logger.info(f"✅ {cozulen_sayisi} bekleyen tahmin sonuçlandırıldı.")

    return takip


def basari_raporu_yazdir(takip):
    cozulen = takip[takip["Durum"] == "cozuldu"].copy()

    if cozulen.empty:
        print(
            "\n📊 Henüz sonuçlanmış tahmin yok "
            "(yeni sonuç CSV'leri eklendikçe otomatik değerlendirilecek).\n"
        )
        return

    toplam = len(cozulen)
    top1_isabet = cozulen["Top1Isabet"].astype(int).sum()
    top3_isabet = cozulen["Top3Isabet"].astype(int).sum()
    agf_isabet = cozulen["AGFTop1Isabet"].astype(int).sum()

    print(f"\n{'=' * 60}")
    print(f"  📊 MODEL BAŞARI RAPORU (toplam {toplam} sonuçlanmış koşu)")
    print(f"{'=' * 60}")
    print(
        f"  🎯 Model Top-1 (favori kazandı):  "
        f"{top1_isabet}/{toplam}  (%{100 * top1_isabet / toplam:.1f})"
    )
    print(
        f"  🎯 Model Top-3 (kazanan ilk 3'te): "
        f"{top3_isabet}/{toplam}  (%{100 * top3_isabet / toplam:.1f})"
    )
    print(
        f"  📈 AGF (piyasa) Top-1 karşılaştırma: "
        f"{agf_isabet}/{toplam}  (%{100 * agf_isabet / toplam:.1f})"
    )

    # --- Ayrışma analizi: model ile AGF FARKLI at seçtiği koşularda kim haklı?
    # Modelin piyasaya gerçekten bir şey katıp katmadığını asıl bu satır gösterir.
    if "ModelFavori" in cozulen.columns and "AGFFavori" in cozulen.columns:
        farkli = cozulen[cozulen["ModelFavori"] != cozulen["AGFFavori"]]
        n_farkli = len(farkli)

        print(f"{'-' * 60}")

        if n_farkli > 0:
            model_galip = int(farkli["Top1Isabet"].astype(int).sum())
            agf_galip = int(farkli["AGFTop1Isabet"].astype(int).sum())
            ikisi_yanlis = n_farkli - model_galip - agf_galip

            print(f"  🎯 AYRIŞMA ANALİZİ (Model ≠ AGF, {n_farkli} koşu)")
            print(
                f"     Model haklı : {model_galip}/{n_farkli} "
                f"(%{100 * model_galip / n_farkli:.1f})"
            )
            print(
                f"     AGF   haklı : {agf_galip}/{n_farkli} "
                f"(%{100 * agf_galip / n_farkli:.1f})"
            )
            print(f"     İkisi de yanlış: {ikisi_yanlis}")
            print(f"     Fark (model - AGF): {model_galip - agf_galip:+d}")

            if n_farkli < 30:
                print("     (az veri)")
        else:
            print("  🎯 AYRIŞMA ANALİZİ: model ve AGF hep aynı atı seçmiş.")

    # --- Adil Top-3 kıyası: AGF'nin ilk 3'ü vs modelin ilk 3'ü, AYNI
    # koşularda. AGF Top-3 sadece bu sürümden sonra kaydedilen koşular
    # için var, bu yüzden model Top-3'ü de aynı alt kümede gösteriliyor.
    if "AGFTop3Isabet" in cozulen.columns:
        agf3 = pd.to_numeric(cozulen["AGFTop3Isabet"], errors="coerce")
        maske = agf3.notna()
        n3 = int(maske.sum())

        if n3 > 0:
            agf3_isabet = int(agf3[maske].sum())
            model3_ayni = int(
                cozulen.loc[maske, "Top3Isabet"].astype(int).sum()
            )

            print(f"{'-' * 60}")
            print(f"  ⚖️ ADİL TOP-3 KIYASI (aynı {n3} koşuda)")
            print(
                f"     Model Top-3: {model3_ayni}/{n3} "
                f"(%{100 * model3_ayni / n3:.1f})"
            )
            print(
                f"     AGF   Top-3: {agf3_isabet}/{n3} "
                f"(%{100 * agf3_isabet / n3:.1f})"
            )

    # --- Segment kırılımı: handikap / maiden / eküri.
    def _segment_yaz(ad, sutun, deger):
        if sutun not in cozulen.columns:
            return

        sayi = pd.to_numeric(cozulen[sutun], errors="coerce")
        alt = cozulen[sayi == deger]
        n = len(alt)

        if n == 0:
            return

        t1 = int(alt["Top1Isabet"].astype(int).sum())
        t3 = int(alt["Top3Isabet"].astype(int).sum())
        a1 = int(alt["AGFTop1Isabet"].astype(int).sum())
        az_veri = "  (az veri)" if n < 30 else ""

        print(
            f"     {ad:<16} n={n:<4} "
            f"Model T1 %{100 * t1 / n:4.1f} | "
            f"Model T3 %{100 * t3 / n:4.1f} | "
            f"AGF T1 %{100 * a1 / n:4.1f}{az_veri}"
        )

    print(f"{'-' * 60}")
    print("  🔎 SEGMENT KIRILIMI")
    _segment_yaz("Handikap", "Handikap", 1)
    _segment_yaz("Handikap değil", "Handikap", 0)
    _segment_yaz("Maiden", "Maiden", 1)
    _segment_yaz("Maiden değil", "Maiden", 0)
    _segment_yaz("Eküri var", "EkuriVar", 1)
    _segment_yaz("Eküri yok", "EkuriVar", 0)

    print(f"{'=' * 60}\n")


# ============================================================================
# ANA PROGRAM
# ============================================================================

def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║    🏇 YARIŞ TAHMİN SİSTEMİ v4.4 - XGBOOST RANKING (EKÜRİ EKLENDİ)         ║
║    CSV: 100 Dosyada Checkpoint | Feature: 25 Günde Checkpoint             ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    os.makedirs(PROGRAM_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    program_files = sorted([
        os.path.join(PROGRAM_DIR, file_name)
        for file_name in os.listdir(PROGRAM_DIR)
        if file_name.lower().endswith(".csv")
    ])

    result_files = sorted([
        os.path.join(RESULT_DIR, file_name)
        for file_name in os.listdir(RESULT_DIR)
        if file_name.lower().endswith(".csv")
    ])

    if not program_files:
        logger.error(
            f"🚫 Program klasöründe CSV bulunamadı: {PROGRAM_DIR}"
        )
        return

    if not result_files:
        logger.error(
            f"🚫 Sonuç klasöründe CSV bulunamadı: {RESULT_DIR}"
        )
        return

    if PREDICT_ONLY_LATEST_PROGRAM and len(program_files) > 1:
        latest_file = max(
            program_files,
            key=os.path.getmtime
        )

        logger.warning(
            f"⚠️ Program klasöründe {len(program_files)} dosya var. "
            f"Sadece en yeni dosya analiz edilecek: "
            f"{os.path.basename(latest_file)}"
        )

        program_files = [latest_file]

    logger.info(
        f"📂 Program CSV: {len(program_files)} | "
        f"Sonuç CSV: {len(result_files)} | "
        f"idman.txt: {'var' if os.path.exists(IDMAN_DOSYA) else 'yok'} | "
        f"hava.txt: {'var' if os.path.exists(HAVA_DOSYA) else 'yok'}"
    )

    history, raw_fingerprint = load_or_build_raw_history(
        result_files
    )

    if history.empty:
        logger.error("🚫 Geçmiş sonuç verisi oluşturulamadı.")
        return

    # Önceki çalıştırmalarda yapılan tahminleri, artık elde olan gerçek
    # sonuçlarla karşılaştırıp otomatik değerlendir ve raporu göster.
    takip = coz_bekleyen_tahminler(history)
    basari_raporu_yazdir(takip)

    if os.path.exists(IDMAN_DOSYA):
        idman_history = load_idman_history()
    else:
        idman_history = pd.DataFrame()
        logger.info(
            f"ℹ️ '{IDMAN_DOSYA}' henüz yok, idman özellikleri nötr "
            "(bilgi yok) olarak kullanılacak."
        )

    if os.path.exists(HAVA_DOSYA):
        hava_history = load_hava_history()
    else:
        hava_history = pd.DataFrame()
        logger.info(
            f"ℹ️ '{HAVA_DOSYA}' henüz yok, hava özellikleri nötr "
            "(bilgi yok) olarak kullanılacak."
        )

    antrenor_kariyer, jokey_kariyer = load_kariyer_istatistikleri()

    if antrenor_kariyer.empty and jokey_kariyer.empty:
        logger.info(
            f"ℹ️ '{ANTRENOR_DOSYA}' / '{JOKEY_DOSYA}' henüz yok, "
            "kariyer istatistikleri nötr (bilgi yok) olarak kullanılacak."
        )

    features = build_or_load_training_features(
        history=history,
        raw_fingerprint=raw_fingerprint,
        idman_history=idman_history,
        hava_history=hava_history,
        antrenor_kariyer=antrenor_kariyer,
        jokey_kariyer=jokey_kariyer
    )

    for program_file in program_files:
        predict_program(
            program_path=program_file,
            history=history,
            features=features,
            raw_fingerprint=raw_fingerprint,
            idman_history=idman_history,
            hava_history=hava_history,
            antrenor_kariyer=antrenor_kariyer,
            jokey_kariyer=jokey_kariyer
        )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        logger.warning(
            "⚠️ Program durduruldu. "
            "Checkpoint kaydedildi; tekrar açınca devam eder."
        )

    except Exception as error:
        logger.error(f"🚫 Beklenmeyen hata: {error}")

        import traceback
        logger.error(traceback.format_exc())
