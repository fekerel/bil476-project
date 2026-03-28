"""
data_loader.py
--------------
UCI Default of Credit Card Clients (id=350) veri setini yükler.
Önce ucimlrepo kütüphanesiyle çeker; başarısız olursa proje dizininde yer alan Excel'e döner.

Her iki yolda da çıkış garantisi:
  - X  : pd.DataFrame, shape (30000, 23), sütunlar X1..X23, dtype int64
  - y  : pd.DataFrame, shape (30000,  1), sütun 'default', dtype int64
  - variables : pd.DataFrame  (her zaman dolu — gerekirse cache'ten yüklenir)

variables cache dosyası:
  UCI'dan başarılı çekildiğinde 'dataset/VARIABLES_INFO_CSV.csv' olarak kaydedilir.
  Sonraki çalıştırmalarda UCI erişilemese bile aynı DataFrame geri döner.
"""

import pandas as pd
from pathlib import Path

FEATURE_COLS    = [f"X{i}" for i in range(1, 24)]   # X1 … X23
TARGET_COL      = "default"
N_EXPECTED      = 30_000
EXCEL_PATH      = Path(__file__).parent / "dataset" / "default of credit card clients.xls"
VARIABLES_INFO_CSV = Path(__file__).parent / "dataset" / "variable_info.csv"

# ── Dışa açık fonksiyon ───────────────────────────────────────────────────────
def load_dataset(excel_path=None, test_excel_fallback=False):
    """
    Veri setini yükler ve normalize eder.

    Returns
    -------
    X : pd.DataFrame        — 23 özellik sütunu (X1..X23)
    y : pd.DataFrame        — 1 hedef sütunu ('default')
    variables : DataFrame | None  — cache varsa her iki yolda da dolu
    """
    if test_excel_fallback:
        X = None
    else:
        X, y, variables = _try_uci()

    if X is None:
        path = Path(excel_path) if excel_path else EXCEL_PATH
        X, y, variables = _load_excel(path)

    X, y = _normalize(X, y)
    _sanity_check(X, y)
    return X, y, variables


# ── Özel yükleyiciler ─────────────────────────────────────────────────────────
def _try_uci():
    try:
        from ucimlrepo import fetch_ucirepo
        repo = fetch_ucirepo(id=350)
        X = repo.data.features.copy()
        y = repo.data.targets.copy()
        variables = repo.variables
        print("✓ Veri UCI reposundan yüklendi.")
        return X, y, variables
    except Exception as e:
        #print(f"UCI repo erişilemedi ({e})\nLocal dosyaya geçiliyor...")
        return None, None, None


def _load_excel(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Local Excel bulunamadı: {path}\n"
            "Dosyayı 'dataset/' klasörüne koyun veya load_dataset(excel_path=...) ile yolu verin."
        )
    df = pd.read_excel(path, header=1, index_col=0)
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, [-1]].copy()
    variables = _load_variables_from_csv()
    return X, y, variables


def _load_variables_from_csv() -> "pd.DataFrame | None":
    """Cache varsa yükler, yoksa None döner ve uyarı verir."""
    if VARIABLES_INFO_CSV.exists():
        df = pd.read_csv(VARIABLES_INFO_CSV)
        #print(f"  (variables lokal'den yüklendi: {VARIABLES_INFO_CSV.name})")
        return df
    print("⚠ variables info csv'si bulunamadı.")
    return None


# ── Normalizasyon ─────────────────────────────────────────────────────────────
def _normalize(X, y):
    # 1. MultiIndex sütunları düzleştir (ucimlrepo bazen döndürür)
    if isinstance(X.columns, pd.MultiIndex):
        X.columns = X.columns.get_level_values(-1)
    if isinstance(y.columns, pd.MultiIndex):
        y.columns = y.columns.get_level_values(-1)

    # 2. Özellik sütun adlarını X1..X23 yap
    if list(X.columns) != FEATURE_COLS:
        if len(X.columns) == 23:
            X.columns = FEATURE_COLS
        else:
            raise ValueError(
                f"Beklenmeyen özellik sütun sayısı: {len(X.columns)} (beklenen 23).\n"
                "Excel dosyası doğru mu? (header=1, index_col=0 ile 23 özellik + 1 hedef bekleniyor)"
            )

    # 3. Hedef sütun adını standartlaştır
    y.columns = [TARGET_COL]

    # 4. Index: 0-tabanlı RangeIndex — her iki kaynakta farklı olabilir
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # 5. dtype: int64 garantisi — Excel bazen float döndürür (NaN yoksa güvenli)
    try:
        X = X.astype("int64")
        y = y.astype("int64")
    except (ValueError, TypeError) as e:
        print(f"⚠ dtype dönüşümü yapılamadı (eksik değer olabilir): {e}")

    return X, y


def _sanity_check(X, y):
    assert list(X.columns) == FEATURE_COLS, "Sütun adları X1..X23 değil!"
    assert list(y.columns) == [TARGET_COL],  "Hedef sütun adı 'default' değil!"
    assert X.index.equals(y.index),           "X ve y index'leri uyuşmuyor!"
    assert X.shape[1] == 23,                  f"Özellik sütun sayısı {X.shape[1]} (beklenen 23)"

    missing_X = X.isnull().sum().sum()
    missing_y = y.isnull().sum().sum()
    if missing_X or missing_y:
        print(f"⚠ Eksik değer: X'te {missing_X}, y'de {missing_y}")

    if len(X) != N_EXPECTED:
        print(f"⚠ Satır sayısı {len(X):,} (beklenen {N_EXPECTED:,})")

    print(f"   X : {X.shape[0]:,} satır × {X.shape[1]} sütun | dtype: {X.dtypes.iloc[0].name}")
    print(f"   y : {y.shape[0]:,} satır × {y.shape[1]} sütun | sütun: '{y.columns[0]}'")
