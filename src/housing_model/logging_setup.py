import logging
import os

def setup_logging(level: str | None = None) -> None:

    # Level seçimi:
    # 1) Funksiya çağırılarkən level verilibsə onu istifadə et
    # 2) Yoxdursa environment variable LOG_LEVEL istifadə et
    # 3) Hər ikisi yoxdursa default "INFO"
    
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO), # logging səviyyəsini təyin edir
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", # mesajın görünüşünü müəyyən edir:
    )
    # %asctime% → vaxt
    # %levelname% → logging səviyyəsi
    # %name% → logger adı (adətən module adı)
    # %message% → mesaj

    # Sonda belə olur:
    # 2026-01-31 18:00:00 | INFO | root | Training started
    # 2026-01-31 18:00:00 | WARNING | root | Learning rate is high
    # 2026-01-31 18:00:00 | ERROR | root | Failed to load dataset
