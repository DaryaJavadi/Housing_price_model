import argparse # Command-line argumentləri oxumaq üçün modul
# (Proqramı dəyişmədən fərqli inputlarla işlədəsən)
import logging # (info, warning, error və s.) üçün
import platform # Python versiyası, OS məlumatları üçün
import sklearn
import yaml 
from pathlib import Path


from .logging_setup import setup_logging
from .config import load_config # YAML config oxuyur
from .data import load_housing, stratified_split # data yükləmə və bölmə
from .train import fit
from .io import write_json
from .versioning import sha256_file, sha256_bytes, sha256_json, short_hash

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    # --config adlı argument əlavə edirik
    # Default olaraq configs/train.yaml istifadə olunacaq

    # yaml faylında modelin hyperparametrləri, data yolları və s. olacaq -> 
    # “bu model necə train olunub?” → 1 fayla baxırsan
    parser.add_argument('--config', default = "configs/train.yaml")
    
    # CLI-dən gələn argumentləri oxuyuruq
    args = parser.parse_args()

    setup_logging()

    # YAML config faylını oxuyub Python obyektinə çeviririk
    cfg = load_config(args.config)

# ----------Hashes for reproducibility-----------
    
    # Config faylının text halını oxuyuruq
    config_text = Path(args.config).read_text(encoding="utf-8") #train.yaml-ın bütün məzmunu oxunur
    config_hash = sha256_bytes(config_text.encode("utf-8")) #YALNIZ bu an üçün hash hesablanır (yaddaşda qalmır)
    data_hash = sha256_file(cfg.data.csv_path) # CSV faylının hash-i çıxarılır
    # hash-lər bunu göstərir -> “Dünənki model bugünkündən fərqlidir, çünki config dəyişib.” Hash-lər dəyişibsə, deməli model də fərqlidir

# ----------------------------------------------- Data part:
    df = load_housing(cfg.data.csv_path)

    X_train, X_test, y_train, y_test = stratified_split(
        df = df,
        target= cfg.data.target,
        stratify_col= cfg.data.stratify_col,
        bins = cfg.data.income_cat_bins,
        labels= cfg.data.income_cat_labels,
        test_size= cfg.data.test_size,
        random_state = cfg.data.random_state
    )

    result = fit(cfg, X_train, y_train, X_test, y_test)

    # Bütün run məlumatlarını saxlayan dictionary
    manifest = {
        'python': platform.python_version(),
        'sklearn': sklearn.__version__,
        'config_path': args.config,
        "config_hash": config_hash,
        'data_path': cfg.data.csv_path,
        "data_hash": data_hash,
        'model_path': result['model_path'],
        "training_profile_path": result["training_profile_path"],
        'metrics': result['metrics'],
        'meta': result['meta'],
    }

    # Config və data hash-lərindən unikal run_id yaradırıq.
    # Məqsəd: hansı run hansı nəticəyə, hansı config və data ilə bağlıdır deyə bilmək.
    manifest["run_id"] = f"{short_hash(config_hash)}_{short_hash(data_hash)}"
    write_json(cfg.output.manifest_path, manifest)

    # runs/
    #  └── 2026-01-31_16-20-10/   ← run_id = unikal
    #       ├── config_hash.txt -> config faylının hash-i (train.yaml)
    #       ├── data_hash.txt -> data faylının hash-i (housing.csv)
    #       ├── metrics.json -> nəticələr (rmse, mae, r2 və s.)
    #       └── model.pkl -> trained modelin saxlandığı fayl


    # Train bitəndə əsas nəticələri log edirik
    logger.info(
        "Done. run_id=%s Test RMSE=%.4f MAE=%.4f R2=%.4f",
        manifest["run_id"],
        result["metrics"]["rmse"],
        result["metrics"]["mae"],
        result["metrics"]["r2"],
    )
    
if __name__ == '__main__':
    main()