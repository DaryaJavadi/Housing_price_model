from dataclasses import dataclass
from pathlib import Path 
import yaml 


# dataclass framework düzəldir, əl ilə yazmamaq üçün.
# Əl ilə yazdığımız __init__, __repr__, __eq__ kimi metodları Python özü yaradır. 

@dataclass(frozen=True) # frozen=True → obyekt immutable olur
class TrainConfig:
    csv_path: str
    target: str 
    stratify_col: str
    test_size: float 
    random_state: int 
    income_cat_bins: list[float] # -> (qruplaşdırma) üçün sərhədlərdir.
    # (income) sütununu kategoriyaya bölmək istəyirsən:
    # 0–30k → bin 1
    # 30k–60k → bin 2
    # 60k–100k → bin 3
    # 100k+ → bin 4
    income_cat_labels: list[int]  # -> Hər bin üçün bir identifikator / kateqoriya adı


@dataclass(frozen=True)
class ModelConfig:
    random_state: int 
    n_jobs: int  # Paralel işləmə üçün CPU core sayı:
    # 1	olarsa -> single-threaded (yəni paralel deyil, normal işləyir)
    # -1 olarsa -> bütün mövcud CPU cores istifadə olunur
    # N > 1 olarsa -> N paralel işçi (worker) ilə işləyir


@dataclass(frozen=True)
class GridConfig:
    enabled: bool # Grid search istifadə olunacaq mı?
    cv: int 
    scoring: str # Scoring metric adı
    param_grid: dict # hyperparameter-lərin axtarışı üçün istifadə olunan dictionary:
    # Açar = ML modelin hyperparameter-i (n_estimators, max_depth, min_samples_split)
    # Dəyər = sınamaq istədiyin variantlar (list)

@dataclass(frozen=True)
class OutputConfig:
    artifacts_dir: str # Bütün nəticələrin saxlanacağı qovluq
    model_path: str 
    metrics_path: str 
    manifest_path: str 

# Bütün app config-ləri birləşdirən DataClass:
@dataclass
class AppConfig:
    data: TrainConfig
    model: ModelConfig
    grid: GridConfig
    output: OutputConfig

# ----------------- Config loader -----------------
# YAML faylı oxuyub AppConfig obyektinə çevirən funksiya
def load_config(path: str) -> AppConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding='utf-8'))

    # YAML-da bölmələri ayırırıq
    data = payload['data']
    model = payload['model']
    grid = payload['grid']
    output = payload['output']

    return AppConfig(
        data=TrainConfig(
            csv_path=data["csv_path"],
            target=data["target"],
            stratify_col=data["stratify_col"],
            test_size=float(data["test_size"]),
            random_state=int(data["random_state"]),
            income_cat_bins=[float(x) if x != ".inf" else float("inf") for x in data["income_cat_bins"]],
            # income_cat_bins → ".inf" varsa inf-ə çeviririk
            income_cat_labels=[int(x) for x in data["income_cat_labels"]],
        ), # label-ları int-ə çeviririk
        model=ModelConfig(
            random_state=int(model["random_state"]),
            n_jobs=int(model["n_jobs"]),
        ),
        grid=GridConfig(
            enabled=bool(grid["enabled"]),
            cv=int(grid["cv"]),
            scoring=str(grid["scoring"]),
            param_grid=grid["param_grid"] or {},
        ),
        output=OutputConfig(
            artifacts_dir=output["artifacts_dir"],
            model_path=output["model_path"],
            metrics_path=output["metrics_path"],
            manifest_path=output["manifest_path"],
        ),
    )

# Məqsəd:

# YAML faylında yazdığımız parametrləri Python obyektinə çevirmək.
# Hər bir parametrin tipi düzgün olsun (float, int, str, bool).
# Parametrlər immutable (dəyişdirilə bilməz) olsun ki, proqramın axışı zamanı təsadüfən dəyişməsin.
# Kodun digər hissələri (main.py, train.py və s.) YAML faylına birbaşa baxmadan rahat işləyə bilsin.