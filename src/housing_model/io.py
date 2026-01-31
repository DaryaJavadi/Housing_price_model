# with open("artifacts/metrics.json", "w") as f:
#     json.dump(metrics, f)

import json 
from pathlib import Path 


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
# parents=True → bütün ancestor qovluqları da yaradır
# exist_ok=True → əgər artıq varsa xəta vermir

def write_json(path: str, payload: dict) -> str: 
    ensure_parent(path) # Parent qovluğun mövcudluğunu təmin edirik
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    # JSON faylını yazırıq, indent=2 → oxunaqlı format
    return path 

# metrics.json faylı run/experiment nəticələrini saxlamaq üçün istifadə olunur.