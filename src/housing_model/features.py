import numpy as np 
from sklearn.preprocessing import FunctionTransformer

# Hər row üçün: column_0 / column_1
def safe_ratio(X: np.ndarray) -> np.ndarray:
    num = X[:, [0]].astype(np.float64) # numerator (üst)
    den = X[:, [1]].astype(np.float64) # denominator (alt)
    out = np.zeros_like(num)
    np.divide(num, den, out = out, where=(den != 0)) # denominator sıfır olan yerlərdə bölmə etməyəcək, nəticə 0 qalacaq
    return out 

    # Nümunə
    # hər müştəri üçün orta satış qiyməti = ümumi satış qiyməti / satışların sayı

def ratio_name(transformer, feature_names_in): # Transformer feature-lar üçün ad generator
    return ['ratio']  # Həmişə 'ratio' adı olacaq

# FunctionTransformer obyektinə çeviririk ki, pipeline-də istifadə olunsun:
ratio_transformer = FunctionTransformer(safe_ratio, feature_names_out=ratio_name)

def safe_log1p(X: np.ndarray) -> np.ndarray: # Log transformation, amma input-da mənfi varsa 0 ilə əvəz olunur:
    X = X.astype(np.float64)
    X = np.clip(X, a_min = 0.0, a_max = None) # mənfi dəyərləri 0 ilə əvəz edirik
    return np.log1p(X) # log1p → log(1 + x)

# Pipeline üçün FunctionTransformer obyekti:
log_transformer = FunctionTransformer(
    safe_log1p, # forward transform
    inverse_func= np.expm1, # inverse → exp(x) - 1
    feature_names_out= 'one-to-one' # feature adları saxlanır
)

