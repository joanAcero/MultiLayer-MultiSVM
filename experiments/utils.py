from __future__ import annotations
import csv, datetime, os, sys, time, warnings
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
warnings.filterwarnings("ignore")

PUBLISHED = {
    "acero2025": {
        "name": "ML-SVM (Acero & Belanche 2025)",
        "Glass":{"acc":0.820},"Breast Cancer":{"acc":0.992},
        "Ionosphere":{"acc":0.950},"Magic":{"acc":0.850},
        "Spambase":{"acc":0.850},"Cover Type":{"acc":0.790},
    },
    "mehrkanoon2018": {"name":"DHNKN (Mehrkanoon & Suykens 2018)",
        "MNIST":{"acc":0.9756,"note":"60k/10k protocol"}},
}
RBF_N_LIMIT = 20_000

def import_ml_msvm():
    experiments_dir = Path(__file__).parent.resolve()
    for base in [experiments_dir.parent, experiments_dir.parent.parent, experiments_dir]:
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        for mod_path in ("mlsvm.ml_msvm","ml_msvm.ml_msvm","ml_msvm"):
            try:
                import importlib
                mod = importlib.import_module(mod_path)
                cls = getattr(mod,"ML_MSVMClassifier")
                if not (isinstance(cls,type) and hasattr(cls,"fit")): continue
                import inspect
                if "dual=False" not in inspect.getsource(cls._make_svm):
                    print("[WARNING] ml_msvm._make_svm does not have dual=False — ArcCos timing will be O(n^2).")
                return cls
            except (ImportError,ModuleNotFoundError,AttributeError): pass
    for f in sorted(experiments_dir.parent.rglob("ml_msvm.py")):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ml_msvm_found",f)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        cls = getattr(mod,"ML_MSVMClassifier",None)
        if cls: print(f"[utils] Loaded from {f}"); return cls
    raise ImportError("Cannot find ML_MSVMClassifier. Run: find ~ -name ml_msvm.py")

class Tee:
    def __init__(self,stream,filepath):
        self._stream=stream
        os.makedirs(os.path.dirname(filepath) or ".",exist_ok=True)
        self._file=open(filepath,"a",buffering=1,encoding="utf-8")
        self._file.write(f"\n{'='*70}\n  Session: {datetime.datetime.now()}\n{'='*70}\n")
    def write(self,data): self._stream.write(data); self._file.write(data); self._file.flush()
    def flush(self): self._stream.flush(); self._file.flush()
    def isatty(self): return False
    def close(self): self._file.close()

class CSVWriter:
    FIELDS=["exp_id","dataset","n_total","n_train","n_test","d","n_classes",
            "model","kernel","L","m","P","split_id","acc","time_s","timestamp"]
    def __init__(self,filepath):
        os.makedirs(os.path.dirname(filepath) or ".",exist_ok=True)
        new=not os.path.exists(filepath)
        self._f=open(filepath,"a",newline="",encoding="utf-8",buffering=1)
        self._w=csv.DictWriter(self._f,fieldnames=self.FIELDS,extrasaction="ignore")
        if new: self._w.writeheader(); self._f.flush()
    def write(self,row):
        row.setdefault("timestamp",datetime.datetime.now().isoformat())
        self._w.writerow(row); self._f.flush()
    def close(self): self._f.close()

_CACHE={}

def _openml(name=None,version=1,data_id=None):
    kw=dict(as_frame=False,return_X_y=True)
    X,y=(fetch_openml(data_id=data_id,**kw) if data_id
         else fetch_openml(name=name,version=version,**kw))
    X=X.astype(np.float64); _,y=np.unique(y,return_inverse=True)
    return X,y.astype(int)

def _load_glass():
    for kw in [{"data_id":41},{"data_id":723},{"name":"glass","version":10},{"name":"glass","version":1}]:
        try: return _openml(**kw)
        except Exception: continue
    raise RuntimeError("Glass not loadable from OpenML — skipping.")

def load(tag,verbose=True,data_dir="data"):
    if tag in _CACHE: return _CACHE[tag]
    if verbose: print(f"  Loading {tag}...",end=" ",flush=True)
    t0=time.perf_counter()
    try:
        if tag=="wine": X,y=load_wine(return_X_y=True); X=X.astype(np.float64)
        elif tag=="breast_cancer": X,y=load_breast_cancer(return_X_y=True); X=X.astype(np.float64)
        elif tag=="ionosphere": X,y=_openml("ionosphere",version=1)
        elif tag=="sonar":      X,y=_openml("sonar",version=1)
        elif tag=="glass":      X,y=_load_glass()
        elif tag=="magic":      X,y=_openml("MagicTelescope",version=1)
        elif tag=="spambase":   X,y=_openml(data_id=44); y=y.astype(int)
        elif tag=="covertype_sub": X,y=_covertype_sub(10_000)
        elif tag=="covertype":  X,y=_openml("covertype",version=3)
        elif tag=="mnist":      X,y=_openml("mnist_784",version=1)
        elif tag=="fashion":    X,y=_openml("Fashion-MNIST",version=1)
        elif tag=="susy":       X,y=_large_cached("susy", data_dir,200_000)
        elif tag=="higgs":      X,y=_large_cached("higgs",data_dir,200_000)
        else: raise ValueError(f"Unknown tag: {tag}")
        if verbose: print(f"done ({time.perf_counter()-t0:.1f}s, shape={X.shape})",flush=True)
        _CACHE[tag]=(X,y); return X,y
    except Exception as e:
        if verbose: print(f"FAILED ({e})",flush=True)
        raise

def _covertype_sub(n):
    X,y=_openml("covertype",version=3)
    idx,_=next(StratifiedShuffleSplit(1,train_size=n,random_state=0).split(X,y))
    return X[idx],y[idx]

def _large_cached(tag,data_dir,n_max):
    os.makedirs(data_dir,exist_ok=True)
    cache=os.path.join(data_dir,f"{tag}_{n_max}.npz")
    if os.path.exists(cache):
        d=np.load(cache); return d["X"].astype(np.float64),d["y"].astype(int)
    ids={"susy":4135,"higgs":23512}
    print(f"\n  Downloading {tag} (will subsample to {n_max:,} immediately)...",flush=True)
    X,y=fetch_openml(data_id=ids[tag],as_frame=False,return_X_y=True)
    X=X.astype(np.float64); _,y=np.unique(y,return_inverse=True); y=y.astype(int)
    if len(y)>n_max:
        idx=np.random.default_rng(0).choice(len(y),n_max,replace=False)
        X,y=X[idx],y[idx]
    np.savez_compressed(cache,X=X,y=y)
    print(f"  Cached {len(y):,} rows → {cache}")
    return X,y

def make_splits(X,y,regime,n_repeats,seed=0):
    if regime in (1,2):
        sss=StratifiedShuffleSplit(n_repeats,test_size=0.1,random_state=seed)
        return [(X[tr],y[tr],X[te],y[te]) for tr,te in sss.split(X,y)]
    elif regime==3:
        splits=[]
        for s in range(n_repeats):
            Xtr,Xte,ytr,yte=train_test_split(X,y,train_size=10_000,test_size=2_000,stratify=y,random_state=s)
            splits.append((Xtr,ytr,Xte,yte))
        return splits
    elif regime==4:
        sss=StratifiedShuffleSplit(n_repeats,test_size=0.2,random_state=seed)
        return [(X[tr],y[tr],X[te],y[te]) for tr,te in sss.split(X,y)]
    raise ValueError(f"Unknown regime {regime}")

def pipe(*steps):
    return Pipeline([("scaler",StandardScaler())]+[(f"s{i}",s) for i,s in enumerate(steps)])

def make_linear_svm():
    return pipe(LinearSVC(C=1.0,dual=False,max_iter=10_000,random_state=0))
def make_rbf_svm():
    return pipe(SVC(kernel="rbf",C=1.0,gamma="scale",random_state=0))
def make_nystroem_svm(P):
    return Pipeline([("scaler",StandardScaler()),
                     ("nystroem",Nystroem(kernel="rbf",n_components=P,gamma=None,random_state=0)),
                     ("svm",LinearSVC(C=1.0,dual=False,max_iter=10_000,random_state=0))])
def make_mlmsvm(ML_MSVM, L, m, P, kernel, C_values=None, final_C=1.0, seed=0):
    if C_values is None:
        C_values = [1.0] if m == 1 else list(np.logspace(-2, 1, num=m))
    return pipe(ML_MSVM(num_layers=L, svms_per_block=m, C_values=C_values,
        rff_features=P, kernel=kernel, arc_cosine_degree=1,
        final_C=final_C, random_state=seed))
def make_flat_rff(ML_MSVM,P,kernel="rbf"):
    return make_mlmsvm(ML_MSVM,0,1,P,kernel)

def ms_for(d):
    if d<=20:    base=[1,2,3,max(4,d//4),d//2,d,2*d]
    elif d<=100: base=[1,2,3,d//4,d//2,d]
    else:        base=[1,2,3,d//16,d//8]
    return sorted(set(max(1,v) for v in base))

def hms(s): return f"{int(s//3600):02d}h{int(s%3600//60):02d}m{int(s%60):02d}s"

def banner(title,*lines):
    w=max(len(title),max((len(l) for l in lines),default=0))+4
    print(f"\n{'█'*w}\n  {title}")
    for l in lines: print(f"  {l}")
    print(f"{'█'*w}\n",flush=True)

if __name__=="__main__":
    print("Testing import_ml_msvm()...")
    try:
        ML=import_ml_msvm(); print(f"  OK: {ML.__module__}")
    except ImportError as e:
        print(f"  FAILED: {e}")