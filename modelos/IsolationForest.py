import math
import cupy as cp
from modelos.base import BaseAnomalyDetector

GAMMA_EULER = 0.5772156649  # constante de Euler-Mascheroni

def harmonic_number(n):
    if n <= 1:
        return 0
    return math.log(n) + GAMMA_EULER

def c_factor(n):
    if n <= 1:
        return 0
    return 2 * harmonic_number(n - 1) - (2 * (n - 1) / n)

class iTree:
    """
    Árbol de Isolation Forest almacenado como arrays de nodos en GPU.
    Construcción iterativa (sin recursión).
    """
    __slots__ = (
        "feat", "thr", "left", "right", "size", "is_leaf",
        "max_height", "n_nodes"
    )

    def __init__(self, X: cp.ndarray, max_height: int, split_selection_random: bool = True):
        """
            X: cp.ndarray [n_samples, n_features] - ya en GPU
            max_height: profundidad máxima (~ ceil(log2(sample_size)))
            split_selection_random: si True, umbral aleatorio uniforme entre min y max
                                    si False, umbral aleatorio entre valores presentes en la muestra
        """
        n, d = X.shape
        self.max_height = max_height

        # Estructuras de nodos (listas en Python -> luego a cupy)
        feat_list   = []
        thr_list    = []
        left_list   = []
        right_list  = []
        size_list   = []
        leaf_list   = []

        # Pila de construcción: (indices_submuestra (cp.ndarray), depth, node_index)
        # Cada push añade un nodo (pendiente de resolver hijos)
        stack = []

        # Creamos el nodo raíz
        root_idx = 0
        feat_list.append(-1)     # placeholder
        thr_list.append(cp.nan)  # placeholder
        left_list.append(-1)
        right_list.append(-1)
        size_list.append(int(n))
        leaf_list.append(False)
        stack.append((cp.arange(n, dtype=cp.int32), 0, root_idx))

        while stack:
            subset_idx, depth, node_id = stack.pop()
            subset_size = int(subset_idx.size)

            # Criterio de parada: altura o tamaño
            if depth >= max_height or subset_size <= 1:
                leaf_list[node_id] = True
                size_list[node_id] = subset_size
                continue

            # Elegir atributo
            attr = int(cp.random.randint(d))

            # min/max del atributo en el subset
            col_vals = X[subset_idx, attr]
            min_v = cp.min(col_vals)
            max_v = cp.max(col_vals)

            if not bool((max_v > min_v).item()):
                # Todos iguales: hoja
                leaf_list[node_id] = True
                size_list[node_id] = subset_size
                continue

            # Elegir umbral
            if split_selection_random:
                split_val = cp.random.uniform(min_v, max_v).astype(cp.float32)
            else:
                # umbral aleatorio tomado de valores presentes
                # Elegimos un índice aleatorio dentro del subset
                ridx = cp.random.randint(subset_size)
                split_val = col_vals[ridx].astype(cp.float32)

            # Split (todo en GPU)
            left_mask = col_vals < split_val
            left_idx = subset_idx[left_mask]
            right_idx = subset_idx[~left_mask]

            # Si un lado queda vacío, marcamos hoja (caso degenerado)
            if left_idx.size == 0 or right_idx.size == 0:
                leaf_list[node_id] = True
                size_list[node_id] = subset_size
                continue

            # Fijar datos del nodo actual
            feat_list[node_id] = attr
            thr_list[node_id] = split_val
            leaf_list[node_id] = False
            size_list[node_id] = subset_size

            # Crear nodos hijos y encolar
            left_id = len(feat_list)
            right_id = left_id + 1

            feat_list.extend([-1, -1])
            thr_list.extend([cp.float32(cp.nan), cp.float32(cp.nan)])
            left_list.extend([-1, -1])
            right_list.extend([-1, -1])
            size_list.extend([int(left_idx.size), int(right_idx.size)])
            leaf_list.extend([False, False])

            left_list[node_id] = left_id
            right_list[node_id] = right_id

            stack.append((right_idx, depth + 1, right_id))
            stack.append((left_idx,  depth + 1, left_id))

        # Convertimos a arrays en GPU
        self.n_nodes = len(feat_list)
        # feat/thr/left/right/size/is_leaf deben ser cp.ndarray
        self.feat   = cp.asarray(feat_list, dtype=cp.int32)
        self.thr     = cp.asarray(cp.stack([cp.asarray(x, dtype=cp.float32) for x in thr_list]))
        self.left    = cp.asarray(left_list, dtype=cp.int32)
        self.right   = cp.asarray(right_list, dtype=cp.int32)
        self.size    = cp.asarray(size_list, dtype=cp.int32)
        self.is_leaf = cp.asarray(leaf_list, dtype=cp.bool_)

    def path_length_batch(self, X: cp.ndarray) -> cp.ndarray:
        """
        Calcula la longitud de camino para TODAS las filas de X a la vez.
        X: cp.ndarray [n_samples, n_features]
        return: cp.ndarray [n_samples]
        """
        n = X.shape[0]
        # Nodo actual por muestra
        curr = cp.zeros(n, dtype=cp.int32)
        done = cp.zeros(n, dtype=cp.bool_)
        # Depth acumulada (sin el término c(size), que se sumará al llegar a hoja)
        depth = cp.zeros(n, dtype=cp.float32)
        gamma32 = cp.float32(GAMMA_EULER)

        ar = cp.arange(n, dtype=cp.int32)

        # Bucle acotado por max_height + 2 por seguridad
        # Salimos si todas las muestras han llegado a hoja
        max_steps = int(self.max_height + 2)
        for _ in range(max_steps):
            if bool(cp.all(done)):  # condición rápida (reduce kernels si ya terminamos)
                break

            # Atributos del nodo actual por muestra
            node = curr
            leaf = self.is_leaf[node]
            # Añadir c(size) a las que llegan a hoja y marcarlas como done
            to_close = (~done) & leaf
            if bool(cp.any(to_close)):
                sz = self.size[node[to_close]].astype(cp.float32)
                # c(size) se calcula en CPU (escalar) pero es por nodo; necesitamos vectorizar:
                # aproximamos con fórmula continua usando log para evitar overhead Python:
                # c(n) ~ 2*(ln(n-1)+gamma) - 2*(n-1)/n  para n>1, y 0 si n<=1
                szf = sz.astype(cp.float32)
                # manejo de casos n<=1
                c_sz = cp.where(
                    szf <= 1.0,
                    cp.float32(0.0),
                    2.0 * (cp.log(szf - 1.0) + cp.float32(gamma32)) - 2.0 * (szf - 1.0) / szf
                )
                depth[to_close] = depth[to_close] + c_sz
                done[to_close] = True

            # Para las no terminadas, hacemos el paso de split
            still = (~done)
            if not bool(cp.any(still)):
                break

            node_s  = node[still]
            feat_s  = self.feat[node_s]
            thr_s   = self.thr[node_s]
            left_s  = self.left[node_s]
            right_s = self.right[node_s]

            # Valor X[row, feat[row]]
            xvals = X[ar[still], feat_s]
            go_left = xvals < thr_s

            # Actualizar profundidad + nodo
            depth[still] = depth[still] + 1.0
            curr[still]  = cp.where(go_left, left_s, right_s)

        return depth

# ------------------------------
# Bosque de aislamiento
# ------------------------------

class IsolationForest(BaseAnomalyDetector):
    """
    Isolation Forest en GPU:
        - Árbol iterativo (sin recursión) con almacenamiento por arrays de nodos.
        - Evaluación batch de path length.
        - Sin transfers fuera de lo imprescindible.
    """
    def __init__(self,
                 n_trees: int = 100,
                 sample_size: int = 256,
                 split_selection_random: bool = True,
                 contamination: float | None = None,
                 random_state: int | None = None):
        self.n_trees = int(n_trees)
        self.sample_size = int(sample_size)
        self.split_selection_random = bool(split_selection_random)
        self.contamination = contamination
        self.threshold_ = None
        self.trees: list[iTree] = []
        self._denom = max(c_factor(self.sample_size), 1e-8)  # escalar CPU
        if random_state is not None:
            # RNG CPU para reproducibilidad de decisiones discretas;
            # para cupy se puede usar cp.random.seed desde fuera si deseas.
            import numpy as _np
            self._rng = _np.random.default_rng(random_state)

    # ---------- Hooks de tu pipeline ----------
    def preprocess(self, X, retrain: bool = True):
        """
        Recibe cudf.DataFrame o pandas.DataFrame.
        Devuelve cp.ndarray en GPU.
        """
        # Si viene en cudf -> .values es cupy array; si es pandas -> lo pasamos a cupy
        try:
            # cudf
            vals = X.values
            return vals if isinstance(vals, cp.ndarray) else cp.asarray(vals)
        except Exception:
            # pandas / numpy
            return cp.asarray(X.values if hasattr(X, "values") else X)

    # ------------------------------------------

    def fit(self, X):
        X = cp.asarray(X, dtype=cp.float32)
        n = int(X.shape[0])
        # altura límite teórica
        height_limit = int(math.ceil(math.log2(max(self.sample_size, 2))))

        self.trees = []

        # tamaño del pool (ajusta: 4, 8 o 16 según la GPU)
        n_streams = 16
        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]

        # Reservamos lista de futuros "ligeros" (referencias) para sincronizar al final
        pending = []

        for t in range(self.n_trees):
            s = streams[t % n_streams]
            with s:
                # muestreo en GPU
                if self.sample_size < n:
                    idx = cp.random.permutation(n)[:self.sample_size]
                else:
                    idx = cp.random.randint(0, n, size=self.sample_size)
                sample = X[idx, :]

                # construir el árbol dentro del stream
                tree = iTree(sample,
                            max_height=height_limit,
                            split_selection_random=self.split_selection_random)
                self.trees.append(tree)
                pending.append(s)

        # Sincroniza todos los streams (asegura que todos los árboles están listos)
        for s in pending:
            s.synchronize()

        # Umbral
        if self.contamination is None:
            self.threshold_ = 0.5
        else:
            scores_train = self.anomaly_score(X)  # cp.ndarray
            q = float(1.0 - float(self.contamination))
            q = min(max(q, 0.0), 1.0)
            self.threshold_ = float(cp.quantile(scores_train, q))
        return self

    def anomaly_score(self, X):
        X = cp.asarray(X, dtype=cp.float32)
        n = int(X.shape[0])
        # Acumulamos longitudes medias
        acc = cp.zeros(n, dtype=cp.float32)
        for t in self.trees:
            h = t.path_length_batch(X)  # cp.ndarray [n]
            acc += h
        Eh = acc / float(len(self.trees)) if self.trees else acc
        # score = 2^(-E[h]/c(n_subsample))
        scores = cp.exp2(-Eh / self._denom)
        return scores

    def predict(self, X, threshold=None):
        X = cp.asarray(X, dtype=cp.float32)
        tau = float(self.threshold_ if threshold is None else threshold)
        return self.anomaly_score(X) >= tau



# isolation_forest_cpu.py
import math
import numpy as np
from joblib import Parallel, delayed
from modelos.base import BaseAnomalyDetector

GAMMA_EULER = 0.5772156649

def harmonic_number(n: int) -> float:
    if n <= 1:
        return 0.0
    return math.log(n) + GAMMA_EULER

def c_factor(n: int) -> float:
    if n <= 1:
        return 0.0
    return 2.0 * harmonic_number(n - 1) - (2.0 * (n - 1) / n)

# =========================
# Árbol CPU (NumPy)
# =========================
class iTreeCPU:
    """
    Árbol de Isolation Forest en CPU con almacenamiento por arrays NumPy.
    Construcción iterativa (sin recursión).
    """
    __slots__ = ("feat", "thr", "left", "right", "size", "is_leaf", "max_height", "n_nodes")

    def __init__(self, X: np.ndarray, max_height: int, split_selection_random: bool = True):
        n, d = X.shape
        self.max_height = max_height

        feat_list, thr_list = [], []
        left_list, right_list = [], []
        size_list, leaf_list = [], []

        # Pila: (idx_submuestra [np.int32], depth, node_id)
        stack = []

        # Nodo raíz
        feat_list.append(-1)
        thr_list.append(np.nan)
        left_list.append(-1)
        right_list.append(-1)
        size_list.append(int(n))
        leaf_list.append(False)

        stack.append((np.arange(n, dtype=np.int32), 0, 0))

        rng = np.random  # sin reproducibilidad; máximo rendimiento

        while stack:
            subset_idx, depth, node_id = stack.pop()
            subset_size = int(subset_idx.size)

            # Parada por altura o tamaño
            if depth >= max_height or subset_size <= 1:
                leaf_list[node_id] = True
                size_list[node_id] = subset_size
                continue

            # Atributo aleatorio
            attr = int(rng.randint(0, d))

            # Valores de la columna en el subset
            col_vals = X[subset_idx, attr]

            # min/max (ruta original para comparar justo con GPU)
            min_v = np.min(col_vals)
            max_v = np.max(col_vals)

            if not (max_v > min_v):
                # Sin rango -> hoja
                leaf_list[node_id] = True
                size_list[node_id] = subset_size
                continue

            # Umbral
            if split_selection_random:
                split_val = rng.uniform(min_v, max_v)
            else:
                ridx = int(rng.randint(0, subset_size))
                split_val = float(col_vals[ridx])

            # Split
            left_mask = col_vals < split_val
            left_idx = subset_idx[left_mask]
            right_idx = subset_idx[~left_mask]

            if left_idx.size == 0 or right_idx.size == 0:
                leaf_list[node_id] = True
                size_list[node_id] = subset_size
                continue

            # Fijar nodo
            feat_list[node_id] = attr
            thr_list[node_id] = float(split_val)
            leaf_list[node_id] = False
            size_list[node_id] = subset_size

            # Crear hijos
            left_id = len(feat_list)
            right_id = left_id + 1

            feat_list.extend([-1, -1])
            thr_list.extend([np.nan, np.nan])
            left_list.extend([-1, -1])
            right_list.extend([-1, -1])
            size_list.extend([int(left_idx.size), int(right_idx.size)])
            leaf_list.extend([False, False])

            left_list[node_id] = left_id
            right_list[node_id] = right_id

            # Apilar primero el derecho (LIFO) para procesar izquierda después
            stack.append((right_idx, depth + 1, right_id))
            stack.append((left_idx,  depth + 1, left_id))

        # Listas -> arrays NumPy
        self.n_nodes = len(feat_list)
        self.feat   = np.asarray(feat_list, dtype=np.int32)
        self.thr    = np.asarray(thr_list, dtype=np.float32)
        self.left   = np.asarray(left_list, dtype=np.int32)
        self.right  = np.asarray(right_list, dtype=np.int32)
        self.size   = np.asarray(size_list, dtype=np.int32)
        self.is_leaf= np.asarray(leaf_list, dtype=np.bool_)

    def path_length_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorizado por pasos (sin recursión), imitando la versión GPU.
        """
        n = X.shape[0]
        curr = np.zeros(n, dtype=np.int32)
        done = np.zeros(n, dtype=bool)
        depth = np.zeros(n, dtype=np.float32)

        ar = np.arange(n, dtype=np.int32)
        gamma32 = np.float32(GAMMA_EULER)

        max_steps = int(self.max_height + 2)
        for _ in range(max_steps):
            if np.all(done):
                break

            node = curr
            leaf = self.is_leaf[node]

            to_close = (~done) & leaf
            if np.any(to_close):
                szf = self.size[node[to_close]].astype(np.float32)
                c_sz = np.zeros_like(szf, dtype=np.float32)
                valid = szf > 1.0
                c_sz[valid] = 2.0 * (np.log(szf[valid] - 1.0) + gamma32) - 2.0 * (szf[valid] - 1.0) / szf[valid]
                depth[to_close] = depth[to_close] + c_sz
                done[to_close] = True


            still = (~done)
            if not np.any(still):
                break

            node_s  = node[still]
            feat_s  = self.feat[node_s]
            thr_s   = self.thr[node_s]
            left_s  = self.left[node_s]
            right_s = self.right[node_s]

            xvals = X[ar[still], feat_s]
            go_left = xvals < thr_s

            depth[still] = depth[still] + 1.0
            curr[still]  = np.where(go_left, left_s, right_s)

        return depth


# =========================
# Bosque CPU (paralelismo extremo con procesos)
# =========================
class IsolationForestCPU(BaseAnomalyDetector):
    """
    Isolation Forest en CPU con construcción de árboles en paralelo (joblib, backend 'loky').
    """
    def __init__(self,
                 n_trees: int = 100,
                 sample_size: int = 256,
                 split_selection_random: bool = True,
                 contamination: float | None = None,
                 n_jobs: int = -1  # todos los cores por defecto
                 ):
        self.n_trees = int(n_trees)
        self.sample_size = int(sample_size)
        self.split_selection_random = bool(split_selection_random)
        self.contamination = contamination
        self.threshold_ = None
        self.trees: list[iTreeCPU] = []
        self._denom = max(c_factor(self.sample_size), 1e-8)
        self.n_jobs = n_jobs

    # ---------- Hooks de tu pipeline ----------
    def preprocess(self, X, retrain: bool = True, y=None):
        """
        Acepta pandas/cudf/np. Devuelve np.ndarray en CPU.
        """
        # cudf -> pasa a pandas -> numpy
        if hasattr(X, "to_pandas"):
            X = X.to_pandas()
        if hasattr(X, "values"):
            return np.asarray(X.values, dtype=np.float32)
        return np.asarray(X, dtype=np.float32)

    # ------------------------------------------

    def _build_one_tree(self, X_np: np.ndarray, height_limit: int) -> iTreeCPU:
        # muestreo CPU súper rápido
        n = X_np.shape[0]
        if self.sample_size < n:
            idx = np.random.permutation(n)[:self.sample_size]
        else:
            idx = np.random.randint(0, n, size=self.sample_size)
        sample = X_np[idx, :]
        return iTreeCPU(sample, max_height=height_limit, split_selection_random=self.split_selection_random)

    def fit(self, X):
        X_np = self.preprocess(X).astype(np.float32, copy=False)
        n = int(X_np.shape[0])
        height_limit = int(math.ceil(math.log2(max(self.sample_size, 2))))

        # Construcción de árboles en paralelo con procesos (sin GIL)
        trees = [self._build_one_tree(X_np, height_limit) for _ in range(self.n_trees)]
        self.trees = trees

        # Umbral
        if self.contamination is None:
            self.threshold_ = 0.5
        else:
            scores_train = self.anomaly_score(X_np)
            q = float(1.0 - float(self.contamination))
            q = min(max(q, 0.0), 1.0)
            self.threshold_ = float(np.quantile(scores_train, q))
        return self

    def anomaly_score(self, X, y=None):
        X_np = self.preprocess(X, retrain=False).astype(np.float32, copy=False)
        n = int(X_np.shape[0])
        acc = np.zeros(n, dtype=np.float32)
        # Suma de longitudes medias (secuencial; suele ser rápido)
        for t in self.trees:
            h = t.path_length_batch(X_np)
            acc += h
        Eh = acc / float(len(self.trees)) if self.trees else acc
        scores = np.exp2(-Eh / self._denom)
        return scores

    def predict(self, X, threshold=None):
        tau = float(self.threshold_ if threshold is None else threshold)
        return self.anomaly_score(X) >= tau
