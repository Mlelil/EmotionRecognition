"""Mini app SER (Speech Emotion Recognition) : un bouton, ta voix, deux modèles.

    clic -> enregistrement 3 s -> log-Mel -> CNN + ResNet-18 -> probabilité de chaque émotion

Fichier unique et autonome :
  * les poids sont détectés automatiquement : il suffit de laisser les dossiers exportés par les
    notebooks (best_model.pt + normalization_stats.json) quelque part à côté de app.py ;
  * les bibliothèques manquantes sont installées toutes seules au premier lancement.

Lancement :  python app.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- dépendances
REQUIRED = {  # module importé -> paquet pip
    "numpy": "numpy", "torch": "torch", "torchvision": "torchvision", "librosa": "librosa",
    "soxr": "soxr", "matplotlib": "matplotlib", "sounddevice": "sounddevice",
}


def ensure_dependencies():
    missing = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print(f"Installation des dépendances manquantes : {', '.join(missing)} (une seule fois)…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    os.execv(sys.executable, [sys.executable, *sys.argv])  # relance le script avec les nouveaux paquets


ensure_dependencies()

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torchvision.models import resnet18  # noqa: E402

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- prétraitement
# Identique à ser_model1.py / ser_model2.py (ne rien changer : c'est ce qui a servi à l'entraînement)
SAMPLE_RATE = 16_000
NUM_SAMPLES = 48_000            # 3 s
DURATION_S = NUM_SAMPLES / SAMPLE_RATE
WIN_LENGTH, HOP_LENGTH, N_FFT, N_MELS = 400, 160, 512, 64
EXPECTED_SHAPE = (64, 301)


def standardize_audio(mono: np.ndarray, sr: int) -> np.ndarray:
    """Mono, 16 kHz, exactement 3 s (crop / zero-padding centré)."""
    mono = np.asarray(mono, dtype=np.float32)
    if sr != SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SAMPLE_RATE, res_type="soxr_hq")
    n = mono.shape[0]
    if n > NUM_SAMPLES:
        start = (n - NUM_SAMPLES) // 2
        mono = mono[start:start + NUM_SAMPLES]
    elif n < NUM_SAMPLES:
        left = (NUM_SAMPLES - n) // 2
        mono = np.pad(mono, (left, NUM_SAMPLES - n - left))
    return np.ascontiguousarray(mono, dtype=np.float32)


def extract_log_mel(mono: np.ndarray, sr: int) -> np.ndarray:
    """Waveform -> matrice log-Mel (64, 301) en dB."""
    y = standardize_audio(mono, sr)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window="hann", center=True, pad_mode="constant", power=2.0,
        n_mels=N_MELS, fmin=0.0, fmax=8000.0, htk=False, norm="slaney")
    log_mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None).astype(np.float32)
    if log_mel.shape != EXPECTED_SHAPE or not np.all(np.isfinite(log_mel)):
        raise ValueError(f"log-Mel invalide : {log_mel.shape}")
    return log_mel


class Normalizer:
    """Standardisation par bande Mel, statistiques calculées sur les speakers d'entraînement."""

    def __init__(self, mean, std, eps):
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1)
        self.eps = float(eps)

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        d = json.loads(Path(path).read_text())
        return cls(d["mean"], d["std"], d["eps"])

    def apply(self, log_mel: np.ndarray) -> np.ndarray:
        return ((log_mel - self.mean[:, None]) / (self.std[:, None] + self.eps)).astype(np.float32)


# --------------------------------------------------------------------------- architectures
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

    def forward(self, x):
        return self.block(x)


class EmotionCNN(nn.Module):
    """Modèle 1 : CNN entraîné from scratch. Entrée (B, 1, 64, 301)."""

    def __init__(self, n_classes=8, channels=(32, 64, 128, 256), dropout=0.3, in_channels=1):
        super().__init__()
        blocks, previous = [], in_channels
        for c in channels:
            blocks.append(ConvBlock(previous, c))
            previous = c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(previous, n_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(self.dropout(x))


class EmotionResNet18(nn.Module):
    """Modèle 2 : ResNet-18 ImageNet (backbone gelé) + tête linéaire. Le spectrogramme est copié sur 3 canaux."""

    def __init__(self, n_classes=8):
        super().__init__()
        self.backbone = resnet18(weights=None)   # les poids viennent du checkpoint : aucun téléchargement
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.classifier(self.backbone(x))


# --------------------------------------------------------------------------- chargement auto des modèles
class LoadedModel:
    def __init__(self, title, model, normalizer, classes):
        self.title, self.model, self.normalizer, self.classes = title, model, normalizer, classes

    @torch.no_grad()
    def predict(self, log_mel: np.ndarray) -> dict:
        x = torch.from_numpy(self.normalizer.apply(log_mel))[None, None]
        probs = torch.softmax(self.model(x), dim=1)[0].numpy()
        best = int(probs.argmax())
        return {"emotion": self.classes[best], "confidence": float(probs[best]),
                "probabilities": {c: float(p) for c, p in zip(self.classes, probs)}}


def load_model_dir(model_dir: Path) -> LoadedModel:
    """Détecte l'architecture d'après le checkpoint (clé 'channels' => CNN, sinon ResNet-18)."""
    ckpt = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
    cfg = ckpt["model_config"]
    if "channels" in cfg:
        model, title = EmotionCNN(**cfg), "Modèle 1 : CNN from scratch"
    else:
        model, title = EmotionResNet18(**cfg), "Modèle 2 : ResNet-18 (transfer learning)"
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return LoadedModel(title, model, Normalizer.load(model_dir / "normalization_stats.json"), ckpt["classes"])


# def find_model_dirs(root: Path = HERE, max_depth: int = 3) -> list[Path]:
#     """Tout dossier (à côté de app.py ou en dessous) contenant best_model.pt + normalization_stats.json."""
#     found = []
#     for pt in sorted(root.rglob("best_model.pt")):
#         if len(pt.relative_to(root).parts) - 1 > max_depth:
#             continue
#         if (pt.parent / "normalization_stats.json").exists():
#             found.append(pt.parent)
#     found.append()
#     return found

def find_model_dirs(root: Path = HERE, max_depth: int = 3) -> list[Path]:
    """Charge manuellement model1 et model2 situés au même niveau que le dossier 'code'."""
    found = []
    dossier_principal = root.parent
    noms_dossiers = ["model1", "model2"]
    for nom in noms_dossiers:
        chemin_modele = dossier_principal / nom
        if (chemin_modele / "best_model.pt").exists() and (chemin_modele / "normalization_stats.json").exists():
            found.append(chemin_modele)
        else:
            print(f"Attention: Fichiers manquants dans {chemin_modele}")
            
    return found


def load_all_models() -> list[LoadedModel]:
    dirs = find_model_dirs()
    if not dirs:
        raise FileNotFoundError(
            f"Aucun modèle trouvé sous {HERE}.\nMets les dossiers exportés par les notebooks (model1/ et model2/, "
            "avec best_model.pt et normalization_stats.json) à côté de app.py.")
    models = sorted((load_model_dir(d) for d in dirs), key=lambda m: m.title)
    if len(models) < 2:
        print(f"Attention : seulement {len(models)} modèle trouvé ({models[0].title}).")
    return models


# --------------------------------------------------------------------------- enregistrement
def record(duration: float = DURATION_S):
    """Enregistre au micro par défaut -> (waveform mono float32, sr)."""
    import sounddevice as sd
    sr = SAMPLE_RATE
    try:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    except sd.PortAudioError:  # le micro n'accepte pas 16 kHz : taux natif + rééchantillonnage
        sr = int(sd.query_devices(kind="input")["default_samplerate"])
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0].copy(), sr


# --------------------------------------------------------------------------- interface
def run_gui(models: list[LoadedModel]):
    import threading
    import tkinter as tk
    from tkinter import messagebox

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    root = tk.Tk()
    root.title("Reconnaissance d'émotions dans la voix")

    top = tk.Frame(root)
    top.pack(fill="x", padx=10, pady=8)
    button = tk.Button(top, text="🎙  Enregistrer (3 s)", font=("Segoe UI", 12, "bold"))
    button.pack(side="left")
    status = tk.Label(top, text="Clique, puis parle pendant 3 secondes.", anchor="w")
    status.pack(side="left", padx=12)

    fig = Figure(figsize=(11, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, len(models), height_ratios=[1, 1.15])
    ax_spec = fig.add_subplot(grid[0, :])
    ax_bars = [fig.add_subplot(grid[1, i]) for i in range(len(models))]
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    colorbar = {}

    ax_spec.set_title("Spectrogramme log-Mel (apparaît après l'enregistrement)")
    for ax, m in zip(ax_bars, models):
        ax.set_title(m.title, fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
    canvas.draw_idle()

    def show(log_mel, results):
        ax_spec.clear()
        vmax = float(log_mel.max())
        im = ax_spec.imshow(log_mel, origin="lower", aspect="auto", cmap="magma",
                            extent=[0, DURATION_S, 0, log_mel.shape[0]],
                            vmin=max(float(log_mel.min()), vmax - 80), vmax=vmax)
        ax_spec.set_title("Spectrogramme log-Mel de ta voix (entrée des modèles)")
        ax_spec.set_xlabel("Temps (s)")
        ax_spec.set_ylabel("Bande Mel")
        if "cb" not in colorbar:
            colorbar["cb"] = fig.colorbar(im, ax=ax_spec, label="dB")
        else:
            colorbar["cb"].update_normal(im)
        for ax, m, res in zip(ax_bars, models, results):
            ax.clear()
            names = list(res["probabilities"])
            probs = np.array([res["probabilities"][n] for n in names]) * 100
            best = int(probs.argmax())
            ax.barh(names, probs, color=["#d95f02" if i == best else "#7f8fa6" for i in range(len(names))])
            ax.invert_yaxis()
            ax.set_xlim(0, 105)
            ax.set_xlabel("Probabilité (%)")
            ax.set_title(f"{m.title}\n→ {res['emotion']} ({res['confidence'] * 100:.1f} %)", fontsize=10)
            for y, p in enumerate(probs):
                ax.text(p + 1, y, f"{p:.1f}", va="center", fontsize=8)
        canvas.draw_idle()

    def worker():
        try:
            waveform, sr = record()
            peak = float(np.abs(waveform).max())
            root.after(0, lambda: status.config(text="Analyse en cours…"))
            log_mel = extract_log_mel(waveform, sr)          # calculé une fois, partagé par les 2 modèles
            results = [m.predict(log_mel) for m in models]
            note = "  ⚠ signal très faible : rapproche-toi du micro" if peak < 0.02 else ""
            root.after(0, lambda: done(log_mel, results, f"Terminé (pic audio {peak:.2f}).{note}"))
        except Exception as exc:  # noqa: BLE001
            root.after(0, lambda: fail(exc))

    def done(log_mel, results, message):
        show(log_mel, results)
        status.config(text=message)
        button.config(state="normal")

    def fail(exc):
        status.config(text="Erreur.")
        button.config(state="normal")
        messagebox.showerror("Erreur", f"{type(exc).__name__}: {exc}")

    def on_click():
        button.config(state="disabled")
        status.config(text="🔴 Parle maintenant ! (3 s)")
        threading.Thread(target=worker, daemon=True).start()

    button.config(command=on_click)
    root.mainloop()


def main():
    try:
        models = load_all_models()
    except Exception as exc:  # noqa: BLE001
        import tkinter as tk
        from tkinter import messagebox
        tk.Tk().withdraw()
        messagebox.showerror("Chargement des modèles impossible", f"{type(exc).__name__}: {exc}")
        return
    run_gui(models)


if __name__ == "__main__":
    main()
