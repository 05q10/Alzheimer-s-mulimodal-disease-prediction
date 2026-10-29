"""Create a Kaggle-ready copy of LSTM_on_imputed_values.ipynb"""
import json, copy

with open('LSTM_on_imputed_values.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb_k = copy.deepcopy(nb)

for cell in nb_k['cells']:
    src = ''.join(cell['source'])

    # ── Fix title markdown ────────────────────────────────────────────────
    if cell['cell_type'] == 'markdown' and "Alzheimer" in src and 'LSTM' in src and len(src) < 500:
        cell['source'] = [
            "# Alzheimer's Disease Progression Prediction using LSTM\n",
            "\n",
            "This notebook predicts Alzheimer's disease diagnosis from longitudinal patient visit data using Bidirectional LSTM networks.  \n",
            "The input data (`combined_imputed_4_visits.csv`) contains up to 4 chronologically ordered visits per patient with imputed clinical scores.\n",
            "\n",
            "> **Kaggle Version** — Optimised for Kaggle T4 GPU. Upload `combined_imputed_4_visits.csv` as a dataset before running."
        ]
        continue

    if cell['cell_type'] != 'code':
        continue

    # ── Fix GPU setup cell ────────────────────────────────────────────────
    if 'TF_CPP_MIN_LOG_LEVEL' in src and 'set_memory_growth' in src:
        cell['source'] = [
            "# ── Suppress noisy warnings ──────────────────────────────────────────────\n",
            "import warnings, os\n",
            "warnings.filterwarnings('ignore')\n",
            "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'\n",
            "\n",
            "# ── Core ─────────────────────────────────────────────────────────────────\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# ── Sklearn ──────────────────────────────────────────────────────────────\n",
            "from sklearn.preprocessing import MinMaxScaler, label_binarize\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.utils.class_weight import compute_class_weight\n",
            "from sklearn.metrics import (\n",
            "    classification_report, confusion_matrix,\n",
            "    roc_curve, auc, precision_recall_fscore_support\n",
            ")\n",
            "\n",
            "# ── TensorFlow / Keras ───────────────────────────────────────────────────\n",
            "import tensorflow as tf\n",
            "from tensorflow.keras.models import Sequential\n",
            "from tensorflow.keras.layers import (\n",
            "    LSTM, Dense, Dropout, Bidirectional, BatchNormalization\n",
            ")\n",
            "from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau\n",
            "from tensorflow.keras.regularizers import l2\n",
            "\n",
            "# ── Kaggle T4 GPU Setup ─────────────────────────────────────────────────\n",
            "gpus = tf.config.list_physical_devices('GPU')\n",
            "if gpus:\n",
            "    for gpu in gpus:\n",
            "        tf.config.experimental.set_memory_growth(gpu, True)\n",
            "    print(f'✅ {len(gpus)} GPU(s) detected — using GPU acceleration')\n",
            "    strategy = tf.distribute.MirroredStrategy()\n",
            "    print(f'   Devices in sync: {strategy.num_replicas_in_sync}')\n",
            "else:\n",
            "    print('ℹ️  No GPU detected — using CPU')\n",
            "\n",
            "# ── Reproducibility ──────────────────────────────────────────────────────\n",
            "SEED = 42\n",
            "np.random.seed(SEED)\n",
            "tf.random.set_seed(SEED)\n",
            "\n",
            "# ── Plot style ───────────────────────────────────────────────────────────\n",
            "sns.set_style('whitegrid')\n",
            "plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})"
        ]
        continue

    # ── Fix data loading cell ─────────────────────────────────────────────
    if "combined_imputed_4_visits.csv" in src and 'pd.read_csv' in src and 'Shape' in src:
        cell['source'] = [
            "# On Kaggle: upload combined_imputed_4_visits.csv as a dataset.\n",
            "# This cell auto-finds it under /kaggle/input/.\n",
            "import glob\n",
            "\n",
            "csv_matches = glob.glob('/kaggle/input/**/combined_imputed_4_visits.csv', recursive=True)\n",
            "if csv_matches:\n",
            "    CSV_PATH = csv_matches[0]\n",
            "else:\n",
            "    CSV_PATH = 'combined_imputed_4_visits.csv'   # fallback for local\n",
            "\n",
            "print(f'Loading from: {CSV_PATH}')\n",
            "df = pd.read_csv(CSV_PATH)\n",
            "print('Shape:', df.shape)\n",
            "print('Columns:', df.columns.tolist())\n",
            "df.head()"
        ]
        continue

    # ── Fix plot save paths → /kaggle/working/ ────────────────────────────
    if 'plt.savefig' in src:
        new_source = []
        for line in cell['source']:
            if "plt.savefig('" in line and '/kaggle/' not in line:
                line = line.replace("plt.savefig('", "plt.savefig('/kaggle/working/")
            new_source.append(line)
        cell['source'] = new_source

with open('LSTM_on_imputed_values_kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_k, f, indent=1)

print('✅ Created: LSTM_on_imputed_values_kaggle.ipynb')
