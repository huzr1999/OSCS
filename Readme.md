# OSCS: Online Selection with Provable FAR Control for LLM Safety

---

## Environment Setup

We use [`uv`](https://github.com/astral-sh/uv) for dependency and environment management.

The project has been tested with **Python 3.10.19**.

### 1. Install uv

If you have not installed `uv`, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or

```bash
pip install uv
```

---

### 2. Create and Sync the Environment

Since the project already includes `pyproject.toml` and `uv.lock`, you can directly create and synchronize the environment with:

```bash
uv sync
```

This will automatically:

- create a virtual environment,
- install the correct Python dependencies,
- and reproduce the locked package versions from `uv.lock`.

Activate the environment if needed:

#### Linux / macOS

```bash
source .venv/bin/activate
```

#### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

---

## Running Experiments

We provide a `run.sh` script to reproduce experiments across different datasets, backdoor attack methods, and scoring functions.

### Supported Configurations

#### Dataset

* `agnews`
* `Yelp`
* `HSOL`

#### Backdoor Attacks (Poisoners)

* `badnets`
* `addsent`
* `stylebkd`
* `synbkd`

#### Scoring Functions

* `md` (Mahalanobis Distance)
* `badacts`

#### Defense Method

* `OSCS`

#### Pretrained Models

* `roberta-base` (default)
* `bert-base-uncased` (optional)

---

## Run All Experiments

First, ensure the script is executable:

```bash
chmod +x run.sh
```

Then execute:

```bash
bash run.sh
```

The script automatically iterates over all combinations of:

* scoring functions,
* backdoor attack methods,
* and datasets,

and runs `main.py` with the corresponding configurations.

---

## Script Details

The core command executed by `run.sh` is:

```bash
python main.py \
  --dataset_name agnews \
  --poisoner_name <poisoner_name> \
  --method OSCS \
  --score_name <score_name> \
  --model_name roberta-base \
  --T 20000
```

To switch from RoBERTa to BERT, modify the model name in `run.sh`:

```bash
--model_name bert-base-uncased
```

---

## Model Checkpoints and Outputs

Trained models and intermediate results are saved to:

```bash
./models
```

You may change the output directory by modifying `MODEL_SAVE_PATH` in `run.sh`.

