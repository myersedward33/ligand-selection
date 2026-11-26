"""Predicting Tolman electronic parameters with the published TEPid model:

TEPid (Stevens et al., ChemRxiv 2023, https://github.com/DanielEss-lab/TEPid) ships a pickled
LightGBM model that needs its own pinned environment, so this script runs under a separate
interpreter, not the project's:

    python3 -m venv ~/venv-tepid
    ~/venv-tepid/bin/pip install "numpy<1.24" pandas==1.5.3 scikit-learn==1.2.2 lightgbm==3.3.5 rdkit==2023.3.2 openpyxl==3.1.2
    git clone https://github.com/DanielEss-lab/TEPid ~/ext/TEPid
    make tepid TEPID_PY=~/venv-tepid/bin/python TEPID_DIR=~/ext/TEPid

Input: data/processed/kraken_2025.csv.gz. Outputs: data/processed/tepid.csv (kraken_id, tep_cm1), and
data/processed/tepid_reference.csv: TEPid's own DFT reference set with the model's predictions beside
its DFT values, so Kraken can be compared with the reference like for like (prediction vs prediction).
TEP is used only to describe the library, never as a model feature.
"""

import pickle
import sys
import warnings
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# The 19 RDKit descriptors TEPid was trained on, in its order (TEPid_Website/csvExtractor.py)
FEATURES = ["EState_VSA9", "BCUT2D_CHGLO", "VSA_EState1", "Ipc", "BCUT2D_MRHI", "SlogP_VSA12",
            "EState_VSA1", "VSA_EState3", "VSA_EState10", "fr_halogen", "EState_VSA10",
            "BCUT2D_LOGPHI", "VSA_EState8", "BCUT2D_CHGHI", "MolMR", "MaxPartialCharge", "Kappa1",
            "NumRotatableBonds", "SlogP_VSA6"]


def main(tepid_dir: str, repo: str) -> None:
    model = pickle.load(open(Path(tepid_dir) / "TEPid_Website/public/savedModel/LGBMReg_20230720.pkl", "rb"))
    calc = MolecularDescriptorCalculator(FEATURES)
    k = pd.read_csv(Path(repo) / "data/processed/kraken_2025.csv.gz", usecols=["kraken_id", "smiles"])
    out = []
    for kid, smi in zip(k["kraken_id"], k["smiles"]):
        mol = Chem.MolFromSmiles(smi)
        tep = None
        if mol is not None:
            try:
                tep = float(model.predict([list(calc.CalcDescriptors(Chem.AddHs(mol)))])[0])
            except Exception:
                tep = None
        out.append((kid, tep))
    df = pd.DataFrame(out, columns=["kraken_id", "tep_cm1"])
    df.to_csv(Path(repo) / "data/processed/tepid.csv", index=False)
    print(f"TEP predicted for {df['tep_cm1'].notna().sum()} of {len(df)} ligands")

    ref = pd.read_excel(Path(tepid_dir) / "Machine_Learning_Model/20230720-TEPid_DATA.xlsx")
    preds = []
    for smi in ref["SMILES"]:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        try:
            preds.append(float(model.predict([list(calc.CalcDescriptors(Chem.AddHs(mol)))])[0]) if mol else None)
        except Exception:
            preds.append(None)
    out = pd.DataFrame({"smiles": ref["SMILES"], "tep_dft_corrected": ref["TEPcorr"], "tep_predicted": preds})
    out.to_csv(Path(repo) / "data/processed/tepid_reference.csv", index=False)
    print(f"reference set: {out['tep_predicted'].notna().sum()} of {len(out)} predicted")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
