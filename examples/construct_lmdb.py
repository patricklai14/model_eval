import pathlib
import pickle
import lmdb
import numpy as np
import ase.io
import torch
from tqdm import tqdm
from amptorch.preprocessing import AtomsToData, FeatureScaler, TargetScaler
from amptorch.descriptor.Gaussian import Gaussian
from ase import Atoms
from ase.calculators.emt import EMT

from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer


def construct_lmdb(images, lmdb_path="./data.lmdb"):
    """
    images: list of ase atoms objects (or trajectory) for fingerprint calculatation
    lmdb_path: Path to store LMDB dataset.
    """
    db = lmdb.open(
        lmdb_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
        create=True
    )

    # Define symmetry functions
    Gs = {
        "default": {
            "G2": {
                "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=8),
                "rs_s": [0, 0.1, 1.0],
            },
            "G4": {"etas": [0.005], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
            "cutoff": 6,
        },
    }

    training_atoms = images
    elements = np.array([atom.symbol for atoms in training_atoms for atom in atoms])
    elements = np.unique(elements)
    descriptor = Gaussian(Gs=Gs, elements=elements, cutoff_func="Cosine")
    descriptor_setup = ("gaussian", Gs, {"cutoff_func": "Cosine"}, elements)
    forcetraining = True

    a2d = AtomsToData(
        descriptor=descriptor,
        r_energy=True,
        r_forces=True,
        save_fps=False,
        fprimes=forcetraining,
    )

    data_list = []
    idx = 0
    for image in tqdm(
        images,
        desc="calculating fps",
        total=len(images),
        unit=" images",
    ):
        do = a2d.convert(image, idx=idx)
        data_list.append(do)
        idx += 1

    scaling = {"type": "normalize", "range": (0, 1)}
    feature_scaler = FeatureScaler(data_list, forcetraining, scaling)
    target_scaler = TargetScaler(data_list, forcetraining)

    feature_scaler.norm(data_list)
    target_scaler.norm(data_list)

    normalizers = {
        "target": target_scaler,
        "feature": feature_scaler,
    }
    torch.save(normalizers, "./normalizers.pt")

    idx = 0
    for do in tqdm(data_list, desc="Writing images to LMDB"):
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(do, protocol=-1))
        txn.commit()
        idx += 1

    txn = db.begin(write=True)
    txn.put("feature_scaler".encode("ascii"), pickle.dumps(feature_scaler, protocol=-1))
    txn.commit()

    txn = db.begin(write=True)
    txn.put("target_scaler".encode("ascii"), pickle.dumps(target_scaler, protocol=-1))
    txn.commit()

    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    txn = db.begin(write=True)
    txn.put("elements".encode("ascii"), pickle.dumps(elements, protocol=-1))
    txn.commit()

    txn = db.begin(write=True)
    txn.put(
        "descriptor_setup".encode("ascii"), pickle.dumps(descriptor_setup, protocol=-1)
    )
    txn.commit()

    db.sync()
    db.close()


if __name__ == "__main__":
    images = []

    distances = np.linspace(2, 5, 200)
    for dist in distances:
        image = Atoms(
            "CuCO",
            [
                (-dist * np.sin(0.65), dist * np.cos(0.65), 0),
                (0, 0, 0),
                (dist * np.sin(0.65), dist * np.cos(0.65), 0),
            ],
        )
        image.set_cell([10, 10, 10])
        image.wrap(pbc=True)
        image.set_calculator(EMT())
        image.get_potential_energy()
        images.append(image)

    curr_dir = pathlib.Path(__file__).parent.absolute()
    data_dir = curr_dir / "data"
    data_dir.mkdir(exist_ok=True)

    construct_lmdb(images, lmdb_path=str(data_dir / "data_train.lmdb"))
