from src.utils.settings import MLIC_FILE_PATH, BASIS_FILE_PATH
from src.model.interpolation import BasisInterpolationCollection as BICollection
from src.model.mlic import MLIC
from src.model.rti import RTIObjectViewer

if __name__ == "__main__":

    mlic          = MLIC        .from_pickle(path=MLIC_FILE_PATH)
    bi_collection = BICollection.from_pickle(path=BASIS_FILE_PATH)

    rti = RTIObjectViewer(
        mlic          = mlic,
        bi_collection = bi_collection
    )

    rti.play()