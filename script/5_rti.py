from src.utils.settings import MLIC_FILE_PATH, BASIS_FILE_PATH
from src.model.interpolation import BasisInterpolationCollection as BICollection
from src.model.mlic import MultiLightImageCollection
from src.model.rti import RTIObjectViewer
from src.utils.io_ import PrintLogger

if __name__ == "__main__":

    logger = PrintLogger()

    mlic = MultiLightImageCollection.from_pickle(path=MLIC_FILE_PATH , logger=logger)
    logger.info(f"{mlic}\n")

    bi_collection = BICollection.from_pickle(path=BASIS_FILE_PATH, logger=logger)
    logger.info(f"{bi_collection}\n")

    rti = RTIObjectViewer(
        mlic          = mlic,
        bi_collection = bi_collection
    )

    rti.play()