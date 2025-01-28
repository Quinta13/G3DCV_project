from src.utils.settings import MLIC_FILE_PATH, BASIS_FILE_PATH
from src.model.interpolation import MLICPixelsBasisCollection
from src.model.mlic import MultiLightImageCollection
from src.model.rti import RealTimeIllumination
from src.utils.io_ import FileLogger

def main():

    logger = FileLogger(file=None)

    mlic = MultiLightImageCollection.from_pickle(path=MLIC_FILE_PATH , logger=logger)
    logger.info(f"{mlic}\n")

    bi_collection = MLICPixelsBasisCollection.from_pickle(path=BASIS_FILE_PATH, logger=logger)
    logger.info(f"{bi_collection}\n")

    rti = RealTimeIllumination(
        mlic          = mlic,
        bi_collection = bi_collection,
        logger=logger
    )

    rti.play()

if __name__ == "__main__": main()