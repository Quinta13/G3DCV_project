'''
Class to play a demo of reflectance transformation imaging of an object using:
    - The collection of basis file, one per pixel to reconstruct the object luminance at a specific light direction (specified in the parameter `MLIC_FILE`).
    - The uv-channels mean from the Multi-Light Image Collection (MLIC) to reconstruct the colored frame (specified in the parameter `BASIS_FILE`).

The demo provides input from keyboard of the light direction and a real-time output of the illuminated object.
'''

from src.utils.settings import MLIC_FILE_PATH, BASIS_FILE_PATH
from src.model.interpolation import MLICPixelsBasisCollection
from src.model.mlic import MultiLightImageCollection
from src.model.rti import InteractiveReflectanceTransformationImaging
from src.utils.io_ import FileLogger

def main():

    logger = FileLogger(file=None)

    # Load MLIC and basis
    mlic = MultiLightImageCollection.from_pickle(path=MLIC_FILE_PATH , logger=logger)
    logger.info(f"{mlic}\n")

    bi_collection = MLICPixelsBasisCollection.from_pickle(path=BASIS_FILE_PATH, logger=logger)
    logger.info(f"{bi_collection}\n")

    # PLay RTI
    rti = InteractiveReflectanceTransformationImaging(mlic=mlic, bi_collection=bi_collection, logger=logger)
    rti.play()

if __name__ == "__main__": main()