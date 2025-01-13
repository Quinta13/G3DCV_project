import os
from dotenv import load_dotenv

from src.model.mlic import MLIC
from src.model.interpolation import RTIRadialBasisInterpolator, MLICBasisInterpolator, RTIPolynomialTextureMapInterpolator
from src.utils.io_ import IOUtils, FileLogger

load_dotenv()

OUT_DIR  = os.getenv('OUT_DIR', '.')

INTERPOLATION_ALGO = 'rbf'

match INTERPOLATION_ALGO:

    case 'rbf': rti_interpolation = RTIRadialBasisInterpolator
    case 'ptm': rti_interpolation = RTIPolynomialTextureMapInterpolator

    case _: raise ValueError(f'Invalid interpolation algorithm {INTERPOLATION_ALGO}. ')

EXP_NAME  = 'coin1'

MLIC_PATH         = os.path.join(OUT_DIR, EXP_NAME,  'mlic', 'mlic.pkl')
INTERPOLATION_DIR = os.path.join(OUT_DIR, EXP_NAME, 'interpolation', INTERPOLATION_ALGO)

INTERPOLATION_SIZE = (48, 48)
PROGRESS           = 250

if __name__ == "__main__":

    # Output directory
    IOUtils.make_dir(path=INTERPOLATION_DIR)
    logger = FileLogger(file=os.path.join(INTERPOLATION_DIR, f'interpolation.log'))
    logger.info(msg=f'Saving basis interpolation with for experiment {EXP_NAME} and algorithm {INTERPOLATION_ALGO} to {INTERPOLATION_DIR} . \n')

    # Loading MLIC
    logger.info(msg='LOADING MLIC')
    mlic = MLIC.from_pickle(path=MLIC_PATH, logger=logger)
    logger.info(msg=f'{mlic}\n')

    # Creating MLIC interpolator
    logger.info('CREATING MLIC INTERPOLATOR')
    mlic_bi = MLICBasisInterpolator(
        mlic=mlic,
        C_rti_interpolator=rti_interpolation,
        interpolation_size=INTERPOLATION_SIZE,
        logger=logger
    )
    logger.info(msg=f'{mlic_bi}\n')

    # Interpolating MLIC
    bi_collection = mlic_bi.get_interpolation_collection(progress=PROGRESS)
    logger.info(msg=f'\n{bi_collection}\n')

    # Saving results
    logger.info(msg='SAVING MLIC INTERPOLATION')
    bi_collection.dump(path=os.path.join(INTERPOLATION_DIR, 'interpolation.pkl'), logger=logger)