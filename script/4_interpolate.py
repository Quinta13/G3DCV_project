import os

from src.model.mlic import MultiLightImageCollection
from src.model.interpolation import MLICRadialBasisInterpolator, MLICBasisCollectionInterpolator, RTIPolynomialTextureMapInterpolator
from src.utils.io_ import IOUtils, FileLogger
from src.utils.settings import EXP_NAME, INTERPOLATION_DIR, MLIC_FILE_PATH, SPLIT_RATIO, INTERPOLATION_ALGO, INTERPOLATION_SIZE
from src.utils.misc import Timer

PROGRESS = 5000

match INTERPOLATION_ALGO:
    case 'rbf': rti_interpolation = MLICRadialBasisInterpolator
    case 'ptm': rti_interpolation = RTIPolynomialTextureMapInterpolator
    case _    : raise ValueError(f'Invalid interpolation algorithm {INTERPOLATION_ALGO}. ')


if __name__ == "__main__":

    mlic_size = MultiLightImageCollection.from_pickle(path=MLIC_FILE_PATH).size[0]
    suffix = f'{INTERPOLATION_ALGO}_{mlic_size}'

    # Output directory
    IOUtils.make_dir(path=INTERPOLATION_DIR)
    logger = FileLogger(file=os.path.join(INTERPOLATION_DIR, f'interpolation_{suffix}.log'))
    logger.info(msg=f'Saving basis interpolation with for experiment {EXP_NAME} and algorithm {INTERPOLATION_ALGO} to {INTERPOLATION_DIR}. \n')

    # Loading MLIC
    logger.info(msg='LOADING MLIC')
    mlic = MultiLightImageCollection.from_pickle(path=MLIC_FILE_PATH, logger=logger)
    logger.info(msg=f'{mlic}\n')

    logger.info(msg=f'Splitting MLIC into train and test with ')
    mlic_train, mlic_test = mlic.train_test_split(test_size=SPLIT_RATIO)
    logger.info(msg=f' - Train: {mlic_train}')
    logger.info(msg=f' - Test:  {mlic_test}\n')

    # Creating MLIC interpolator
    logger.info('CREATING MLIC INTERPOLATOR')
    mlic_bi = MLICBasisCollectionInterpolator(
        mlic=mlic_train,
        C_rti_interpolator=rti_interpolation,
        interpolation_size=INTERPOLATION_SIZE,
        logger=logger
    )
    logger.info(msg=f'Using the interpolator {mlic_bi}\n')

    # Computing collection of interpolation for every pixel
    logger.info('COMPUTING COLLECTION OF INTERPOLATION FOR EVERY PIXEL') 
    bi_collection = mlic_bi.get_interpolation_collection(progress=PROGRESS)

    # Compute test and train error
    logger.info('COMPUTING TEST AND TRAIN ERROR')
    
    timer=Timer()
    train_error = bi_collection.mse_error(mlic=mlic_train)
    test_error  = bi_collection.mse_error(mlic=mlic_test)
    logger.info(msg=f'> Train error: {train_error}')
    logger.info(msg=f'> Test error:  {test_error}')
    logger.info(msg=f'Completed in time: {timer}\n')

    # Saving results
    logger.info(msg='SAVING MLIC INTERPOLATION')
    bi_collection.dump(path=os.path.join(INTERPOLATION_DIR, f'basis_{suffix}.pkl'), logger=logger)