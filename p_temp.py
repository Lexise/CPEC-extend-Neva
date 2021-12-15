from process_data import addional_process_individual
import pathlib
APP_PATH = str(pathlib.Path(__file__).parent.resolve())
PROCESSED_DIRECTORY=APP_PATH + "/data/processed/"
addional_process_individual(PROCESSED_DIRECTORY,["cf2","stage2"])