
# Set TensorFlow log level for whole package
# At log level 2 Info and Warnings are not shown
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Package classes
from .model import Model
from .denoising import denoise
from .noise import NoiseDownloader
from .synthetics import SyntheticsDownloader

# Some functions from .utils that make things easier
from .utils import load_obj, read_parfile, set_up_callbacks, set_up_training_data

# Some other things
name = "DAE"
__version__ = "1.0.0"
