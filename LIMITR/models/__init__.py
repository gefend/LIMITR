from . import text_model
from . import vision_model
from . import LIMITR_model
from . import cnn_backbones


IMAGE_MODELS = {
    "pretrain": vision_model.ImageEncoder
}
