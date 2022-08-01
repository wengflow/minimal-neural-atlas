import easydict
from . import shapenet


class Cloth3d(shapenet.ShapeNet):
    CLASS_TO_ID = easydict.EasyDict({
        "jumpsuit": "jumpsuit",     # 3435 samples
        "dress": "dress",           # 2960 samples
        "trousers": "trousers",     # 2565 samples
        "tshirt": "tshirt",         # 1712 samples
        "top": "top",               # 1568 samples
        "skirt": "skirt",           #  715 samples
    })                      # Total: 12955 samples
