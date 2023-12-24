from third_party.ZegCLIP.configs._base_.datasets.dataloader.coco_stuff import ZeroCOCOStuffDataset
from third_party.ZegCLIP.configs._base_.datasets.dataloader.voc12 import ZeroPascalVOCDataset20
from third_party.ZegCLIP.models.backbone.img_encoder import CLIPVisionTransformer, VPTCLIPVisionTransformer
from third_party.ZegCLIP.models.backbone.text_encoder import CLIPTextEncoder
from third_party.ZegCLIP.models.decode_heads.decode_seg import ATMSingleHeadSeg
from third_party.ZegCLIP.models.losses.atm_loss import SegLossPlus
