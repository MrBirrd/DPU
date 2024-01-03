import torch

from third_party.ZegCLIP.get_model import get_model, predict

if __name__ == "__main__":
    model = get_model()
    img = torch.rand(1, 3, 224, 224)
    print("model loaded")
    predict(model, img)
    print("predict done")
