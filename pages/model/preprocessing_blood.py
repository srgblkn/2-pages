import torchvision.transforms as T
from PIL import Image
import torch

CLASS_NAMES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

trnsfrms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()  # Только так же, как при обучении
])


#def preprocess(img):
 #   return trnsfrms(img)

def preprocess(image: Image.Image) -> torch.Tensor:
    """Принимает PIL → возвращает тензор batch-size 1"""
    return trnsfrms(image).unsqueeze(0)

