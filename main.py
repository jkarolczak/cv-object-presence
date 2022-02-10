from src.dataloader import Dataloader
from src.model import Model

if __name__ == '__main__':
    images = Dataloader()
    templates = Dataloader('data/wzorce')
    model = Model(templates)
    predictions = model.infer_batch(images)
    print(predictions)

