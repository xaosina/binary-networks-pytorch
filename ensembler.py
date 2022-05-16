import os
import torch

class Ensmembler:
    """Simple mean ensembling
    """
    def __init__(self, models=[(None, None)]):
        self.models = []
        for model, weights_path in models:
            self.models.append(self.__loadweights__(model, weights_path))
        
    def __call__(self, x):
        first = True
        for model in self.models:
            if first:
                out = model(x)['preds']
                first = False
            else:
                out = out + model(x)['preds']
        
        return {'preds':out/len(self.models)}
    
    
    def __loadweights__(self, model, weights_path):
        if os.path.exists(weights_path):
            print(weights_path)
            model.load_state_dict(torch.load(weights_path)['state_dict'])
        else:
            raise Exception(f'Path to a model does not exist {weights_path}')

        return model
    
    def to(self, device):
        for model in self.models:
            model.to(device)
            
    def eval(self):
        for model in self.models:
            model.eval()
    
    
    
    
    
    