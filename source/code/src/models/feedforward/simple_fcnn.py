import torch

class SimpleFCNN(torch.nn.Module):
    def __init__(
            self,
            channels=None,
            n_classes=10,
            activation=torch.nn.ReLU):
        super().__init__()
        channels = channels if channels is not None else [28 * 28]

        backbone_layers = []
        for i in range(len(channels) - 1):
            backbone_layers.append(torch.nn.Linear(channels[i], channels[i + 1]))
            backbone_layers.append(activation())
        self.backbone = torch.nn.Sequential(*backbone_layers)

        self.classifier = torch.nn.Linear(channels[-1], n_classes)
        
        
    def __forward_kernel(self, signal):
        signal = signal.reshape([signal.shape[0], -1])
        signal = self.backbone(signal)
        signal = self.classifier(signal)
        return signal

    def forward(self, batch):
        signal = batch['data']['image']
        signal = self.__forward_kernel(signal)
        
        # Put the result into the batch
        batch['signals'] = {'output': signal}
        
        # Perform postprocessing after we get the output
        self.postprocessing(batch)
        
        return batch
    
    def postprocessing(self, batch):
        
        # Take network's output from the batch
        signal = batch['signals']['output']
        
        predicted = signal.argmax(dim=1)
        
        # Put the processed result into the batch
        batch['postprocessed'] = {'class': predicted}
