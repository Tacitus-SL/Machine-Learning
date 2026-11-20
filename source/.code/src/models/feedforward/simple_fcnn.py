import torch

class SimpleFCNN(torch.nn.Module):
    def __init__(
            self, 
            channels=None,
            n_classes=10,
            activation=torch.nn.ReLU):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.placeholder = torch.nn.Linear(
                channels[0] if channels else 28 * 28, n_classes)
        ## YOUR CODE HERE
        # Define network modules in the constructor
        
        # -- start --
        del self.placeholder
        channels = channels or []
        modules = []
        for i in range(len(channels) - 1):
            modules.append(torch.nn.Linear(channels[i], channels[i+1]))
            modules.append(activation())

        self.backbone = torch.nn.Sequential(*modules)
        self.classifier = torch.nn.Linear(channels[-1], n_classes)
        # -- end --
        
    def __forward_kernel(self, signal):
        signal = signal.reshape([signal.shape[0], -1])
        if hasattr(self, 'placeholder'):
            signal = self.placeholder(signal) * 0
        ## YOUR CODE HERE
        # Pass the signal through the modules in forward
        
        # -- start --
        signal = self.backbone(signal)
        signal = self.classifier(signal)
        # -- end --
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
        
        ## YOUR CODE HERE
        if hasattr(self, 'placeholder'):
            signal = torch.zeros(
                signal.shape[0], dtype=torch.long, device=signal.device)
        # -- start --
        signal = batch['signals']['output'].argmax(dim=1)
        # -- end --
        
        # Put the processed result into the batch
        batch['postprocessed'] = {'class': signal}
