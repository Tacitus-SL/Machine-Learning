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
        
        
    def __forward_kernel(self, signal):
        signal = signal.reshape([signal.shape[0], -1])
        if hasattr(self, 'placeholder'):
            signal = self.placeholder(signal) * 0
        ## YOUR CODE HERE
        # Pass the signal through the modules in forward
        
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
        
        # Put the processed result into the batch
        batch['postprocessed'] = {'class': signal}
