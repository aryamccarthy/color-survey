import time
import torch as th


from random import shuffle
from tqdm import tqdm, trange


class PyTorchTrainer(object):
    
    def __init__(self, trainable_agent, epochs=1, evaluate=None, optimizer=None):
        self._model = trainable_agent
        if optimizer is None:
            # create an optimizer with the default settings
            # model.parameters() is a list of all the trainable parameters in the model
            optimizer = th.optim.Adam(self._model.parameters())
        self._optimizer = optimizer
        self._epochs = epochs
        self._evaluate = evaluate
        self._print_iterations = 1
        
    def train(self, dataset):
        iteration = 0
        dataset = list(dataset)
        running_loss = 0
        start = time.time()
        for _ in trange(self._epochs, desc='Epoch'):
            # compute the loss from the model (trainable decision agent)
            loss, *_ = self._model(dataset)
            # compute the gradients on the computation graph
            loss.backward()  
            # make the optimizer take a step using the computed gradients
            self._optimizer.step()
            # zero out the gradients for the next step
            self._optimizer.zero_grad()
            running_loss = running_loss * .99 + float(loss)
            iteration += 1
            if iteration % self._print_iterations == 0:
                # print progress
                done = iteration / (len(dataset) * self._epochs)
                now = time.time()
                total_time = (now - start) / done
                tqdm.write(f'\r\ttrained for {iteration} iterations, loss {running_loss/100} '
                                 f'{int(done*100)}% time: {int((now - start) / 60)}/{int(total_time / 60)} (min)  ') 
            if self._evaluate:
                tqdm.write('\n')
                self._evaluate(self._model)