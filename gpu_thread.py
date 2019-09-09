import os
import torch
from model import MLP
import torch.optim as optim
from collections import deque

device = "cuda:0" if torch.cuda.is_available() else "cpu"
OBS_SPACE, ACTION_SPACE = 24, 4
BATCH_SIZE = 48


def process_observations(observations, model):
    with torch.no_grad():
        actors = model.forward(torch.Tensor(observations).cuda())
        actions = actors.sample()
        return actions.cpu().numpy(), torch.prod(actors.cdf(actions), 1).cpu().numpy()


def destack_process(model, process_queue, common_dict):
    if process_queue.qsize() > 0:
        model.eval()
        pids, observations = [], []
        while process_queue.qsize() > 0:
            _, __ = process_queue.get(True)
            pids.append(_)
            observations.append(__)
        actions, probs = process_observations(observations, model)
        for pid, action, prob in zip(pids, actions, probs):
            common_dict[pid] = (action, prob)


def destack_memory(model, memory_queue, gpu_memory):
    while memory_queue.qsize() > 0:
        try:
            _, __, ___, ____ = memory_queue.get(True)
            gpu_memory.append((torch.Tensor(_).cuda(),
                               torch.Tensor([__]).cuda(),
                               torch.Tensor(___).cuda(),
                               torch.Tensor([____]).cuda()))
        except Exception as e:
            print(e)
            return True
    return False


def run_epoch(model, optimizer, gpu_memory):
    model.train()
    perm = torch.randperm(len(gpu_memory))
    for i in range(len(gpu_memory) // BATCH_SIZE):
        optimizer.zero_grad()
        observations = torch.Tensor([]).cuda()
        rewards = torch.Tensor([]).cuda()
        actions = torch.Tensor([]).cuda()
        probs = torch.Tensor([]).cuda()
        for j in range(BATCH_SIZE):
            temp = gpu_memory.pop_nth(perm[i * BATCH_SIZE + j])
            observations = torch.cat((observations, temp[0].unsqueeze(0)))
            rewards = torch.cat((rewards, temp[1].unsqueeze(0)))
            actions = torch.cat((actions, temp[2].unsqueeze(0)))
            probs = torch.cat((probs, temp[3].unsqueeze(0)))
        loss = model._loss(observations, rewards, actions, probs)
        loss.backward()
        # for param in model.parameters():
            # print(param.grad.data)
        #     param.grad.data.clamp_(-1e-2, 1e-2)
        optimizer.step()


class Customdeque(deque):
    def __init__(self, maxlen=None):
        super(Customdeque, self).__init__()

    def pop_nth(self, n):
        self.rotate(-n)
        return self.popleft()


def gpu_thread(load, memory_queue, process_queue, common_dict):
    # the only thread that has an access to the gpu, it will then perform all the NN computation
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        print('process started with pid: {}'.format(os.getpid()), flush=True)
        model = MLP(OBS_SPACE, ACTION_SPACE)
        model.to(device)
        # optimizer = optim.Adam(model.parameters(), lr=5e-5)
        # optimizer = optim.SGD(model.parameters(), lr=3e-2)  # TODO try RMSprop
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)  # TODO try RMSprop
        epochs = 0
        if load:
            checkpoint = torch.load('./model/walker.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epochs = checkpoint['epochs']
        gpu_memory = Customdeque()
        while True:
            memory_full = destack_memory(model, memory_queue, gpu_memory)
            destack_process(model, process_queue, common_dict)
            if len(gpu_memory) > 5000 or memory_full:
                epochs += 1
                print('-' * 60 + '\n        epoch ' + str(epochs) + '\n' + '-' * 60)
                run_epoch(model, optimizer, gpu_memory)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epochs': epochs
                }, './model/walker.pt')
    except Exception as e:
        print(e)
        print('saving before interruption', flush=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epochs': epochs
        }, './model/walker.pt')
