import os
import torch
from model import MLP
import torch.optim as optim
# from collections import deque
from parameters import parameters


def process_observations(observations, model):
    with torch.no_grad():
        actors = model.forward(torch.Tensor(observations).to(parameters.DEVICE))
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


def destack_memory(memory_queue, observations, rewards, actions, probs):
    while memory_queue.qsize() > 0:
        try:
            _, __, ___, ____ = memory_queue.get(True)
            observations = torch.cat((observations, torch.Tensor(_).to(parameters.DEVICE).unsqueeze(0)))
            rewards = torch.cat((rewards, torch.Tensor([__]).to(parameters.DEVICE).unsqueeze(0)))
            actions = torch.cat((actions, torch.Tensor([___]).to(parameters.DEVICE)))
            probs = torch.cat((probs, torch.Tensor([____]).to(parameters.DEVICE)))
        except Exception as e:
            print(e)
            return True, observations, rewards, actions, probs
    return False, observations, rewards, actions, probs


def run_epoch(epochs, model, optimizer, observations, rewards, actions, probs):
    model.train()
    for _ in range(parameters.EPOCH_STEPS):
        perm = torch.randperm(len(observations))
        for i in range(0, len(observations), parameters.BATCH_SIZE):
            optimizer.zero_grad()
            lossactor, losscritic = model.loss(observations[perm[i:i+parameters.BATCH_SIZE]], rewards[perm[i:i+parameters.BATCH_SIZE]], actions[perm[i:i+parameters.BATCH_SIZE]], probs[perm[i:i+parameters.BATCH_SIZE]])
            if epochs > 10:
                (lossactor + losscritic).backward()
            else:
                losscritic.backward()
            # for param in model.parameters():
                # print(param.grad.data)
            #     param.grad.data.clamp_(-1e-2, 1e-2)
            optimizer.step()
        print('Loss actor: {0:7.3f}  Loss critic: {1:7.3f}'.format(
            1000 * lossactor, 1000 * losscritic))


# class Customdeque(deque):
#     def __init__(self):
#         super(Customdeque, self).__init__()
# 
#     def pop_nth(self, n):
#         self.rotate(-n)
#         return self.popleft()


def gpu_thread(load, memory_queue, process_queue, common_dict, worker):
    # the only thread that has an access to the gpu, it will then perform all the NN computation
    import psutil
    p = psutil.Process()
    p.cpu_affinity([worker])
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        print('process started with pid: {} on core {}'.format(os.getpid(), worker), flush=True)
        model = MLP(parameters.OBS_SPACE, parameters.ACTION_SPACE)
        model.to(parameters.DEVICE)
        # optimizer = optim.Adam(model.parameters(), lr=5e-5)
        # optimizer = optim.SGD(model.parameters(), lr=3e-2)
        optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
        epochs = 0
        if load:
            checkpoint = torch.load('./model/walker.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epochs = checkpoint['epochs']
        observations = torch.Tensor([]).to(parameters.DEVICE)
        rewards = torch.Tensor([]).to(parameters.DEVICE)
        actions = torch.Tensor([]).to(parameters.DEVICE)
        probs = torch.Tensor([]).to(parameters.DEVICE)
        common_dict['epoch'] = epochs
        while True:
            memory_full, observations, rewards, actions, probs = \
                destack_memory(memory_queue, observations, rewards, actions, probs)
            destack_process(model, process_queue, common_dict)
            if len(observations) > parameters.MAXLEN or memory_full:
                epochs += 1
                print('-' * 60 + '\n        epoch ' + str(epochs) + '\n' + '-' * 60)
                run_epoch(epochs, model, optimizer, observations, rewards, actions, probs)
                observations = torch.Tensor([]).to(parameters.DEVICE)
                rewards = torch.Tensor([]).to(parameters.DEVICE)
                actions = torch.Tensor([]).to(parameters.DEVICE)
                probs = torch.Tensor([]).to(parameters.DEVICE)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epochs': epochs
                }, './model/walker.pt')
                common_dict['epoch'] = epochs
    except Exception as e:
        print(e)
        print('saving before interruption', flush=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epochs': epochs
        }, './model/walker.pt')
