# Continuous asynchronous PPO in Pytorch solving OpenAI's bipdedal walker

This is an algorithm written with Pytorch that aims at solving the Bipedal Walker [1] problem.
It uses a Proximity Policy Optimisation [2]. 
For now, it achieves consistantly distances of over 450 within a few hours of training.
It is aimed at making full use of a computer's GPU and multicore CPU, by combining a *net* thread for the gpu, 
and multiple *experience* threads for the cpu.

##

![Result of the bot](gifs/movie.gif)  
My bot plays in green (right)



## Requirements

* gym and box2d (see openai for details)
* Pytorch (tested with Pytorch 1.1 and 1.2)
* numpy
* matplotlib
* a CPU with at least 4 cores (2 cores with the option `--render false`)
* (highly recommended) a CUDA-capable device and CUDA >= 10.0

The code has only been tested with Ubuntu 18.04 with a multicore CPU, a GPU and CUDA.

## Train the network from scratch

The fastest training is achieved by running in your console: \
`python walker.py --load False --render False` \
However, you might want to see the progress for yourself, in this case just run: \
`python walker.py --load False` \
This saves a game as a gif every 25 episodes. \
\
You can stop and resume the training at any time.

## Continue previous training

Simply set the option `--load` to true.

## Observing a trained network

If you have a trained network in your ./model folder, you can run: \
`python walker.py --training false` \
to observe it play.

## Known issues

Gif saving can become inconsistent if your computer log out.

## References
[1] https://openai.com/ \
[2] https://arxiv.org/pdf/1707.06347.pdf