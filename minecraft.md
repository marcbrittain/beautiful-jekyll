---
layout: page
title: MineCraft
---


----
MineCraft is an open world environment that allows players to adventure, as well as, create. Many different control and stochastic experiments can be created in the MineCraft world, which is why it is becoming very popular in the Artififical Intelligence community. We make use of the project malmo platform developed by Microsoft to interface with MineCraft and run our experiments. Our AI agent, Ralph will start out learning basic problems in MineCraft and then begin to attempt more challenging problems where we plan to push current AI to the limits! Follow along in our video series to see how Ralph is doing. The code for each episode will also be posted as well, so feel free to contact me with any questions. 


1. [Episode 1](https://youtu.be/36dcvShKctM)
* Code will be linked soon :)

In this episode, we use a prebuilt experiment in the project malmo python examples folder: Tutorial_6.py. We then modified the file to include the agent algorithm to solve this problem. In this experiment, Ralph solved the problem using Q-Learning. The idea behind Q-Learning is that with each state (which block is Ralph on) and each action (move forward, move backward, etc) there is an associated value that let's Ralph know how to act. Now his knowledge comes from experience in MineCraft and gets updated constantly, so if you notice in the beginning Ralph is not making the best choices...But he then learns the correct actions and is able to solve this experiment. 
