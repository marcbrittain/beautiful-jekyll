---
layout: post
title: AI in MineCraft: Episode 
tags: [MineCraft, AI, python, keras, tensorflow]
permalink: /2018/09/06/minecraft-episode/
---


In this episode, we use a prebuilt experiment in the project malmo python examples folder: Tutorial_6.py. We then modified the file to include the agent algorithm to solve this problem. In this experiment, Ralph solved the problem using Q-Learning. The idea behind Q-Learning is that with each state (which block is Ralph on) and each action (move forward, move backward, etc) there is an associated value that let’s Ralph know how to act. Now his knowledge comes from experience in MineCraft and gets updated constantly, so if you notice in the beginning Ralph is not making the best choices…But he then learns the correct actions and is able to solve this experiment.
