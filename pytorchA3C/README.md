A3C's Implement
======
To train the model: <br>
`python main.py --num-processes 16 --env-name PongDeterministic-v3`

The model will be saved as `A3C.t7` and the record will be saved as `record.txt`

You can add `--loadmodel 1` to load pretrained model named `A3C.t7`

Also you can change `--env-name` to train on different Atari

Performance on Breakout(8 threads for 20 h ):<br>

![](https://github.com/tankche1/Deep-Reinforcement-Learning/blob/master/pytorchA3C/Breakout.png)
<br><br><br>
Performance on Pong:<br>
![](https://github.com/tankche1/Deep-Reinforcement-Learning/blob/master/pytorchA3C/Pong.png)

