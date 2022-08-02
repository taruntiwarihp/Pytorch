# class_to_id = {
#     'GG' : 0, 
#     'QS' : 1, 
#     'JK' : 2,
# }

# from matplotlib import pyplot as plt
# import seaborn as sns
# import numpy as np


# accuracy = np.array([
# [0.99130435, 1.        , 1.        ],
# [0.99130435, 1.        , 1.        ],
# [1.        , 1.        , 0.98412698],
# [1.        , 0.96153846, 1.        ],
# [1.        , 0.95454545, 1.        ],
# [1., 1., 1.],
# [0.98      , 0.96774194, 1.        ],
# [1., 1., 1.],
# [1., 1., 1.],
# [1., 1., 1.],
# ])

# precision = np.array([
# [0.99130435, 1.        , 1.        ],
# [0.99130435, 1.        , 1.        ],
# [1.        , 1.        , 0.98412698],
# [1.        , 0.96153846, 1.        ],
# [1.        , 0.95454545, 1.        ],
# [1., 1., 1.],
# [0.98      , 0.96774194, 1.        ],
# [1., 1., 1.],
# [1., 1., 1.],
# [1., 1., 1.],
# ])

# recall = np.array([
# [1.  , 1.  , 0.98],
# [1.        , 1.        , 0.97916667],
# [0.99145299, 1.        , 1.        ],
# [1.        , 1.        , 0.98039216],
# [0.99270073, 1.        , 1.        ],
# [1., 1., 1.],
# [1.        , 1.        , 0.95833333],
# [1., 1., 1.],
# [1., 1., 1.],
# [1., 1., 1.],
# ])

# f1 = np.array([
# [0.99563319, 1.        , 0.98989899],
# [0.99563319, 1.        , 0.98947368],
# [0.99570815, 1.        , 0.992     ],
# [1.        , 0.98039216, 0.99009901],
# [0.996337  , 0.97674419, 1.        ],
# [1., 1., 1.],
# [0.98989899, 0.98360656, 0.9787234 ],
# [1., 1., 1.],
# [1., 1., 1.],
# [1., 1., 1.],
# ])

# print(np.mean(accuracy, axis=0))
# print(np.mean(precision, axis=0))
# print(np.mean(recall, axis=0))
# print(np.mean(f1, axis=0))

# # epoch = range(1, 11)

# # sns.set()

# # fig = plt.subplots(figsize =(24, 8))

# # barWidth = 0.25
# # br1 = np.arange(len(epoch))
# # br2 = [x + barWidth for x in br1]
# # br3 = [x - barWidth for x in br1]

# # plt.bar(br1, accuracy[:, 0], label='GG', color ='r', edgecolor ='grey')
# # plt.bar(br2, accuracy[:, 1], label='QS', color ='g', edgecolor ='grey')
# # plt.bar(br3, accuracy[:, 2], label='JK', color ='b', edgecolor ='grey')

# # plt.xlabel('Fold')
# # plt.ylabel('Score')
# # plt.title('Class Accuracy')
# # plt.legend()
# # plt.savefig('abc.png', dpi=1000)
# # plt.show()
'''
Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
'''

hyp = {
    'LSTM Hidden Layer' : 100,
    'K FOld' : 10,
    'Chunk Lenghth' : 10,
    'Overlap length' : 5,
    'Pretraining Epoch' : 50,
    'Pretraining Warmup Step' : 300,
    'Finetuning Epoch' : 50,
    'Learning Rate' : 3e-5,
    'Loss Function' : 'Cross Entropy Loss',
    'Optimizer' : 'ADAMW',
    'Scheduler' : 'Linear',
}

import numpy as np

accuracy = np.array([
[0.99074074, 1.        , 1.        ],
[1., 1., 1.],
[1.        , 0.96551724, 0.98076923],
[1., 1., 1.],
[1.        , 1.        , 0.98148148],
[1.        , 0.92857143, 0.98      ],
[0.99115044, 1.        , 1.        ],
[1.        , 1.        , 0.96226415],
[0.98319328, 1.        , 1.        ],
[1., 1., 1.],
])

precision = np.array([
[0.99074074, 1.        , 1.        ],
[1., 1., 1.],
[1.        , 0.96551724, 0.98076923],
[1., 1., 1.],
[1.        , 1.        , 0.98148148],
[1.        , 0.92857143, 0.98      ],
[0.99115044, 1.        , 1.        ],
[1.        , 1.        , 0.96226415],
[0.98319328, 1.        , 1.        ],
[1., 1., 1.],
])

f1 = np.array([
[0.99534884, 0.98461538, 1.        ],
[1., 1., 1.],
[0.9958159 , 0.98245614, 0.98076923],
[1., 1., 1.],
[0.99555556, 1.        , 0.99065421],
[0.99591837, 0.96296296, 0.97029703],
[0.99555556, 1.        , 0.99099099],
[0.99559471, 0.98507463, 0.98076923],
[0.99152542, 1.        , 0.9787234 ],
[1., 1., 1.],
])

recall = np.array([
[1.        , 0.96969697, 1.        ],
[1., 1., 1.],
[0.99166667, 1.        , 0.98076923],
[1., 1., 1.],
[0.99115044, 1.        , 1.        ],
[0.99186992, 1.        , 0.96078431],
[1.        , 1.        , 0.98214286],
[0.99122807, 0.97058824, 1.        ],
[1.        , 1.        , 0.95833333],
[1., 1., 1.],
])

# print(np.mean(accuracy, axis=0))
# print(np.mean(precision, axis=0))
# print(np.mean(recall, axis=0))
# print(np.mean(f1, axis=0))

'''
[0.99650845 0.98940887 0.99045149]
[0.99650845 0.98940887 0.99045149]
[0.99659151 0.99402852 0.98820297]
[0.99653144 0.99151091 0.98922041]
'''


A = np.mean(accuracy, axis=1).tolist()
B = np.mean(precision, axis=1).tolist()
C = np.mean(recall, axis=1).tolist()
D = np.mean(f1, axis=1).tolist()

for (a, b, c, d) in zip(A, B, C, D):

    print('|'.join([str(a), str(b), str(c), str(d)]))