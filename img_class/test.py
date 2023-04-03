import numpy as np

eff = np.array([
    [9953, 47, 0, 0, 0],
    [47, 9855, 0, 98, 0],
    [0, 0, 10000, 0, 0],
    [0, 98, 0, 9902, 0],
    [0, 0, 0, 0, 10000]
])

print(eff.sum(axis=1))
print(eff.sum(axis=0))

alex = np.array([
    [9182, 332, 192, 215, 79],
    [332, 7703, 1075, 549, 341],
    [192, 1037, 7583, 926, 262],
    [215, 587, 888, 8254, 56],
    [79, 341, 262, 56, 9262]
])

print(alex.sum(axis=1))
print(alex.sum(axis=0))

vgg = np.array([
    [8735, 363, 208, 518, 176],
    [363, 8332, 18, 971, 316],
    [208, 96, 8739, 793, 164],
    [518, 893, 871, 7227, 491],
    [176, 316, 164, 491, 8853]
])

print(vgg.sum(axis=1))
print(vgg.sum(axis=0))

resnet = np.array([
    [7868, 274, 281, 1347, 230],
    [274, 8340, 463, 665, 258],
    [281, 406, 8739, 338, 236],
    [1347, 722, 281, 7227, 423],
    [230, 258, 236, 423, 8853]
])

print(resnet.sum(axis=1))
print(resnet.sum(axis=0))

inception = np.array([
    [7423, 16, 15, 2472, 74],
    [16, 5158, 3042, 1476, 308],
    [15, 3245, 3750, 2507, 483],
    [2472, 1273, 2710, 1553, 1992],
    [74, 308, 483, 1992, 7143]
])

print(inception.sum(axis=1))
print(inception.sum(axis=0))

mobile = np.array([
    [7452, 189, 198, 2068, 93],
    [189, 4305, 2489, 2565, 452],
    [198, 3011, 4250, 1793, 748],
    [2068, 2043, 2315, 1747, 1827],
    [93, 452, 748, 1827, 6880]
])

print(mobile.sum(axis=1))
print(mobile.sum(axis=0))

model_stats = {
    'AlexNet': alex,
    'VGG16BN': vgg,
    'ResNet50': resnet,
    'MobileNetV2': mobile,
    'InceptionV3': inception,
    'AttentionEfficientNet': eff
}

import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt




# x_axis_labels = ['Anthracnose', 'Bacterial Black Spot', 'Healthy', 'Nutritional Deficiency', 'Powdery Mildew']
# y_axis_labels = ['F', 'G', 'H', 'I', 'J']

# for cls in list(model_stats.keys()):

#     cm = model_stats[cls]
#     plt.figure(figsize = (10,7))
#     plt.title('Confusion Matrix for {}'.format(cls), fontsize = 20)
    
#     group_counts = ["{0:0.0f}\n".format(value) for value in cm.flatten()]

#     group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]

#     box_labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_counts,group_percentages)]
#     box_labels = np.asarray(box_labels).reshape(cm.shape[0],cm.shape[1])

#     sn.heatmap(
#         cm, annot=box_labels, fmt='', cmap='Blues',
#         xticklabels=x_axis_labels, yticklabels=x_axis_labels,
#     )

#     # sn.heatmap(cm/np.sum(cm),  annot=True, 
#     #             fmt='.2%', cmap='Blues', xticklabels=x_axis_labels, yticklabels=x_axis_labels)

#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.tight_layout()
#     plt.savefig('docs_new/{}_cm_hy.png'.format(cls.lower()), dpi=1000)
#     plt.show()

# from pytablewriter import MarkdownTableWriter

# for cls in list(model_stats.keys()):
#     cm = model_stats[cls]

#     class_acc = cm.diagonal()/cm.sum(axis=1)
#     class_acc[np.isnan(class_acc)]=0
#     class_precision = cm.diagonal()/cm.sum(axis=1)
#     class_precision[np.isnan(class_precision)]=0
#     class_recall = cm.diagonal()/cm.sum(axis=0)
#     class_recall[np.isnan(class_recall)]=0
#     class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)
#     class_f1[np.isnan(class_f1)]=0

#     print()
#     writer = MarkdownTableWriter(
#         table_name=cls,
#         headers=["Matrix"] + x_axis_labels,
#         value_matrix=[
#             ["Accuracy"] + class_acc.tolist(),
#             ["Precision"] + class_precision.tolist(),
#             ["Recall"] + class_recall.tolist(),
#             ["F1 Score"] + class_f1.tolist(),
#         ],
#     )

#     writer.write_table()