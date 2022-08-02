data = {'frsjnvjn': 
    {'DC_SERVICE': 
        {'1.0': 
            {'count': 4, 'filename':
                                {'sources/mortalitymodel/pipelines/DC2_auw_mortalitymodel.yaml',
                                 'sources/mortalitymodel/pipelines/DC2auw_penddelt.yamI',
                                 'sources/mortalitymodel/pipelines/DCZ_auw pendelt.yaml',
                                 'sources/prd_ind_mainframe/pipelines/DCZ_auw ML UNDRRITE PRODDATA PENDDISK.yaml'}}}}}
data = dict(data)

# data1 = list(list(data[list(data.keys())[0]].values())[0].values())[0]
# k = list(data1[list(data1.keys())[1]])
# print(k)

import yaml

class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            assert key not in mapping
            mapping.append(key)
        return super().construct_mapping(node, deep) 



with open('data.yaml', 'r') as f:
    my_dict = yaml.load(f, Loader=UniqueKeyLoader)


print(my_dict)