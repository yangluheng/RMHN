import tensorflow as tf

"""
配置文件类
Profile class
"""
class Config():
    """
    初始化
    initialization
    """
    def __init__(self, d='', l=''):
        dataset = d
        language = l
        prefix=''
        if dataset=='DWY15k':
            prefix='data/data/' + str(dataset) + '/'
        else:
            prefix = 'data/' + str(dataset) + '/' + str(language)
        self.kg1 = prefix + '/triples_1'        #中文图Chinese KG
        self.kg2 = prefix + '/triples_2'        #英文图English KG
        self.e1 = prefix + '/ent_ids_1'         #中文实体 Chinese entities
        self.e2 = prefix + '/ent_ids_2'         #英文实体 English entities
        self.ill = prefix + '/ref_ent_ids'      #对齐实体对 aligned entity pairs
        self.ill_r = prefix + '/ref_r_ids'      #对齐关系  aligned relation
        self.vec = prefix + '/vectorList.json'  #golve vector
        self.save_suffix = str(dataset)+'_'+str(language)

        self.epochs = 60
        self.pre_epochs = 50
        self.train_batchnum =200
        self.test_batchnum = 200
        self.dim = 300
        self.act_func = tf.nn.relu

        """
        这个函数设计的目标是：1）能像triplet loss一样灵活；
        2）能够自适应不同的数据分布；3）能够像contrastive loss一样计算高效。
        
        """
        self.gamma = 1  # margin based loss
        self.lama=0.001 # the balance parameter of overall loss
        self.k = 90  # number of negative samples for each positive one
        self.seed = 3  # 30% of seeds
        self.lr = 0.0001 # learning rate
