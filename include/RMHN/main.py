import datetime

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import argparse
from include.Config import Config
from include.Model import build, training, test1
from include.Load import *

import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

'''
Followed the code style of the followings:
https://github.com/1049451037/GCN-Align
https://github.com/StephanieWyt/HGCN-JE-JR
https://github.com/Peter7Yao/RNM
Thanks for their outstanding contributions.
'''

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,default='DBP15k', help='DBP15k，DWY')
parser.add_argument('--lang', type=str, default='zh_en',help='zh_en, ja_en and fr_en')

args = parser.parse_args()

if __name__ == '__main__':
    time1=time.localtime(time.time())
    print("begin time：",time.strftime("%Y-%m-%d %H:%M:%S",time1))
    start = datetime.datetime.now()

    print("Executing"+args.dataset)
    config = Config(args.dataset,args.lang)
    e1 = set(loadfile(config.e1, 1))
    e2 = set(loadfile(config.e2, 1))
    e = len(e1 | e2)            
    print(e)
    ILL = loadfile(config.ill, 2)   # [(0, 10500), (1, 10501), (2, 10502), (3, 10503), (4, 10504)]
    illL = len(ILL) #15000
    np.random.shuffle(ILL)          
    s = args.lang + ",epochs=" +str(config.epochs)+"，pre_epochs="+str(config.pre_epochs)+\
        "，train_batchnum="+str(config.train_batchnum)+"，test_batchnum="+str(config.test_batchnum)+\
        "，k="+str(config.k)+"，seed="+str(config.seed)
 
    train = np.array(ILL[:illL // 10 * config.seed])   #4500
    test = ILL[illL // 10 * config.seed:]
    # print(train,len(train))
    """
    [[22129 37240]
     [ 5946 16446]
     [ 2449 12949]
     ...
     [21029 36364]
     [21047 35687]
     [ 7535 18035]] 4500

    [(4946, 15446), (2132, 12632), (1233, 11733), (604, 11104)] 10500
    """
    # print(test,len(test))
    
    ILL_r = loadfile(config.ill_r, 2)
    
    KG1 = loadfile(config.kg1, 3)
    KG2 = loadfile(config.kg2, 3)

    """
    [(3329, 42, 4592), (24424, 819, 22802), (24436, 1126, 29860)] 70414
    [(35437, 2152, 35055), (16605, 2241, 34776), (37558, 2217, 19906)] 95142
    """
    # print(KG1,len(KG1))
    # print(KG2,len(KG2))

    r_kg_1 = set()  
    r_kg = set()  

    for tri in KG1:
        r_kg_1.add(tri[1])
        r_kg.add(tri[1])
        # print(tri)
        """
        (3118, 1123, 9427)
        (9984, 1252, 24843)
        (23621, 603, 8178)
        (9275, 156, 21281)
        """
    
    for tri in KG2:
        r_kg.add(tri[1])

    output_h, output_r, loss_pre, loss_transr, loss_all, M0, rel_type  = \
        build(config.dim, config.act_func, config.gamma, config.k, config.vec, e,e1,e2,ILL_r, KG1,KG2,KG1 + KG2)
    training(output_h, loss_pre, loss_transr, loss_all, config.lr, config.epochs, config.pre_epochs, train, e,
                         config.k, config.save_suffix, config.dim, config.train_batchnum, test, M0,
                         KG1,KG2, KG1 + KG2, rel_type, output_r, s,len(r_kg_1), len(r_kg), ILL_r)
    # test1(outvec, outvec_r, r_kg_1, M0, train, ILL_r,rel_type, test, None, None, KG1, KG2)
    # sess.close()
    time2 = time.localtime(time.time())
    print("end time：",time.strftime("%Y-%m-%d %H:%M:%S", time2))
    end = datetime.datetime.now()
    print("time use",end-start)






    
