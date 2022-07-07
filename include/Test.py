import datetime

import numpy as np
import scipy.spatial
import time
# from .Model import ss

time1 = time.localtime(time.time())
print(time.strftime("%Y-%m-%d %H:%M:%S", time1))
time1 = time.strftime("%Y-%m-%d%H-%M-%S", time1)
time1 = "result/result+" + time1 + ".txt"
file = open(time1, "w")

def update_ss(output,KG1,KG2,neg2_left,neg_right):
    L1 = np.array([e1 for e1,r, e2 in KG1])  # Select the node in KG1 from the test entity pair
    L2= np.array([e2 for e1,r, e2 in KG1])  # 从Select the node in KG2 from the test entity pair
    R1 = np.array([e1 for e1, r, e2 in KG2])  # Select the node in KG1 from the test entity pair
    R2 = np.array([e2 for e1, r, e2 in KG2])  # Select the node in KG2 from the test entity pair
    L1_vec=np.array(output[L1],dtype='float16')
    L2_vec=np.array(output[L2],dtype='float16')
    R1_vec = np.array(output[R1],dtype='float16')
    R2_vec=np.array(output[R2],dtype='float16')
    KG_vec = np.array(output, dtype='float16')  # 实体向量
    neg1_vec=[]
    neg2_vec=[]

    for list in neg2_left:
        for l in [list]:
            neg1_vec.append(output[l])
    for list in neg_right:
        for l in [list]:
            neg2_vec.append(output[l])
    neg1_vec=np.array(neg1_vec)
    neg2_vec=np.array(neg2_vec)
   
    sim1=batch(L1,L2,L1_vec,L2_vec)
    sim2 = batch(R1, R2, R1_vec, R2_vec)
    # sim1=scipy.spatial.distance.cdist(L1_vec, L2_vec, metric='cityblock')
    # sim2=scipy.spatial.distance.cdist(R1_vec, R2_vec, metric='cityblock')
    # sim1_neg=scipy.spatial.distance.cdist(neg1_vec, KG_vec, metric='cityblock')
    # sim2_neg = scipy.spatial.distance.cdist(neg2_vec, KG_vec, metric='cityblock')
    sim1_neg=batch(neg2_left, neg_right,neg1_vec, KG_vec)
    sim2_neg = batch(neg2_left, neg_right, neg2_vec, KG_vec)
    # SS=ss(sim1,sim2,sim1_neg,sim2_neg)
    SS=sim1_neg
    return SS

def batch(L1,L2,L1_vec,L2_vec):
    batchnum = 100
    t = len(L1)
    t1 = len(L2)
    sim=np.array([])
    for p in range(batchnum):  # 采用批处理
        head = int(t / batchnum * p)
        head1 = int(t / batchnum * p)
        if p == batchnum - 1:
            tail = t
            tail1 = t1
        else:
            tail = int(t / batchnum * (p + 1))
            tail1 = int(t1 / batchnum * (p + 1))
        sim_bat = scipy.spatial.distance.cdist(  # 对齐实体向量与实体向量L1距离
            L1_vec[head:tail], L2_vec[head1:tail1], metric='cityblock')
        np.append(sim,sim_bat)
    return sim

def get_hits(vec, vec_r, l1, M0, ref_data, rel_type, test_pair, sim_e, sim_r,KG1,KG2,s,neg2_left,neg_right,top_k=(1,3,10)):
    ref = set()

    time11 = time.localtime(time.time())
    time11 = time.strftime("%Y-%m-%d %H:%M:%S", time11)
    print("begin time：", time11)
    start = datetime.datetime.now()

    for pair in ref_data:   
        ref.add((pair[0], pair[1]))
    r_num = len(vec_r)//2   
    # print(r_num)
    
    kg = {} #Node corresponding relationship - node
    rel_ent = {}    #Entity corresponding to the relationship
    for tri in M0:  #Add inverted graph, adjacency matrix
        if tri[0] == tri[2]:        #5729	13	5729
            continue
        if tri[0] not in kg:    
            kg[tri[0]] = set()
        if tri[2] not in kg:
            kg[tri[2]] = set()
        if tri[1] not in rel_ent:
            rel_ent[tri[1]] = set()
        
        kg[tri[0]].add((tri[1], tri[2]))    
        kg[tri[2]].add((tri[1]+r_num, tri[0])) 
        rel_ent[tri[1]].add((tri[0], tri[2]))   
        # print(kg)
        # print(rel_ent)

    L = np.array([e1 for e1, e2 in test_pair])  
    R = np.array([e2 for e1, e2 in test_pair])  
    Lvec = vec[L]   
    Rvec = vec[R]  
    # print(L)
    # print(R)
    # print(Lvec)
    # print(Rvec)

    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    print("entity sim",sim)
    if sim_e is None:
        sim_e = sim
    
    R_set = {}  
    
    for i in range(len(L)): 
        j = sim_e[i, :].argsort()[0]    
        if sim_e[i,j] >= 5: 
            continue
        if j in R_set and sim_e[i, j] < R_set[j][1]:
            ref.remove((L[R_set[j][0]], R[j]))  
            ref.add((L[i], R[j]))               
            R_set[j] = (i, sim_e[i, j])         
        if j not in R_set:
            ref.add((L[i], R[j]))   
            R_set[j] = (i, sim_e[i, j]) 
    
    if sim_r is None:   
        sim_r = scipy.spatial.distance.cdist(vec_r[:l1], vec_r[l1:r_num], metric='cityblock')
    print("relation sim",sim_r)

    ref_r = set()   
    for i in range(l1): 
        j = sim_r[i, :].argsort()[0]    
        if sim_r[i,j] < 3:              
            ref_r.add((i, j+l1))        
            ref_r.add((i+r_num,j+l1+r_num))
    
    e_index=0
    for i in range(len(L)):  
        rank = sim[i, :].argsort()[:800]    
        e_index=len(rank)
        for j in rank:
            if R[j] in kg:  
                match_num = 0   
                for n_1 in kg[L[i]]:    
                    for n_2 in kg[R[j]]:
                        if (n_1[1], n_2[1]) in ref and (n_1[0], n_2[0]) in ref_r:   
                            w = rel_type[str(n_1[0]) + ' ' + str(n_1[1])] * rel_type[str(n_2[0]) + ' ' + str(n_2[1])]
                            match_num += w  
                sim[i,j] -= 10 * match_num / (len(kg[L[i]]) + len(kg[R[j]])+match_num)   
               
    # file.writelines("********************************")
    sim_r = scipy.spatial.distance.cdist(vec_r[:l1], vec_r[l1:r_num], metric='cityblock')
    print(sim_r)
    r_index=0
    for i in range(l1): 
        rank = sim_r[i, :].argsort()[:200]   
        r_index=len(rank)
        for j in rank:
            if i in rel_ent and j+l1 in rel_ent:   
                match_num = 0
                for n_1 in rel_ent[i]:  
                    for n_2 in rel_ent[j+l1]:   
                        if (n_1[0],n_2[0]) in ref and (n_1[1],n_2[1]) in ref:   
                            match_num += 1 
                sim_r[i,j] -= 200 * match_num / (len(rel_ent[i])+len(rel_ent[j+l1])+match_num)    
              
                # file.writelines(str(sim_r[i,j]))
    # count=0
    # SS=update_ss(vec,KG1,KG2,neg2_left,neg_right)
    # print("SS:",SS)
    # for i in range(len(L)):
    #     rank=sim[i,:].argsort()
    #     if rank[0]!=i:
    #         count+=1
    #         if len(rank)*count>=SS-i:
    #             sim[i,0]=i
    # print(count)

    mrr_l = []  # KG1MRR
    mrr_r = []  # KG2MRR
    mr_l = []  # KG1MR
    mr_r = []  # KG2MR

    top_lr = [0] * len(top_k)   
    for i in range(Lvec.shape[0]):  
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]  
        # print(rank)
        mrr_l.append(1.0 / (rank_index+1))      
        mr_l.append(rank_index+1)
        for j in range(len(top_k)): 
            if rank_index < top_k[j]:
                top_lr[j] += 1  

    top_rl = [0] * len(top_k)   
    # print(top_rl)

    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]  
        mrr_r.append(1.0 / (rank_index+1))     
        mr_r.append(rank_index+1)
        for j in range(len(top_k)):     
            if rank_index < top_k[j]:
                top_rl[j] += 1 


    # file.write("sim_e：" + str(sim_e))
    # file.write("sim_r：" + str(sim_r))

    print('Entity Alignment (left):')
    file.write(s+"，Entity ranking="+str(e_index)+"，Relation ranking="+str(r_index)+"\n")
    file.write("Entity Alignment (left):\n")
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
        file.writelines('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100)+"\n")
    print('MRR: %.4f' % (np.mean(mrr_l)))
    print('MR: %.4f' % (np.mean(mr_l)))
    file.writelines('MRR: %.4f' % (np.mean(mrr_l))+"\n")
    file.writelines('MR: %.4f' % (np.mean(mr_l))+"\n")
    print('Entity Alignment (right):')
    file.write("Entity Alignment (right):\n")
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
        file.writelines('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100)+"\n")
    print('MRR: %.4f' % (np.mean(mrr_r)))
    print('MR: %.4f' % (np.mean(mr_r)))
    file.writelines('MRR: %.4f' % (np.mean(mrr_r))+"\n")
    file.writelines('MR: %.4f' % (np.mean(mr_r))+"\n")
    time2 = time.localtime(time.time())
    time2 = time.strftime("%Y-%m-%d %H:%M:%S", time2)
    print("end time", time2)
    end = datetime.datetime.now()
    print("Time use", end - start)
    return sim, sim_r,ref

def get_rel_hits(vec, vec_r, l1, M0, ref_data, rel_type, test_pair, ILL_r, top_k=(1, 10)):
    time11 = time.localtime(time.time())
    time11 = time.strftime("%Y-%m-%d %H:%M:%S", time11)
    print("begin time：", time11)
    start = datetime.datetime.now()

    L = np.array([e1 for e1, e2 in test_pair])  
    R = np.array([e2 for e1, e2 in test_pair])  
    Lvec = vec[L]  
    Rvec = vec[R]   
    time1 = time.localtime(time.time())
    print(time.strftime("%Y-%m-%d %H:%M:%S", time1))
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    print(sim)
    
    ref = set() 
    for pair in ref_data:
        ref.add((pair[0], pair[1]))
    
    for i in range(len(L)):
        j = sim[i, :].argsort()[0]
        if sim[i,j] < 5:
            ref.add((L[i],R[j]))   

    rel_ent = {}    
    for tri in M0:
        if tri[0] == tri[2]:
            continue
        if tri[1] not in rel_ent:
            rel_ent[tri[1]] = set()

        rel_ent[tri[1]].add((tri[0], tri[2]))
    
    L_r = np.array([r1 for r1, r2 in ILL_r])   
    R_r = np.array([r2 for r1, r2 in ILL_r])   
    Lvec_r = vec_r[L_r] 
    Rvec_r = vec_r[R_r] 
    # print(L_r)
    # print(R_r)
    # print(Lvec_r)
    # print(Rvec_r)
    
    sim_r = scipy.spatial.distance.cdist(Lvec_r, Rvec_r, metric='cityblock')
    
    for i in range(len(ILL_r)):
        rank = sim_r[i, :].argsort()[:300]
        for j in rank:
            if L_r[i] in rel_ent and R_r[j] in rel_ent:
                match_num = 0
                for n_1 in rel_ent[L_r[i]]:
                    for n_2 in rel_ent[R_r[j]]:
                        if (n_1[0],n_2[0]) in ref and (n_1[1],n_2[1]) in ref:
                            match_num += 1
                sim_r[i,j] -= 200 * match_num / (len(rel_ent[L_r[i]])+len(rel_ent[R_r[j]]))
                
    
    top_lr = [0] * len(top_k)
    for i in range(Lvec_r.shape[0]):
        rank = sim_r[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1

    top_rl = [0] * len(top_k)
    # print(top_rl)

    for i in range(Rvec_r.shape[0]):
        rank = sim_r[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1

    print('Relation Alignment (left):')
    file.writelines("Relation Alignment (left):\n")
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(ILL_r) * 100))
        file.writelines('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(ILL_r) * 100)+"\n")
    
    print('Relation Alignment (right):')
    file.writelines("Relation Alignment (right):\n")
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(ILL_r) * 100))
        file.writelines('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(ILL_r) * 100)+"\n")
    file.close()
    time2 = time.localtime(time.time())
    time2 = time.strftime("%Y-%m-%d %H:%M:%S", time2)
    print("end time", time2)
    end = datetime.datetime.now()
    print("time use", end - start)
    return
