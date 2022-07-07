import re

def load(file):
    with open(file=file,encoding="utf-8") as f:
        list=[]
        for line in f:
            res = []
            a = line[:-1].split('\t')
            s=a[1].split("http://dbpedia.org/ontology/")[1]
            res.append(int(a[0]))
            res.append(s)
            list.append(res)
    print(list)
    return list


def write(list1,list2):
    dict={}
    f = open("D:\\Python\\RNM\\data\\data\\DWY15K\\ref_r_ids", "w")
    for list in list1:
        dict[list[1]]=list[0]
    for list in list2:
        if list[1] in dict:
            # print(str(dict[list[1]]))
            # print(list[0])
            # print(list[1])

            s=str(dict[list[1]])+"\t"+str(list[0])+"\t"+str(list[1])
            f.write(s)
            f.write("\n")
    f.close()
    print("success")




if __name__ == '__main__':
    file1="D:\\Python\\HTRS\\data\\data\\DWY15K\\rel_ids_1"
    file2="D:\\Python\\HTRS\\data\\data\\DWY15K\\rel_ids_2"
    list1=load(file1)
    list2=load(file2)
    write(list1,list2)
