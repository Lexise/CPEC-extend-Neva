import re
import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.cluster import KMeans,DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from os import listdir,unlink,walk
from os.path import isfile, join
from zipfile import ZipFile
from pathlib import Path
import time
import itertools
import os
from sklearn.model_selection import train_test_split
#import subprocess
from clingo_asp_compute import compute_extensions

def process_extension_individual(question, semantics, processed_dir, upload_dir, extenion_dir, eps, minpts,n_cluster): # for "other" situation, when user want to select their own semantics or semantic pairs
    for item in semantics:
        if item == 'stable':
            asp_encoding = "stable_web.dl"
            end = "STB"
        elif item == 'preferred':
            asp_encoding = "prefex.dl"
            end = "PR"
        elif item == 'stage':
            asp_encoding = "stage-cond-disj.dl"
            end = "STG"
        elif item == 'semi-stable':
            asp_encoding = "semi-stable_web.txt"
            asp_encoding = "semi-stable_web.txt"
            end = "SEMI-STB"
        elif item == 'cf2':
            asp_encoding = "cf2_web.dl"
            end = "CF2"
        elif item == 'stage2':
            asp_encoding = "stage2_web.txt"
            end = "STG2"
        else:
            return False
        extension_file = "{}.EE_{}".format(question, end)
        compute_extensions(upload_dir +question,asp_encoding,extenion_dir+extension_file)
        # os.system(
        #     "D:/test2/clingo-4.5.4-win64/clingo.exe {} data/app_uploaded_files/{} 0 > data/extension_sets/{}".format(
        #         asp_encoding, question, extension_file))
        process_data_two_sets(processed_dir, upload_dir + question, extenion_dir + extension_file, eps, minpts,
                     n_cluster, item)

def addional_process_individual(processed_dir, semantics):
    if len(semantics)==1:
        return False
    semantic1=semantics[0]
    semantic2=semantics[1]
    processed_data_stage2 = pd.read_pickle(processed_dir + semantic1+"_processed_data.pkl")
    processed_data_cf2 = pd.read_pickle(processed_dir + semantic2+"_processed_data.pkl")
    # X_train,  y_test = train_test_split(processed_data2, test_size=0.33)
    # y_test.to_pickle(PROCESSED_DIRECTORY +"cf2_processed_data_small.pkl")
    arguments = processed_data_stage2.arg.append(processed_data_cf2.arg)
    common_all = set(arguments[0]).intersection(*arguments)

    feature1 = find_feature_group(common_all, processed_data_stage2, processed_data_cf2)
    feature2 = find_feature_group(common_all, processed_data_cf2, processed_data_stage2)
    sum_diff = pd.DataFrame({
        "semantics": [semantic1, semantic2],
        "feature_arguments": [feature1, feature2],
    })
    sum_diff.to_pickle(processed_dir + "group_feature.pkl")
    common_data = pd.merge(processed_data_cf2, processed_data_stage2, on=['arg'], how='inner')
    present_data1 = processed_data_cf2[~processed_data_cf2.id.isin(common_data.id_x)]
    present_data1["category"] = "only_"+semantic2
    present_data2 = processed_data_stage2[~processed_data_stage2.id.isin(common_data.id_y)]
    present_data2["category"] = "only_"+semantic1
    present_common = processed_data_cf2[processed_data_cf2.id.isin(common_data.id_x)]
    present_common["category"] = semantic1+"_and_"+semantic2

    processed_data = pd.concat([present_data1, present_data2, present_common])
    processed_data.to_pickle(processed_dir + "CombinedProcessed_data.pkl")

def clean_folder(folder_path):
    if len(listdir(folder_path))!=0:
        removed=[]
        for the_file in listdir(folder_path):
            file_path = join(folder_path, the_file)
            try:
                if isfile(file_path):
                    unlink(file_path)
                    removed.append(the_file)
            except Exception as e:
                print(e)
        return removed
    else:
        return []


def get_color_label(processed_data, color_label,groups_set):
    processed_data['color'] = processed_data[color_label]
    if color_label =='category':

        colors=['#e5f5f9','#2ca25f']
        if len(groups_set)<3:
            processed_data["color"]='#2ca25f'
        else:
            for x in groups_set:
                if 'and' in x:
                    processed_data["color"].replace({x: '#99d8c9'}, inplace=True)
                else:
                    processed_data["color"].replace({x: colors[0]}, inplace=True)
                    del colors[0]
    else:
        colors=['#e41a1c','#377eb8','#4daf4a']
        for x in range(0,len(groups_set)):
            processed_data["color"].replace({groups_set[x]: colors[x]}, inplace=True)

# def change_to_hotpot(answer, item):
#
#       returnlist = [0]*len(item)
#       for ele in answer:
#         if ele in item:
#           returnlist[item.index(ele)]=1
#       return returnlist


def f_comma(my_str, group, char=','):
        my_str = str(my_str)
        return char.join(my_str[i:i+group] for i in range(0, len(my_str), group))


def process_data(dir, arguments_file, answer_sets, eps, minpts, n_cluster, semantic):

    #os.system(
    #    "D:/test2/clingo-4.5.4-win64/clingo.exe prefex.dl apx_files/AachenerVerkehrsverbund_26October2020.zip_train+metro+tram+bus.lp.apx 0 > extension_sets/test.EE_PR")
    start = time.process_time()

    with open(arguments_file, 'r') as file:
        question = file.read()
    itemlist = [s for s in re.findall(r"arg[(]a(.*?)[)].", question)]
    itemlist.sort()
    #column_arg = [str(s) for s in itemlist]    #test autoencoder
    with open(answer_sets, 'r') as file:
        answer = file.read()

    test = answer.split("Answer:")
    del test[0]
    #indexlist = [int(s.split("\n",1)[0]) for s in test]
    arg_len=len(test)
    if arg_len == 0:
        return False
    indexlist = range(1,arg_len+1)
    transfered=[]
    arguments=[]


    for s in test:
      temp1=re.findall(r"^in(.*)", s, re.M)
      if temp1:
        temp2=[s for s in re.findall(r'\d+', temp1[0])]
        bool_represent = np.in1d(itemlist, temp2) #boolean list representation
        temp2=frozenset(temp2)
        one_answer=bool_represent.astype(int) #to int list
        transfered.append( one_answer)
        arguments.append(temp2)
      else:
        arguments.append(set())
        transfered.append([])

    #for_auto_reduction = pd.DataFrame(data=transfered, index=indexlist, columns=column_arg)  # test autoencoder



    if "preferred_stable" in semantic:
        difference="not_defeated"
        big="preferred"
        small="stable"
    elif "stable_stage" in semantic:
        difference="nrge"
        big="stage"
        small="stable"
    elif "stage2_stable" in semantic:
        difference = "not_defeated"
        big = "stage2"
        small = "stable"
    elif "stable_cf2" in semantic:
        difference = "not_defeated"
        big = "cf2"
        small = "stable"
    elif 'cf2_stage2' in semantic:
        pass#要不要用“groups”作为区分 only cf2, cf2 _tage2的标准

    not_defeated1 = []
    for s in test:
        temp1 = re.findall(difference+"(.*)", s, re.M)
        if len(temp1) != 0:
            temp2 = [s for s in re.findall(r'\d+', temp1[0])]
            not_defeated1.append(temp2)
        else:
            not_defeated1.append(0)
    processed_data = pd.DataFrame({
        'id': indexlist,
        'in': transfered,
        'arg': arguments,
        difference: not_defeated1,
    })
    processed_data["groups"] = np.where(processed_data[difference] == 0, small, big)

    print("generally process answer sets( read data, one hot, group label, ): ", time.process_time() - start)
    start2 = time.process_time()
    if eps != "" and eps != "Eps":
        processed_data = clustering_dbscan(processed_data, float(eps), int(minpts))
    else:
        processed_data = clustering_dbscan(processed_data)
    print("dbscan clustering: ", time.process_time() - start2)

    processed_data = dimensional_reduction(processed_data)
    y = np.array([np.array(xi) for xi in transfered])    #to change and test###############减少需要内存大小
    processed_data =dimensional_reduction_autoencoding(y, processed_data)
    #processed_data =dimensional_reduction_autoencoding(for_auto_reduction, processed_data)
    start3 = time.process_time()

    if n_cluster !="" and n_cluster !="Cluster Num":
        processed_data= clustering_km(processed_data,int(n_cluster))
    else:
        processed_data= clustering_km(processed_data)
    print("kmeans clustering: ", time.process_time() - start3)

    start4 = time.process_time()
    all_arguments = [item for sublist in arguments for item in sublist]
    frequency = np.array([])
    for argument in itemlist:
        #count = 0

        # for arg_list in arguments:
        #     if argument in arg_list:
        #         count += 1
        frequency=np.append(frequency,all_arguments.count(argument))
    rate=frequency / len(processed_data) *100
    bar_data = pd.DataFrame({
        # "index":itemlist,
        "argument": itemlist,#argument
        "frequency": frequency,
        "rate": rate
    })
    print("bar chart(argument frequency): ", time.process_time() - start4)
    #correlation matrix
    start5 = time.process_time()
    all_occurence=pd.DataFrame([x for x in processed_data['in']], columns=[str(x) for x in itemlist])
    to_drop = bar_data.loc[bar_data['rate'].isin([0, 100])].argument
    all_occurence.drop([str(x) for x in to_drop], axis='columns', inplace=True)

    temp = all_occurence.astype(int)
    correlation_matrix = temp.corr()
    print("create correlation matrix: ", time.process_time() - start5)

    #find features:
    start6 = time.process_time()
    common_all = set(arguments[0]).intersection(*arguments)
    cluster_feature_db=find_feature_cluster(common_all,processed_data,"db")
    cluster_feature_km=find_feature_cluster(common_all,processed_data,"km")
    group_feature=find_feature_cluster(common_all,processed_data,"groups")
    #
    #
    clean_folder(dir)
    group_feature.to_pickle(dir + "group_feature.pkl")
    cluster_feature_db.to_pickle(dir + "db_cluster_feature.pkl")
    cluster_feature_km.to_pickle(dir + "km_cluster_feature.pkl")
    print("create feature report: ", time.process_time() - start6)
    processed_data.to_pickle(dir + "processed_data.pkl")
    bar_data["argument"]=[str(x)+"argument" for x in itemlist]
    bar_data.to_pickle(dir + "bar_data.pkl")
    correlation_matrix.to_pickle(dir + "correlation_matrix.pkl")

    # create a ZipFile object
    start7 = time.process_time()
    file_name=arguments_file.split("/")[-1]
    zipname = big+"_"+file_name.strip("apx") + "zip"
    path = Path(dir)
    zip_dir= str(path.parent) +"/processed_zip/"
    with ZipFile(zip_dir + zipname, 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in walk(dir):
            for filename in filenames:

                    # create complete filepath of file in directory
                    filePath = join(folderName, filename)
                    # Add file to zip
                    zipObj.write(filePath, arcname=filename)
    print("zip files: ", time.process_time() - start7)
    return True
    #return processed_data,bar_data,correlation_matrix,cluster_feature_db,cluster_feature_km,group_feature


def clustering_km( data, cluster_num=2):
    km = KMeans(n_clusters=cluster_num, precompute_distances='auto').fit_predict(list(data['in']))
    data['km_cluster_label'] = km
    return data

def clustering_dbscan( data, eps=1.7, minpoint=7):
    c = DBSCAN(eps=eps, min_samples=minpoint).fit_predict(list(data['in']))
    data['db_cluster_label'] = c
    return  data

def dimensional_reduction_autoencoding(data, processed_data)->pd.DataFrame:
    start = time.process_time()

    encoding_dim = 2
    original_shape=data.shape[1]
    input_df = Input(shape=(original_shape,))
    encoded = Dense(encoding_dim, activation='relu' )(input_df) #linear activity_regularizer=regularizers.l1(1e-4), bias_regularizer=regularizers.l2(1e-5),
    decoded = Dense(original_shape, activation='sigmoid')(encoded)

    # encoder
    autoencoder = Model(input_df, decoded)

    # intermediate result
    encoder = Model(input_df, encoded)

    # This is our encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder =Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(data, data,#可能要dataz这个形式有问题，下载然后
                    epochs=150, #50
                    batch_size=256,
                    shuffle=True,
                    #validation_data=(data_val, data_val)
                   )
    encoded_X_train = encoder.predict(data).T
    processed_data['auto_position_x'] = encoded_X_train[0]
    processed_data['auto_position_y'] = encoded_X_train[1]
    print("Autoencoding dimensional reduciton: ", time.process_time() - start)
    return processed_data


def dimensional_reduction(data):
    start1 = time.process_time()
    result2 = TSNE(n_components=2).fit_transform(list(data['in'])).T
    data['tsne_position_x'] = result2[0]
    data['tsne_position_y'] = result2[1]
    print("Tsne dimensional reduciton: ", time.process_time() - start1)
    svd = TruncatedSVD(n_components=2, n_iter=7)
    start2 = time.process_time()
    result = svd.fit_transform(list(data['in'])).T
    print("SVD dimensional reduciton: ", time.process_time() - start2)
    data['svd_position_x'] = result[0]
    data['svd_position_y'] = result[1]
    return data

def add_to(feature, combine):
  mask=[combine.issubset(x) for x in feature]
  if any(mask):
    super_f=list(itertools.compress(feature,mask))
    for x in super_f:
      feature.remove(x)

    feature.append(combine)

def find_feature_cluster(common_all, data, labels):  #clustered data
    if labels !="groups":
        labels=labels+"_cluster_label"


    clusters = data[labels].unique().tolist()
    if len(clusters)==1 or len(clusters)>50:
        return pd.DataFrame([])
    all_feature = []
    cluster_with_feature = clusters.copy()
    for cluster in clusters:
        feature = []
        current_cluster = data[data[labels] == cluster]
        all_lists = list(current_cluster.arg)
        common_links = set(all_lists[0]).intersection(*all_lists[1:])
        common_links = common_links - common_all

        #other_arguments = [item for sublist in other_cluster_arg for item in sublist]
        # mask = [a not in other_arguments for a in common_links]
        # if any(mask):
        #     feature = list(itertools.compress(common_links, mask))

        #2021.02.21 对于单一的feature的处理
        has_single_feature=0
        other_cluster_arg_flat_list = [x for x in data[data[labels] != cluster].arg]
        other_cluster_arg_combine = set([item for sublist in other_cluster_arg_flat_list for item in sublist])
        #other_cluster_arg_combine=set(other_cluster_arg_combine)
        for x in common_links:
            if x not in other_cluster_arg_combine:
                feature.append(x)
                has_single_feature=1
        if has_single_feature:
            all_feature.append(str(feature).strip('[]'))






        else:
            other_cluster_arg = [set(x) for x in data[data[labels] != cluster].arg]
            if not any(common_links <= x for x in other_cluster_arg):
                feature.append(common_links)
                for x in range(len(common_links), 1, -1):
                    temp = False
                    combinitions = list(itertools.combinations(common_links, x - 1))  #对于15长度的common_links, 3003 for x =11  太多了
                    for combine in combinitions:
                        combine = set(combine)
                        if not any(combine.issubset(x) for x in other_cluster_arg):
                            temp = True
                            add_to(feature, combine)
                    if not temp:
                        break
            if not feature:
                cluster_with_feature.remove(cluster)
            else:
                all_feature.append(str(feature[0]))

    sum_diff = pd.DataFrame({
        labels: cluster_with_feature,
        "feature_arguments": all_feature,
    })

    return sum_diff



def find_feature_group(common_all, data, otherdata):  #clustered data

    feature = []
    arguments1=data.arg
    common_links1 = set(arguments1[0]).intersection(*arguments1)
    # arguments2 = data2.arg
    # common_links2 = set(arguments2[0]).intersection(*arguments2)
    # arguments=arguments1.append(arguments2)
    # common_all=set(arguments[0]).intersection(*arguments)
    common_links=common_links1- common_all
    has_single_feature = 0
    other_cluster_arg_flat_list = otherdata.arg
    other_cluster_arg_combine = set([item for sublist in other_cluster_arg_flat_list for item in sublist])
    all_feature = []
    for x in common_links:
        if x not in other_cluster_arg_combine:
            feature.append(x)
            has_single_feature = 1
    if has_single_feature:
        all_feature.append(str(feature).strip('[]'))
    else:
        other_cluster_arg = [set(x) for x in otherdata.arg]
        if not any(common_links <= x for x in other_cluster_arg):
            feature.append(common_links)
            for x in range(len(common_links), 1, -1):
                temp = False
                combinitions = list(
                    itertools.combinations(common_links, x - 1))  # 对于15长度的common_links, 3003 for x =11  太多了
                for combine in combinitions:
                    combine = set(combine)
                    if not any(combine.issubset(x) for x in other_cluster_arg):
                        temp = True
                        add_to(feature, combine)
                if not temp:
                    break
        if not feature:
            return "no feature arguments"
        else:
            all_feature.append(str(feature[0]))

    # sum_diff = pd.DataFrame({
    #     labels: cluster_with_feature,
    #     "feature_arguments": all_feature,
    # })

    return all_feature










def process_data_two_sets(dir, arguments_file, answer_sets, eps, minpts, n_cluster, semantic): #process need to handlle two semantics seperately

    #os.system(
    #    "D:/test2/clingo-4.5.4-win64/clingo.exe prefex.dl apx_files/AachenerVerkehrsverbund_26October2020.zip_train+metro+tram+bus.lp.apx 0 > extension_sets/test.EE_PR")
    start = time.process_time()

    with open(arguments_file, 'r') as file:
        question = file.read()
    itemlist = [s for s in re.findall(r"arg[(]a(.*?)[)].", question)]
    itemlist.sort()
    #column_arg = [str(s) for s in itemlist]    #test autoencoder
    with open(answer_sets, 'r') as file:
        answer = file.read()

    test = answer.split("Answer:")
    del test[0]
    #indexlist = [int(s.split("\n",1)[0]) for s in test]
    arg_len=len(test)
    if arg_len == 0:
        return False
    indexlist = range(1,arg_len+1)
    transfered=[]
    arguments=[]


    for s in test:
      temp1=re.findall(r"^in(.*)", s, re.M)
      if temp1:
        temp2=[s for s in re.findall(r'\d+', temp1[0])]
        bool_represent = np.in1d(itemlist, temp2) #boolean list representation
        temp2=frozenset(temp2)
        one_answer=bool_represent.astype(int) #to int list
        transfered.append( one_answer)
        arguments.append(temp2)
      else:
        arguments.append(set())
        transfered.append([])

    #for_auto_reduction = pd.DataFrame(data=transfered, index=indexlist, columns=column_arg)  # test autoencoder




    processed_data = pd.DataFrame({
        'id': indexlist,
        'in': transfered,
        'arg': arguments,
    })

    print("generally process answer sets( read data, one hot, group label, ): ", time.process_time() - start)
    start2 = time.process_time()
    if eps != "" and eps != "Eps":
        processed_data = clustering_dbscan(processed_data, float(eps), int(minpts))
    else:
        processed_data = clustering_dbscan(processed_data)
    print("dbscan clustering: ", time.process_time() - start2)

    processed_data = dimensional_reduction(processed_data)
    y = np.array([np.array(xi) for xi in transfered])    #to change and test###############减少需要内存大小
    processed_data =dimensional_reduction_autoencoding(y, processed_data)
    #processed_data =dimensional_reduction_autoencoding(for_auto_reduction, processed_data)
    start3 = time.process_time()

    if n_cluster !="" and n_cluster !="Cluster Num":
        processed_data= clustering_km(processed_data,int(n_cluster))
    else:
        processed_data= clustering_km(processed_data)
    print("kmeans clustering: ", time.process_time() - start3)

    start4 = time.process_time()
    all_arguments = [item for sublist in arguments for item in sublist]
    frequency = np.array([])
    for argument in itemlist:
        #count = 0

        # for arg_list in arguments:
        #     if argument in arg_list:
        #         count += 1
        frequency=np.append(frequency,all_arguments.count(argument))
    rate=frequency / len(processed_data) *100
    bar_data = pd.DataFrame({
        # "index":itemlist,
        "argument": itemlist,#argument
        "frequency": frequency,
        "rate": rate
    })
    print("bar chart(argument frequency): ", time.process_time() - start4)
    #correlation matrix
    start5 = time.process_time()
    all_occurence=pd.DataFrame([x for x in processed_data['in']], columns=[str(x) for x in itemlist])
    to_drop = bar_data.loc[bar_data['rate'].isin([0, 100])].argument
    all_occurence.drop([str(x) for x in to_drop], axis='columns', inplace=True)

    temp = all_occurence.astype(int)
    correlation_matrix = temp.corr()
    print("create correlation matrix: ", time.process_time() - start5)

    #find features:
    start6 = time.process_time()
    common_all = set(arguments[0]).intersection(*arguments)
    cluster_feature_db=find_feature_cluster(common_all,processed_data,"db")
    cluster_feature_km=find_feature_cluster(common_all,processed_data,"km")
    #
    #

    processed_data_dir = dir + semantic
    #group_feature.to_pickle(dir + "group_feature.pkl")
    cluster_feature_db.to_pickle(processed_data_dir + "_db_cluster_feature.pkl")
    cluster_feature_km.to_pickle(processed_data_dir + "_km_cluster_feature.pkl")
    print("create feature report: ", time.process_time() - start6)
    processed_data.to_pickle(processed_data_dir + "_processed_data.pkl")
    bar_data["argument"]=[str(x)+"argument" for x in itemlist]
    bar_data.to_pickle(processed_data_dir + "_bar_data.pkl")
    correlation_matrix.to_pickle(processed_data_dir + "_correlation_matrix.pkl")

    # create a ZipFile object
    start7 = time.process_time()
    file_name=arguments_file.split("/")[-1]
    zipname = semantic+"_"+file_name.strip("apx") + "zip"
    path = Path(dir)
    zip_dir= str(path.parent) +"/processed_zip/"
    with ZipFile(zip_dir + zipname, 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in walk(dir):
            for filename in filenames:

                    # create complete filepath of file in directory
                    filePath = join(folderName, filename)
                    # Add file to zip
                    zipObj.write(filePath, arcname=filename)
    print("zip files: ", time.process_time() - start7)
    return True

