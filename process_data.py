import re
import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from file_manage import clean_folder
from sklearn.cluster import KMeans,DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from os import listdir,unlink,walk
from os.path import isfile, join
from zipfile import ZipFile
from pathlib import Path
import time
import itertools
from BayesianOptimization import bayesian_optimization
from rq import get_current_job
import os
from sklearn.model_selection import train_test_split
#import subprocess
import textwrap
from clingo_asp_compute import compute_extensions

import random


def get_colors(n):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append('rgb' + str((r, g, b)))
    return ret


def get_color_label(processed_data, color_label, groups_set):
    processed_data['color'] = processed_data[color_label]
    # if color_label =='category':  #seperate semantic extensions
    # d8b365
    # f5f5f5
    # 5ab4ac
    colors = ['#FFF8E3', '#c7eae5']  # f6e8c3
    if len(groups_set) < 2:
        processed_data["color"] = '#c7eae5'
    else:
        for x in groups_set:
            if 'and' in x:
                processed_data["color"].replace({x: '#F0F0F0'}, inplace=True)  # f5f5f5
            else:
                processed_data["color"].replace({x: colors[0]}, inplace=True)
                del colors[0]
class Process_data:
    def __init__(self, stage):
        self.stage = stage
        #self.processed_data = pd.DataFrame()




    def process_extension_individual(self,question, item, processed_dir, upload_dir, extenion_dir): # for "other" situation, when user want to select their own semantics or semantic pairs

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
                asp_encoding = "semi-cond-disj.dl"

                end = "SEMI-STB"
            elif item == 'cf2':
                asp_encoding = "cf2_web.dl"
                end = "CF2"
            # elif item == 'stage2':
            #     asp_encoding = "stage2_web.txt"
            #     end = "STG2"
            else:
                return False
            extension_file = "{}.EE_{}".format(question, end)
            self.stage='extension computing of a single semantics'
            compute_extensions(upload_dir +question,asp_encoding,extenion_dir+extension_file)

            return extension_file
            # os.system(
            #     "D:/test2/clingo-4.5.4-win64/clingo.exe {} data/app_uploaded_files/{} 0 > data/extension_sets/{}".format(
            #         asp_encoding, question, extension_file))
            # process_data_two_sets(processed_dir, upload_dir + question, extenion_dir + extension_file, eps, minpts,
            #              n_cluster, item)





    def find_semantic_files(files,item):#find corresponding extension file
        if item == 'stable':

            end = "STB"
        elif item == 'preferred' or item== 'preferred_stable':

            end = "PR"
        elif item == 'stage' or item=='stable_stage':
            asp_encoding = "stage-cond-disj.dl"
            end = "STG"
        elif item == 'semi-stable':

            end = "SEMI-STB"
        elif item == 'cf2' or item == 'stable_cf2':

            end = "CF2"
        # elif item == 'stage2' or item=='stage2_stable':
        #
        #     end = "STG2"
        ends='.EE_'+end
        for x in files:
            if x.endswith(ends):
                return x
        return None

    def addional_process_individual(self,processed_dir, semantics):
        self.stage='combine extensions of two semantics'
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

        feature1 = self.find_feature_group(common_all, processed_data_stage2, processed_data_cf2)
        feature2 = self.find_feature_group(common_all, processed_data_cf2, processed_data_stage2)
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



    def get_catogery(self,processed_dir, semantics):
        self.stage='get semantics label'
        if len(semantics)==1:
            return False
        semantic1=semantics[0]
        semantic2=semantics[1]
        processed_data_stage2 = pd.read_pickle(processed_dir + semantic1 + "_processed_data.pkl")
        processed_data_cf2 = pd.read_pickle(processed_dir + semantic2 + "_processed_data.pkl")
        common_data = pd.merge(processed_data_cf2, processed_data_stage2, on=['arg'], how='inner')
        #present_data1 = processed_data_cf2[~processed_data_cf2.id.isin(common_data.id_x)]
        processed_data_stage2["category"] = "only_"+semantic1
        processed_data_stage2[processed_data_stage2.id.isin(common_data.id_x)]["category"]=semantic1+" and " + semantic2
        #present_data2 = processed_data_stage2[~processed_data_stage2.id.isin(common_data.id_y)]
        processed_data_cf2["category"] = "only_"+semantic2
        processed_data_cf2[processed_data_cf2.id.isin(common_data.id_x)]["category"] = semantic1+" and " + semantic2
        processed_data_stage2.to_pickle(processed_dir + semantic1 + "_processed_data.pkl")
        processed_data_cf2.to_pickle(processed_dir + semantic2 + "_processed_data.pkl")









        # else:
        #     # ef8a62
        #     # ffffff
        #     # 999999
        #     colors=['#f6e8c3','#f5f5f5','#c7eae5'] #['#e41a1c','#377eb8','#4daf4a']
        #     for x in range(0,len(groups_set)):
        #         processed_data["color"].replace({groups_set[x]: colors[x]}, inplace=True)

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

    def clustering_km(self, data, cluster_num=2):
        self.stage = 'Kmeans clustering'
        km = KMeans(n_clusters=cluster_num, precompute_distances='auto').fit_predict(list(data['in']))
        data['km_cluster_label'] = [i + 1 for i in km]
        return data

    def clustering_dbscan(self, data, eps=1.7, minpoint=7):
        self.stage = 'DBscan clustering'
        c = DBSCAN(eps=eps, min_samples=minpoint).fit_predict(list(data['in']))
        data['db_cluster_label'] = [i + 1 for i in c]
        return data

    def dimensional_reduction_autoencoding(self, data, processed_data) -> pd.DataFrame:
        start = time.process_time()
        self.stage = 'dimentional reduction autoencoder'
        encoding_dim = 2
        original_shape = data.shape[1]
        input_df = Input(shape=(original_shape,))
        encoded = Dense(encoding_dim, activation='relu')(
            input_df)  # linear activity_regularizer=regularizers.l1(1e-4), bias_regularizer=regularizers.l2(1e-5),
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
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(data, data,
                        epochs=150,  # 50
                        batch_size=256,
                        shuffle=True,
                        # validation_data=(data_val, data_val)
                        )
        encoded_X_train = encoder.predict(data).T
        processed_data['auto_position_x'] = encoded_X_train[0]
        processed_data['auto_position_y'] = encoded_X_train[1]
        print("Autoencoding dimensional reduciton: ", time.process_time() - start)
        return processed_data

    def dimensional_reduction(self, data):
        start1 = time.process_time()
        self.stage = 'dimentional reduction Tsne'
        result2 = TSNE(n_components=2).fit_transform(list(data['in'])).T
        data['tsne_position_x'] = result2[0]
        data['tsne_position_y'] = result2[1]
        print("Tsne dimensional reduciton: ", time.process_time() - start1)
        svd = TruncatedSVD(n_components=2, n_iter=7)
        start2 = time.process_time()
        self.stage = 'dimentional reduction svd'
        result = svd.fit_transform(list(data['in'])).T
        print("SVD dimensional reduciton: ", time.process_time() - start2)
        data['svd_position_x'] = result[0]
        data['svd_position_y'] = result[1]
        return data

    def add_to(self, feature, combine):
        mask = [combine.issubset(x) for x in feature]
        if any(mask):
            super_f = list(itertools.compress(feature, mask))
            for x in super_f:
                feature.remove(x)

            feature.append(combine)

    def find_feature_cluster(self, common_all, data, labels):  # clustered data
        if labels != "groups":
            labels = labels + "_cluster_label"

        clusters = data[labels].unique().tolist()
        if len(clusters) == 1 or len(clusters) > 50:
            return pd.DataFrame([])
        all_feature = []
        cluster_with_feature = clusters.copy()
        for cluster in clusters:
            feature = []
            current_cluster = data[data[labels] == cluster]
            all_lists = list(current_cluster.arg)
            common_links = set(all_lists[0]).intersection(*all_lists[1:])
            common_links = common_links - common_all

            # other_arguments = [item for sublist in other_cluster_arg for item in sublist]
            # mask = [a not in other_arguments for a in common_links]
            # if any(mask):
            #     feature = list(itertools.compress(common_links, mask))

            # 2021.02.21 for single feature
            has_single_feature = 0
            other_cluster_arg_flat_list = [x for x in data[data[labels] != cluster].arg]
            other_cluster_arg_combine = set([item for sublist in other_cluster_arg_flat_list for item in sublist])
            # other_cluster_arg_combine=set(other_cluster_arg_combine)
            for x in common_links:
                if x not in other_cluster_arg_combine:
                    feature.append(x)
                    has_single_feature = 1
            if has_single_feature:
                all_feature.append(str(feature).strip('[]'))






            else:
                other_cluster_arg = [set(x) for x in data[data[labels] != cluster].arg]
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
                                self.add_to(feature, combine)
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

    def find_feature_group(self,common_all, data, otherdata):  # clustered data

        feature = []
        arguments1 = data.arg
        common_links1 = set(arguments1[0]).intersection(*arguments1)
        # arguments2 = data2.arg
        # common_links2 = set(arguments2[0]).intersection(*arguments2)
        # arguments=arguments1.append(arguments2)
        # common_all=set(arguments[0]).intersection(*arguments)
        common_links = common_links1 - common_all
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
                        itertools.combinations(common_links,
                                               x - 1))  # for common_links whose length is bigger than 15, 3003 iteration for x =11. Just too many
                    for combine in combinitions:
                        combine = set(combine)
                        if not any(combine.issubset(x) for x in other_cluster_arg):
                            temp = True
                            self.add_to(feature, combine)
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

    def process_data(self,dir, arguments_file, answer_sets, eps, minpts, n_cluster, use_optim,semantic):

        #os.system(
        #    "D:/test2/clingo-4.5.4-win64/clingo.exe prefex.dl apx_files/AachenerVerkehrsverbund_26October2020.zip_train+metro+tram+bus.lp.apx 0 > extension_sets/test.EE_PR")
        start = time.process_time()
        self.stage='data preprocess: extraction and transformation'
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

        print("1.progress chldren{}, super{}".format(os.getpid(), os.getppid()))

        if "preferred_stable" in semantic:
            difference="not_defeated"
            big="preferred"
            small="preferred and stable"
            #middle='preferred and stable'
        elif "stable_stage" in semantic:
            difference="nrge"
            big="stage"
            small="stage and stable"
            #middle = 'stage and stable'
        # elif "stage2_stable" in semantic:
        #     difference = "not_defeated"
        #     big = "stage2"
        #     small = "stage2 and stable"
            #middle = 'stage2 and stable'
        elif "stable_cf2" in semantic:
            difference = "not_defeated"
            big = "cf2"
            small = "cf2 and stable"
            #middle = 'cf2 and stable'
        else:
            raise Exception

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
        print("2.progress chldren{}, super{}".format(os.getpid(), os.getppid()))
        print("generally process answer sets( read data, one hot, group label, ): ", time.process_time() - start)
        start2 = time.process_time()
        self.stage='bayesian optimization on DBscan '
        if use_optim:
            eps, minpts = bayesian_optimization(processed_data, 'dbscan')

            processed_data = self.clustering_dbscan(processed_data, eps, minpts)
        else:
            if eps ==None or minpts == None:
                return 'parameter mistakes'
            else:
                processed_data = self.clustering_dbscan(processed_data, float(eps), int(minpts))
        # if eps != None:
        #     if minpts != None:
        #         processed_data = clustering_dbscan(processed_data, float(eps), int(minpts))
        #     else:
        #         processed_data = clustering_dbscan(processed_data, float(eps))
        # else:
        #     if minpts != None:
        #         processed_data = clustering_dbscan(data=processed_data, minpoint= int(minpts))
        #     else:
        #         processed_data = clustering_dbscan(processed_data)
        print("dbscan clustering: ", time.process_time() - start2)

        processed_data = self.dimensional_reduction(processed_data)
        y = np.array([np.array(xi) for xi in transfered])    #to change and test###############减少需要内存大小
        processed_data =self.dimensional_reduction_autoencoding(y, processed_data)
        #processed_data =dimensional_reduction_autoencoding(for_auto_reduction, processed_data)
        start3 = time.process_time()
        if use_optim:
            self.stage = 'bayesian optimization on DBscan '
            n_cluster = bayesian_optimization(processed_data, 'kmeans')
            processed_data = self.clustering_km(processed_data, n_cluster)
        else:
            if n_cluster ==None:
                return 'parameter_mistakes'
            else:
                processed_data=self.clustering_km(processed_data,int(n_cluster))

                # if n_cluster !=None and n_cluster !="2":
                #     processed_data= clustering_km(processed_data,int(n_cluster))
                # else:
                #     processed_data= clustering_km(processed_data)
        print("kmeans clustering: ", time.process_time() - start3)
        self.stage = 'creating bar chart '
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
        self.stage = "creating correlation matrix"
        start5 = time.process_time()
        all_occurence=pd.DataFrame([x for x in processed_data['in']], columns=[str(x) for x in itemlist])
        to_drop = bar_data.loc[bar_data['rate'].isin([0, 100])].argument
        all_occurence.drop([str(x) for x in to_drop], axis='columns', inplace=True)

        temp = all_occurence.astype(int)
        correlation_matrix = temp.corr()
        print("create correlation matrix: ", time.process_time() - start5)
        print("4.progress chldren{}, super{}".format(os.getpid(), os.getppid()))
        #find features:
        self.stage = "find identifiers"
        start6 = time.process_time()
        common_all = set(arguments[0]).intersection(*arguments)
        cluster_feature_db=self.find_feature_cluster(common_all,processed_data,"db")
        cluster_feature_km=self.find_feature_cluster(common_all,processed_data,"km")
        group_feature=self.find_feature_cluster(common_all,processed_data,"groups")
        #
        #
        clean_folder(dir)
        group_feature.to_pickle(dir + "group_feature.pkl")
        cluster_feature_db.to_pickle(dir + "db_cluster_feature.pkl")
        cluster_feature_km.to_pickle(dir + "km_cluster_feature.pkl")
        print("create feature report: ", time.process_time() - start6)
        self.stage='store processed data'
        processed_data.to_pickle(dir + "processed_data.pkl")
        bar_data["argument"]=[str(x)+"argument" for x in itemlist]
        bar_data.to_pickle(dir + "bar_data.pkl")
        correlation_matrix.to_pickle(dir + "correlation_matrix.pkl")

        # create a ZipFile object
        start7 = time.process_time()
        file_name=arguments_file.split("/")[-1]

        parameter='-Eps('+str(eps)+')-MinP('+str(minpts)+')-Cluster_num('+str(n_cluster)+')'
        zipname = big+"_"+file_name.strip(".apx") +parameter+ ".zip"
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






    def initial_process_individual(self,dir, arguments_file, answer_sets,semantic):
        start = time.process_time()
        self.stage='data preprocess: extraction and transformation'
        with open(arguments_file, 'r') as file:
            question = file.read()
        itemlist = [s for s in re.findall(r"arg[(]a(.*?)[)].", question)]
        itemlist.sort()
        # column_arg = [str(s) for s in itemlist]    #test autoencoder
        with open(answer_sets, 'r') as file:
            answer = file.read()

        test = answer.split("Answer:")
        del test[0]
        # indexlist = [int(s.split("\n",1)[0]) for s in test]
        arg_len = len(test)
        if arg_len == 0:
            return False
        indexlist = range(1, arg_len + 1)
        transfered = []
        arguments = []

        for s in test:
            temp1 = re.findall(r"^in(.*)", s, re.M)
            if temp1:
                temp2 = [s for s in re.findall(r'\d+', temp1[0])]
                bool_represent = np.in1d(itemlist, temp2)  # boolean list representation
                temp2 = frozenset(temp2)
                one_answer = bool_represent.astype(int)  # to int list
                transfered.append(one_answer)
                arguments.append(temp2)
            else:
                arguments.append(set())
                transfered.append([])

        # for_auto_reduction = pd.DataFrame(data=transfered, index=indexlist, columns=column_arg)  # test autoencoder

        processed_data = pd.DataFrame({
            'id': indexlist,
            'in': transfered,
            'arg': arguments,
        })
        processed_data.to_pickle(dir+semantic+'_processed_data.pkl')
        print("generally process answer sets( read data, one hot, group label, ): ", time.process_time() - start)
        return transfered,arguments, itemlist


    def process_data_two_sets(self, dir, arguments_file, transfered, arguments, itemlist, eps, minpts, n_cluster, use_optim, semantic): #process need to handlle two semantics seperately
        #transfered, arguments, itemlist=initial_process_individual(dir, arguments_file, answer_sets,semantic)

        # start = time.process_time()
        #
        # with open(arguments_file, 'r') as file:
        #     question = file.read()
        # itemlist = [s for s in re.findall(r"arg[(]a(.*?)[)].", question)]
        # itemlist.sort()
        # #column_arg = [str(s) for s in itemlist]    #test autoencoder
        # with open(answer_sets, 'r') as file:
        #     answer = file.read()
        #
        # test = answer.split("Answer:")
        # del test[0]
        # #indexlist = [int(s.split("\n",1)[0]) for s in test]
        # arg_len=len(test)
        # if arg_len == 0:
        #     return False
        # indexlist = range(1,arg_len+1)
        # transfered=[]
        # arguments=[]
        #
        #
        # for s in test:
        #   temp1=re.findall(r"^in(.*)", s, re.M)
        #   if temp1:
        #     temp2=[s for s in re.findall(r'\d+', temp1[0])]
        #     bool_represent = np.in1d(itemlist, temp2) #boolean list representation
        #     temp2=frozenset(temp2)
        #     one_answer=bool_represent.astype(int) #to int list
        #     transfered.append( one_answer)
        #     arguments.append(temp2)
        #   else:
        #     arguments.append(set())
        #     transfered.append([])
        #
        # #for_auto_reduction = pd.DataFrame(data=transfered, index=indexlist, columns=column_arg)  # test autoencoder
        #
        #
        #
        #
        # processed_data = pd.DataFrame({
        #     'id': indexlist,
        #     'in': transfered,
        #     'arg': arguments,
        # })
        #
        # print("generally process answer sets( read data, one hot, group label, ): ", time.process_time() - start)
        start2 = time.process_time()
        processed_data=pd.read_pickle(dir+semantic+'_processed_data.pkl')
        self.stage='bayesian optimization on DBscan '
        if use_optim:

            eps,minpts = bayesian_optimization(processed_data, 'dbscan')
            processed_data = self.clustering_dbscan(processed_data, eps, minpts)
        else:
            if n_cluster == None:
                return 'parameter_mistakes'
            else:

                processed_data = self.clustering_dbscan(processed_data, float(eps), int(minpts))

        # if eps != "" and eps != "Eps":
        #     processed_data = clustering_dbscan(processed_data, float(eps), int(minpts))
        # else:
        #     processed_data = clustering_dbscan(processed_data)
        print("dbscan clustering: ", time.process_time() - start2)

        processed_data = self.dimensional_reduction(processed_data)
        y = np.array([np.array(xi) for xi in transfered])    #descreaed required memory size
        processed_data =self.dimensional_reduction_autoencoding(y, processed_data)
        #processed_data =dimensional_reduction_autoencoding(for_auto_reduction, processed_data)
        start3 = time.process_time()



        # if n_cluster !="" and n_cluster !="Cluster Num":
        #     processed_data= clustering_km(processed_data,int(n_cluster))
        # else:
        #     processed_data= clustering_km(processed_data)
        #
        self.stage = 'bayesian optimization on Kmeans '
        if use_optim:

            n_cluster = bayesian_optimization(processed_data, 'kmeans')
            processed_data = self.clustering_km(processed_data, n_cluster)
        else:
            if n_cluster ==None:
                return 'parameter_mistakes'
            else:
                processed_data= self.clustering_km(processed_data,int(n_cluster))
        print("kmeans clustering: ", time.process_time() - start3)
        start4 = time.process_time()
        self.stage='creating bar chart'
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
        self.stage="creating correlation matrix"
        all_occurence=pd.DataFrame([x for x in processed_data['in']], columns=[str(x) for x in itemlist])
        to_drop = bar_data.loc[bar_data['rate'].isin([0, 100])].argument
        all_occurence.drop([str(x) for x in to_drop], axis='columns', inplace=True)

        temp = all_occurence.astype(int)
        correlation_matrix = temp.corr()
        print("create correlation matrix: ", time.process_time() - start5)

        #find features:
        start6 = time.process_time()
        self.stage = "find identifiers"
        common_all = set(arguments[0]).intersection(*arguments)
        cluster_feature_db=self.find_feature_cluster(common_all,processed_data,"db")
        cluster_feature_km=self.find_feature_cluster(common_all,processed_data,"km")
        #
        #

        processed_data_dir = dir + semantic
        #group_feature.to_pickle(dir + "group_feature.pkl")
        cluster_feature_db.to_pickle(processed_data_dir + "_db_cluster_feature.pkl")
        cluster_feature_km.to_pickle(processed_data_dir + "_km_cluster_feature.pkl")
        print("create feature report: ", time.process_time() - start6)
        self.stage = "storing processed data"
        processed_data.to_pickle(processed_data_dir + "_processed_data.pkl")
        bar_data["argument"]=[str(x)+"argument" for x in itemlist]
        bar_data.to_pickle(processed_data_dir + "_bar_data.pkl")
        correlation_matrix.to_pickle(processed_data_dir + "_correlation_matrix.pkl")

        # create a ZipFile object
        start7 = time.process_time()
        file_name=arguments_file.split("/")[-1]
        parameter = '-Eps(' + str(eps) + ')-MinP(' + str(minpts) + ')-Cluster_num(' + str(n_cluster) + ')'
        zipname = semantic + "_" + file_name.strip(".apx") + parameter + ".zip"
        #zipname = semantic+"_"+file_name.strip("apx") + "zip"
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

