import re
import os
import sys
import string
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from math import ceil
from difflib import SequenceMatcher

now = datetime.now().strftime("%Y%m%d%H%M%S")

#region Functions
# Подготовка строк к векторизации
def preprocessing(line:str):
    line = line.lower()
    line = re.sub(r"[{}]".format(string.punctuation), "", line)
    return line

# Кластеризация
def kmeans_clustering(dataset:list, n_clusters:int, save_to_file:bool=False) -> list:
    r""" Функция кластеризации набора данных
    Параметры:
        dataset - список данных
        n_clusters - количество кластеров
        save_to_file - сохранять ли кластеры в файлы?

    Возвращает:
        Список кластеров
    """

    # Создание модели
    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocessing, token_pattern = r"\b\S{1,}\b")
    tfidf = tfidf_vectorizer.fit_transform(dataset)
    kmeans = KMeans(n_clusters = n_clusters, max_iter = 500).fit(tfidf)

    # Список списков
    clusters = [None] * n_clusters
    for index in range(len(dataset)):
        if clusters[kmeans.labels_[index]] == None:
            clusters[kmeans.labels_[index]] = [dataset[index]]
        else:
            clusters[kmeans.labels_[index]].append(dataset[index])

    # Сохранение кластеров по своим файлам
    if save_to_file == True:
        os.makedirs("Data\\Data_{}\\Clusters".format(now), exist_ok=True)
        for cluster_indx in range(len(clusters)):
            file = open("Data\\Data_{}\\Clusters\\cluster_{}.txt".format(now, cluster_indx), "w", encoding = "utf-8")
            for line in clusters[cluster_indx]:
                file.write("{}\n".format(line))
            file.close()

    return clusters

# Извлечение ключевых слов из кластера
def cluster_key_words_extractor(cluster:list) -> list:
    r""" Функция извлечения ключевых слов из кластера
    Параметры:
        cluster - список текстовых данных

    Возвращает:
        Список ключевых слов
    """

    # Вектроизация слов
    vectorizer = TfidfVectorizer(preprocessor = preprocessing, token_pattern = r"\b\S{1,}\b")
    tfidf = vectorizer.fit_transform(cluster)

    # Ниже страшные вещи (желательно исправить)
    # Массив (индекс, значение)
    index_value = {i[1]:i[0] for i in vectorizer.vocabulary_.items()}
    line_word_weights = []
    for row in tfidf:
        line_word_weights.append({index_value[column]:value for (column,value) in zip(row.indices,row.data)})

    # Словарь (слово, значение - "важность слова (чем чаще, тем важнее)")
    word_weights = {}
    for line in line_word_weights:
        for word in line:
            if word_weights.get(word) == None:
                word_weights[word] = 1 - line[word]
            else:
                word_weights[word] += 1 - line[word]

    # Список ключевых слов
    key_words = []
    for word in word_weights:
        value = word_weights[word] / len(word_weights) * 100
        if value > 1:
            key_words.append(word)

    return key_words

# Шаблонизатор строки по ключевым словам
def regex_maker(line:str, key_words:list, ex_key_words:list=[]) -> str:
    r""" Функция генерации регулярного выражения
    Параметры:
        line - список сообщений
        key_words - список ключевых слов, которые должны сохраниться в строке
        ex_key_words - список слов, которые должны быть заменены (по-умолчанию - [])

    Возвращает:
        Регулярное выражение для данной строки, включая один список слов и исключая другой
    """

    # Удаление "лишних"(?) символов
    line = re.sub(r"[!№#%.,:;*?\\/()+-]", "", line).replace('>', '').replace('<', '')

    # Удаление слов, которые нужно исключить, из списка ключевых слов
    for ex_word in ex_key_words:
        if ex_word in key_words:
            key_words.remove(ex_word)

    # Обработанная строка (не ключевые слова и значения заменнены на
    # соответствующие им местозаполнители)
    new_line = ""
    for word in line.split():
        if word.lower() in key_words:
            new_line += "{} ".format(word)
        else:
            if word.isdigit():
                new_line += "%d "
            else:
                new_line += "%w "

    new_line, line = (new_line + "_").split(), ""

    index = 0
    while index < len(new_line):
        if new_line[index] == "%d":
            rept = 1
            while index < len(new_line) and new_line[index + 1] == "%d":
                rept += 1
                index += 1
            if rept == 1:
                #line += "%d "
                line += "%d1 "
            else:
                #line += "%d{{1,{}}} ".format(rept)
                line += "%d{} ".format(rept)
        elif new_line[index] == "%w":
            rept = 1
            while index < len(new_line) and new_line[index + 1] == "%w":
                rept += 1
                index += 1
            if rept == 1:
                #line += "%w "
                line += "%w1 "
            else:
                #line += "%w{{1,{}}} ".format(rept)
                line += "%w{} ".format(rept)
        elif new_line[index] == "_":
            break
        else:
            line += "{} ".format(new_line[index])
        index += 1

    return line

# Функция включения одного регулярного выражения в другое
def regex_union(str1:str, str2:str) -> str:
    r""" Функция объединения двух регулярных выражений по наибольшему вхождению символа
    Параметры:
        str1 - первая строка, содержащая регулярное выражение
        str2 - вторая строка, содержащая регулярное выражение

    Возвращает:
        Строку содержащую регулярное выражение
    """

    str1 = str1.split()
    str2 = str2.split()

    result = ""
    for x in range(len(str1)):
        if str1[x] == str2[x]:
            result += "{} ".format(str1[x])
        elif (re.match("%w\d+", str1[x]) and re.match("%w\d+", str2[x])) or (re.match("%d\d+", str1[x]) and re.match("%d\d+", str2[x])):
            symbol = re.findall(r'\w', str1[x])
            num1 = re.findall(r'\d+', str1[x])
            num2 = re.findall(r'\d+', str2[x])
            result += "%{}{} ".format(symbol[0], num1[0]) if num1 > num2 else "%{}{} ".format(symbol[0], num2[0])
        else:
            return None

    return result

# Финальные обработка регулярного выражения перед выводом
def regex_postprocessing(templates:list) -> list:
    r""" Функция конечной обработки регулярных выражений
    Параметры:
        templates - список текстовых данных, содержащих регулярные выражения
        
    Возвращает:
        Список регулярных выражений
    """

    tmp_templates = []
    for template in templates:
        tmp = ""
        for word in template[0].split():
            if re.match('%w\d+', word):
                num = re.findall(r'\d+', word)[0]
                if num == '1':
                    tmp += "%w "
                else:
                    tmp += "%w{{1,{}}} ".format(num)
            elif re.match('%d\d+', word):
                num = re.findall(r'\d+', word)[0]
                if num == '1':
                    tmp += "%d "
                else:
                    tmp += "%d{{1,{}}} ".format(num)
            else:
                tmp += "{} ".format(word)
        tmp_templates.append([tmp, template[1]])
    return tmp_templates

# Функция обработки шаблонов
def regex_combiner(templates:list) -> list:
    r""" Функция объединения и обработки регулярных выражений, из большого количества регулярных выражений, делает поменьше
    Параметры:
        templates - список регулярных выражений

    Возвращает:
        Список [регулярное выражение, количество строк]
    """
    
    tmp_templates = []
    for j in range(10):
        if len(templates) == 0:
            break
        counter = 0
        first = templates.pop(0)
        index = 0
        while index < len(templates):
            if SequenceMatcher(a=first, b=templates[index]).ratio() > 0.9 and len(first.split()) == len(templates[index].split()):
                tmp = regex_union(first, templates[index])
                if tmp == None:
                    index += 1
                    continue
                else:
                    first = tmp
                templates.remove(templates[index])
                counter += 1
                if index > 1:
                    index -= 1
                else:
                    index = 0
            else: 
                index += 1

        if counter != 0:
            tmp_templates.append([first, counter + 1])
    return regex_postprocessing(tmp_templates)
#endregion


def main(argv):
    if len(argv) < 3:
        print("Missing args")
        print("Try Bank.py {dataset_path} {number_of_clusters}")
        return

    print("Start.")
    
    print("Clustering.")
    dataset = open(argv[1], encoding = "utf-8").read().split("\n")
    clusters = kmeans_clustering(dataset, int(argv[2]))

    # Шаблонизация
    print("Creating templates.")
    os.makedirs("Data\\Data_{}\\Templates".format(now), exist_ok=True)
    all_templates = []
    for i, cluster in enumerate(clusters):
        print("\tCluster [{} / {}]:\t0%   ".format(i + 1, len(clusters)), end = "\r")  # progress bar
        key_words_list = cluster_key_words_extractor(cluster)
        
        templates = []
        for j, line in enumerate(cluster):
            templates.append(regex_maker(line, key_words_list))
            print("\tCluster [{} / {}]:\t{}% ".format(i + 1, len(clusters), ceil(j / len(cluster) * 99)), end = "\r")  # progress bar
        
        templates = regex_combiner(templates)
        all_templates.extend(templates)

        # Сохранение шаблонов по кластерам
        file = open("Data\\Data_{}\\Templates\\cluster_{}.txt".format(now, i), "w", encoding = "utf-8")
        for line in templates:
            file.write("({})-\t{}\n".format(line[1], line[0]))
        file.close()
        print("\tCluster [{} / {}]:\t100% ".format(i + 1, len(clusters)), end = "\r")  # progress bar

    # Сохранение всех шаблонов в один файл
    file = open("Data\\Data_{0}\\templates_{0}.txt".format(now), "w", encoding = "utf-8")
    for line in all_templates:
        file.write("({})-\t{}\n".format(line[1], line[0]))
    file.close()

    print()
    print("Done.")

#main(sys.argv)
main(["", "Xmpls\\sms_dataset_10000.txt", 28])

