# -*- coding: utf-8 -*-
#     natural language neural network
#     each encountered signal (character) creates a Node
#     each Node can connect to (up to N) previously evoked Nodes to create a new Node
import operator
import re
import codecs
from os import listdir, path
from collections import Counter, OrderedDict
import json
import difflib
import numpy
from timeit import itertools
import csv

def count_patterns(text_list,n):
    ngram = zip(*[text_list[i:] for i in range(n)])
    count = Counter(ngram)
#     count = {}
#     for i in range(l,len(text)):
#         pattern = text[i-l:i+1]
#         if ' ' in pattern:
#             continue         
#         if(pattern not in count):
#             count[pattern] = len(re.findall(re.escape(pattern), text))
#     count = sorted(count.iteritems(), key=operator.itemgetter(1), reverse = True)

    f = codecs.open('freq'+str(n),'w','utf-8')
    f.write(('\n'.join([(''.join(y for y in x[0])) + ', ' + str(x[1]) for x in count.items()])))
    f.close()
        
    return count

def replace_patterns(text, count, threshold):
    replaced = 0
    for c in count:
        if c[1] > threshold:
            continue
        replaced += 1
        if(replaced % 1000 == 0):
            print(replaced)

        text = re.sub(' '+re.escape(c[0])+' ', ' * ', text)
    return replaced, text

def merge_pattern(text_list, pattern):
    separator = unichr(449)
    pattern1 = separator.join(pattern)
    pattern2 = ''.join(pattern)
    text = separator.join(text_list)
    text = re.sub(pattern1, pattern2, text)
    return text.split(separator)
    
    

def clean_text(text):
    text = text.lower()
    text = re.sub('\r\n', ' ', text)
    text = re.sub('([^\w\s])([^ ])','\\1 \\2', text)
    text = re.sub('([^ ])([^\w\s])','\\1 \\2', text)
    text = re.sub('\s+', ' ', text)
#     text = re.sub('.+\n\*\*\* start .+ \*\*\*\n', '',text)
#     text = re.sub('\n\*\*\* end .+ \*\*\*\n.+', '',text)
    return text

def combine_text():
    dir = 'res/'
    text = ""
    txtfiles = [ f for f in listdir(dir) if f.endswith('.txt') ]
    for tf in txtfiles:
        f = codecs.open(dir+tf,'r', "utf-8")
        text += f.read()
        f.close()
    f = codecs.open('text','w', "utf-8");
    f.write(text)
    f.close()
    return text

def get_difference(diffs):
    difference = []
    d1 = []
    d2 = []
    for d in diffs:
        if(unicode(d).startswith('-')):
            d1.append(d[2:])
        if(unicode(d).startswith('+')):
            d2.append(d[2:])
        if(unicode(d).startswith(' ')):
            if(len(d1) > 0 and len(d2) > 0):
                difference.append((tuple(d1),tuple(d2)))
            d1 = []
            d2 = []
    return(tuple(difference))

def get_matches(s1,sentences):
    matches = []
    for s2 in sentences:
#         l = len(s2)
#         if(s1 != s2 and s1[:l/2] == s2[:l/2] and s1[l/2+1:] == s2[l/2+1:]):
        if(s1[:-1] == s2[:-1] and s1[-1] != s2[-1]):
            matches.append(s2[-1])
    return matches

def list2file(l,filename):
    f = open(filename, 'w')
#     cw = csv.writer(f, delimiter='|')
    json.dump(l,f,indent=4, separators=(',', ': '),default=str)
#     for k,v in dict:
#         f.write(u' '.join(k).encode('utf-8')+','+str(v)+'\n')
#         cw.writerow([(str(k)),str(v)])

    f.close()

def file2list(filename):
#     dict = sorted(dict.iteritems(), key = operator.itemgetter(1), reverse=True)
    f = open(filename, 'r')
    l = json.load(f)    
#     for l in f.readlines():
#         l = l.split(',')
#         dict[l[0]] = l[1]
#     l = f.readlines()
#     f.close()
    return l


def calc_distances(seqs,words):
#     if(path.isfile('distmat')):
#         f = open('distmat','r')
#         distmat = numpy.load(f)
#         f.close()
#         return distmat
    words = dict(words)
    if(path.isfile('distdict')):
        distlist = file2list('distdict')
        return distlist
    d = {}
    for seq in seqs:
        s = list(seq)
        w = s.pop(N/2)
        s = tuple(s)
        if s not in d:
            d[s] = {}
        if w not in d[s]:
            d[s][w] = 0
        d[s][w] += 1
#     distmat = numpy.zeros(shape=(len(words),len(words)))
    distdict={}
    print(len(d)) 
    for s in d:
        pairs = itertools.permutations(d[s],2)
        for (w1,w2) in pairs:
            w = [w1,w2]
            c = [words[w1],words[w2]]
            w = tuple([x for (y,x) in sorted(zip(c,w))])
#                 dist = float(d[s][w[0]])+float(d[s][w[1]])
            dist = (float(d[s][w[0]])/c[0])*(float(d[s][w[1]])/c[1])
#             if w1 not in distdict:
#                 distdict[w1] = {}
#             if w2 not in distdict[w1]:
#                 distdict[w1][w2] = 0
#             distdict[w1][w2] += dist 
#             if w2 not in distdict:
#                 distdict[w2] = {}
#             if w1 not in distdict[w2]:
#                 distdict[w2][w1] = 0
            if w not in distdict:
                distdict[w] = 0
            distdict[w] += dist 
#                 distmat[i][j] += dist
#                 distmat[j][i] += dist
#     f = open('distmat', 'w')
#     numpy.save(f, distmat)
#     f.close()
    distlist = sorted(distdict.iteritems(), key = operator.itemgetter(1), reverse=True)
    list2file(distlist, 'distdict')
#     d = sorted(d.iteritems(), key = operator.itemgetter(1), reverse=True)
#     list2file(d, 'd')
    return distlist
       
def merge_closest(closest,distances,words):
    distdict = {}
    words = dict(words)
    k1 , v1 = closest
    for k,v in distances:
        if k1 == k:
            continue
        l = len(set(k1).intersection(set(k)))
        if l>0:
            w = list(set(k1).union(set(k)))
            c = [words[i] for i in w]
            w = tuple([x for (y,x) in sorted(zip(c,w))])
            if w not in distances:
                distdict[w] = 0
            distdict[w] += l*float(v)/len(k1)
        else:
            distdict[tuple(k)] = v
    distances = sorted(distdict.iteritems(), key = operator.itemgetter(1), reverse=True)
    return distances 
            
            
       
#    input: distance database
#    output: words clusters
#    method:
#     while distances length > 8
#        find closest words
#        merge closest words
def find_clusters(distances,words):
    l = len(distances)
    while True:
        closest = distances[0]
        distances = merge_closets(closest,distances,words)
        if(len(distances) < l/10):
            return distances
#     cluster_dict = {}
#     c=0
#     for k,v in distances:
#         if(w%1000 == 0):
#             print(w)        
#         find most similar
#         a = numpy.argmax(distances)
#         (i,j) = numpy.unravel_index(a, (len(words),len(words)))
#         distances[i][j] = 0
#         distances[j][i] = 0
#         k,v = d
#         link most similar
#         s1 = k[0]
#         s2 = k[1]
#         if(s1 in cluster_dict and s2 in cluster_dict):
#             continue
#             for s in cluster_dict:
#                 if cluster_dict[s] == cluster_dict[s2]:
#                     cluster_dict[s] = cluster_dict[s1]
#         elif s1 in cluster_dict:
#             cluster_dict[s2] = cluster_dict[s1]
#         elif s2 in cluster_dict:
#             cluster_dict[s1] = cluster_dict[s2]
#         else:
#         if(s1 not in cluster_dict and s2 not in cluster_dict):
#             cluster_dict[s1] = c
#             cluster_dict[s2] = c
#             c+=1
#     return cluster_dict
            
f = codecs.open('text','r', "utf-8")
text = f.read()
f.close()
# text = combine_text()
text = clean_text(text)

N = 7
K = 1000
L = 10000

words = text.split()
print('counting words')
if(path.isfile('word_count')):
    word_count = file2list('word_count')
else:
    word_count = Counter(words)
    word_count = sorted(word_count.iteritems(), key = operator.itemgetter(1), reverse=True)
    list2file(word_count,'word_count')    

# replaced, text = replace_patterns(text, count, K)
# print(replaced)

print('counting sentences')
    # sentences = text.split('.')
if(path.isfile('sentences')):
    sentences = file2list('sentence')
else:
    sentences = zip(*[words[i:] for i in range(N)])
    list2file(sentences, 'sentences')
if(path.isfile('sentence_count')):
    sentence_count = file2list('sentence_count')
else:
    sentence_count = Counter(sentences)
    # sentence_prob = {}
    # for k,v in sentence_count.items():
    #     sentence_prob[k] =v/word_count
    sentence_count = sorted(sentence_count.iteritems(), key = operator.itemgetter(1), reverse=True)
    list2file(sentence_count, 'sentence_count')

# words, counts = zip(*word_count) 
print('calculating distances')
distances = calc_distances(sentences, word_count)
print('finding clusters')
cluster_dict = find_clusters(distances,word_count)
print('writing to file')
cluster_dict = sorted(cluster_dict, key = operator.itemgetter(1), reverse=True)
list2file(cluster_dict, 'cluster_dict')


# sentences = list(set(sentences))
# word_sentences = []
# for i in range(len(sentences)):
#     word_sentences.append(sentences[i].split())
# sentences = word_sentences
# differences = []
# clusters =[]
# for i in range(len(sentences)):
#     print(i)
#     s = sentences[i]
#     matches = get_matches(s,sentences)
#     if(len(matches)>0):
#         matches.append(s[-1])
#         matches = tuple(sorted(matches))
#         if matches not in clusters:
#             clusters.append(matches)
#             f = open('clusters','a')
#             f.write(str(matches)+'\n')
#             f.close()
#             print(matches)
#     matches = difflib.get_close_matches(s, sentences, cutoff=0.7)
#     for j in range(len(matches)): 
#         m = matches[j] 
#         diffs = list(difflib.ndiff(s, m))
#         difference = get_difference(diffs)    
#         for d in difference:
#             differences.append(d)
#             print(str(s) + ' : ' + str(m))
#             count = Counter(differences)
#             count = sorted(count.iteritems(), key = operator.itemgetter(1), reverse=True)
#             f = open('differences','w')
#             for k,v in count:
#                 f.write(str((k,v))+'\n')
#             f.close()    
#             print(d)

         

# text_list = list(text)
# used = set()
# count = count_patterns(text_list, 2)
# pattern = count.most_common(1)[0][0]
# j=0
# i=0
# n=L
# while n >= L:
#     print(i)
#     i+=1
#     if( len(set(pattern).intersection(used)) > 0):
#         count = count_patterns(text_list, 2)
#         used = set()
#         j=0
#     used = used.union(set(pattern))
#     pattern,n = count.most_common()[j]
#     print(n)
#     j+=1
#     text_list = merge_pattern(text_list, pattern)
 
# new_list = []
# for i in text_list:
#     if len(i) == 1:
#         new_list.append('*')
#     else:
#         new_list.append(i)
# text_list = new_list

# layers = {}
# for i in range(N):
#     layers[i] = count_patterns(text, i)
    
#     replaced = 1
#     while(replaced > 0):
#         layers[i] = count_patterns(text, i)
#         replaced, text = replace_patterns(text, layers[i],K)
#         print(replaced)
    
# in_dict = True 
# while len(symbol_dict)>0 and in_dict:
#     layers[1] = count_patterns(text, 1)
#     pattern = layers[1][0][0]
#     print(pattern)
#     if pattern in symbol_dict:
#         text = re.sub(re.escape(pattern), symbol_dict[pattern], text)
#         symbol_dict.pop(pattern)
#     else:
#         in_dict = False
#         print('not in dict')
#         break
        

f = codecs.open('text1','w', "utf-8")
f.write(''.join(text))
f.close()

