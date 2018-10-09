# Python program for insert and search
# operation in a Trie
import  os
import cv2
import numpy as np
from os.path import basename
import re

dict={}
char_to_id={}


class TrieNode:

    # Trie node class
    def __init__(self):
        self.children = [None] * 100

        # isEndOfWord is True if node represent the end of the word
        self.isEndOfWord = False


class Trie:

    # Trie data structure class
    def __init__(self):
        self.root = self.getNode()
        self.count=0

    def getNode(self):

        # Returns new trie node (initialized to NULLs)
        return TrieNode()

    def _charToIndex(self, ch):

        # private helper function
        # Converts key current character into index
        # use only 'a' through 'z' and lower case
        return char_to_id[ch]

        #return ord(ch) - ord('a')

    def insert(self, key):

        # If not present, inserts key into trie
        # If the key is prefix of trie node, 
        # just marks leaf node
        pCrawl = self.root
        length = len(key)
        for level in range(length):
            index = self._charToIndex(key[level])

            # if current character is not present
            if not pCrawl.children[index]:
                pCrawl.children[index] = self.getNode()
            pCrawl = pCrawl.children[index]

        # mark last node as leaf
        pCrawl.isEndOfWord = True

    def search(self, key):

        # Search key in the trie
        # Returns true if key presents 
        # in trie, else false
        pCrawl = self.root
        length = len(key)
        for level in range(length):
            index = self._charToIndex(key[level])
            if not pCrawl.children[index]:
                return False
            pCrawl = pCrawl.children[index]

        return pCrawl != None and pCrawl.isEndOfWord

    def word_to_vector(self,fileName):

        # Search key in the trie
        # Returns true if key presents
        # in trie, else false
        global str
        key=""
        filepath='literature_labels/%s'%fileName
        with open(filepath) as fp:
            line = fp.readline()
            line=line.strip()
            key=line

        f = open('label_word_id/%s' % fileName, "w+")
        print(fileName)
        #key.replace(" ", "")

        key=re.sub('[\s+]', '', key)
        print(key)

        if key=="$" or key=="":
            return  False


        pCrawl = self.root
        length = len(key)
        s=""
        start=0
        prev=0
        level=0
        cnt=0

        while level<length:
            cnt+=1
            if cnt>30:
                return False
            index = self._charToIndex(key[level])
            if not pCrawl.children[index]:
                string=key[start:prev+1]

                if string not in dict.keys():
                    f.close()
                    os.remove('label_word_id/%s' % fileName)
                    return False

                if string=="o" and prev!=0:
                    start = prev + 1
                    pCrawl = self.root
                    continue

                id=dict[string]
                #f.write('{}'.format(id))
                f.write(str(id)+'\n')
                start=prev+1
                pCrawl = self.root
                continue
            if level==length-1 and pCrawl.isEndOfWord==False:
                string = key[start:prev + 1]

                if string not in dict.keys():
                    f.close()
                    os.remove('label_word_id/%s' % fileName)
                    return False

                if string=="o" and prev!=0:
                    start = prev + 1
                    pCrawl = self.root
                    continue


                id = dict[string]
                #f.write('{}'.format(id))
                f.write(str(id)+'\n')
                start = prev + 1
                pCrawl = self.root
                continue

            if pCrawl.isEndOfWord:
                prev=level
            level+=1
            pCrawl = pCrawl.children[index]



        string=(key[start:level])

        if string not in dict.keys():
            f.close()
            os.remove('label_word_id/%s' % fileName)
            return False

        if string == "o" and prev != 0:
            return True

        id = dict[string]
        #f.write('{}'.format(id))
        f.write(str(id) + '\n')
        self.count+=1
        return  True







# driver function

# Trie object
t = Trie()

def cre_dict():
    filepath = 'C:/Users/Badhon/Desktop/imran/bangla_letters.txt'

    cnt = 1
    with open(filepath) as fp:
        line = fp.readline()
        splited = line.split(' ')
        for item in splited:
            item = item.rstrip()
            dict[item] = cnt

        while line:
            # print("Line {}: {}".format(cnt, line.strip()))
            line = fp.readline()
            splited = line.split(' ')
            cnt += 1

            for item in splited:
                item = item.rstrip()
                # t.insert(item)
                dict[item] = cnt

    print(cnt)


def cre_char_id():
    filepath = 'C:/Users/Badhon/Desktop/imran/character_list.txt'

    cnt = 1
    with open(filepath) as fp:
        line = fp.readline()
        splited = line.split(' ')
        for item in splited:
            item = item.rstrip()
            char_to_id[item] = cnt

        while line:
            # print("Line {}: {}".format(cnt, line.strip()))
            line = fp.readline()
            splited = line.split(' ')
            cnt += 1

            for item in splited:
                item = item.rstrip()
                # t.insert(item)
                char_to_id[item] = cnt




def populate_trie():
    filepath = 'C:/Users/Badhon/Desktop/imran/bangla_letters.txt'


    with open(filepath) as fp:
        line = fp.readline()
        splited = line.split(' ')
        for item in splited:
            item = item.rstrip()
            t.insert(item)

        while line:
            # print("Line {}: {}".format(cnt, line.strip()))
            line = fp.readline()
            splited = line.split(' ')


            for item in splited:
                item = item.rstrip()
                t.insert(item)

def check():
    filepath = 'C:/Users/Badhon/Desktop/imran/bangla_letters.txt'

    with open(filepath) as fp:
        line = fp.readline()
        splited = line.split(' ')
        for item in splited:
            item = item.rstrip()
            print(t.search(item))

        while line:
            # print("Line {}: {}".format(cnt, line.strip()))
            line = fp.readline()
            splited = line.split(' ')

            for item in splited:
                item = item.rstrip()
                print(t.search(item))


def create_label_id():
    path = 'C:/Users/Badhon/PycharmProjects/imran/literature_labels'

    for fileName in os.listdir(path):
        base=os.path.splitext(fileName)[0]
        #f = open('label_word_id/%s.txt' % base, "w+")
        #f.close()
        #filePath="C:/Users/Badhon/PycharmProjects/imran/literature_labels/%s.txt" % base
        t.word_to_vector(fileName)





def main():

    output = ["Not present in trie",
              "Present in trie"]


    cre_dict()
    cre_char_id()

    populate_trie()
    #t.insert("sp")
    create_label_id()
    print(t.count)





if __name__ == '__main__':
    main()