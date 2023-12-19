#-*- coding: utf-8 -*-
#!/usr/bin/env python

import sys
import numpy
import argparse
import jiwer

def editDistance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.

    Main algorithm used is dynamic programming.

    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

def getStepList(r, h, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.

    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    '''
    x = len(r)
    y = len(h)
    list = []
    while True:
        if x == 0 and y == 0: 
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] and r[x-1] == h[y-1]: 
            list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y-1]+1:
            list.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1]+1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]

def alignedPrint(list, r, h, result):
    '''
    This funcition is to print the result of comparing reference and hypothesis sentences in an aligned way.
    
    Attributes:
        list   -> the list of steps.
        r      -> the list of words produced by splitting reference sentence.
        h      -> the list of words produced by splitting hypothesis sentence.
        result -> the rate calculated based on edit distance.
    '''
    print("REF:", end=" ")
    for i in range(len(list)):
        if list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print(" "*(len(h[index])), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) < len(h[index2]):
                print(r[index1] + " " * (len(h[index2])-len(r[index1])), end=" ")
            else:
                print(r[index1], end=" "),
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(r[index], end=" "),
    print("\nHYP:", end=" ")
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(" " * (len(r[index])), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print(h[index2] + " " * (len(r[index1])-len(h[index2])), end=" ")
            else:
                print(h[index2], end=" ")
        else:
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print(h[index], end=" ")
    print("\nEVA:", end=" ")
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print("D" + " " * (len(r[index])-1), end=" ")
        elif list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print("I" + " " * (len(h[index])-1), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print("S" + " " * (len(r[index1])-1), end=" ")
            else:
                print("S" + " " * (len(h[index2])-1), end=" ")
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(" " * (len(r[index])), end=" ")
    print("\nWER: " + result)

def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split()) 
    """
    # build the matrix

    d = editDistance(r, h)

    # find out the manipulation steps
    list = getStepList(r, h, d)


    # print the result in aligned way
    sed = float(d[len(r)][len(h)])
    if len(r)==0: return -1,-1,[]
    result = sed / len(r) * 100
    str_result = str("%.2f" % result) + "%"
    alignedPrint(list, r, h, str_result)
    return result, sed ,list


def word_statistics(ref,hypo, steps):
    hypo_len=len(ref)
#    global sub,dele,ins,sub_ref,dele_ref,ins_dele
    ind_sub=[i for i in range(len(steps)) if steps[i]=='s' ]
    ind_dele=[i for i in range(len(steps)) if steps[i]=='d' ]
    ind_ins=[i for i in range(len(steps)) if steps[i]=='i' ]
    for i in ind_dele: hypo.insert(i,'_')
    for i in ind_ins: ref.insert(i,'_')
    sub=[hypo[i] for i in ind_sub]
    sub_ref=[ref[i] for i in ind_sub]
    dele=[ref[i] for i in ind_dele]
    ins=[hypo[i] for i in ind_ins]
    return sub, sub_ref,dele,ins, hypo_len

def write2file(filename,table_,ref_length):
        table=[w for w in table_]
        print('=============================================================== ',file=stats)
        print('Number words: %d/%d = %.2f %%' % (len(table),ref_length, len(table)/ref_length),file=stats)
        print('--------------------------------------------------------------- ',file=stats)
        dic ={w:0 for w in table}
        for w in table:
            dic[w]+=1
        dic_sort=dict(sorted(dic.items(), key=lambda item: item[1],reverse=True))
        row_format = '{:<20}' * 3
        print(row_format.format('ref','hypo','frequency'),file=stats)
        print('--------------------------------------------------------------- ',file=stats)
        for k, v in dic_sort.items():
            print(row_format.format(*k,str(v)),file=stats)
        print('=============================================================== ',file=stats)
        print('\n',file=stats)


parser = argparse.ArgumentParser(description='pynn')
parser.add_argument('--hypo', help='hypothesis', required=True)
parser.add_argument('--ref', help='stm reference', required=True)
parser.add_argument('--ref-field', help='the text field order begining from 0', type=int, default=4)
parser.add_argument('--hyp-field', help='the text field order begining from 0', type=int, default=1)
parser.add_argument('--word-stats-file', help='path to file for word statistics', type=str, default='')
parser.add_argument('--delimiter', help="space, tab", type=str, default="space")
parser.add_argument('--no_uid', help="if this is set hypo and ref needs same number of lines", action="store_true")

if __name__ == '__main__':
#    global sub, ins, dele, sub_ref, ins_ref, dele_ref
    sub=[]; ins=[]; dele=[]; sub_ref=[]; dele_ref=[]; ins_ref=[]; ref_length=0
    hypos = {}
    args = parser.parse_args()
    with open(args.hypo, 'r',encoding="utf8") as hypo:
        for idx, line in enumerate(hypo):
            tokens = line.split()
            uid, hypo = tokens[0], tokens[args.hyp_field:]
            #if args.delimiter == "tab":
            #    tokens = line.split("\t")
            #    uid, hypo = tokens[0], tokens[args.hyp_field]
           # uid = uid if not args.no_uid else idx
            hypos[uid] = hypo
    rf = args.ref_field
    err = 0
    l = 0
    n = 0
    n10 = 0
    num_ref = sum(1 for _ in open(args.ref))

    if args.no_uid:
        assert num_ref == len(hypos)

    jiwer_refs = list()
    jiwer_hypos = list()
    with open(args.ref, 'r', encoding="utf8") as ref_f:
        for idx,line in enumerate(ref_f):
            idx +=1
            if line.startswith(";;"): continue
            if args.delimiter == "space":
                tokens = line.split()
                uid = tokens[0]
                ref = tokens[rf:]
            if args.delimiter == "tab":
                tokens = line.split("\t")
                uid, ref = tokens[0], tokens[rf].split()
            uid = uid if not args.no_uid else idx

            #print("u:{} ref:{}".format( uid, ref))

            #print(hypos[str(uid)])
            #if uid not in hypos: hypo=""
            if str(uid) not in hypos: continue
            else: hypo = hypos[str(uid)]
            print('uttid: %s\n' % uid)     

            jiwer_refs.append(str(ref))
            jiwer_hypos.append(str(hypo))
            WER, sed,list = wer(ref, hypo)
            l += len(ref)

            #print("WER:davor {} uid: {} in hypos: {}".format(WER, uid, uid in hypos))
            if args.word_stats_file and list:
              sub_,sub_ref_, dele_, ins_,ref_length_ = word_statistics(ref,hypo,list)
              sub+=sub_ ; sub_ref+=sub_ref_ ; dele+=dele_ ; ins+=ins_ ; ref_length+=ref_length_
            if WER ==-1:
                print('Empty utt: %s' % uid)
            print('-----------------------------------------------------------------')
            #print('%s: %.2f' % (uid, wer))
            #print("HERE: {}".format(uid in hypos))
            err += sed
            #l += len(ref)
            if WER > 10: n10 += 1
            n += 1
    WER = float(err) / l * 100
    print("====================================================================")
    print('Overall WER: %.2f %%, Error Utter: %0.2f %%' % (WER, float(n10)/n*100))
    if args.word_stats_file:
        with open(args.word_stats_file, 'w',encoding="utf8") as stats: 
            write2file(stats,zip(sub_ref,sub),ref_length)
            write2file(stats,zip(dele,['-'for s in dele]),ref_length)
            write2file(stats,zip(['-' for s in ins],ins),ref_length)

    print("====================================================================")
    print("JIWER")
    print("#UTTS: {}".format(len(jiwer_hypos)))
    print("WER: {}".format(jiwer.wer(jiwer_refs, jiwer_hypos)))
    print("CER: {}".format(jiwer.cer(jiwer_refs, jiwer_hypos)))
#    filename1 = sys.argv[1]
#    filename2 = sys.argv[2]
#    with open(filename1, 'r', encoding="utf8") as ref:
#        r = ref.read().split()
#    with open(filename2, 'r', encoding="utf8") as hyp:
#        h = hyp.read().split()
#    wer(r, h)   
