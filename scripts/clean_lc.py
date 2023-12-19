import os
import argparse
import re


parser = argparse.ArgumentParser()
parser.add_argument('-inf',help='input file')
parser.add_argument('-o', help='out file', required=True)
parser.add_argument('-i', help='index of text', type=int, required=True)
parser.add_argument('-lc', help='set if also lowercase', action='store_true')
parser.add_argument('-splitter', help="tab, or space splitted columns", default="space")

args = parser.parse_args()



def prepare_unpunct_text(text):
    """
       Given a text, normalizes it to subsequently restore punctuation
    """
    formatted_txt = text.replace('\n', '').strip()
   # formatted_txt = formatted_txt.lower()
    formatted_txt_lst = formatted_txt.split(" ")
    punct_strp_txt = [strip_punct(i) for i in formatted_txt_lst]
    normalized_txt = " ".join([i for i in punct_strp_txt if i])
    #return normalized_txt.replace('-', ' ')
    return normalized_txt

def strip_punct(wrd):
    """
        Given a word, strips non aphanumeric characters that precede and follow it
    """
    if not wrd:
        return wrd
                                                                
    while not wrd[-1:].isalnum():
        if not wrd:
            break
        wrd = wrd[:-1]
                                                                                                    
    while not wrd[:1].isalnum():
        if not wrd:
            break
        wrd = wrd[1:]
                                                                                                                                        
    return wrd

if args.lc:
    lc_f = open(f"{args.o[:-4]}_lc{args.o[-4:]}", "w")

with open(args.inf, "r") as i:
    with open(args.o, "w") as o:
        for line in i:

            if args.splitter == "space": 
                if len(line.split())-1 < args.i: 
                    continue; 
                tokens, text = line.split()[:args.i] ," ".join(line.split()[args.i:])
            if args.splitter == "tab": 
                if len(line.split("\t"))-1 < args.i: 
                    continue; 
                tokens, text = line.split("\t")[:args.i], line.split("\t")[args.i]
            text = prepare_unpunct_text(text)
            o.write("\t".join(tokens) + "\t" + text +"\n")
            if args.lc: lc_f.write("\t".join(tokens) + "\t" + text.lower() +"\n");
            
if args.lc:
    lc_f.close()
