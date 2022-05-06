#predict splicing sites with Spliceator
import Bio

#import files from raw_seq folder
with open('lncRNA_list.txt') as f:
    lines = f.readlines()

#remove the second line in each file(single strand probability, which is not necessary in this task.)
for l in lines:
    l = l.replace('\n','')
    records = list(SeqIO.parse("dat/"+l, "fasta"))
    print(records)


