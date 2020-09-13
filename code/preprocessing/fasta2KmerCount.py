import itertools
import sys

k=int(sys.argv[2])
bases=['A','T','G','C']
kmers=[''.join(p) for p in itertools.product(bases, repeat=k)]

fastaFile=sys.argv[1]
f=open(fastaFile,"r")
lines=f.read().splitlines()
f.close()
Count={}
print("\t".join(kmers))
for  i in range(1,len(lines),2):
     count = lines[i-1]
     for j in range(0,len(kmers)):
         count = count + "\t" + str( (lines[i].upper()).count(kmers[j]) )
     print(count)
