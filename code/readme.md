To train a model using all sequence, and epigenetic features, use the code `runModel_all.py`

`runModel_all.py` takes 10 arguments when executed from command line

- `argv[1]` is the folder containing the csv files for histone, CTCF, and RAD21 signals.
- `argv[2]` number of bins to be used for epigenetic features.
- `argv[3]` number of bins to be used for sequence features.
- `argv[4]` file containing frequencies of 64 3-mer count
- `argv[5]` file containing promoter count
- `argv[6]` file containing phastcons scores
- `argv[7]` file containing LINE transposon coverage
- `argv[8]` file containing SINE transposon coverage
- `argv[9]` file containing LTR transposon coverage
- `argv[6]` file containing DNA transposon coverage

For example to train model for cell line GM12878 using 5 bins for epigenetic features and 1 bin for sequence features,

` python3.6 runModel_all.py ../histones/gm12878/ 5 1 ../kmer/rao_gm12878.3mer.csv ../Sequence/gm12878.promoters.csv ../conservation/gm12878.phastcons.txt ../Repeats/gm12878.LTR.txt ../Repeats/gm12878.LINE.txt ../Repeats/gm12878.SINE.txt ../Repeats/gm12878.DNA.txt`
