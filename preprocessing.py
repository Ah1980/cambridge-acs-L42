import numpy

# File names
headerf1 = "GSE9006-GPL96_series_matrix.header.txt"
headerf2 = "GSE9006-GPL97_series_matrix.header.txt"
rownamef1 = "GSE9006-GPL96_series_matrix.rowname.txt"
rownamef2 = "GSE9006-GPL97_series_matrix.rowname.txt"
contentf1 = "GSE9006-GPL96_series_matrix.content.txt"
contentf2 = "GSE9006-GPL97_series_matrix.content.txt"

# Read files
header1 = numpy.loadtxt(headerf1, dtype='str', delimiter='\t')
header2 = numpy.loadtxt(headerf2, dtype='str', delimiter='\t')

rowname1 = numpy.loadtxt(rownamef1, dtype='str', delimiter='\t')
rowname2 = numpy.loadtxt(rownamef2, dtype='str', delimiter='\t')

content1 = numpy.loadtxt(contentf1, dtype='float', delimiter='\t')
content2 = numpy.loadtxt(contentf2, dtype='float', delimiter='\t')

# Clean up input
id1 = [h[1:-1] for h in header1[0][1:]]
id2 = [h[1:-3] for h in header2[0][1:]]

rowname1 = [r[1:-1] for r in rowname1]
rowname2 = [r[1:-1] for r in rowname2]

def to_health_class(s):
    if s == '"Illness: Healthy"':
        return 0
    elif s == '"Illness: Type 1 Diabetes"':
        return 1
    elif s == '"Illness: Type 2 Diabetes"':
        return 2
    else:
        raise Exception


health_classes = [to_health_class(h) for h in header1[1][1:]]

health_classes = numpy.array(health_classes)

# Merge 2 files
exp1 = numpy.transpose(content1)
backup = numpy.transpose(content2)
exp2 = numpy.zeros(backup.shape)

for i in range(len(id2)):
    exp2[id1.index(id2[i])] = backup[i]

exp_all = numpy.array([numpy.concatenate([exp1[i], exp2[i]]) for i in range(exp1.shape[0])])


# Split training/testing set
training_idx = range(0, 10) + range(30,40) + range(107, 117)
testing_idx = [i for i in range(0, 117) if i not in training_idx]

X = exp_all[training_idx]
Y = health_classes[training_idx]

X_test = exp_all[testing_idx]
