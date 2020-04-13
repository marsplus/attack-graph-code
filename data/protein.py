

with open('attribute_similarity_matrix_cosine.txt', 'r') as fid:
    data = fid.readlines()

ret = []
mapping = []
for idx, line in enumerate(data[3:]):
    L = line.split('\t')
    prot_name = L[0]
    mapping.append('{}\t{}\n'.format(idx, prot_name))
    ret.append('\t'.join(L[3:]))
    
with open('protein_adj_mat.txt', 'w') as fid:
    for item in ret:
        fid.write(item)


with open('nodeID_to_protein.txt', 'w') as fid:
    for item in mapping:
        fid.write(item)



