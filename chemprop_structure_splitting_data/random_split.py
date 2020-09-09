import random


f = open('CCSD_exp_poly.csv', 'r')
df = open('CCSD_exp_poly_train.csv', 'w')
ddf = open('CCSD_exp_poly_val.csv', 'w')
dddf = open('CCSD_exp_poly_test.csv', 'w')
lines = f.readlines()
df.write('smiles,hf\n')
ddf.write('smiles,hf\n')
dddf.write('smiles,hf\n')


for line in lines[1:]:
    print(line)
    random_number = random.uniform(0,1)
    if random_number < 0.8:
        df.write(line)
    else:
        random_num = random.uniform(0,1)
        if random_num < 0.5:
            ddf.write(line)
        else:
            dddf.write(line)

f.close()
df.close()
ddf.close()
dddf.close()
