import xlrd
from rmgpy.molecule import Molecule
#from rmgpy.data.thermo import convertRingToSubMolecule
import csv

smi_list={}
overall_mol=[]
QM_num=0
no=0
def Average(lst): 
    return sum(lst) / len(lst)     
def conversion(ringsmi):
    rawsmi=ringsmi.split('[')
    secondsmi=''
    finalsmi=''
    for clip in rawsmi:
        secondsmi+=clip
    thirdsmi=secondsmi.split(']')
    for clip in thirdsmi:
        finalsmi+=clip
    return finalsmi

def QM(smi):
    QM_num=0
    for char in smi:
        if char in ['C','N','O','c','n','o']:
            QM_num+=1
    return QM_num
def str_to_mol(s):
    if s.startswith('InChI'):
        mol = Molecule().fromInChI(s, backend='rdkit-first')
    else:
        mol = Molecule().fromSMILES(s, backend='rdkit-first')
    return mol

mol_base_dic={}
smi_base_dic={}

'''
with open('QM1to7_evaluation_no_res_test.csv') as df:
    csvr=csv.reader(df)
    q=0
    for moledata in csvr:
        if q<1:
            q+=1
        else:
            q+=1
            smi=moledata[0]
            mse=moledata[4]
            mol=Molecule().fromSMILES(smi,backend='rdkit-first')
            mono_rings_base, poly_rings_base = mol.getDisparateRings()
            if len(mono_rings_base)==1:
                for ring in mono_rings_base:
                    ringmol = convertRingToSubMolecule(ring)[0]
#                    smi_base_dic[smi]=ringmol
                    smi_base_dic[smi]=mse


for smi_base in smi_base_dic:
    mol_base_dic[smi_base]={}
    mse_base=smi_base_dic[smi_base]
    mol_base=Molecule().fromSMILES(smi_base,backend='rdkit-first')
    mono_rings_base, poly_rings_base = mol_base.getDisparateRings()
    for ring in mono_rings_base:
        ringmol = convertRingToSubMolecule(ring)[0]
        mol_base_dic[smi_base][ringmol]=float(mse_base)
    for ring in poly_rings_base:
        ringmol = convertRingToSubMolecule(ring)[0]
        mol_base_dic[smi_base][ringmol]=float(mse_base)

#for 3~7member cyclic top 10 error analysis
for j in range (3,8):
    with open('QM{}_cyclic_evaluation_no_res_cleared_11ver.csv'.format(j)) as csvf:
        csvr=csv.reader(csvf)
        i=0
        yes=0
        test_mol={}
        print('\nQM{}: '.format(j))
        for moledata in csvr:
            if i<1:
                i+=1
            elif i>0 and i<11:
                mse_list=[]
                mse_dic={}
                smi_len_list=[]
                smi=moledata[0]
                mse=moledata[4]
                mol=Molecule().fromSMILES(smi, backend='rdkit-first')
                test_mol[smi]=float(mse)
                mono_rings, poly_rings = mol.getDisparateRings()
                for ring in mono_rings:
                    ringmol = convertRingToSubMolecule(ring)[0]
                    for smi_base in mol_base_dic:
                        for ringmol_base in mol_base_dic[smi_base]:
                            if ringmol.is_equal(ringmol_base):
                                ringsmi=ringmol.toSMILES()
                                newsmi=conversion(ringsmi)
                                mse_dic[smi_base]=mol_base_dic[smi_base][ringmol_base]
                                smi_len_list.append(QM(smi_base))
                                mse_list.append(mol_base_dic[smi_base][ringmol_base])
                                yes=1
                                if QM(smi_base)==min(smi_len_list):
                                    min_smi=smi_base
                            else:
                                ringsmi=ringmol.toSMILES()
                                newsmi=conversion(ringsmi)
                if yes:
#                    print('{}: {} with mse {} has subring {} ( {} )  with biggest mse {}'.format(i,smi,test_mol[smi],newsmi,min_smi,mse_dic[min_smi]))
                    print('{} {} {} {} {} {}'.format(i,smi,test_mol[smi],newsmi,min_smi,mse_dic[min_smi]))
                else:
                    print('{}: {} with mse {} has subring {}'.format(i,smi,test_mol[smi],newsmi))
                yes=0
                i+=1
'''
'''
#for specific ring detection
#with open('QM6_cyclic_evaluation_no_res_cleared_test.csv') as csvf:
with open('QM1to7_evaluation_no_res_test.csv') as csvf:
    csvr=csv.reader(csvf)
    i=0
    yes=0
    test_mol={}
    sp_ringsmi='OC1=C(O)C(=O)OC=N1'
    sp_ringmol=Molecule().fromSMILES(sp_ringsmi, backend='rdkit-first')
    mono_rings, poly_rings = sp_ringmol.getDisparateRings()
    sp_ringmol =convertRingToSubMolecule(mono_rings[0])[0]
    for moledata in csvr:
        if i<1:
            i+=1
        elif i>0:
            smi=moledata[0]
            mse=moledata[4]
            mol=Molecule().fromSMILES(smi, backend='rdkit-first')
            mono_rings, poly_rings = mol.getDisparateRings()
            for ring in mono_rings:
                ringmol = convertRingToSubMolecule(ring)[0]
                ringsmi=ringmol.toSMILES()
                if ringmol.is_equal(sp_ringmol):
                    yes=1
            if yes:
                print('{}: {} with mse {}'.format(i,smi,mse))
            yes=0
            i+=1
'''
'''
#for resonance cyclic mol test
for j in range (3,8):
    with open('QM{}_cyclic_evaluation_cleared.csv'.format(j)) as csvf:
        csvr=csv.reader(csvf)
        counter=0
        i=0
        yes=0
        test_mol={}
        print('\nQM{}: '.format(j))
        for moledata in csvr:
            if i<1:
                i+=1
            elif i>0 :
                mse_list=[]
                mse_dic={}
                mollist=[]
                smi_len_list=[]
                smi=moledata[0]
                mse=moledata[4]
                mol=Molecule().fromSMILES(smi, backend='rdkit-first')
                mollist=mol.generate_resonance_structures()
                
                test_mol[smi]=float(mse)
                if len(mollist)>1:
                    print('{}: {} {}'.format(i,smi,test_mol[smi]))
                    counter+=1
                yes=0
                i+=1
'''
'''
for j in range(3,8):
    print('QM{}:'.format(j))
    with open('QM{}_cyclic_evaluation_3depth_CN_diff_cleared.csv'.format(j)) as csvf:
            smi_to_test=[]
            csvr=csv.reader(csvf)
            i=0
            k=0
            mol_base={}
            for moledata in csvr:
                if i<1:
                    i+=1
                elif i in range(1,6):
                    smi=moledata[0]
                    h_mse=moledata[4]
                    mol_base[smi]=h_mse
                    smi_to_test.append(smi)
                    i+=1
                else:
                    smi=moledata[0]
                    h_mse=moledata[4]
                    mol_base[smi]=h_mse
                    i+=1
            for smi_base in smi_to_test:
                mse_list=[]
                print('{} :'.format(smi_base))
                mol_to_test=Molecule().fromSMILES(smi_base, backend='rdkit-first')
                mono, poly = mol_to_test.getDisparateRings()
                ringmol_to_test=convertRingToSubMolecule(mono[0])[0]
                for smi in mol_base:
                    mol=Molecule().fromSMILES(smi, backend='rdkit-first')
                    monos, polys = mol.getDisparateRings()
                    ringmol=convertRingToSubMolecule(monos[0])[0]
                    if ringmol.is_equal(ringmol_to_test):
                        k+=1
                        ringsmi=ringmol.toSMILES()
                        newsmi=conversion(ringsmi)
                        rmse=pow(float(mol_base[smi]),0.5)
                        mse_list.append(rmse)
                        
#                print('average root square error:{}\nnum:{}'.format(Average(mse_list),k))
                k=0

'''
'''
#for resonance detection and writing
ringmol_list=[]
with open('QM1to7_CN_diff.csv') as df:
    csvr=csv.reader(df)
    q=0
    for moledata in csvr:
        if q<1:
            q+=1
        else:
            q+=1
            smi=moledata[0]
            mol=Molecule().fromSMILES(smi,backend='rdkit-first')
            mono_rings_base, poly_rings_base = mol.getDisparateRings()
            for ring in mono_rings_base:
                ringmol = convertRingToSubMolecule(ring)[0]
                ringmol_list.append(ringmol)
            for ring in poly_rings_base:
                ringmol = convertRingToSubMolecule(ring)[0]
                ringmol_list.append(ringmol)
for i in range(3,8):
    fwrite=open('QM{}_cyclic_for_evaluation_no_res_cleared.csv'.format(i),'w')
    with open('QM{}_cyclic_for_evaluation_no_res.csv'.format(i)) as csvf:
            i=0
            k=0
            mol_base={}
            for moledata in csvf:
                i+=1
                line=moledata.strip().split()
                smi=line[0]
                h_true=line[1]
                mol=Molecule().fromSMILES(smi,backend='rdkit-first')
                mono_rings, poly_rings = mol.getDisparateRings()
                ring=mono_rings[0]
                rinmol=convertRingToSubMolecule(ring)[0]
                if ringmol in ringmol_list:
                    fwrite.write('{} {}\n'.format(smi,h_true))
                    k+=1
            print('i:{} k:{}'.format(i,k))
    fwrite.close
'''
'''
#for cycli resonance detection after deleting the side chain
mol_list={}
with open('trainingset_high_level_with_exp+bacfit.csv') as df:
    csvr=csv.reader(df)
    q=0
    for moledata in csvr:
        if q<1:
            q+=1
        elif q>0:
            q+=1
            smi=moledata[0]
            mol=Molecule().fromSMILES(smi,backend='rdkit-first')
            cc_true=moledata[1]
            mol_list[mol]=cc_true

with open('QM7_cyclic_evaluation_no_res_cleared_F_11ver.csv') as csvf:
    csvd=csv.reader(csvf)
    q=0
    i=0
    for line in csvd:
        if q<1:
            q+=1
        elif q>0 and i<11:
            smi=line[0]
            dft_true=line[1]
            mol=Molecule().fromSMILES(smi,backend='rdkit-first')
            for base_mol in mol_list:
                if base_mol.is_equal(mol):
                    i+=1
                    print('{} true value: {} {}'.format(smi,mol_list[base_mol],dft_true))'''
'''
pass_list=[]
file=xlrd.open_workbook('bac_test.xlsx')
sheet = file.sheet_by_index(0)
print('sheet.name:',sheet.name,'sheet.nrows:',sheet.nrows,'sheet.ncols:',sheet.ncols)
fwrite=open('bac_test.csv','w')
writer=csv.writer(fwrite)
for i in range(0,sheet.nrows):
    info=str(sheet.cell_value(i,0))
    h_f=sheet.cell_value(i,1)
    mol=str_to_mol(info)
    smi=mol.toSMILES()
    writer.writerow([smi,h_f])'''
'''
fwrite=open('134k_all_for_chemprop_train.csv')

with open('134k_all_for_chemprop.csv') as df:
    csvr=csv.reader(df)
    q=0
    for line in csvr:
        if q<1:
            q+=1
        elif q>0:
            smi=line[0]
            if smi in pass_list and 'F' not in smi:
                hf=line[1]
                fwrite.write('{} {}\n'.format(smi,hf))'''

'''
with open('134k_withF_ok_to_use.csv') as csvf:
    csvd=csv.reader(csvf)
    q=0
    i=0
    for line in csvd:
        if q<1:
            q+=1
        elif q>0:
            string=line[0]
            cc_pred=line[1]
            mol=str_to_mol(string)
            mol_list.append(mol)
'''
mol_list=['C',
'CC',
'CCC',
'CCCC',
'CCCCC',
'CCCCCC',
'CCCCCCC',
'CCCCCCCC'
]
'''fw=open('c_test.csv','w')
with open('trainingset_high_level.csv') as csvf:
    csvd=csv.reader(csvf)
    q=0
    i=0
    exist=0
    for line in csvd:
        if q<1:
            q+=1
        elif q>0:
            smi=line[0]
            cc_true=line[1]
            #mol=Molecule().fromSMILES(smi,backend='rdkit-first')
            if smi in mol_list:
                fw.write('{} {}\n'.format(smi,cc_true))'''

sum_list=[]
mol_list={}

with open('134k_withF.csv') as df:
    csvr=csv.reader(df)
    q=0
    for moledata in csvr:
        if q<1:
            q+=1
        elif q>0:
            q+=1
            smi=moledata[0]
            b3lyp_hf=moledata[1]
            mol=Molecule().fromSMILES(smi,backend='rdkit-first')
            mol_list[mol]=b3lyp_hf

with open('trainingset_high_level+TRC+bacfit(source).csv') as csvf:
    csvd=csv.reader(csvf)
    q=0
    i=0
    for line in csvd:
        if q<1:
            q+=1
        elif q>0:
            smi=line[0]
            h_f=line[1]
            source=str(line[2])
            mol=Molecule().fromSMILES(smi,backend='rdkit-first')
            for base_mol in mol_list:
                if base_mol.is_equal(mol):
                    if mol not in sum_list:
                        mol_list.remove(base_mol)
                        i+=1
                        #sum_list.append(mol)
                        print('{} {} > {} {}'.format(i,smi,h_f,mol_list[base_mol],source))