import argparse
import itertools
import os
import numpy as np
import json
from functools import reduce
import operator
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import shutil
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Process phylogenetic data with k-mer and depth range")
    parser.add_argument('--phy_folder', type=str, default='phy', help="Folder containing .phy files")
    parser.add_argument('--kmer_range', type=int, nargs='+', default=[2,3,4,5,6], help="Range of k-mers to analyze")
    parser.add_argument('--out_name', type=str, default="out", help="Outgroup species name")
    parser.add_argument('--num_cores', type=int, default=None, help="Number of cores for parallel processing")
    parser.add_argument('--output_folder', type=str, default='output_jsons', help="Folder for output JSON files")
    return parser.parse_args()

args = parse_args()

# Parse command-line arguments to variables
phy_folder = args.phy_folder
kmer_range = args.kmer_range
out_name = args.out_name
num_cores = args.num_cores
output_folder = args.output_folder
os.makedirs(output_folder, exist_ok=True)

# define nucleotides to numeric values map
char_to_num = {
	"a": [[0], 1], "t": [[1], 1], "c": [[2], 1], "g": [[3], 1], "u": [[1], 1],
	"A": [[0], 1], "T": [[1], 1], "C": [[2], 1], "G": [[3], 1], "U": [[1], 1],
	"M": [[0, 2], 0.5],  "m": [[0, 2], 0.5],  # M: A or C
	"R": [[0, 3], 0.5],  "r": [[0, 3], 0.5],  # R: A or G
	"W": [[0, 1], 0.5],  "w": [[0, 1], 0.5],  # W: A or T
	"S": [[2, 3], 0.5],  "s": [[2, 3], 0.5],  # S: C or G
	"Y": [[1, 2], 0.5],  "y": [[1, 2], 0.5],  # Y: C or T
	"K": [[1, 3], 0.5], "k": [[1, 3], 0.5],  # K: G or T
	"B": [[1, 2, 3], 1/3],  "b": [[1, 2, 3], 1/3],  # B: C, G or T (not A)
	"D": [[0, 1, 3], 1/3],  "d": [[0, 1, 3], 1/3],  # D: A, G or T (not C)
	"H": [[0, 1, 2], 1/3],  "h": [[0, 1, 2], 1/3],  # H: A, C or T (not G)
	"V": [[0, 2, 3], 1/3],  "v": [[0, 2, 3], 1/3],  # V: A, C or G (not T)
	"N": [[0, 1, 2, 3], 0.25],  "n": [[0, 1, 2, 3], 0.25],  "?": [[0, 1, 2, 3], 0.25]  # N: A, C, G, or T
}
char_to_num_ignore = {
	"a": 0, "t": 1,  "c": 2, "g": 3,  "u": 1,
	"A": 0, "T": 1,  "C": 2, "G": 3,  "U": 1, 
}

def write_json_output(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f)

def obtain_256_pattern_count(seq_length, outgroups, out_line_idx, combination,
							 char_to_num = char_to_num, char_to_num_ignore = char_to_num_ignore):
	all_nucleotides = "ACTGUatcguMRWSYKBDHVNmrwsykbdhvn?"
	ignore_nucleotides = "MRWSYKBDHVNmrwsykbdhvn?"
	feature_2_1 = np.zeros(256, dtype = int)
	feature_2_2 = np.zeros(256, dtype = int)
				
	for i in range(seq_length):
		current_nucleotides = [
			outgroups[1][out_line_idx][i],
			combination[0][1][i],
			combination[1][1][i],
			combination[2][1][i]
		]

		motif1, motif2, motif3, motif4 = [], [], [], []

		if all(nuc in all_nucleotides for nuc in current_nucleotides):
			ignore = sum(nuc in ignore_nucleotides for nuc in current_nucleotides)

			if ignore <= 3:
				if ignore > 0:
					motif1 = [char_to_num.get(outgroups[1][out_line_idx][i])]
					motif2.append( char_to_num.get(combination[0][1][i]) )
					motif3.append( char_to_num.get(combination[1][1][i]) )
					motif4.append( char_to_num.get(combination[2][1][i]) )
					for j in range(len(motif1)):
						for k in range(len(motif1[j][0])):
							each_motif1 = str(motif1[j][0][k])
							factor1 = motif1[j][1]

							for l in range(len(motif2)):
								for m in range(len(motif2[l][0])):
									each_motif2 = str(motif2[l][0][m])
									factor2 = motif2[l][1]

									for n in range(len(motif3)):
										for o in range(len(motif3[n][0])):
											each_motif3 = str(motif3[n][0][o])
											factor3 = motif3[n][1]

											for p in range(len(motif4)):
												for q in range(len(motif4[p][0])):
													each_motif4 = str(motif4[p][0][q])
													factor4 = motif4[p][1]
													kmerNd10 = int((each_motif1+ each_motif2+ each_motif3+ each_motif4), 4)
													feature_2_1[kmerNd10] += 1 * factor1 * factor2 * factor3 * factor4   
				else:
					motif1 = str(char_to_num_ignore.get(outgroups[1][out_line_idx][i]))
					motif2 = str(char_to_num_ignore.get(combination[0][1][i]))
					motif3 = str(char_to_num_ignore.get(combination[1][1][i]))
					motif4 = str(char_to_num_ignore.get(combination[2][1][i]))

					kmerNd10 = int((motif1+ motif2+ motif3+ motif4), 4)
					feature_2_2[kmerNd10] += 1

	return feature_2_1, feature_2_2

def obtain_kmer_count(seq_length, outgroups, out_line_idx,combination, K, char_to_num = char_to_num):
	
	aaaa1,aaad1,aaca1,aacc1,aacd1,abaa1,abab1,abad1,abba1,abbb1,abbd1,abca1,abcb1,abcc1,abcd1 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	aaaa2,aaad2,aaca2,aacc2,aacd2,abaa2,abab2,abad2,abba2,abbb2,abbd2,abca2,abcb2,abcc2,abcd2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	feature64_1 = [[0]*64, [0]*64, [0]*64, [0]*64]
	feature64_2 = [[0]*64, [0]*64, [0]*64, [0]*64]
	
	for i in range(seq_length):
		if i < seq_length + 1 - (K):

			elements_to_check = (
				outgroups[1][out_line_idx][i:(i + K)] +
				combination[0][1][i:(i + K)] +
				combination[1][1][i:(i + K)] +
				combination[2][1][i:(i + K)])
			
			if all(element in "ACTGUatcguMRWSYKBDHVNmrwsykbdhvn?" for element in elements_to_check):
									
				if not all(element in "ACTGUatcgu" for element in elements_to_check):

					feature_spe1, feature_spe2, feature_spe3, feature_spe4 = [], [], [], []
					kmer_spe1, kmer_spe2, kmer_spe3, kmer_spe4 = [], [], [], []
					kmer_spe_list = [kmer_spe1, kmer_spe2, kmer_spe3, kmer_spe4]
					kmer_motif1, kmer_motif2, kmer_motif3, kmer_motif4 = [], [], [], []
					
					for n, species in enumerate([feature_spe1, feature_spe2, feature_spe3, feature_spe4]):
						feature_spen = species

						if n == 0:
							kmerd4 = []
							kmer1d4 = outgroups[1][out_line_idx][i:i + K]
							kmerNd4 = kmer1d4 

							for char in kmerNd4:
								kmerd4.append( char_to_num.get(char)  )
							feature_spen.append( kmerd4  )
						else:				 

							kmerd4 = []
							kmer1d4 = combination[n-1][1][i:i + K]
							kmerNd4 = kmer1d4 

							for char in kmerNd4:
								kmerd4.append(char_to_num.get(char))
							feature_spen.append( kmerd4  )

					for n, species in enumerate([feature_spe1, feature_spe2, feature_spe3, feature_spe4]):
						feature_spen = species
						kmer_spen = kmer_spe_list[n]

						for l in range(len(feature_spen)):
							combinations_kmer = list(itertools.product(*[kmer[0] for kmer in feature_spen[l]]))
					
							for combo in combinations_kmer:
								kmer_spen.append([combo, reduce(operator.mul, [feature_spen[l][i][1] for i in range(len(feature_spen[l]))], 1)])
					if K ==3:

						for n in range(len(feature_spe1[0][0][0])):
							base1d4 = feature_spe1[0][0][0][n] * 10**(3-1-0)
							factor1 = feature_spe1[0][0][1]
							for o in range(len(feature_spe1[0][1][0])):
								base2d4 = feature_spe1[0][1][0][o] * 10**(3-1-1)
								factor2 = feature_spe1[0][1][1]
								for p in range(len(feature_spe1[0][2][0])):
									base3d4 = feature_spe1[0][2][0][p] * 10**(3-1-2)
									factor3 = feature_spe1[0][2][1]
									feature64_1[0][int(str(base1d4 + base2d4 + base3d4), 4)] += factor1 * factor2 * factor3

					set1 = [item[0] for item in kmer_spe1]
					set2 = [item[0] for item in kmer_spe2]
					set3 = [item[0] for item in kmer_spe3]
					set4 = [item[0] for item in kmer_spe4]

					value1 = kmer_spe1[0][1]
					value2a , value3a ,value3b ,value4a ,value4b ,value4c = 0,0,0,0,0,0
											
					for e in range(len(kmer_spe2)):
						if kmer_spe2[e][0] in set1:
							value2a +=kmer_spe2[e][1]

					kmer_motif2 = [["AA", min(value2a, 1)], ["AB", max(1 - value2a, 0)]]

					for e in range(len(kmer_spe3)):
						if kmer_spe3[e][0] in set1:
							value3a +=kmer_spe3[e][1]
						elif kmer_spe3[e][0] in set2:
							value3b +=kmer_spe3[e][1]
					for l in range(len(kmer_motif2)):
						if kmer_motif2[l][1] != 0:
							kmer_motif3.append([kmer_motif2[l][0]+"A", kmer_motif2[l][1]*min(1, value3a)])
							kmer_motif3.append([kmer_motif2[l][0]+"B", kmer_motif2[l][1]*min(1, value3b, max(1-value3a, 0))])
							kmer_motif3.append([kmer_motif2[l][0]+"C", kmer_motif2[l][1]*max(1-value3a-value3b, 0)])

					for e in range(len(kmer_spe4)):
						if kmer_spe4[e][0] in set1:
							value4a +=kmer_spe4[e][1]
						elif kmer_spe4[e][0] in set2:
							value4b +=kmer_spe4[e][1]
						elif kmer_spe4[e][0] in set3:
							value4c +=kmer_spe4[e][1]

					for l in range(len(kmer_motif3)):
						if kmer_motif3[l][1] != 0:
							kmer_motif4.append([kmer_motif3[l][0]+"A", kmer_motif3[l][1]*min(1, value4a)])
							kmer_motif4.append([kmer_motif3[l][0]+"B", kmer_motif3[l][1]*min(1, value4b, max(1-value4a, 0))])
							kmer_motif4.append([kmer_motif3[l][0]+"C", kmer_motif3[l][1]*min(1, value4c, max(1-value4a-value4b, 0))])
							kmer_motif4.append([kmer_motif3[l][0]+"D", kmer_motif3[l][1]*max(1-value4a-value4b-value4c, 0)])
					for l in range(len(kmer_motif4)):
						combined_str = kmer_motif4[l][0]
						coefficient_product = kmer_motif4[l][1]
						if combined_str == "AAAA":
							aaaa1 += coefficient_product
						elif combined_str == "AAAB":
							aaad1 += coefficient_product
						elif combined_str == "AAAC":
							aaad1 += coefficient_product
						elif combined_str == "AAAD":
							aaad1 += coefficient_product
						elif combined_str == "AABA":
							aaca1 += coefficient_product
						elif combined_str == "AABB":
							aacc1 += coefficient_product
						elif combined_str == "AABC":
							aacd1 += coefficient_product
						elif combined_str == "AABD":
							aacd1 += coefficient_product
						elif combined_str == "AACA":
							aaca1 += coefficient_product
						elif combined_str == "AACB":
							aacd1 += coefficient_product
						elif combined_str == "AACC":
							aacc1 += coefficient_product
						elif combined_str == "AACD":
							aacd1 += coefficient_product
						elif combined_str == "ABAA":
							abaa1 += coefficient_product
						elif combined_str == "ABAB":
							abab1 += coefficient_product
						elif combined_str == "ABAC":
							abad1 += coefficient_product
						elif combined_str == "ABAD":
							abad1 += coefficient_product
						elif combined_str == "ABBA":
							abba1 += coefficient_product
						elif combined_str == "ABBB":
							abbb1 += coefficient_product
						elif combined_str == "ABBC":
							abbd1 += coefficient_product
						elif combined_str == "ABBD":
							abbd1 += coefficient_product
						elif combined_str == "ABCA":
							abca1 += coefficient_product
						elif combined_str == "ABCB":
							abcb1 += coefficient_product
						elif combined_str == "ABCC":
							abcc1 += coefficient_product
						elif combined_str == "ABCD":
							abcd1 += coefficient_product												
						else:
							print("other motif",i,combined_str,coefficient_product)

				else:
					feature_spes = ['', '', '', '']
					for n in range(0, 4):
						if n == 0:
							kmer1d4 = outgroups[1][out_line_idx][i:i + K]
							feature_spes[n] = kmer1d4
						else:
							kmer1d4 = combination[n-1][1][i:i + K]
							feature_spes[n] = kmer1d4
					feature_spe1, feature_spe2, feature_spe3, feature_spe4 = feature_spes
					set1 = set(feature_spe1)
					set2 = set(feature_spe2)
					set3 = set(feature_spe3)
					set4 = set(feature_spe4)

					if feature_spe2 == feature_spe1:
						kmer_motif2 = "AA"
					else:
						kmer_motif2 = "AB"
					if feature_spe3 == feature_spe1:
						kmer_motif3 = kmer_motif2+"A"
					elif feature_spe3 == feature_spe2:
							kmer_motif3 = kmer_motif2+"B"
					else:
						kmer_motif3 = kmer_motif2+"C"
					if feature_spe4 == feature_spe1:
						kmer_motif4 = kmer_motif3+"A"
					elif feature_spe4 == feature_spe2:
						kmer_motif4 = kmer_motif3+"B"
					elif feature_spe4 == feature_spe3:
						kmer_motif4 = kmer_motif3+"C"
					else:
						kmer_motif4 = kmer_motif3+"D"

					combined_str = kmer_motif4

					if combined_str == "AAAA":
						aaaa2 += 1
					elif combined_str == "AAAD":
						aaad2 += 1
					elif combined_str == "AACA":
						aaca2 += 1
					elif combined_str == "AACC":
						aacc2 += 1
					elif combined_str == "AACD":
						aacd2 += 1
					elif combined_str == "ABAA":
						abaa2 += 1
					elif combined_str == "ABAB":
						abab2 += 1
					elif combined_str == "ABAD":
						abad2 += 1
					elif combined_str == "ABBA":
						abba2 += 1
					elif combined_str == "ABBB":
						abbb2 += 1
					elif combined_str == "ABBD":
						abbd2 += 1
					elif combined_str == "ABCA":
						abca2 += 1
					elif combined_str == "ABCB":
						abcb2 += 1
					elif combined_str == "ABCC":
						abcc2 += 1
					elif combined_str == "ABCD":
						abcd2 += 1												
					else:
						print("other motif",i,combined_str,1)
					if K ==3:
						for p in range(4):
							num_list = [char_to_num_ignore[nuc] for nuc in feature_spes[p]]
							feature64_2[p][int(''.join(map(str, num_list)), 4)] += 1
	return ([aaaa1,aaad1,aaca1,aacc1,aacd1,abaa1,abab1,abad1,abba1,abbb1,abbd1,abca1,abcb1,abcc1,abcd1], 
		 	[aaaa2,aaad2,aaca2,aacc2,aacd2,abaa2,abab2,abad2,abba2,abbb2,abbd2,abca2,abcb2,abcc2,abcd2], 
			feature64_1, feature64_2)

def process_combination(outgroups, out_line_idx, combination, seq_length, kmer_range=kmer_range,  
                        file_name = None, output_folder = None, char_to_num=char_to_num, char_to_num_ignore=char_to_num_ignore, target3=None, target2=None):
	
    #edge_1 = [[i, j] for i, j in itertools.combinations(list(range(15)), 2)] # 15 nodes
    #edge_2 = [[i, j] for i, j in itertools.combinations(list(range(256)), 2)] # 256 nodes

    # Obtain pattern and kmer counts
    feature_2_1, feature_2_2 = obtain_256_pattern_count(
        seq_length=seq_length, outgroups=outgroups, out_line_idx=out_line_idx,
        combination=combination, char_to_num=char_to_num, char_to_num_ignore=char_to_num_ignore)

    # Generate k-mer counts
    feature_kmer_1, feature_kmer_2 = [], []
    for K in kmer_range:
        temp1, temp2, temp3, temp4= obtain_kmer_count(
            seq_length=seq_length, outgroups=outgroups, out_line_idx=out_line_idx,
            combination=combination, K=K, char_to_num=char_to_num)
        feature_kmer_1.append(temp1)
        feature_kmer_2.append(temp2)
        if K ==3:
            feature64 = (np.array(temp3) + np.array(temp4)).tolist()
            feature64_1 = temp3
            feature64_2 = temp4

    feature_kmer = np.add(feature_kmer_1, feature_kmer_2).tolist()

    with open(f"{file_name}-out.txt", 'r') as file:
        lines_json = [line.strip() for line in file]
        if len(lines_json) >= 2:
            second_line = lines_json[1].split('\t')
            feature_1 = [float(column) for column in second_line[6:21]]

    with open(f"{file_name}-out-filtered.txt", 'r') as file:
        lines = file.readlines()  
        if len(lines) == 1:
            target1 = 8


        else:
            target1 = 2
            values = lines[1].strip().split()



    hyde_folder = f'{phy_file.replace(".phy", "")}_hyde'
    os.makedirs(hyde_folder, exist_ok=True)
    shutil.move(f"{file_name}-out.txt", os.path.join(hyde_folder, f"{file_name}-out.txt"))
    shutil.move(f"{file_name}-out-filtered.txt", os.path.join(hyde_folder, f"{file_name}-out-filtered.txt"))

    denominator = feature_1[3] - feature_1[6] + feature_1[8] - feature_1[6]





    data = {
        "labels_1": feature_1,
        "labels_2": (np.array(feature_2_1) + np.array(feature_2_2)).tolist(),
        "labels_3": feature_kmer,
        "labels_4": feature64,
        "target1": target1,
        "target2": target2
    }
    output_file = os.path.join(hyde_folder, f"{file_name}.json")
    write_json_output(data, output_file)
    print(f'Generated JSON file: {output_file}')


def generate_comb_seq_file(seq_length, outgroups, out_line_idx, combination, spes, phy_file_path, imap_file_path, spe_ind_dict):
	file_name_parts = [outgroups[0][out_line_idx]] + ([non_out_line[0] for non_out_line in combination])
	def get_species_index(part, species_lists):
		for i, species_list in enumerate(species_lists):
			if part in species_list:
				return i

	species_lists = [spe_ind_dict[species_name] for species_name in spe_ind_dict if species_name != out_name]
	spe_count = len(species_lists)
	target_indices = [get_species_index(part, species_lists) for part in file_name_parts[1:]]
	target_string = ''.join(map(str, target_indices))
	target3 = int(target_string, spe_count)
	species_lists4 = [['flo.CS12', 'flo.CS13', 'flo.CS14', 'flo.CS15', 'flo.CS2337', 'flo.CS2338', 'flo.CS2341', 'flo.CS2350', 'flo.CS2358', 'flo.CS2359', 'thxn.JM313', 'thxn.JM57', 'thxn.JM84', 'thxn.JM86', 'thxn.MJ12-3221', 'thxn.MJ12-3233', 'thxn.MJ12-3308', 'txn.MJ11-3339', 'txn.MJ11-3340', 'txn.MJ11-3460'], ['ros.CAM1841', 'ros.CAM1880', 'ros.CAM2045', 'ros.CAM2059', 'ros.CAM2519', 'ros.CAM2552', 'ros.CJ2071', 'ros.CJ531', 'ros.CJ533', 'ros.CJ546', 'vul.CS10', 'vul.CS3603', 'vul.CS3605', 'vul.CS3606', 'vul.CS3612', 'vul.CS3614', 'vul.CS3615', 'vul.CS3617', 'vul.CS3618', 'vul.CS3621', 'mal.CS1002', 'mal.CS1011', 'mal.CS1815', 'mal.CS21', 'mal.CS22', 'mal.CS24', 'mal.CS586', 'mal.CS594', 'mal.CS604', 'mal.CS615', 'ama.JM160', 'ama.JM216', 'ama.JM293', 'ama.JM48', 'ama.MJ11-3188', 'ama.MJ11-3189', 'ama.MJ11-3202', 'ama.MJ12-3217', 'ama.MJ12-3258', 'ama.MJ12-3301', 'melG.CAM1349', 'melG.CAM1422', 'melG.CAM2035', 'melG.CAM8171', 'melG.CAM8216', 'melG.CAM8218', 'melG.CJ13435', 'melG.CJ9315', 'melG.CJ9316', 'melG.CJ9317'], [], ['chi.CAM25091', 'chi.CAM25137', 'chi.CAM580', 'chi.CAM582', 'chi.CAM585', 'chi.CAM586', 'chi.CJ553', 'chi.CJ560', 'chi.CJ564', 'chi.CJ565', 'zel.CS1', 'zel.CS1028', 'zel.CS1029', 'zel.CS1030', 'zel.CS1033', 'zel.CS1035', 'zel.CS2', 'zel.CS2262', 'zel.CS273', 'zel.CS30']]
	target_indices4 = [get_species_index(part.split("_")[-1], species_lists4) for part in file_name_parts[1:]]
	target_string4 = ''.join(map(str, target_indices4))
	target2 = int(target_string4, 4)
	file_name = '_'.join(file_name_parts)
	file_path = os.path.join(output_folder, f'{file_name}.txt')
	if len(spes) == len(combination) and len(outgroups[1]) == 1:
		file_path =phy_file_path
		each_imap = imap_file_path
		num_ind = 4
		outgroup_name = out_name
	else:
		if not os.path.exists(f"{file_path}"):
			with open(file_path, 'w') as output_file:
					output_file.writelines(f"{outgroups[0][out_line_idx]}\t{outgroups[1][out_line_idx]}\n")
					for n in range(len(combination)):
						output_file.writelines(f"{combination[n][0]}\t{combination[n][1]}\n")
		each_imap = os.path.join(phy_folder, f'{file_name}_imap')    

		with open(each_imap, 'w') as output_file:
				num_ind = 1
				output_file.write(f"{outgroups[0][out_line_idx]}\tout\n") 
				for n in range(len(combination)):
					output_file.write(f"{combination[n][0]}\tsp{n}\n") 
					num_ind += 1
		outgroup_name = "out"
	if not os.path.exists(f"{file_name}-out.txt"):
		os.system(f"python3 run_hyde.py -i {file_path} -m {each_imap} -o {outgroup_name} -n {num_ind} -t 4 -s {seq_length} --prefix {file_name}")

	if not os.path.exists(f"{file_name}-out-filtered.txt"):
		os.system(f"python3 individual_hyde.py -i {file_path} -m {each_imap} -tr {file_name}-out-filtered.txt -o {outgroup_name} -n 4 -t 4 -s {seq_length} --prefix {file_name}")

	print(f'Generated file: {file_name}.txt')
	return file_name, file_path, target3, target2

def process_task(spes, outgroups, out_line_idx, combination,
				 seq_length, kmer_range, phy_file_path, imap_file_path,
				 output_folder, char_to_num, char_to_num_ignore, spe_ind_dict):
		
	file_name, file_path, target3, target2 = generate_comb_seq_file(seq_length=seq_length,outgroups=outgroups, out_line_idx=out_line_idx, combination=combination, 
						  spes=spes, phy_file_path=phy_file_path, imap_file_path=imap_file_path, spe_ind_dict = spe_ind_dict)

	process_combination(outgroups=outgroups, out_line_idx=out_line_idx, combination=combination, 
					   seq_length=seq_length, kmer_range=kmer_range, 
                        file_name = file_name, output_folder = output_folder, char_to_num=char_to_num, char_to_num_ignore=char_to_num_ignore, target3=target3, target2=target2)
	if file_path != phy_file_path:
		os.remove(file_path)

def process_phy_file(phy_file, phy_folder = phy_folder, 
					 kmer_range = kmer_range, num_cores=None):
	
	phy_file_path = os.path.join(phy_folder, phy_file)
	imap_file = phy_file.replace('.phy', '.imap')
	imap_file_path = os.path.join(phy_folder, imap_file)
	spes = []
	outgroups = []
	spe_ind_dict = {}

	if os.path.exists(imap_file_path):
		with open(imap_file_path, 'r') as file:
			num_ind = 0
			for line in file:
				parts = line.strip().split()
				if len(parts) < 2:
					continue
				if len(spes) == 0:
					spes.append([parts[1], [parts[0]]])
				else: 
					found = False
					for n in range(len(spes)):
						if parts[1] == spes[n][0]:
							spes[n][1].append(parts[0])
							found = True
							break
					if not found:
						spes.append([parts[1], [parts[0]]])
				num_ind += 1

				m_ind, m_spe = parts
				if m_spe not in spe_ind_dict:
					spe_ind_dict[m_spe] = []
				spe_ind_dict[m_spe].append(m_ind)
	

				
	with open(phy_file_path, 'r') as file:
		for line in file:
			parts = line.strip().split()
			for n in range(len(spes)):
				if parts[0] in spes[n][1]:
					if len(spes[n]) < 3:
						spes[n].append([parts[1]])
					else:
						spes[n][2].append(parts[1]) 

	seq_length = len(spes[0][2][0])
	
	for n in range(len(spes)):
		if spes[n][0] == out_name:
			outgroups = spes[n][1:]
			outline = n
		else:
			del spes[n][0]
	spes.pop(outline)
	spes = [list(item) for sublist in spes for item in zip(*sublist)]
	combination_number = (len(spes) * (len(spes) - 1) * (len(spes) - 2) // (6 * 24)) + 1
	print("combination_number", combination_number)

	max_cores = multiprocessing.cpu_count()

	if num_cores is None:
		num_cores = max_cores
	elif num_cores > max_cores:
		print(f"Specified cores ({num_cores}) exceed available cores ({max_cores}). Using {max_cores} cores instead.")
		num_cores = max_cores

	# create the combination of four sequences: 1 outgroup + 3 individuals
	for out_line_idx in range(len(outgroups[0])):
		line_combinations = itertools.combinations(spes, 3)

		with ProcessPoolExecutor(max_workers=num_cores) as executor:
			futures = []

			for combination in line_combinations:
				futures.append(executor.submit(process_task, spes, outgroups, out_line_idx, combination, 
								   seq_length, kmer_range, phy_file_path, imap_file_path, 
								   output_folder, char_to_num, char_to_num_ignore, spe_ind_dict))

			for future in as_completed(futures):
				try:
					future.result()
				except Exception as e:
					print(f"Task failed with exception: {e}")

	print(f'Finished processing {phy_file}')

# Printout arguments
print("\n" + "="*40)
print(f"{'Argument':<20} | {'Value'}")
print("="*40)
print(f"{'--phy_folder':<20} | {phy_folder}")
print(f"{'--kmer_range':<20} | {kmer_range}")
print(f"{'--out_name':<20} | {out_name}")
print(f"{'--num_cores':<20} | {num_cores}")
print(f"{'--output_folder':<20} | {output_folder}")
print("="*40 + "\n")

# process each phy file in the folder
phy_files = [f for f in os.listdir(phy_folder) if f.endswith('.phy')]

for phy_file in phy_files:
	process_phy_file(phy_file=phy_file, phy_folder = phy_folder, 
					 kmer_range = kmer_range, num_cores=num_cores)
