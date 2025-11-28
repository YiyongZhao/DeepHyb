<div align="center" style="display: flex; align-items: center; justify-content: center;">
  <h2 style="margin: 0;">DeepHyb</h2>
</div>

**DeepHyb** enables inference of historical hybrid speciation events using a Convolutional Neural Network (CNN) with embedded comprehensive multiple sequence alignment (MSA) features.

- **PyPI**: [DeepHyb] no link available
- **GitHub**: [DeepHyb](https://github.com/XinzhengDu/DeepHyb)
- **License**: MIT License
- **Release Date**: October 2025
- **Contacts**: 
  - Xinzheng Du: [xzdu@ustc.edu.cn](mailto:duxz@ustc.edu.cn) (Division of Life Sciences and Medicine, University of Science and Technology of China)
  - Yiyong Zhao (Corresponding Author): [yiyong.zhao@yale.edu](mailto:yiyong.zhao@yale.edu) (Department of Biomedical Informatics & Data Science, Yale University)                                                                    


![Version](https://img.shields.io/badge/Version-1.0.0-blue)
[![Documentation Status](http://readthedocs.org/projects/deephyb/badge/?version=latest)](http://deephyb.readthedocs.io)
[![DeepHyb Issues](https://img.shields.io/badge/DeepHyb--Issues-blue)](https://github.com/XinzhengDu/DeepHyb/issues)
![Build Status](https://travis-ci.org/XinzhengDu/DeepHyb.svg?branch=main)
[![PyPI](https://img.shields.io/pypi/v/DeepHyb.svg)](https://pypi.python.org/pypi/DeepHyb)

### Introduction
DeepHyb is the first deep learning tool that directly analyzes raw DNA multiple sequence alignments (MSAs) to infer historical hybridization events. It addresses the limitation of traditional tools (e.g., HyDe) that can detect hybridization but fail to classify non-hybrid evolutionary scenarios. 

Key capabilities of DeepHyb:
1. Accurately detect hybridization events and distinguish between hybrid and non-hybrid lineages.
2. Identify parental lineages and hybrid offspring from MSA data.
3. Leverage four embedded MSA features (15_summary_site_patterns, 256_one_base_site_patterns, 75_summary_kmer_site_patterns, 256_seq_kmer_patterns) for robust inference.
4. Provide interpretability via SHAP (SHapley Additive exPlanations) analysis, validating classic introgression signals (ABBA-BABA patterns) and uncovering a novel predictive pattern (ABAA).

DeepHyb was validated on genomic data from 92 *Heliconius* butterflies (a model system for evolutionary biology), achieving over 93% accuracy in classifying species relationships (hybrid + non-hybrid cases).
<p align="center">
</p>
*Figure: DeepHyb’s CNN architecture, processing 4 MSA-derived features via distinct branches to predict hybridization status (Target1) and parental/hybrid identities (Target2).*


### Clone and Install Environment
```bash
# One-click installation via conda (https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
git clone https://github.com/XinzhengDu/DeepHyb.git
cd DeepHyb/
conda env create -f environment.yml # environment.yml not included; use pip for dependencies below
conda activate DeepHyb

# Alternative installation via pip
# Required dependencies (Python 3.7+):
pip install torch scikit-learn numpy pandas tqdm shap argparse tensorflow
# Install HyDe separately for preliminary hybridization signal comparison:
pip install hyde-python
```

### Install from PyPI with pip
```bash
pip install DeepHyb # Cannot install from PyPI currently
```

## Usage 
### Datasets for CNN Training
[JSON](https://www.json.org/json-en.html) is used as the data interchange format, storing MSA-derived features and labels. Each JSON file contains three key components:
1. **Feature data**: 4 feature sets from MSAs (15_summary_site_patterns, 256_one_base_site_patterns, 75_summary_kmer_site_patterns, 256_seq_kmer_patterns).
2. **Hybridization signals**: Preliminary results from HyDe (e.g., z-scores, p-values).
3. **Taxonomic labels**: Annotations for outgroups, parental lineages, and hybrid lineages (defining model targets).

The 15_summary_site_patterns (Feature 1) correspond to the following patterns (used for node labeling in feature processing):
| **Pattern** | **Node ID** |
|:-------------| :------------:|
| AAAA | 0 |
| AAAD | 1 |
| AACA | 2 |
| AACC | 3 |
| AACD | 4 |
| ABAA | 5 |
| ABAB | 6 |
| ABAD | 7 |
| ABBA | 8 |
| ABBB | 9 |
| ABBD | 10 |
| ABCA | 11 |
| ABCB | 12 |
| ABCC | 13 |
| ABCD | 14 |

Example JSON structure:
```javascript
{
  "15_summary_site_patterns": [12, 8, 5, ..., 3], // Length: 15
  "256_one_base_site_patterns": [5, 18, 220, ..., 101], // Length: 256
  "75_summary_kmer_site_patterns": [[3, 7, ..., 2], [5, 9, ..., 4], ..., [1, 6, ..., 0]], // Length: 5 (k=2-6) × 15
  "256_seq_kmer_patterns": [[2, 5, ..., 1], [7, 3, ..., 4], ..., [9, 1, ..., 6]], // Length: 4 (individuals) × 64
  "target1": 2, // 2 = Hybrid, 8 = Non-hybrid
  "target2": 15 // 64 categories (outgroup + parental/hybrid combinations)
}
```

### Example Input Format: Multiple Sequence Alignment (MSA)
DeepHyb requires MSA files in **PHYLIP format (no header)**. The number of sequences must be >4, and all sequences must have the same length.

#### Key MSA Requirements:
1. Place PHYLIP files (extension: .phy) in the "phy" folder.
2. Concatenate orthologous coding genes into a supermatrix (ideal length: >50,000 bp for sufficient inference power, per coalescent model estimates).
3. Convert VCF files to PHYLIP format (post-reference alignment) using tools like [GATK](https://gatk.broadinstitute.org/hc/en-us) for individual-level genomic data.

#### Example PHYLIP File (species-level):
```
-----------heliconius_species.phy-----------------------------------------------------------------------------------------
outgroup	GAAGTTAGTA-TGA-ACTGATTAGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
sp1_ind1	GAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
sp1_ind2	GAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
sp2_ind1	GAC-TTAGT-CTGATACTGATGAGGTTCCTTGAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
sp2_ind2	GAC-TTAGT-CTGATACTGATGAGGTTCCTTGAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
hybrid_ind1	GAAGTTAGTA-TGA-ACTGATGAGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
hybrid_ind2	GAAGTTAGTA-TGA-ACTGATGAGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
```

#### Example PHYLIP File (population-level):
```
-----------heliconius_population.phy---------------------------------------------------------------------------------------------
outgroup_pop1	GAAGTTAGTA-TGA-ACTGATTAGGTTCCTTGAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
sp1_pop1	GAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
sp1_pop2	GAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
sp2_pop1	GAC-TTAGT-CTGATACTGATGAGGTTCCTTGAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
sp2_pop2	GAC-TTAGT-CTGATACTGATGAGGTTCCTTGAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
hybrid_pop1	GAAGTTAGTA-TGA-ACTGATGAGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
hybrid_pop2	GAAGTTAGTA-TGA-ACTGATGAGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
```

### MSA Feature Extraction (msa_sitepattern_counter.py)
This script processes PHYLIP files to generate JSON feature files and runs HyDe for preliminary hybridization signals.

```bash
# Basic usage: Process PHYLIP files in "phy" folder, k-mer range 2-6, outgroup named "outgroup"
python msa_sitepattern_counter.py \
  --phy_folder ./phy \
  --kmer_range 2 3 4 5 6 \
  --out_name outgroup \
  --num_cores 4 \
  --output_folder ./output_jsons
```

#### Key Parameters for msa_sitepattern_counter.py:
| Parameter         | Type       | Description                                                                 | Default       |
|:------------------|:-----------|:-----------------------------------------------------------------------------|:--------------|
| --phy_folder      | STR        | Folder containing PHYLIP (.phy) files                                        | ./phy         |
| --kmer_range      | LIST[INT]  | Range of k-mer lengths for 75_summary_kmer_site_patterns (k=2-6)            | 2 3 4 5 6     |
| --out_name        | STR        | Name of the outgroup (must match sequence names in .phy)                     | outgroup      |
| --num_cores       | INT        | Number of CPU cores for parallel processing                                  | Max available |
| --output_folder   | STR        | Folder to save JSON feature files and HyDe results                           | ./output_jsons|

### CNN Training & Inference (DeepHybCNN.py)
This script trains the CNN model on JSON features, evaluates performance, and saves the trained model.

#### Quick Start:
```bash
# Train model: 10,000 epochs, batch size 512, early stopping at loss < 0.01
python DeepHybCNN.py \
  --train_folder ./output_jsons/train \
  --test_folder ./output_jsons/test \
  --epochs 10000 \
  --batch_size 512 \
  --lr 0.0003 \
  --maxloss 0.01 \
  --model_save_path ./models/deephyb_cnn.pth \
  --loss_save_path ./logs/deephyb_loss.json
```

#### Train with Pretrained Model:
```bash
# Load pretrained model (no training, only inference)
python DeepHybCNN.py \
  --test_folder ./output_jsons/test \
  --load_path ./models/deephyb_cnn.pth
```

#### Key Parameters for DeepHybCNN.py:
| Parameter         | Type       | Description                                                                 | Default               |
|:------------------|:-----------|:-----------------------------------------------------------------------------|:----------------------|
| --train_folder    | STR        | Folder containing training JSON files                                        | ./output_jsons/train  |
| --test_folder     | STR        | Folder containing testing JSON files                                         | ./output_jsons/test   |
| --epochs          | INT        | Maximum training epochs                                                      | 10000                 |
| --batch_size      | INT        | Batch size for training/inference                                            | 512                   |
| --lr              | FLOAT      | Learning rate (Adam optimizer)                                               | 0.0003                |
| --maxloss         | FLOAT      | Early stopping threshold (stop if avg loss < maxloss)                        | 0.01                  |
| --model_save_path | STR        | Path to save trained model (.pth)                                            | ./models/deephyb_cnn.pth |
| --loss_save_path  | STR        | Path to save training loss history (JSON)                                    | ./logs/deephyb_loss.json |
| --load_path       | STR        | Path to load pretrained model (skips training)                               | None                  |

### SHAP Interpretability Analysis
Run SHAP to quantify feature importance (focus on 256_one_base_site_patterns, the top contributor to predictions):
```bash
# Example: Analyze model predictions with SHAP
python shap_analysis.py \
  --model_path ./models/deephyb_cnn.pth \
  --test_folder ./output_jsons/test \
  --output_plot ./plots/shap_feature_importance.png
```

#### Key SHAP Findings:
- Top predictive patterns: ABBA (classic introgression), BABA (classic introgression), ABAA (novel pattern).
- Highest pattern importance near wing-pattern loci (e.g., *HmB* in *Heliconius*), confirming adaptive introgression.


## Output
All outputs are saved in the "output_files" folder by default:
1. **Model outputs**: Trained CNN model (.pth), loss history (JSON).
2. **Prediction results**: 
   - Target1 (Hybrid/Non-hybrid): Accuracy, precision, recall, F1-score, confusion matrix.
   - Target2 (Parental/hybrid identity): Classification accuracy, confusion matrix.
3. **SHAP results**: Feature importance plots, SHAP value matrices.
4. **HyDe results**: Raw/filtered hybridization signals (from msa_sitepattern_counter.py).


## Bug Reports
Report bugs, request features, or ask questions via the [GitHub Issues page](https://github.com/XinzhengDu/DeepHyb/issues). Include:
- Your input PHYLIP file (or sample).
- Command run and error logs (if any).


## Contributing
We welcome contributions to improve DeepHyb! Follow these steps:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add [feature/bug fix]: description"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a Pull Request.


## Version History
| Version | Date       | Changes                                                                 |
|:--------|:-----------|:-------------------------------------------------------------------------|
| 1.0.0   | October 2025 | Initial release: MSA feature extraction, CNN training, SHAP analysis.    |


## License
DeepHyb is licensed under the [MIT LICENSE](LICENSE).