# CryoETSim
The official implementations and example dataset for paper "CryoETSim: Segmentation-aware adaptive cryo-ET imaging simulation".

## 1 Introduction
We introduce CryoETSim, a simulation framework that integrates realistic biological structure modeling with transmission electron microscopy imaging physics. CryoETSim incorporates membrane morphologies segmented directly from volume electron microscopy data, producing near-realistic structural environments. It further implements a generative noise-modeling pipeline that captures contrast transfer function effects, noise statistics, and acquisition artifacts, thereby enhancing the realism of simulated tilt series and tomograms. We demonstrate the utility of CryoETSim datasets in tomogram enhancement and missing-wedge evaluation tasks, where they significantly improve performance on experimental data and yield new insights.

## 2 User guidance
Please refer to the supplementary materials for step-by-step user guidance.

## 3 Example dataset
The example dataset and detailed implementations of noise synthesizer will be available on a Google Drive Link.


## Acknowledgement
We sinceresly thank following work with their [open-sourced code](https://github.com/anmartinezs/polnet) as our reference (Their repository is under License Apache 2.0): <br>
A. Martinez-Sanchez, L. Lamm, M. Jasnin and H. Phelippeau, "Simulating the Cellular Context in Synthetic Datasets for Cryo-Electron Tomography," in IEEE Transactions on Medical Imaging, vol. 43, no. 11, pp. 3742-3754, Nov. 2024, doi: 10.1109/TMI.2024.3398401.
