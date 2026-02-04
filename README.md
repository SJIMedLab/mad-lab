# MAD-Lab

**MAD-Lab** (MRV Artifact Diffusion Laboratory) is a research-oriented framework for modeling flow-related artifacts in magnetic resonance venography (MRV) using diffusion-based generative models.

This repository provides an experimental system for studying imaging uncertainty and artifact patterns in MRV, with the goal of improving the reliability and consistency of image interpretation in suspected cerebral venous sinus thrombosis (CVST).

---

## Overview

Flow-related artifacts are common in MRV and may closely mimic venous sinus thrombosis, leading to substantial interreader variability and reduced diagnostic reliability.  
MAD-Lab is designed to explore this problem from a *methodological and experimental* perspective.

The core idea is to use **3D denoising diffusion probabilistic models (DDPMs)** to explicitly model the spatial distribution and severity of MRV artifacts, and to evaluate their impact on:

- Interreader agreement in artifact interpretation  
- Robustness of downstream artifact recognition models  
- Imaging uncertainty rather than disease diagnosis  

This project focuses on *artifact modeling*, not artifact removal or automated clinical diagnosis.

---

## Key Features

- 3D diffusion-based modeling of MRV flow-related artifacts  
- Synthetic artifact generation for data augmentation and analysis  
- Support for reader study–oriented evaluation (e.g., interreader agreement)  
- Downstream artifact recognition experiments under controlled conditions  
- Modular, research-friendly codebase intended for reproducibility  

---

## Scope and Intended Use

MAD-Lab is provided **for research and educational purposes only**.

- This project **does not** generate or replace clinical diagnostic images  
- This project **does not** output CVST diagnoses or clinical decisions  
- This project is **not** intended for real-time or clinical deployment  

All experiments are conducted in retrospective, research-only settings.

---

## Relationship to the Associated Study

This repository accompanies a retrospective diagnostic reliability study investigating whether diffusion-based MRV artifact modeling can reduce interreader variability and improve interpretation consistency.

The generative models in MAD-Lab are used solely for artifact modeling and experimental analysis, and not as clinical decision-support systems.

---

## Data Availability

Due to patient privacy and ethical restrictions, **no real patient imaging data are included** in this repository.

Users are expected to prepare their own datasets in accordance with local regulations, institutional review board (IRB) requirements, and applicable data protection laws.

Synthetic data generation modules are provided for research experimentation only.

---

## License

This project is released under the **Apache License 2.0**.

You are free to use, modify, and distribute this software in compliance with the terms of the license.  
See the [LICENSE](LICENSE) file for details.

---

## Disclaimer

This software is provided *as is*, without warranty of any kind.

It is **not intended for clinical diagnosis, treatment planning, or medical decision-making**.  
Any use of this software in clinical or regulatory settings is the sole responsibility of the user.

---

## Citation

If you use MAD-Lab in academic work, please cite the associated publication:

> *Diffusion-Based Modeling of MRV Artifacts for Improving Diagnostic Reliability in Cerebral Venous Sinus Thrombosis*  
> JAMA Network Open (under review)

---

## Contact

For academic questions or collaboration inquiries, please contact the corresponding author of the associated study.
