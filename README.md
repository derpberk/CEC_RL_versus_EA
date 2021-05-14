# A Dimensional Comparison between Evolutionary Algorithm and Deep Reinforcement Learning Methodologies for Autonomous Surface Vehicles with Water Quality Sensors

Comparison between Deep Reinforcement Learning algorithms and Evolutionary Algorithms for the Patrolling Problem in the Ypacaraí case.

## Abstract

The monitoring of water resources using Autonomous Surface Vehicles with water-quality sensors has been a recent approach due to the advances in unmanned transportation technology.
 The Ypacaraí Lake, the biggest water resource in Paraguay, suffers from a major contamination problem because of cyanobacteria blooms. 
In order to supervise the blooms using these on-board sensor modules, a Non-Homogeneous Patrolling Problem (a NP-hard problem) must be solved in a feasible amount of time. 
A dimensionality study is addressed to compare the most common methodologies, Evolutionary Algorithms and Deep Reinforcement Learning, 
in different map scales and fleet sizes with changes in the environmental conditions.
 The results determined that Deep Q-Learning overcomes the evolutionary method in terms of sample-efficiency by a 50-70\% in higher resolutions.
 Furthermore, it reacts better than the Evolutionary Algorithm in high space-state actions. 
In contrast, the evolutionary approach shows a better efficiency in lower resolutions and needs fewer parameters to synthesize robust solutions. 
This study reveals that Deep Q-learning approaches exceed in efficiency for the Non-Homogeneous Patrolling Problem but with many hyper-parameters involved in the stability and convergence.

This repository corresponds to a published paper in MDPI Sensors:

https://www.mdpi.com/1077678

Please, cite using:

## Citing the Project

To cite this repository in publications:

```bibtex
@Article{yanesDRLvsEV,
 AUTHOR = {Yanes Luis, Samuel and Gutiérrez-Reina, Daniel and Toral Marín, Sergio},
 TITLE = {A Dimensional Comparison between Evolutionary Algorithm and Deep Reinforcement Learning Methodologies for Autonomous Surface Vehicles with Water Quality Sensors},
 JOURNAL = {Sensors},
 VOLUME = {21},
 YEAR = {2021},
 NUMBER = {8},
 ARTICLE-NUMBER = {2862},
 URL = {https://www.mdpi.com/1424-8220/21/8/2862},
 ISSN = {1424-8220},
 DOI = {10.3390/s21082862}
}

```
