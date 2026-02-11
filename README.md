
# Introduction

Data2ensembles is a module that was writen to analyse high resolution relaxometry experiments and high field relaxation experiments. This can be done using the model free approach or from an MD trajectory. This module is not intended to be a tool for others to use, though you are welcome to, but exists primarily as a record of the development in the article 
```
How flexible is a DNA duplex ? An investigation by NMR relaxometry and molecular dynamics simulations
https://doi.org/10.26434/chemrxiv-2026-wzcx6
```

We are in the process of writing a more general implimentation of the ideas in this code. We decided to go for a full rewrite to help build a more modular code that uses modern accelerators. If you are interested in contributing to this development or would like you use it then please get in touch!

# Installation 

To install this module I recommend making a dedicated python enviroment
```
python3.10 -m venv <path-to-new-venv>
source <path-to-new-venv>/bin/activate
git clone https://github.com/L-Siemons/data2ensemble.git
cd data2ensemble
pip3 install . 
```

If you have any questions then please get in touch!
### Thanks For reading! Lucas Siemons

