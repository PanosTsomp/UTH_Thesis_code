# Disease Detection with Machine Learning
This repository contains four distinct AI models designed for brain tumor segmentation using the BraTs dataset. The code was developed as part of a thesis project in the Department of Digital Systems at the University of Thessaly.


## Abstract
The incorporation of Artificial Intelligence (AI) in medical imaging has significantly
fostered progress in neuroimaging, thereby facilitating the precision and individualiza-
tion of brain tumor diagnosis and treatment. This dissertation aims to compare four var-
ious deep learning architectures, i.e., U-Net, Vision Transformer (ViT), DeepMedic, and
TransBTS, based on three-dimensional (3D) brain tumor segmentation using the bench-
marked BraTS dataset. Through extensive literature review, this research highlights key
clinical requirements, such as precise tumor delineation and the possibility of feasible
integration of models into clinical workflow, along with current technical challenges,
such as the complexity of 3D imaging, data heterogeneity, and interpretability. Their
assessment is carried out in the absence of preprocessing methods or data augmentation,
thus making it possible to investigate their inherent robustness and ability to generalize
under constrained processing circumstances. The results show significant differences
between the different architectures; the U-Net and TransBTS models perform consis-
tently well even in these limited settings, whereas the ViT model is particularly strong at
encoding global spatial information. DeepMedic performs significantly worse, showing
an increased dependency on data augmentation techniques. The results stress the para-
mount importance of selecting suitable architectures along with careful data preparation
and processing to arrive at clinically trustworthy outcomes. Specific emphasis is laid on
the need for model interpretability, with tools such as Grad-CAM, to boost the confidence
of clinical experts and facilitate the embedding of artificial intelligence in daily clinical
routine. Meanwhile, the study suggests specific directions for future research targeting
the sustainable and efficient integration of AI into clinical settings.

## University of Thessaly Institutional Repository
https://ir.lib.uth.gr/xmlui/handle/11615/86950
