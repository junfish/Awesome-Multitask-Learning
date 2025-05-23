# Awesome-Multitask-Learning
<!-- This repository periodicly updates the the papers and resources of MTL. -->
Awesome-Multitask-Learning
Welcome to Awesome-Multitask-Learning, a curated repository of papers, code, and datasets in the field of Multi-Task Learning (MTL). This repo is designed to serve both newcomers and experienced researchers seeking a comprehensive understanding of the evolution, methods, and applications of MTL—from classical approaches to modern deep learning and pre-trained foundation models.

This repository is grounded in our three-part survey published in Harvard Data Science Review (HDSR):

📘 Part I: Fundamentals — introduces MTL from its origins, with formal definitions, taxonomies, and core motivations.

📗 Part II: Regularization and Optimization — dives deep into the theoretical techniques enabling task-sharing, including loss functions, inductive biases, and training dynamics.

📙 Part III: Applications — showcases real-world deployment of MTL methods, and how recent developments like pre-trained foundation models (PFMs) reshape the practical landscape of multi-task systems.

Together, these papers span from 1997 to 2024, mapping the MTL journey across generations of machine learning paradigms. Each method in this repo is linked to its source paper and, when available, code implementations.


## Table of Contents:
- [Awesome-Multitask-Learning](#awesome-multitask-learning)
  - [Table of Contents:](#table-of-contents)
  - [Existing survey papers](#existing-survey-papers)
  - [Datasets](#datasets)
    - [Regression task](#regression-task)
    - [Classification task](#classification-task)
    - [Dense prediction task](#dense-prediction-task)
  - [Methods](#methods)
    - [Traditional era](#traditional-era)
      - [Feature Selection](#feature-selection)
      - [Feature Transformation](#feature-transformation)
      - [Low-Rank Factorization](#low-rank-factorization)
      - [Decomposition](#decomposition)
      - [Priori Sharing](#priori-sharing)
      - [Task Clustering](#task-clustering)
    - [Deep Learning Era](#deeplearning-era)
      - [Feature Fusion](#feature-fusion)
      - [Cascading](#cascading)
      - [Knowledge Distilation](#knowledge-distilation)
      - [Cross-Task Attention](#cross-task-attention)
      - [Scalarization](#scalarization)
      - [Multi-objective Optimization (MOO)](#multi-objective-optimization-moo)
      - [Adversarial Training](#adversarial-training)
      - [Mixture of Experts](#mixture-of-experts)
      - [Graph-based](#graph-based)
      

<!-- - [Architectures](#architectures)
  - [Encoder-based](#encoder)
  - [Decoder-based](#decoder)
  - [Other](#otherarchitectures)
- [Neural Architecture Search](#nas)
- [Optimization strategies](#optimization)
- [Transfer learning](#transfer) -->

<a name="survey"></a>
## Existing survey papers
<!-- - <a name="vandenhende2020revisiting"></a> Vandenhende, S., Georgoulis, S., Van Gansbeke, W., Proesmans, M., Dai, D., & Van Gool, L. 
*[Multi-Task Learning for Dense Prediction Tasks: A Survey](https://ieeexplore.ieee.org/abstract/document/9336293)*,
T-PAMI, 2020. [[PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch)] -->
- **A survey on multi-task learning** \
*Yu Zhang and Qiang Yang*
IEEE Transactions on Knowledge and Data Engineering 2022. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9392366) \
March 31, 2021
- **Multi-task learning for dense prediction tasks: A survey** \
*Simon Vandenhende, Stamatios Georgoulis, Marc Proesmans, Dengxin Dai and Luc Van Gool* \
IEEE Transactions on Pattern Analysis and Machine Intelligence 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9336293)] \
Jan 26, 2021
- **A Brief Review of Deep Multi-task Learning and Auxiliary Task Learning** \
*Partoo Vafaeikia and Khashayar Namdar and Farzad Khalvati* \
arXiv 2020. [[Paper](https://arxiv.org/abs/2007.01126)] \
Jul 02, 2020
- **Multi-Task Learning with Deep Neural Networks: A Survey** \
*Michael Crawshaw* \
arXiv 2020. [[Paper](https://arxiv.org/abs/2009.09796)] \
Sep 10, 2020
- **A brief review on multi-task learning** [[Paper](https://link.springer.com/article/10.1007/s11042-018-6463-x)] \
*Kim-Han Thung, Chong-Yaw Wee* \
Multimedia Tools and Applications 2018. \
Aug 08, 2018
- **An overview of multi-task learning in deep neural networks** \
*Sebastian Ruder* \
arXiv 2017. [[Paper](https://arxiv.org/abs/1706.05098)] \
Jun 15, 2017


<a name="datasets"></a>

## Datasets
<a name = "regression"></a>
### Regression task
- **Synthetic Data**
This dataset is often artificially defined by researchers, thus different from one another. The features are often generated via drawing random variables from a shared distribution and adding irrelevant variants from other distributions, and the corresponding responses are produced by a specific computational method.

- [School Data](http://www.bristol.ac.uk/cmm/learning/)
This dataset comes from the Inner London Education Authority (ILEA) and
contains 15,362 records of student examination, which are described by 27 student- and school-specific features from 139 secondary schools. The goal is to predict exam scores from 27 features.

- [SARCOS Data](http://gaussianprocess.org/gpml/data/)
This dataset is in humanoid robotics consists of 44,484 training examples and
4, 449 test examples. The goal of learning is to estimate inverse dynamics model of a 7 degrees-of-freedom (DOF) SARCOS anthropomorphic robot arm.

- **Computer Survey Data**
This dataset is from a survey on the likelihood (11 point scale from 0 to 10) of purchasing personal computers. There are 20 computer models as examples, each of which contains 13 computer descriptions (e.g., price, CPU speed, and screen size) and 6 subject-level covariates (e.g., gender, computer knowledge, and work experience) as features and ratings of 179 subjects as targets, i.e., tasks.
- [Climate Dataset](https://www.cambermet.co.uk/(S(jevn1q55xcd0n42oueeyrp45))/default.aspx)
This real-time dataset is collected from a sensor network (e.g., anemometer,
thermistor, and pressure transducer) of four climate stations—Cambermet, Chimet, Sotonmet and Bramblemet—in the south on England, which can represent 4 tasks as needed. The archived data are reported in 5-minute intervals, including ∼ 10 climate signals (e.g., wind speed, wave period, barometric pressure, and water temperature).

<a name = "classification"></a>
### Classification task
- [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)
This dataset is a collection of approximately 19, 000 netnews articles, organized into 20 hierarchical newsgroups according to the topic, such as root categories (e.g., comp,rec, sci, and talk) and sub-categories (e.g., comp.graphics, sci.electronics, and talk.politics.guns). Users can design different combinations as multiple text classifications tasks.

- [Reuters-21578 Collection](http://www.daviddlewis.com/resources/testcollections/reuters21578/)
This text collection contains 21578 documents from Reuters newswire dating back to 1987. These documents were assembled and indexed with more than 90 correlated categories—5 top categories (i.e., exchanges, orgs, people, place, topic), and each of them includes variable sub-categories。

- [MultiMNIST](http://www.cs.toronto.edu/~tijmen/affNIST/)
This dataset is a MTL version of MNIST dataset9. By overlaying multiple images together, traditional digit classification is converted to a MTL problem, where classifying the digits on the different positions are considered as distinctive tasks.

- [ImageCLEF-2014](https://www.imageclef.org/2014/adaptation)
This dataset is a benchmark for domain adaptation challenge, which contains 2, 400 images of 12 common categories selected from 4 domains: Caltech 256, ImageNet 2012, Pascal VOC 2012, and Bing.

- [Office-Caltech](https://www.v7labs.com/open-datasets/office-caltech-10)
This dataset is a standard benchmark for domain adaption in computer vision, consisting of real-world images of 10 common categories from Office dataset and Caltech-256 dataset. There are 2,533 images from 4 distinct domains/tasks: Amazon, DSLR, Webcam, and Caltech.

- [Office-31](https://opendatalab.com/Office-31)
This dataset consists of 4,110 images from 31 object categories across 3 domains/tasks: Amazon, DSLR, and Webcam.
  
- [Office-Home Dataset.](https://www.hemanthdv.org/officeHomeDataset.html)
This dataset is collected for object recognition to validate domain adaptation models in the era of deep learning, which includes 15,588 images images in office and home settings (e.g., alarm clock, chair, eraser, keyboard, telephone, etc.) organized into 4 domains/tasks: Art (paintings, sketches and artistic depictions), Clipart (clipart images), Product (product images from www.amazon.com), and Real-World (real-world objects captured with a regular camera).

- [DomainNet](http://ai.bu.edu/M3SDA/)
This dataset is annotated for the purpose of multi-source unsupervised domain adaptation (UDA) research. It contains ∼ 0.6 million images from 345 categories across 6 distinct domains, e.g., sketch, infograph, quickdraw, real, etc.

- [EMMa](https://emma.stanford.edu/)
This dataset comprises more than 2.8 million objects from Amazon product listings, each annotated with images, listing text, mass, price, product ratings, and its position in Amazon’s product-category taxonomy. It includes a comprehensive taxonomy of 182 physical materials, and objects are annotated with one or more materials from this taxonomy. EMMa offers a new benchmark for multi-task learning in computer vision and NLP, allowing for the addition of new tasks and object attributes at scale.

- [SYNTHIA](https://synthia-dataset.net/)
This dataset is a synthetic dataset created to address the need for a large and diverse collection of images with pixel-level annotations for vision-based semantic segmentation in urban scenarios, particularly for autonomous driving applications. It consists of precise pixel-level semantic annotations for 13 classes, including sky, building, road, sidewalk, fence, vegetation, lane-marking, pole, car, traffic signs, pedestrians, cyclists, and miscellaneous objects.

<a name = "dense-prediction-task"></a>
### Dense prediction task

- [CityScapes](https://www.cityscapes-dataset.com/)
This dataset consists of 5,000 images with high quality annotations
and 20,000 images with coarse annotations from 50 different cities, which contains 19 classes for semantic urban scene understanding. 

- [NYU-Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
This dataset is comprised of 1,449 images from 464 indoor scenes across 3 cities, which contains 35,064 distinct objects of 894 different classes. The dense per-pixel labels of class, instance, and depth are used in many computer vision tasks, e.g., semantic segmentation, depth prediction, and surface normal estimation.

- [PASCAL VOC Project](http://host.robots.ox.ac.uk/pascal/VOC/)
This project provides standardized image datasets for object class recognition and also has run challenges evaluating performance on object class recognition from 2005 to 2012, where [VOC07](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html), [VOC08](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/index.html), and [VOC12](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) are commonly used for MTL research. The multiple tasks covers classification, detection (e.g., body part, saliency, semantic edge), segmentation, attribute prediction, surface normals prediction, etc. 

- [Taskonomy](https://github.com/StanfordVL/taskonomy/tree/master/data#downloading-the-dataset-new-oct-2021)
This dataset is currently the most diverse product for computer vision in
MTL, consisting of 4 million samples from 3D scans of ∼ 600 buildings. This product is a dictionary of 26 tasks (e.g., 2D, 2.5D, 3D, semantics, etc.) as a computational taxonomic map for task transfer learning. 

<a name = "methods"></a>
## Methods

<a name ="traditional-era"></a>
### Traditional era

<a name = "feature-selection"></a>
#### Feature Selection
- **Adaptive multi-task sparse learning with an application to fMRI study** [[Paper](https://epubs.siam.org/doi/abs/10.1137/1.9781611972825.19)] \
*Xi Chen, Jinghui He, Rick Lawrence and Jaime G Carbonell* \
Proceedings of the 2012 SIAM International Conference on Data Mining 2012. 

- **Multi-stage multi-task feature learning** [[Paper](https://proceedings.neurips.cc/paper/2012/hash/2ab56412b1163ee131e1246da0955bd1-Abstract.html)] \
*Pinghua Gong, Jieping Ye and Changshui Zhang* \
Advances in neural information processing systems 2012. 

- **Sparse Multi-Task Lasso** \
*Aurelie C Lozano and Grzegorz Swirszcz* \
Proceedings of the 29th International Coference on International Conference on Machine Learning 2012. [[Paper](https://dl.acm.org/doi/abs/10.5555/3042573.3042652)]

- **Modeling disease progression via fused sparse group lasso** \
*Jiayu Zhou, Jun Liu, Vaibhav A Narayan and Jieping Ye* \
Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining 2012. [[Paper](https://dl.acm.org/doi/abs/10.1145/2339530.2339702?casa_token=19FLLXrMooMAAAAA:cV8xhbjZigE8Nh89yeBCsAz3Bpzp9gs4vAeUpWTvy-N52l_iOXpL-MJ3JO13zDiVmqkrr-4aMQup)]

- **A multi-task learning formulation for predicting disease progression** \
*Jiayu Zhou, Lei Yuan, Jun Liu and Jieping Ye* \
Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining 2011. [[Paper](https://dl.acm.org/doi/abs/10.1145/2020408.2020549)]

- **Dirty Block-Sparse Model** \
*Ali Jalali, Sujay Sanghavi, Chao Ruan and Pradeep Ravikumar* \
Advances in neural information processing systems 2010. [[Paper](https://proceedings.neurips.cc/paper/2010/hash/00e26af6ac3b1c1c49d7c3d79c60d000-Abstract.html)] 


- **Adaptive Sparse Multi-Task Lasso** \
*Seunghak Lee, Jun Zhu and Eric Xing* \
Advances in neural information processing systems 2010. [[Paper](https://proceedings.neurips.cc/paper/2010/hash/3cf166c6b73f030b4f67eeaeba301103-Abstract.html)]
- **Multi-Task Feature Selection** \
*Guillaume Obozinski, Ben Taskar and Michael Jordan* \
researchgate 2006. [[Paper](https://www.researchgate.net/profile/Guillaume-Obozinski/publication/228666354_Multi-task_feature_selection/links/0a85e53b2c728afb52000000/Multi-task-feature-selection.pdf)]   

- **A probabilistic framework for multi-task learning** \
*Jian Zhang* \
Ph.D. Thesis 2006. [[Paper](https://cse.sustech.edu.cn/faculty/~zhangy/papers/Yu_Zhang_PhD_Thesis.pdf)] 




<a name = "feature-transformation"></a>
#### Feature Transformation

- **Multi-task learning for multiple language translation** [[Paper](https://aclanthology.org/P15-1166.pdf)] \
*Daxiang Dong, Hua Wu, Wei He, Dianhai Yu and Haifeng Wang* \
Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing 2015. 

- **A convex formulation for learning shared structures from multiple tasks** [[Paper](https://dl.acm.org/doi/abs/10.1145/1553374.1553392?casa_token=l-kQ9i88wJAAAAAA:K6TvNZGzne_TONBUpUNuYLRtY0QnPpF3GhKtiexV9EUtqOfOBEpQXI_5JlJuyi3_3apLj7pTVJhQ)] \
*Jianhui Chen, Lei Tang, Jun Liu and Jieping Ye* \
Proceedings of the 26th annual international conference on machine learning 2009. 

- **Multi-task feature learning** [[Paper](https://proceedings.neurips.cc/paper_files/paper/2006/hash/0afa92fc0f8a9cf051bf2961b06ac56b-Abstract.html)] \
*Andreas Argyriou, Theodoros Evgeniou and Massimiliano Pontil* \
Advances in neural information processing systems 2006. 

- **A framework for learning predictive structures from multiple tasks and
unlabeled data** [[Paper](https://www.jmlr.org/papers/volume6/ando05a/ando05a.pdf)] \
*Rie Kubota Ando, Tong Zhang and Peter Bartlett* \
Journal of Machine Learning Research 2005. 






<a  name = "low-rank-factorization"></a>
#### Low-Rank Factorization

- **Learning Linear and Nonlinear Low-Rank Structure in Multi-Task Learning** [[Paper](https://ieeexplore.ieee.org/abstract/document/9875058/)] \
*Yi Zhang, Yu Zhang and Wei Wang* \
IEEE Transactions on Knowledge and Data Engineering 2023

- **Multi-stage multi-task learning with reduced rank** [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/10261)] \
*Lei Han and Yu Zhang* \
Proceedings of the AAAI Conference on Artificial Intelligence 2016. 

- **Multitask learning meets tensor factorization: task imputation via convex optimization** [[Paper](https://proceedings.neurips.cc/paper/2014/hash/6d9cb7de5e8ac30bd5e8734bc96a35c1-Abstract.html)] \
*Kishan Wimalawarne, Masashi Sugiyama and Ryota Tomioka* \
Advances in neural information processing systems 2014

- **Multilinear multitask learning** [[Paper](https://proceedings.mlr.press/v28/romera-paredes13.html)] \
*Bernardino Romera-Paredes, Hane Aung, Nadia Bianchi-Berthouze and Massimiliano Pontil* \
Proceedings of the 30th International Conference on Machine Learning 2013

- **An accelerated gradient method for trace norm minimization** [[Paper](https://dl.acm.org/doi/abs/10.1145/1553374.1553434?casa_token=9J7YNGdK0jEAAAAA:enMIpGXqhGpAPKlYswpPOyLqlztQx9OwdTHnKGXV5L0HRvx0Wz2UjJY3cP73VTU84PqG7R6k-dJu)] \
*Shuiwang Ji and Jieping Ye* \
Proceedings of the 26th annual international conference on machine learning 2009. 






<a name = "decomposition"></a>
#### Decomposition

- **Learning incoherent sparse and low-rank patterns from multiple tasks** [[Paper](https://dl.acm.org/doi/abs/10.1145/2086737.2086742?casa_token=bqKYlMtacY4AAAAA:M3s9srPyiaWN8bMbbTUQY29nn1ezG1zhXqC-I67l4-IqyHuo60h86ucD4y_NDYbWVV68pK1XJCcT)] \
*Jianhui Chen, Ji Liu and Jieping Ye* \
ACM Transactions on Knowledge Discovery from Data 2012. 

- **Multi-level lasso for sparse multi-task regression** [[Paper](https://dl.acm.org/doi/abs/10.5555/3042573.3042652)] \
*Aurelie C Lozano and Grzegorz Swirszcz* \
Proceedings of the 29th International Coference on International Conference on Machine Learning 2012. 

- **Robust multi-task feature learning** [[Paper](https://dl.acm.org/doi/abs/10.1145/2339530.2339672casa_token=eakEuKmTUtEAAAAA:xeumdsvJ0ojB-uWcRnMUjscW-G4ry1fqlEveVLvlt5rnV5M1ZgxJxc1N0h3unKi9_JJrgK4kF4rq)] \
*Pinghua Gong, Jieping Ye and Changshui Zhang* \
Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining 2012. 

- **Integrating low-rank and group-sparse structures for robust multi-task learning** [[Paper](https://dl.acm.org/doi/abs/10.1145/2020408.2020423?casa_token=ztgYGGR1stsAAAAA:awKc_tKgcDBo6oL8s7kwngbfIqMi_O5gePUIUQiZUtaBNvgmCU670NbfPdO487XFCUoJOGH_wsAb)] \
*Jianhui Chen, Jiayu Zhou and Jieping Ye* \
Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining 2011. 


- **A dirty model for multi-task learning** [[Paper](https://proceedings.neurips.cc/paper/2010/hash/00e26af6ac3b1c1c49d7c3d79c60d000-Abstract.html)] \
*Ali Jalali, Sujay Sanghavi, Chao Ruan and Pradeep Ravikumar* \
Advances in neural information processing systems 2010.

- **A framework for learning predictive structures from multiple tasks and unlabeled data.** [[Paper](https://www.jmlr.org/papers/volume6/ando05a/ando05a.pdf?ref=ruder.io)] \
*Rie Kubota Ando, Tong Zhang and Peter Bartlett* \
Journal of Machine Learning Research 2005


<a name = "priori-sharing"></a>
#### Priori Sharing

- **A convex formulation for learning task relationships in multi-task learning.** [[Paper](https://arxiv.org/abs/1203.3536)] \
*Zhang, Yu and Yeung, Dit--Yan* \
arXiv preprint arXiv:1203.3536 2012

- **Hierarchical multitask structured output learning for large-scale sequence segmentation.** [[Paper](https://proceedings.neurips.cc/paper/2011/hash/ac796a52db3f16bbdb6557d3d89d1c5a-Abstract.html)] \
*G{\"o}rnitz, Nico and Widmer, Christian and Zeller, Georg and Kahles, Andr{\'e} and R{\"a}tsch, Gunnar and Sonnenburg, S{\"o}ren* \
Advances in Neural Information Processing Systems 2011

- **Large margin multi-task metric learning.** [[Paper](https://proceedings.neurips.cc/paper/2010/hash/087408522c31eeb1f982bc0eaf81d35f-Abstract.html)] \
*Evgeniou, Theodoros and Pontil, Massimiliano* \
Advances in neural information processing systems 2010

- **Multi-task learning using generalized t process.** [[Paper](https://proceedings.mlr.press/v9/zhang10c.html)] \
*Zhang, Yu and Yeung, Dit--Yan* \
JMLR Workshop and Conference Proceedings 2010

- **Multi-task learning via conic programming.** [[Paper](https://proceedings.neurips.cc/paper/2007/hash/67f7fb873eaf29526a11a9b7ac33bfac-Abstract.html)] \
*Kato, Tsuyoshi and Kashima, Hisashi and Sugiyama, Masashi and Asai, Kiyoshi* \
Advances in Neural Information Processing Systems 2007

- **Multi-task Gaussian process prediction.** [[Paper](https://proceedings.neurips.cc/paper_files/paper/2007/hash/66368270ffd51418ec58bd793f2d9b1b-Abstract.html)] \
*Bonilla, Edwin V and Chai, Kian and Williams, Christopher* \
Advances in Neural Information Processing Systems 2007

- **Learning multiple tasks with kernel methods.** [[Paper](https://www.jmlr.org/papers/volume6/evgeniou05a/evgeniou05a.pdf)] \
*Evgeniou, Theodoros and Micchelli, Charles A and Pontil, Massimiliano and Shawe-Taylor, John* \
Journal of machine learning research 2005

- **Regularized multi--task learning.** [[Paper](https://dl.acm.org/doi/abs/10.1145/1014052.1014067)] \
*Parameswaran, Shibin and Weinberger, Kilian Q* \
Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining 2004




<a name = "task-clustering"></a>
#### Task Clustering

- **Learning tree structure in multi-task learning.** [[Paper](https://dl.acm.org/doi/abs/10.1145/2783258.2783393)] \
*Han, Lei and Zhang, Yu* \
Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining 2015

- **Clustered Multi-Task Learning Via Alternating Structure Optimization.** [[Paper](https://proceedings.neurips.cc/paper/2011/hash/a516a87cfcaef229b342c437fe2b95f7-Abstract.html)] \
*Zhou, Jiayu and Chen, Jianhui and Ye, Jieping* \
Advances in Neural Information Processing Systems 2011

- **A Framework for Learning Predictive Structures from Multiple Tasks and Unlabeled Data.** [[Paper](https://dl.acm.org/doi/10.5555/1046920.1194905)] \
*Ando, Rie Kubota and Zhang, Tong* \
The Journal of Machine Learning Research 2005



<a name = "deeplearning-era"></a>
### Deep Learning Era

<a name = "feature-fusion"></a>
#### Feature Fusion




<table style="width:100%">
  <!-- <tr>
    <th style="width:50%">Paper</th>
    <th style="width:50%">Network</th>
  </tr> -->
  <tr>
    <tr>
    <td style="width:50%">
      <b>Latent multi-task architecture learning</b> <a href="https://ojs.aaai.org/index.php/AAAI/article/view/4410"><i>[Paper]</i></a> 
      <br>
      <b>Authors:</b> Sebastian Ruder, Joachim Bingel, Isabelle Augenstein and Anders Sogaard<br>
      <b>Publisher:</b> Proceedings of the AAAI Conference on Artificial Intelligence <br>
      <b>Year:</b> 2019 <br>
    </td>
    <td style="width:50%">
      <img src="img/sluice_block.png" alt="Description of image" style="max-width:100%">
    </td> 
  </tr>
  <tr>
    <td style="width:50%">
      <b>Nddr-cnn: Layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction</b> <a href="https://scholar.google.com/scholar?hl=en&as_sdt=0%2C39&q=%E2%80%9CNddr-cnn%3A+Layerwise+feature+fusing+in+multi-task+cnns+by+neural+discriminative+dimensionality+reduction%2C&btnG="><i>[Paper]</i></a> 
      <a href = "https://github.com/ethanygao/NDDR-CNN"><i>[Code]</i></a>
      <br>
      <b>Authors:</b> Yuan Gao, Jiayi Ma, Mingbo Zhao , Wei Liu and Alan L Yuille<br>
      <b>Publisher:</b> Proceedings of the IEEE/CVF conference on computer vision and pattern recognition <br>
      <b>Year:</b> 2019 <br>
    </td>
    <td style="width:50%">
      <img src="img/nddr.png" alt="Description of image" style="max-width:100%">
    </td> 
  </tr>
    <td style="width:50%">
      <b>Cross-stitch networks for multi-task learning</b> <a href="https://openaccess.thecvf.com/content_cvpr_2016/html/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.html"><i>[Paper]</i></a> 
      <a href = "https://github.com/helloyide/Cross-stitch-Networks-for-Multi-task-Learning"><i>[Code]</i></a><br>
      <b>Authors:</b> Ishan Misra, Abhinav Shrivastava, Abhinav Gupta and Martial Hebert<br>
      <b>Publisher:</b> CVPR <br>
      <b>Year:</b> 2016 <br>
    </td>
    <td style="width:50%">
      <img src="img/cross_stitch.png" alt="Description of image" style="max-width:100%">
    </td>
    
  </tr>

</table>

<a name = "cascading"></a>
#### Cascading

<table style="width:100%">
  <!-- <tr>
    <th style="width:50%">Paper</th>
    <th style="width:50%">Network</th>
  </tr> -->
    <tr>
    <td style="width:50%">
      <b>  Deep cascade multi-task learning for slot filling in online shopping assistant</b> <a href="https://ojs.aaai.org/index.php/AAAI/article/view/4611"><i>[Paper]</i></a> 
      <a href = "https://github.com/gy910210/DCMTL"><i>[Code]</i></a>
      <br>
      <b>Authors:</b> Yu Gong, Xusheng Luo, Yu Zhu, Wenwu Ou, Zhao Li, Muhua Zhu, Kenny Q. Zhu, Lu Duan and Xi Chen <br>
      <b>Publisher:</b> Proceedings of the AAAI conference on artificial intelligence <br>
      <b>Year:</b> 2019  <br>
    </td>
    <td style="width:50%">
      <img src="img/cascade_4.png" alt="Description of image" style="max-width:100%">
    </td>
    
  </tr>
  <tr>
    <td style="width:50%">
      <b> A hierarchical multi-task approach for learning embeddings from semantic tasks</b> <a href="https://ojs.aaai.org/index.php/AAAI/article/view/4673"><i>[Paper]</i></a> 
      <a href = "https://github.com/huggingface/hmtl"><i>[Code]</i></a>
      <br>
      <b>Authors:</b> Victor Sanh, Thomas Wolf and Sebastian Ruder <br>
      <b>Publisher:</b> Proceedings of the AAAI conference on artificial intelligence <br>
      <b>Year:</b> 2019  <br>
    </td>
    <td style="width:50%">
      <img src="img/cascade_4.png" alt="Description of image" style="max-width:100%">
    </td>
    
  </tr>
  <tr>
    <td style="width:50%">
      <b>Deep multi-task learning with low level tasks supervised at lower layers</b> <a href="https://aclanthology.org/P16-2038/"><i>[Paper]</i></a> 
      <!-- <a href = "https://github.com/VICO-UoE/KD4MTL"><i>[Code]</i></a> -->
      <br>
      <b>Authors:</b> Anders S{\o}gaard and Yoav Goldberg<br>
      <b>Publisher:</b> Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics Volume 2: Short Papers <br>
      <b>Year:</b> 2016 <br>
    </td>
    <td style="width:50%">
      <img src="img/cascade_1.png" alt="Description of image" style="max-width:100%">
    </td>
    
  </tr>
  <tr>
    <td style="width:50%">
      <b>Instance-aware semantic segmentation via multi-task network cascades</b> <a href="https://openaccess.thecvf.com/content_cvpr_2016/html/Dai_Instance-Aware_Semantic_Segmentation_CVPR_2016_paper.html"><i>[Paper]</i></a> 
      <a href = "https://github.com/daijifeng001/MNC"><i>[Code]</i></a>
      <br>
      <b>Authors:</b> Jifeng Dai, Kaiming He and Jian Sun<br>
      <b>Publisher:</b> Proceedings of the IEEE conference on computer vision and pattern recognition <br>
      <b>Year:</b> 2016 <br>
    </td>
    <td style="width:50%">
      <img src="img/cascade_2.png" alt="Description of image" style="max-width:100%">
    </td>
    
  </tr>

  <tr>
    <td style="width:50%">
      <b>A Joint Many-Task Model: Growing a Neural Network for
Multiple NLP Tasks</b> <a href="https://arxiv.org/abs/1611.01587"><i>[Paper]</i></a> 
      <a href = "https://github.com/hassyGo/charNgram2vec"><i>[Code]</i></a>
      <br>
      <b>Authors:</b> Kazuma Hashimoto, Caiming Xiong, Yoshimasa Tsuruoka, Richard Socher <br>
      <b>Publisher:</b> arXiv preprint arXiv:1611.01587 <br>
      <b>Year:</b> 2016 <br>
    </td>
    <td style="width:50%">
      <img src="img/cascade_3.png" alt="Description of image" style="max-width:100%">
    </td>
    
  </tr>


</table>

<a name = "knowledge-distillation"></a>
#### Knowledge Distilation


<table style="width:100%">
  <!-- <tr>
    <th style="width:50%">Paper</th>
    <th style="width:50%">Network</th>
  </tr> -->
    <tr>
    <td style="width:50%">
      <b>Online Knowledge Distillation for Multi-Task Learning</b> <a href="https://openaccess.thecvf.com/content/WACV2023/html/Jacob_Online_Knowledge_Distillation_for_Multi-Task_Learning_WACV_2023_paper.html"><i>[Paper]</i></a> 
      <!-- <a href = "https://github.com/VICO-UoE/KD4MTL"><i>[Code]</i></a> -->
      <br>
      <b>Authors:</b> Geethu Miriam Jacob and Vishal Agarwal<br>
      <b>Publisher:</b> Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision <br>
      <b>Year:</b> 2023 <br>
    </td>
    <td style="width:50%">
      <img src="img/OKD-MTL.png" alt="Description of image" style="max-width:100%">
    </td>
    <tr>
    <td style="width:50%">
      <b>Cross-task knowledge distillation in multi-task recommendation</b> <a href="https://ojs.aaai.org/index.php/AAAI/article/view/20352"><i>[Paper]</i></a> 
      <!-- <a href = "https://github.com/VICO-UoE/KD4MTL"><i>[Code]</i></a> -->
      <br>
      <b>Authors:</b> Chenxiao Yang, Junwei Pan, Xiaofeng Gao, Tingyu Jiang, Dapeng Liu and Guihai Chen<br>
      <b>Publisher:</b> Proceedings of the AAAI Conference on Artificial Intelligence <br>
      <b>Year:</b> 2022 <br>
    </td>
    <td style="width:50%">
      <img src="img/cross_task_kd.png" alt="Description of image" style="max-width:100%">
    </td>
  </tr>
    <tr>
    <td style="width:50%">
      <b>Multi-task self-training for learning general representations</b> <a href="https://openaccess.thecvf.com/content/ICCV2021/html/Ghiasi_Multi-Task_Self-Training_for_Learning_General_Representations_ICCV_2021_paper.html"><i>[Paper]</i></a> 
      <!-- <a href = "https://github.com/VICO-UoE/KD4MTL"><i>[Code]</i></a> -->
      <br>
      <b>Authors:</b> Golnaz Ghiasi, Barret Zoph, Ekin D Cubuk and Quoc V Le and Tsung-Yi Lin<br>
      <b>Publisher:</b> Proceedings of the IEEE/CVF International Conference on Computer Vision <br>
      <b>Year:</b> 2021 <br>
    </td>
    <td style="width:50%">
      <img src="img/MuST.png" alt="Description of image" style="max-width:100%">
    </td>
  </tr>
  <tr>
    <td style="width:50%">
      <b>Knowledge distillation for multi-task learning</b> <a href="https://link.springer.com/chapter/10.1007/978-3-030-65414-6_13"><i>[Paper]</i></a> 
      <a href = "https://github.com/VICO-UoE/KD4MTL"><i>[Code]</i></a><br>
      <b>Authors:</b> WeiHong Li and Hakan Bilen<br>
      <b>Publisher:</b> Computer Vision--ECCV <br>
      <b>Year:</b> 2020 <br>
    </td>
    <td style="width:50%">
      <img src="img/KD4MTL.png" alt="Description of image" style="max-width:100%">
    </td>
    
  </tr>




    
  </tr>

</table>

<a name = "cross_task_attention"></a>
#### Cross-Task Attention

<table style="width:100%">
  <!-- <tr>
    <th style="width:50%">Paper</th>
    <th style="width:50%">Network</th>
  </tr> -->
  <tr>
  <td style="width:50%">
    <b> Demt: Deformable mixer transformer for multi-task learning of dense prediction</b> 
    <a href="https://ui.adsabs.harvard.edu/abs/2023arXiv230103461X/abstract"><i>[Paper]</i></a> 
    <a href = "https://github.com/yangyangxu0/demt"><i>[Code]</i></a>
    <br>
    <b>Authors:</b> Yangyang Xu, Yibo Yang and Lefei Zhang<br>
    <b>Publisher:</b> arXiv e-prints <br>
    <b>Year:</b> 2023 <br>
  </td>
  <td style="width:50%">
    <img src="img/DeMT.png" alt="Description of image" style="max-width:100%">
  </td>
  </tr> 

  <tr>
  <td style="width:50%">
    <b>Exploring relational context for multi-task dense prediction</b> 
    <a href="https://openaccess.thecvf.com/content/ICCV2021/html/Bruggemann_Exploring_Relational_Context_for_Multi-Task_Dense_Prediction_ICCV_2021_paper.html"><i>[Paper]</i></a> 
    <a href = "https://github.com/brdav/atrc"><i>[Code]</i></a>
    <br>
    <b>Authors:</b> David Bruggemann, Menelaos Kanakis, Anton Obukhov, Stamatios Georgoulis and Luc Van Gool<br>
    <b>Publisher:</b> Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition <br>
    <b>Year:</b> 2021 <br>
  </td>
  <td style="width:50%">
    <img src="img/ATRC-module.png" alt="Description of image" style="max-width:100%">
  </td>
  </tr>
  <tr>
    <td style="width:50%">
      <b>Pattern-structure diffusion for multi-task learning</b> <a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Zhou_Pattern-Structure_Diffusion_for_Multi-Task_Learning_CVPR_2020_paper.html"><i>[Paper]</i></a> 
      <!-- <a href = "https://github.com/lorenmt/mtan"><i>[Code]</i></a> -->
      <br>
      <b>Authors:</b> Ling Zhou, Zhen Cui, Chunyan Xu, Zhenyu Zhang, Chaoqun Wang, Tong Zhang and Jian Yang<br>
      <b>Publisher:</b> Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition <br>
      <b>Year:</b> 2020 <br>
    </td>
    <td style="width:50%">
      <img src="img/PSD.png" alt="Description of image" style="max-width:100%">
    </td>
    
  </tr>
  <tr>
    <td style="width:50%">
      <b>Mti-net: Multi-scale task interaction networks for multi-task
learning</b> <a href="https://link.springer.com/chapter/10.1007/978-3-030-58548-8_31"><i>[Paper]</i></a> 
      <a href = "https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch"><i>[Code]</i></a>
      <br>
      <b>Authors:</b> Simon Vandenhende, Stamatios Georgoulis and Luc Van Gool<br>
      <b>Publisher:</b> Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part IV 16 <br>
      <b>Year:</b> 2020 <br>
    </td>
    <td style="width:50%">
      <img src="img/MTI-Net.png" alt="Description of image" style="max-width:100%">
    </td>
    
  </tr>
  <tr>
    <td colspan="2" style="width:100%">
      <b>End-to-end multi-task learning with attention</b> <a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.html"><i>[Paper]</i></a> 
      <a href = "https://github.com/lorenmt/mtan"><i>[Code]</i></a><br>
      <b>Authors:</b> Shikun Liu, Edward Johns and Andrew J Davison<br>
      <b>Publisher:</b> Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition <br>
      <b>Year:</b> 2019 <br>
    </td>

  </tr>
  <tr>
    <td style="width:50%">
      <b>Pattern-affinitive propagation across depth, surface normal and semantic segmentation</b> <a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Pattern-Affinitive_Propagation_Across_Depth_Surface_Normal_and_Semantic_Segmentation_CVPR_2019_paper.html"><i>[Paper]</i></a> 
      <!-- <a href = "https://github.com/lorenmt/mtan"><i>[Code]</i></a> -->
      <br>
      <b>Authors:</b> Zhenyu Zhang, Zhen Cui, Chunyan Xu, Yan Yan, Nicu Sebe and Jian Yang<br>
      <b>Publisher:</b> Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition <br>
      <b>Year:</b> 2019 <br>
    </td>
    <td style="width:50%">
      <img src="img/PAP.png" alt="Description of image" style="max-width:100%">
    </td>
    
  </tr>
  <tr>
    <td style="width:50%">
      <b>Attentive single-tasking of multiple tasks</b> 
      <a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Maninis_Attentive_Single-Tasking_of_Multiple_Tasks_CVPR_2019_paper.html"><i>[Paper]</i></a> 
      <a href = "https://github.com/facebookresearch/astmt"><i>[Code]</i></a>
      <br>
      <b>Authors:</b> Kevis-Kokitsi Maninis, Ilija Radosavovic and Iasonas Kokkinos<br>
      <b>Publisher:</b> Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition <br>
      <b>Year:</b> 2019 <br>
    </td>
    <td style="width:50%">
      <img src="img/ASTMT.png" alt="Description of image" style="max-width:100%">
    </td>
    
  </tr>
  <tr>
    <td style="width:50%">
      <b>Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing</b> <a href="https://openaccess.thecvf.com/content_cvpr_2018/html/Xu_PAD-Net_Multi-Tasks_Guided_CVPR_2018_paper.html"><i>[Paper]</i></a> 
      <!-- <a href = "https://github.com/VICO-UoE/KD4MTL"><i>[Code]</i></a><br> -->
      <br>
      <b>Authors:</b> Dan Xu, Wanli Ouyang, Xiaogang Wang and Nicu Sebe<br>
      <b>Publisher:</b> Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition <br>
      <b>Year:</b> 2018 <br>
    </td>
    <td style="width:50%">
      <img src="img/PAD.png" alt="Description of image" style="max-width:100%">
    </td>
  </tr>






  <!-- <tr>
  <td style="width:50%">
    <b> </b> 
    <a href=""><i>[Paper]</i></a> 
    <a href = ""><i>[Code]</i></a>
    <br>
    <b>Authors:</b> <br>
    <b>Publisher:</b> <br>
    <b>Year:</b> <br>
  </td>
  <td style="width:50%">
    <img src="" alt="Description of image" style="max-width:100%">
  </td>
  
  </tr>  -->
</table>

<a name = "scalarization_approach"></a>
#### Scalarization 
<table style="width:100%">

  <!-- <tr>
    <th style="width:50%">Paper</th>
    <th style="width:50%">Network</th>
  </tr> -->
  <tr>
    <td colspan="2" style="width:100%">
      <b>Recon: Reducing Conflicting Gradients From the Root For Multi-Task Learning</b> <a href="https://openreview.net/forum?id=ivwZO-HnzG_"><i>[Paper]</i></a> 
      <a href = "https://github.com/moukamisama/recon"><i>[Code]</i></a><br>
      <b>Authors:</b> Guangyuan Shi, Qimai Li, Wenlong Zhang, Jiaxin Chen, Xiao-Ming Wu<br>
      <b>Publisher:</b> The Eleventh International Conference on Learning Representations <br>
      <b>Year:</b> 2022 <br>
    </td>
    <!-- <td style="width:50%">
      <img src="img/pad_module.png" alt="Description of image" style="max-width:100%">
    </td> -->
  </tr>
  <tr>
    <td colspan="2" style="width:100%">
      <b>Towards impartial multi-task learning</b> <a href="https://openreview.net/forum?id=IMPnRXEWpvr"><i>[Paper]</i></a> 
      <a href = "https://github.com/lorenmt/mtan"><i>[Code]</i></a><br>
      <b>Authors:</b> Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, Wayne Zhang<br>
      <b>Publisher:</b> ICLR <br>
      <b>Year:</b> 2021 <br>
    </td>
    <!-- <td style="width:50%">
      <img src="img/pad_module.png" alt="Description of image" style="max-width:100%">
    </td> -->
  </tr>
  <tr>
    <td style="width:50%">
      <b>Gradient surgery for multi-task learning</b> <a href="https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html"><i>[Paper]</i></a> 
      <a href = "https://github.com/tianheyu927/PCGrad"><i>[Code]</i></a><br>
      <b>Authors:</b> Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea Finn<br>
      <b>Publisher:</b> Advances in Neural Information Processing Systems <br>
      <b>Year:</b> 2020 <br>
    </td>
    <td style="width:50%">
      <img src="img/pcgrad.png" alt="Description of image" style="max-width:100%">
    </td>
  </tr>
  <tr>
    <td colspan="2" style="width:100%">
      <b>Multi-task learning using uncertainty to weigh losses for scene geometry and semantics</b> <a href="https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html"><i>[Paper]</i></a> 
      <a href = "https://github.com/ranandalon/mtl"><i>[Code]</i></a><br>
      <b>Authors:</b> Alex Kendall, Yarin Gal and Roberto Cipollae<br>
      <b>Publisher:</b> Proceedings of the IEEE conference on computer vision and pattern recognition <br>
      <b>Year:</b> 2018 <br>
    </td>
    <!-- <td style="width:50%">
      <img src="img/pad_module.png" alt="Description of image" style="max-width:100%">
    </td> -->
  </tr>
  <tr>
    <td colspan="2" style="width:100%">
      <b>Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks</b> <a href="http://proceedings.mlr.press/v80/chen18a.html?ref=https://githubhelp.com"><i>[Paper]</i></a> 
      <a href = "https://github.com/LucasBoTang/GradNorm"><i>[Code]</i></a><br>
      <b>Authors:</b> Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee and Andrew Rabinovich<br>
      <b>Publisher:</b> International conference on machine learning <br>
      <b>Year:</b> 2018 <br>
    </td>
    <!-- <td style="width:50%">
      <img src="img/pad_module.png" alt="Description of image" style="max-width:100%">
    </td> -->
  </tr>


</table>



<a name = "Multi-objective Optimization (MOO)"></a>
### Multi-objective Optimization (MOO)
<table style="width:100%">
  <tr>
    <td colspan="2" style="width:100%">
      <b>Mitigating gradient bias in multi-objective learning: A provably convergent approach</b> 
      <a href=""><i>[Paper]</i></a> 
      <a href = ""><i>[Code]</i></a>
      <br>
      <b>Authors:</b>Heshan Devaka Fernando，Han Shen, Miao Liu,Subhajit Chaudhury, Keerthiram Murugesan and Tianyi Chen<br>
      <b>Publisher:</b>The Eleventh International Conference on Learning Representations  <br>
      <b>Year:</b>2022  <br>
    </td>
  </tr>
  <tr>
    <td colspan="2" style="width:100%">
      <b>Multi-task learning as a bargaining game</b> 
      <a href="https://arxiv.org/abs/2202.01017"><i>[Paper]</i></a> 
      <a href ="https://paperswithcode.com/paper/multi-task-learning-as-a-bargaining-game"><i>[Code]</i></a>
      <br>
      <b>Authors:</b>Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik and Ethan Fetaya<br>
      <b>Publisher:</b>arXiv preprint arXiv:2202.01017  <br>
      <b>Year:</b>2022  <br>
    </td>
  </tr>
  <tr>
    <td colspan="2" style="width:100%">
      <b>Multi-task learning with user preferences: Gradient descent with controlled ascent in pareto optimization</b> 
      <a href="https://proceedings.mlr.press/v119/mahapatra20a.html"><i>[Paper]</i></a> 
      <a href = "https://paperswithcode.com/paper/multi-task-learning-with-user-preferences"><i>[Code]</i></a>
      <br>
      <b>Authors:</b>Debabrata Mahapatra and Vaibhav Rajan<br>
      <b>Publisher:</b>International Conference on Machine Learning  <br>
      <b>Year:</b>2020  <br>
    </td>
  </tr>
  <tr>
    <td colspan="1" style="width:100%">
      <b>Pareto multi-task learning</b> 
      <a href="https://proceedings.neurips.cc/paper_files/paper/2019/hash/685bfde03eb646c27ed565881917c71c-Abstract.html"><i>[Paper]</i></a> 
      <a href = "https://paperswithcode.com/paper/pareto-multi-task-learning-1"><i>[Code]</i></a>
      <br>
      <b>Authors:</b>Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qing-Fu Zhang and Sam Kwong<br>
      <b>Publisher:</b>Advances in neural information processing systems  <br>
      <b>Year:</b>2019  <br>
    </td>
  </tr>

  <tr>
    <td colspan="2" style="width:100%">
      <b>Multi-task learning as multi-objective optimization</b> 
      <a href="https://proceedings.neurips.cc/paper_files/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html"><i>[Paper]</i></a> 
      <a href = "https://paperswithcode.com/paper/multi-task-learning-as-multi-objective"><i>[Code]</i></a>
      <br>
      <b>Authors:</b>Ozan Sener and Vladlen Koltun<br>
      <b>Publisher:</b>Advances in neural information processing systems  <br>
      <b>Year:</b>2018  <br>
    </td>
  </tr>
  <tr>
    <td colspan="1" style="width:100%">
      <b>Multicriteria optimization</b> 
      <a href="https://books.google.com.tw/books?hl=en&lr=&id=8wGyB5Sa2CUC&oi=fnd&pg=PA1&dq=Multicriteria+optimization&ots=ahYLzX1okW&sig=P0iD91TO_igyvyaZZxpB96fnB6A&redir_esc=y#v=onepage&q=Multicriteria%20optimization&f=false"><i>[Paper]</i></a> 
      <!-- <a href = ""><i>[Code]</i></a> -->
      <br>
      <b>Authors:</b>Matthias Ehrgott<br>
      <b>Publisher:</b>Springer Science \& Business Media  <br>
      <b>Year:</b>2005  <br>
    </td>
  </tr>
  <tr>
    <td colspan="2" style="width:100%">
      <b>Steepest descent methods for multicriteria optimization</b> 
      <a href="https://link.springer.com/article/10.1007/s001860000043"><i>[Paper]</i></a> 
      <!-- <a href = ""><i>[Code]</i></a> -->
      <br>
      <b>Authors:</b>J{\"o}rg Fliege and Benar Fux Svaiter<br>
      <b>Publisher:</b>Springer<br>
      <b>Year:</b>2000  <br>
    </td>
  </tr>


</table>

<a name = "Adversarial training"></a>
#### Adversarial Training
<table style="width:100%">
  <!-- <tr>
    <th style="width:50%">Paper</th>
    <th style="width:50%">Network</th>
  </tr> -->
  <tr>
    <td style="width:50%">
      <b>Representation disentanglement for multi-task learning with application to fetal ultrasound</b> 
      <a href="https://link.springer.com/chapter/10.1007/978-3-030-32875-7_6"><i>[Paper]</i></a> 
      <a href = "https://paperswithcode.com/paper/190807885"><i>[Code]</i></a>
      <br>
      <b>Authors:</b>Qingjie Meng, Nick Pawlowski, Daniel Rueckert, Bernhard Kainz<br>
      <b>Publisher:</b> Springer <br>
      <b>Year:</b> 2019 <br>
    </td>
    <td style="width:50%">
      <img src="img/rd4mtl.png" alt="Description of image" style="max-width:100%">
    </td>
  </tr>
    <tr>
    <td style="width:50%">
      <b>Multi-task adversarial network for disentangled feature learning</b> 
      <a href="https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Multi-Task_Adversarial_Network_CVPR_2018_paper.html"><i>[Paper]</i></a> 
      <!-- <a href = "https://github.com/FrankWork/fudan_mtl_reviews"><i>[Code]</i></a> -->
      <br>
      <b>Authors:</b> Yang Liu, Zhaowen Wang, Hailin Jin, Ian Wassell<br>
      <b>Publisher:</b> Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition <br>
      <b>Year:</b> 2018 <br>
    </td>
    <td style="width:50%">
      <img src="img/MTAdvN.png" alt="Description of image" style="max-width:100%">
    </td>
  </tr>

  <tr>
    <td style="width:50%">
      <b>Gradient adversarial training of neural networks</b> 
      <a href="https://arxiv.org/abs/1806.08028"><i>[Paper]</i></a> 
      <!-- <a href = "https://paperswithcode.com/paper/190807885"><i>[Code]</i></a> -->
      <br>
      <b>Authors:</b>Ayan Sinha, Zhao Chen, Vijay Badrinarayanan, Andrew Rabinovich<br>
      <b>Publisher:</b> arXiv preprint arXiv:1806.08028 <br>
      <b>Year:</b> 2018 <br>
    </td>
    <td style="width:50%">
      <img src="img/great4mtl.png" alt="Description of image" style="max-width:100%">
    </td>
  </tr>
  <tr>
    <td style="width:50%">
      <b>Adversarial Multi-task Learning for Text Classification</b> 
      <a href="https://arxiv.org/abs/1704.05742"><i>[Paper]</i></a> 
      <a href = "https://github.com/FrankWork/fudan_mtl_reviews"><i>[Code]</i></a>
      <br>
      <b>Authors:</b> Pengfei Liu, Xipeng Qiu, Xuanjing Huang <br>
      <b>Publisher:</b> arXiv preprint arXiv:1704.05742 <br>
      <b>Year:</b> 2017 <br>
    </td>
    <td style="width:50%">
      <img src="img/asp_mtl.png" alt="Description of image" style="max-width:100%">
    </td>
  </tr>


</table>


<a name = "mixture_of_experts"></a>
#### Mixture of Experts
<table style="width:100%">
  <!-- <tr>
    <th style="width:50%">Paper</th>
    <th style="width:50%">Network</th>
  </tr> -->
    <tr>
      <td colspan="2" style="width:100%">
      <b>Mod-Squad: Designing Mixtures of Experts As Modular Multi-Task Learners</b>
      <a href="https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Mod-Squad_Designing_Mixtures_of_Experts_As_Modular_Multi-Task_Learners_CVPR_2023_paper.html"><i>[Paper]</i></a> 
      <!-- <a href = "https://paperswithcode.com/paper/dselect-k-differentiable-selection-in-the#code"><i>[Code]</i></a><br> -->
      <br>
      <b>Authors:</b> Zitian Chen, Yikang Shen, Mingyu Ding, Zhenfang Chen, Hengshuang Zhao, Erik Learned-Miller, Chuang Gan<br>
      <b>Publisher:</b> Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition <br>
      <b>Year:</b> 2023 <br>
    </td>
    <!-- <td style="width:50%">
      <img src="img/pad_module.png" alt="Description of image" style="max-width:100%">
    </td> -->
  </tr>

  <tr>
    <td colspan="2" style="width:100%">
      <b>AdaMV-MoE: Adaptive Multi-Task Vision Mixture-of-Experts</b>
      <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Chen_AdaMV-MoE_Adaptive_Multi-Task_Vision_Mixture-of-Experts_ICCV_2023_paper.html"><i>[Paper]</i></a> 
      <a href = "https://paperswithcode.com/paper/adamv-moe-adaptive-multi-task-vision-mixture#code"><i>[Code]</i></a><br>
      <b>Authors:</b> Tianlong Chen, Xuxi Chen, Xianzhi Du, Abdullah Rashwan, Fan Yang, Huizhong Chen, Zhangyang Wang, Yeqing Li<br>
      <b>Publisher:</b> Proceedings of the IEEE/CVF International Conference on Computer Vision <br>
      <b>Year:</b> 2023 <br>
    </td>
    <!-- <td style="width:50%">
      <img src="img/pad_module.png" alt="Description of image" style="max-width:100%">
    </td> -->
  </tr>
  <tr>
    <td colspan="2" style="width:100%">
      <b>SSummaReranker: A multi-task mixture-of-experts re-ranking framework for abstractive summarization</b> 
      <a href="https://arxiv.org/abs/2203.06569"><i>[Paper]</i></a> 
      <!-- <a href = "https://paperswithcode.com/paper/dselect-k-differentiable-selection-in-the#code"><i>[Code]</i></a><br> -->
      <br>
      <b>Authors:</b> Mathieu Ravaut, Shafiq Joty, Nancy F. Chen<br>
      <b>Publisher:</b> arXiv preprint arXiv:2203.06569 <br>
      <b>Year:</b> 2022 <br>
    </td>
    <!-- <td style="width:50%">
      <img src="img/pad_module.png" alt="Description of image" style="max-width:100%">
    </td> -->
  </tr>
  <tr>
    <td colspan="2" style="width:100%">
      <b>Multi-task learning with calibrated mixture of insightful experts</b>
      <a href="https://ieeexplore.ieee.org/abstract/document/9835373/"><i>[Paper]</i></a> 
      <!-- <a href = "https://paperswithcode.com/paper/dselect-k-differentiable-selection-in-the#code"><i>[Code]</i></a><br> -->
      <br>
      <b>Authors:</b> Sinan Wang, Yumeng Li, Hongyan Li, Tanchao Zhu, Zhao Li, Wenwu Ou<br>
      <b>Publisher:</b> 2022 IEEE 38th International Conference on Data Engineering (ICDE) <br>
      <b>Year:</b> 2022 <br>
    </td>
    <!-- <td style="width:50%">
      <img src="img/pad_module.png" alt="Description of image" style="max-width:100%">
    </td> -->
  </tr>
    <tr>
    <td style="width:50%">
      <b>Eliciting transferability in multi-task learning with task-level mixture-of-experts</b>
      <a href="https://arxiv.org/abs/2205.12701"><i>[Paper]</i></a> 
      <a href = "https://paperswithcode.com/paper/eliciting-transferability-in-multi-task#code"><i>[Code]</i></a><br>
      <b>Authors:</b> Qinyuan Ye, Juan Zha, Xiang Ren<br>
      <b>Publisher:</b> arXiv preprint arXiv:2205.12701 <br>
      <b>Year:</b> 2022 <br>
    </td>
    <td style="width:50%">
      <img src="img/single-moe.png" alt="Description of image" style="max-width:100%">
    </td>
  </tr>
  <tr>
    <td colspan="2" style="width:100%">
      <b>Eliciting and Understanding Cross-Task Skills with Task-Level Mixture-of-Experts</b>
      <a href="https://arxiv.org/abs/2205.12701"><i>[Paper]</i></a> 
      <a href = "https://paperswithcode.com/paper/eliciting-transferability-in-multi-task#code"><i>[Code]</i></a><br>
      <b>Authors:</b> Qinyuan Ye, Juan Zha, Xiang Ren<br>
      <b>Publisher:</b> arXiv preprint arXiv:2205.12701 <br>
      <b>Year:</b> 2022 <br>
    </td>
    <!-- <td style="width:50%">
      <img src="img/pad_module.png" alt="Description of image" style="max-width:100%">
    </td> -->
  </tr>
  <tr>
    <td colspan="2" style="width:100%">
      <b>M³ViT: Mixture-of-Experts Vision Transformer for Efficient Multi-task Learning with Model-Accelerator Co-design</b>
      <a href="https://proceedings.neurips.cc/paper_files/paper/2022/hash/b653f34d576d1790481e3797cb740214-Abstract-Conference.html"><i>[Paper]</i></a> 
      <a href = "https://paperswithcode.com/paper/m-3-vit-mixture-of-experts-vision-transformer#code"><i>[Code]</i></a><br>
      <b>Authors:</b> Hanxue Liang, Zhiwen Fan, Rishov Sarkar, Ziyu Jiang, Tianlong Chen, Kai Zou, Yu Cheng, Cong Hao, Zhangyang Wang<br>
      <b>Publisher:</b> Advances in Neural Information Processing Systems <br>
      <b>Year:</b> 2022 <br>
    </td>
    <!-- <td style="width:50%">
      <img src="img/pad_module.png" alt="Description of image" style="max-width:100%">
    </td> -->
  </tr>
  <tr>
    <td colspan="2" style="width:100%">
      <b>Dselect-k: Differentiable selection in the mixture of experts with applications to multi-task learning</b> 
      <a href="https://proceedings.neurips.cc/paper_files/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html"><i>[Paper]</i></a> 
      <a href = "https://paperswithcode.com/paper/dselect-k-differentiable-selection-in-the#code"><i>[Code]</i></a><br>
      <b>Authors:</b> Hussein Hazimeh, Zhe Zhao, Aakanksha Chowdhery, Maheswaran Sathiamoorthy, Yihua Chen, Rahul Mazumder, Lichan Hong, Ed H. Chi<br>
      <b>Publisher:</b> Advances in Neural Information Processing Systems <br>
      <b>Year:</b> 2021 <br>
    </td>
    <!-- <td style="width:50%">
      <img src="img/pad_module.png" alt="Description of image" style="max-width:100%">
    </td> -->
  </tr>
    <tr>
    <td colspan="2" style="width:100%">
      <b>Sparsely Activated Mixture-of-Experts are Robust Multi-Task Learners</b> 
      <a href="https://arxiv.org/abs/2204.07689"><i>[Paper]</i></a> 
      <!-- <a href = "https://paperswithcode.com/paper/dselect-k-differentiable-selection-in-the#code"><i>[Code]</i></a><br> -->
      <br>
      <b>Authors:</b> Shashank Gupta, Subhabrata Mukherjee, Krishan Subudhi, Eduardo Gonzalez, Damien Jose, Ahmed H. Awadallah, Jianfeng Gao<br>
      <b>Publisher:</b> arXiv preprint arXiv:2204.07689 <br>
      <b>Year:</b> 2021 <br>
    </td>
    <!-- <td style="width:50%">
      <img src="img/pad_module.png" alt="Description of image" style="max-width:100%">
    </td> -->
  </tr>
  <tr>
    <td style="width:50%">
      <b>Modeling task relationships in multi-task learning with multi-gate mixture-of-experts</b> 
      <a href="https://dl.acm.org/doi/abs/10.1145/3219819.3220007"><i>[Paper]</i></a> 
      <a href = "https://paperswithcode.com/paper/modeling-task-relationships-in-multi-task"><i>[Code]</i></a><br>
      <b>Authors:</b> Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong and Ed H Chi<br>
      <b>Publisher:</b> Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery \& data mining <br>
      <b>Year:</b> 2018 <br>
    </td>
    <td style="width:50%">
      <img src="img/multi-moe.png" alt="Description of image" style="max-width:100%">
    </td>
  </tr>
  <tr>
    <td style="width:50%">
      <b>Outrageously large neural networks: The sparsely-gated mixture-of-experts layer</b> 
      <a href="https://arxiv.org/abs/1701.06538"><i>[Paper]</i></a> <br>
      <b>Authors:</b> Noam Shazeer, Azalia Mirhoseini, Krzysztof  Maziarz, Andy Davis, Quoc Le,, Geoffrey Hinton and Jeff Dean<br>
      <b>Publisher:</b> arXiv preprint arXiv:1701.06538 <br>
      <b>Year:</b> 2017 <br>
    </td>
    <td style="width:50%">
      <img src="img/moe.png" alt="Description of image" style="max-width:100%">
    </td>
  </tr>




</table>



<a name = "graph_based"></a>
#### Graph-based
<table style="width:100%">
  <tr>
    <td style="width:50%">
      <b>Relational Multi-Task Learning: Modeling Relations between Data and Tasks</b> 
      <a href="https://arxiv.org/abs/2303.07666"><i>[Paper]</i></a> 
      <a href = "https://paperswithcode.com/paper/relational-multi-task-learning-modeling-1#code"><i>[Code]</i></a>
      <br>
      <b>Authors:</b> Kaidi Cao, Jiaxuan You, Jure Leskovec <br>
      <b>Publisher:</b> ParXiv preprint arXiv:2303.07666 <br>
      <b>Year:</b> 2023<br>
    </td>
    <td style="width:50%">
      <img src="img/metalink.png" alt="Description of image" style="max-width:100%">
    </td>
  </tr>
  <tr>
    <td style="width:50%">
      <b>Multi-label image recognition with graph convolutional networks</b> 
      <a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Multi-Label_Image_Recognition_With_Graph_Convolutional_Networks_CVPR_2019_paper.html"><i>[Paper]</i></a> 
      <a href = "https://paperswithcode.com/paper/multi-label-image-recognition-with-graph#code"><i>[Code]</i></a>
      <br>
      <b>Authors:</b> Zhao-Min Chen, Xiu-Shen Wei, Peng Wang, Yanwen Guo <br>
      <b>Publisher:</b> Proceedings of the IEEE/CVF conference on computer vision and pattern recognition <br>
      <b>Year:</b> 2019 <br>
    </td>
    <td style="width:50%">
      <img src="img/ML-GCN.png" alt="Description of image" style="max-width:100%">
    </td>
  <tr>
    <td colspan="2" style="width:100%">
      <b>Leveraging sequence classification by taxonomy-based multitask
      learning</b> 
      <a href="https://link.springer.com/chapter/10.1007/978-3-642-12683-3_34"><i>[Paper]</i></a> 
      <!-- <a href = "https://github.com/ranandalon/mtl"><i>[Code]</i></a><br> -->
      <br>
      <b>Authors:</b> Christian Widmer, Jose Leiva, Yasemin Altun, Gunnar Rätsch <br>
      <b>Publisher:</b> Research in Computational Molecular Biology: 14th Annual International Conference, RECOMB 2010, Lisbon, Portugal, April 25-28, 2010. Proceedings 14 <br>
      <b>Year:</b> 2010 <br>
    </td>
    <!-- <td style="width:50%">
      <img src="img/pad_module.png" alt="Description of image" style="max-width:100%">
    </td> -->
  </tr>
</table>

✉️ Contact
If you have questions, ideas, or wish to contribute, feel free to reach out:

📧 Jun Yu (juy220@lehigh.edu)
📧 Yong Chen (ychen123@pennmedicine.upenn.edu)

> @article{Jun2024Multi,
	author = {Jun Yu, Xiaokang Liu, Chongliang Luo, Jin Huang, Rong Zhou, Yixin Liu, Jie Hu, Jianmin Chen, Ke Zhang, Dazheng Zhang, Yishan Shen, Eashan Adhikarla, Yutong Dai, Kai Zhang, Zhaoming Kong, Wenxuan Ye, Yilong Yin, Vinod Namboodiri, Brian D. Davison, Jason H. Moore, Yong Chen},
	journal = {Harvard Data Science Review},
	number = {Special Issue X},
	year = {2024},
	month = {may 31},
	note = {https://hdsr.mitpress.mit.edu/pub/xxxxxx},
	publisher = {The MIT Press},
	title = {Multi-Task Learning 1997-2024: Part I Fundamentals},
	volume = { },
}

> @article{Jun2024Multi,
	author = {Xiaokang Liu, Jun Yu, Yutong Dai, Yishan Shen, Jianmin Chen, Jie Hu, Jin Huang, Yixin Liu, Yilong Yin, Vinod Namboodiri, Brian D. Davison, Jason H. Moore, Yong Chen},
	journal = {Harvard Data Science Review},
	number = {Special Issue X},
	year = {2024},
	month = {may 31},
	note = {https://hdsr.mitpress.mit.edu/pub/xxxxxx},
	publisher = {The MIT Press},
	title = {Multi-Task Learning 1997-2024: Part II Regularization and Optimization},
	volume = { },
}

> @article{Jun2024Multi,
	author = {Jun Yu, Jin Huang, Kai Zhang, Yixin Liu, Ke Zhang, Yishan Shen, Dazheng Zhang, Rong Zhou, Xiaokang Liu, Yilong Yin, Vinod Namboodiri, Brian D. Davison, Jason H. Moore, Yong Chen},
	journal = {Harvard Data Science Review},
	number = {Special Issue X},
	year = {2024},
	month = {may 31},
	note = {https://hdsr.mitpress.mit.edu/pub/xxxxxx},
	publisher = {The MIT Press},
	title = {Multi-Task Learning 1997-2024: Part III Applications},
	volume = { },
}
<!-- <a name="architectures"></a>
## Architectures

<!-- <a name="encoder-based"></a>
### Encoder-based

<a name = "decoder-based"></a>
### Decoder-based --> -->


<!-- DO NOT DELETE THIS LINE -->
Visitor Count: ![Visitor Count](https://profile-counter.glitch.me/{junfish}/count.svg)
