# Awesome-Multitask-Learning
<!-- This repository periodicly updates the the papers and resources of MTL. -->
Our survey primarily aims to provide a comprehensive understanding of MTL, encompassing its definition, taxonomy, applications, and their connections and trends. We delve into the various aspects of MTL methods, including the loss function, network architecture, and optimization methods, offering explanations and insights from the perspective of technical details. For each method, we provide the corresponding paper link, as well as the code repository for the MTL methods discussed in the paper. We sincerely hope that this survey aids in your comprehension of MTL and its associated methods. If you have any questions or suggestions, please feel free to contact us.
> [**Unleashing the Power of Multi-Task Learning: A Comprehensive Survey Spanning Traditional, Deep, and Pre-Trained Foundation Model Eras **](https://junfish.github.io/)


## Table of Contents:
- [Awesome-Multitask-Learning](#awesome-multitask-learning)
  - [Table of Contents:](#table-of-contents)
  - [Existing survey papers](#existing-survey-papers)
  - [Datasets](#datasets)
    - [Regression task](#regression-task)
    - [Classification task](#classification-task)
    - [Dense prediction task](#dense-prediction-task)
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
- **An overview of multi-task learning in deep neural networks** \
*Sebastian Ruder* \
arXiv 2017. [[Paper](https://arxiv.org/abs/1706.05098)] \
Jun 15, 2017

- **A brief review on multi-task learning** \
*Kim-Han Thung, Chong-Yaw Wee* \
Multimedia Tools and Applications 2018. [[Paper](https://link.springer.com/article/10.1007/s11042-018-6463-x)] 
Aug 08, 2018

- **Multi-task learning for dense prediction tasks: A survey** \
*Simon Vandenhende, Stamatios Georgoulis, Marc Proesmans, Dengxin Dai and Luc Van Gool* \
IEEE Transactions on Pattern Analysis and Machine Intelligence 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9336293)]
Jan 26, 2021
- **A Brief Review of Deep Multi-task Learning and Auxiliary Task Learning** \
*Partoo Vafaeikia and Khashayar Namdar and Farzad Khalvati* \
arXiv 2020. [[Paper](https://arxiv.org/abs/2007.01126)] \
Jul 02, 2020
- **Multi-Task Learning with Deep Neural Networks: A Survey** 
*Michael Crawshaw* \
arXiv 2020. [[Paper](https://arxiv.org/abs/2009.09796)] 
Sep 10, 2020
- **A survey on multi-task learning** \
*Zhang, Yu and Yang, Qiang*
IEEE Transactions on Knowledge and Data Engineering 2022. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9392366) \
March 31, 2021
<!-- - <a name = "ruder2017overview"></a> S. Ruder, “*[An overview of multi-task learning in deep neural networks](https://arxiv.org/abs/1706.05098),” arXiv preprint arXiv:1706.05098, 2017. -->
<!-- - <a name = "thung2018brief"></a> K.-H. Thung and C.-Y. Wee, “*[A brief review on multi-task learning](https://link.springer.com/article/10.1007/s11042-018-6463-x),” Multimedia Tools and Applications, vol. 77, no. 22, pp. 29705–29725, 2018. -->
<!-- - <a name = "vandenhende2021multi"></a>S. Vandenhende, S. Georgoulis, W. Van Gansbeke, M. Proesmans, D. Dai, and L. Van Gool, “*[Multi-task learning for dense prediction tasks: A survey](https://arxiv.org/abs/2004.13379),” IEEE transactions on pattern analysis and machine intelligence, 2021. -->
<!-- - <a name = "vafaeikia2020brief"></a> P. Vafaeikia, K. Namdar, and F. Khalvati, “*[A brief review of deep multi-task learning and auxiliary task learning](),” arXiv preprint arXiv:2007.01126, 2020. -->
<!-- - <a name = "crawshaw2020multi"></a> M. Crawshaw, “Multi-task learning with deep neural networks: A survey,” arXiv preprint arXiv:2009.09796, 2020. -->
<!-- - <a name = "zhang2021survey"></a> Y. Zhang and Q. Yang, “*[A survey on multi-task learning](https://ieeexplore.ieee.org/document/9392366),” IEEE Transactions on Knowledge and Data Engineering, 2021. -->

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









<!-- DO NOT DELETE THIS LINE -->
Visitor Count: ![Visitor Count](https://profile-counter.glitch.me/{junfish}/count.svg)
