BlockExpertNet trained in two steps: 
1. Train binarized ResNet18
2. Copy weights from ResNet18 into each branch of BlockExpert.
3. Train binarized BlockExpertNet

This approach is derived from the paper *HIGH-CAPACITY EXPERT BINARY NETWORKS*.
https://arxiv.org/pdf/2010.03558.pdf