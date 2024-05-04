# f-Differential-Privacy-Gradient-Descent-with-Adaptive-per-Iteration-Privacy-Budget
Undergraduate thesis

This paper focuses on applying f-DP (Dong, J 2019) framework to a noisy gradient descent algorithm in machine learning. Based on f-DP's unique combination theorem and downsampling theorem, the paper propose a new modified algorithm which is improved from three main aspects: gradient approximation, step size selection, and adaptive noise adjustment. The algorithm can autonomously and dynamically adjust privacy budget allocation, iteration times, and learning rate with adaptive privacy budget allocation function. It is able to characterize differential privacy more accurately than the basic differential privacy framework (ϵ,δ)-DP (Cynthia, D 2008) while improving the learning accuracy of the model.

The following is the pseudocode:

![image text](https://github.com/JiangChengyu457/f-Differential-Privacy-Gradient-Descent-with-Adaptive-per-Iteration-Privacy-Budget/blob/main/pseudocode.png "pseudocode")

Afterwards, the paper implements the pseudocode and conduct experiments on public datasets to test the performance of the algorithm. By adjusting the values of each parameter, the impact of each parameter on the accuracy of the model under the premise of privacy protection will be further analyzed.


Reference:
[1] Dong, A. Roth, and W. Su. Gaussian differential privacy. [J] Journal of the Royal Statistical Society: Series B (with discussion), 2022：5-29. https://doi.org/10.1111/rssb.12455  [2] Cynthia Dwork. "Differential Privacy: A Survey of Results." [J]International Conference on Theory and Applications of Models of Computation Springer, Berlin, Heidelberg, 2008: 112-126.  [3] Cynthia Dwork, Aaron Roth, et al. 2014. The algorithmic foundations of differential privacy. [J] Foundations and Trends® in Theoretical Computer Science 9, 3–4 (2014):211–407.
