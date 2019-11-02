# Label-Embedded-Dictionary-Learning
A class shared dictionary learning method for image classification
Created by Shuai Shao, Rui Xu, Weifeng Liu, Bao-Di Liu, Yan-Jiang Wang from China University of Petroleum.




In this paper, we propose a novel dictionary learning algorithm named label embedded dictionary learning (LEDL). This method introduces the $\ell_1$-norm regularization term to replace the $\ell_0$-norm regularization of LC-KSVD. Compared with $\ell_0$-norm, the sparsity constraint factor of $\ell_1$-norm is unfixed so that the basis vectors can be selected freely for linear fitting. Thus, our proposed LEDL method can get smaller errors than LC-KSVD. In addition, $\ell_1$-norm sparse representation is widely used in many fields so that our proposed LEDL method can be extended and applied easily. We show the difference between our proposed LEDL and LC-KSVD in Figure~\ref{fig:Comparision}. We adopt the alternating direction method of multipliers (ADMM)~\cite{boyd2011distributed} framework and blockwise coordinate descent (BCD)~\cite{liu2014blockwise} algorithm to optimize LEDL. Our work mainly focuses on threefold.
