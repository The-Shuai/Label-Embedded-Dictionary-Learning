function [B,W] = LEDL_SRC_ADMM_train(X,tr_label,Q,par)
% Input X,Y,K,alpha,lambda,Q
% X: training data, d*N；
% Y: label, C*N；
% K is the size of dictionary；
% alpha and lambda are hyperparameters
% Output S,B,W 

K                 = par.method.param.K;
alpha_train       = par.method.param.alpha_train;
lambda            = par.method.param.lambda;
omega             = par.method.param.omega;
rho_train         = par.method.param.rho_train;
gd_train        = par.method.param.gd_train; %梯度上升步长
min_gd_train     = par.method.param.min_gd_train;
maxiter           = par.method.maxiter;


% Compute Y
lable = max(tr_label); % 一共有多少类
Y = zeros(lable,size(X,2));
for i=1:size(X,2)
    Y(tr_label(i),i) = 1;
end

dimX = size(X,1);
dimY = size(Y,1);
dimQ = size(Q,1);
EPS = 1e-16;

Z = zeros(K,size(X,2)); % K*N, Initialize C randomly
delta = zeros(K,size(X,2)); % Initialize Lagragian to be nothing (seems to work well)

% Set the fast soft thresholding function
fast_sthresh = @(x,th) sign(x).*max(abs(x) - th,0);

F = 1 / (1 + lambda);

rand('seed',1);

B = rand(dimX, K)-0.5;
B = B - repmat(mean(B,1), size(B,1),1);
B = B * diag(1./sqrt(sum(B.*B)));

W = rand(dimY, K)-0.5;
W = W - repmat(mean(W,1), size(W,1),1);
W = W * diag(1./sqrt(sum(W.*W)));

A = rand(dimQ, K)-0.5;
A = A - repmat(mean(A,1), size(A,1),1);
A = A * diag(1./sqrt(sum(A.*A)));
iter = 0;

fold = 1000000000000;
fnew = 100000000000;
% while iter<maxiter 
tic;
t0 = cputime;
% while (fold-fnew)/fold>1*1e-6 && iter<maxiter
while iter < maxiter
    iter = iter+1;

    P = B' * B;
    p = B' * X;
    
    R = W' * W;
    r = W' * Y;
    
    T = A' * A;
    t = A' * Q;

    % Updating C while fixing B,W,A,Z,delta
    C = (P + lambda*R + omega*T + rho_train*eye(K)) \ (p + lambda*r + omega*t + rho_train*Z - delta);
    
    % Updating Z while fixing B,W,A,C,delta    
    Z = fast_sthresh(C + delta/rho_train, alpha_train/rho_train);

    % Updating delta while fixing B,W,A,C,Z
    delta = delta + gd_train*(C-Z);
%     delta = (1-step_train)*delta + step_train*(C-Z);
    gd_train = max(min_gd_train, gd_train*0.99);
%     steplen_train = min(max_steplen_train, steplen_train*1.05);

    % update B,W,A
    D = C * C';
    E = X * C';
    e = Y * C';
    f = Q * C';
    D_revise = tril(D,-1)+triu(D,1);
%     clear D; 
    D_2 = B * D_revise;
    for j=1:K
%         B(:,j) = (E(:,j) - B * D_revise(:,j));
        B(:,j) = B * D_revise(:,j);
%         WWWWWWWW=norm(B(:,j)); 
%         B(:,j) = B(:,j) / norm(B(:,j)); 
        W(:,j) = (e(:,j) - W * D_revise(:,j));
        W(:,j) = W(:,j) / norm(W(:,j)); 
        A(:,j) = (f(:,j) - A * D_revise(:,j));
        A(:,j) = A(:,j) / norm(A(:,j));
    end     
%     clear E; clear e; clear D_revise; 
    
%     S = (abs(S)>EPS) .* S; % ???
    
%     update objective function
%     fold = fnew;
%     fnew1 = sum(sum((X - B * C).^2));% 训练集X部分,L2范数的平方
%     fnew2 = lambda * sum(sum((Y - W * C).^2));% 标签Y部分,L2范数的平方
%     fnew3 = omega * sum(sum((Q - A * C).^2));% 标签Y部分,L2范数的平方
%     fnew4 = 2 * alpha_train * (sum(sum(abs(Z))));% 约束项S的L1范数
%     fnew = fnew1 + fnew2 + fnew3 + fnew4;
%     obj(iter,1) = fnew;
% 
%     
%     fprintf('Iteration = %.2f  ',iter);%display which iteration
% %     fprintf('relative error = %.5f  %.5f  %.5f  %.5f',fnew1/size(X,2),fnew2/size(X,2),fnew3,fnew);     
%     
%     fprintf('relative error = %.5f\n',fnew);
% %     fprintf('sparsity = %.5f\n',  length(find(abs(S(:))~=0))/length(S(:)));              

end
% toc;
% aaaaa = num2str(toc);
% t1 = cputime - t0;
% fprintf('convergence time = %.5f\n',t1);
end
   
