Ninput = 4; % alphabet size
Nhidden = 2; 
Ncontrol = 10; % really, 1 should be sufficient
Noutput = 4;
Ntimes = 1;

init_scale = 0.1;
num_iterations = 1500;
batch_size = 5;
num_examples = 5;
learning_rate = 0.3;

RNNnet = RNNmodel(Ninput, Nhidden, Noutput, Ncontrol, learning_rate, init_scale);
RNNnet.weights.w_HC = zeros(Ncontrol, Nhidden);
RNNnet.weights.w_CH = zeros(Nhidden, Ncontrol);
RNNnet.weights.w_CO = zeros(Noutput, Ncontrol);

mat = randi([0 1], num_examples, Ninput, Ntimes);
%{
NNnet = NNmodel(Nhidden, learning_rate, -2, init_scale, 0.00001, 0);
NNnet.setData(mat, zeros(size(mat, 1), 1), mat);
NNnet.configure();
NNnet.weights.W_IH = RNNnet.weights.w_IH;
NNnet.weights.W_HO = RNNnet.weights.w_HO;
NNnet.trainOnline(num_iterations);
[out_data, hid_data] = NNnet.runSet(mat(1, :), 0);
%}

mse_log = RNNnet.trainOnline(mat, mat, Ntimes, batch_size, num_iterations);
outs = RNNnet.predictSample(mat(1, :), Ntimes);
plot(mse_log);

%% 
outs = net.predictSet(mat, Ntimes);
figure(1);
subplot(1, 2, 1);
imagesc(outs);
subplot(1, 2, 2);
imagesc(mat(:, :, 1));