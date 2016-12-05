%% test the wmRNN with AX-CPT

Ninput = 4; % alphabet size
Nhidden = 2; 
Ncontrol = 2; % really, 1 should be sufficient
Noutput = 2;
Ntimes = 2;

init_scale = 0.5;
num_iterations = 15000;
batch_size = 4;
num_examples = 4;
learning_rate = 0.5;

net = RNNmodel(Ninput, Nhidden, Noutput, Ncontrol, learning_rate, init_scale);

inputs = zeros(num_examples, Ninput, Ntimes);
labels = zeros(num_examples, Noutput, Ntimes);
% AX
inputs(1, 1, 1) = 1;
inputs(1, 3, 2) = 1;
labels(1, 1, 2) = 1;

% BX
inputs(2, 2, 1) = 1;
inputs(2, 3, 2) = 1;
labels(2, 2, 2) = 1;

% AY
inputs(3, 1, 1) = 1;
inputs(3, 4, 2) = 1;
labels(3, 2, 2) = 1;

% BY
inputs(4, 2, 1) = 1;
inputs(4, 4, 2) = 1;
labels(4, 1, 2) = 1;
%}
mse_log = net.trainOnline(inputs, labels, Ntimes, batch_size, num_iterations, 1);
%%
% train the network
num_iterations = 1;
batch_size = 1;
mse_log = net.trainOnline(inputs(1, :, :), labels(1, :, :), Ntimes, batch_size, num_iterations, 1);
%% Autoencoder
Ninput = 4; % alphabet size
Nhidden = 3; 
Ncontrol = 10; % really, 1 should be sufficient
Noutput = 4;
Ntimes = 2;

init_scale = 0.5;
num_iterations = 25000;
batch_size = 7;
num_examples = 7;
learning_rate = 0.5;

net = RNNmodel(Ninput, Nhidden, Noutput, Ncontrol, learning_rate, init_scale);

inputs = zeros(num_examples, Ninput, Ntimes);
labels = zeros(num_examples, Noutput, Ntimes);
% AX
inputs(1, 1, 1) = 1;
inputs(1, 3, 2) = 1;
labels(1, 1, 2) = 1;
labels(1, 3, 2) = 1;

% BX
inputs(2, 2, 1) = 1;
inputs(2, 3, 2) = 1;
labels(2, 2, 2) = 1;
labels(2, 3, 2) = 1;

% AY
inputs(3, 1, 1) = 1;
inputs(3, 4, 2) = 1;
labels(3, 1, 2) = 1;
labels(3, 4, 2) = 1;

% BY
inputs(4, 2, 1) = 1;
inputs(4, 4, 2) = 1;
labels(4, 2, 2) = 1;
labels(4, 4, 2) = 1;

% AXY
inputs(5, 1, 1) = 1;
inputs(5, 3, 2) = 1;
inputs(5, 4, 2) = 1;
labels(5, 1, 2) = 1;
labels(5, 3, 2) = 1;
labels(5, 4, 2) = 1;

% BXY
inputs(6, 2, 1) = 1;
inputs(6, 3, 2) = 1;
inputs(6, 4, 2) = 1;
labels(6, 2, 2) = 1;
labels(6, 3, 2) = 1;
labels(6, 4, 2) = 1;

% ABXY
inputs(7, 1, 1) = 1;
inputs(7, 2, 1) = 1;
inputs(7, 3, 2) = 1;
inputs(7, 4, 2) = 1;
labels(7, 1, 2) = 1;
labels(7, 2, 2) = 1;
labels(7, 3, 2) = 1;
labels(7, 4, 2) = 1;

% train the network
mse_log = net.trainOnline(inputs, labels, Ntimes, batch_size, num_iterations, 0);
%%
%% Autoencoder w/ AXCPT
Ninput = 4; % alphabet size
Nhidden = 3; 
Ncontrol = 10; % really, 1 should be sufficient
Noutput = 4;
Ntimes = 2;

init_scale = 0.5;
num_iterations = 25000;
batch_size = 7;
num_examples = 7;
learning_rate = 0.5;

net = RNNmodel(Ninput, Nhidden, Noutput, Ncontrol, learning_rate, init_scale);
RNNnet.weights.w_CO = zeros(Noutput, Ncontrol);

inputs = zeros(num_examples, Ninput, Ntimes);
labels = zeros(num_examples, Noutput, Ntimes);
% AX
inputs(1, 1, 1) = 1;
inputs(1, 3, 2) = 1;
labels(1, 1, 2) = 1;
labels(1, 3, 2) = 1;

% BX
inputs(2, 2, 1) = 1;
inputs(2, 3, 2) = 1;
labels(2, 2, 2) = 1;
labels(2, 3, 2) = 1;

% AY
inputs(3, 1, 1) = 1;
inputs(3, 4, 2) = 1;
labels(3, 1, 2) = 1;
labels(3, 4, 2) = 1;

% BY
inputs(4, 2, 1) = 1;
inputs(4, 4, 2) = 1;
labels(4, 2, 2) = 1;
labels(4, 4, 2) = 1;

% AXY
inputs(5, 1, 1) = 1;
inputs(5, 3, 2) = 1;
inputs(5, 4, 2) = 1;
labels(5, 1, 2) = 1;
labels(5, 3, 2) = 1;
labels(5, 4, 2) = 1;

% BXY
inputs(6, 2, 1) = 1;
inputs(6, 3, 2) = 1;
inputs(6, 4, 2) = 1;
labels(6, 2, 2) = 1;
labels(6, 3, 2) = 1;
labels(6, 4, 2) = 1;

% ABXY
inputs(7, 1, 1) = 1;
inputs(7, 2, 1) = 1;
inputs(7, 3, 2) = 1;
inputs(7, 4, 2) = 1;
labels(7, 1, 2) = 1;
labels(7, 2, 2) = 1;
labels(7, 3, 2) = 1;
labels(7, 4, 2) = 1;

% train the network
mse_log = net.trainOnline(inputs, labels, Ntimes, batch_size, num_iterations, 0);

net.Noutput = 2;
net.biases.b_O = -3 * ones(2, 1);
net.weights.w_HO = (-1 + 2.*rand(net.Noutput,net.Nhidden)) * init_scale;
net.weights.w_CO = (-1 + 2.*rand(net.Noutput,net.Ncontrol)) * init_scale;

num_examples = 4;
batch_size = 4;

inputs = zeros(num_examples, Ninput, Ntimes);
labels = zeros(num_examples, net.Noutput, Ntimes);
% AX
inputs(1, 1, 1) = 1;
inputs(1, 3, 2) = 1;
labels(1, 1, 2) = 1;

% BX
inputs(2, 2, 1) = 1;
inputs(2, 3, 2) = 1;
labels(2, 2, 2) = 1;

% AY
inputs(3, 1, 1) = 1;
inputs(3, 4, 2) = 1;
labels(3, 2, 2) = 1;

% BY
inputs(4, 2, 1) = 1;
inputs(4, 4, 2) = 1;
labels(4, 1, 2) = 1;

mse_log = net.trainOnline(inputs, labels, Ntimes, batch_size, num_iterations, 1);
%% test the wmRNN with AX-CPT, AX given at the same time

Ninput = 4; % alphabet size
Nhidden = 50; 
Ncontrol = 50; % really, 1 should be sufficient
Noutput = 2;
Ntimes = 2;

init_scale = 0.5;
num_iterations = 25000;
batch_size = 4;
num_examples = 4;
learning_rate = 0.75;

net = RNNmodel(Ninput, Nhidden, Noutput, Ncontrol, learning_rate, init_scale);

inputs = zeros(num_examples, Ninput, Ntimes);
labels = zeros(num_examples, Noutput, Ntimes);
% AX
inputs(1, 1, 1) = 1;
inputs(1, 3, 1) = 1;
labels(1, 1, 2) = 1;

% BX
inputs(2, 2, 1) = 1;
inputs(2, 3, 1) = 1;
labels(2, 2, 2) = 1;

% AY
inputs(3, 1, 1) = 1;
inputs(3, 4, 1) = 1;
labels(3, 2, 2) = 1;

% BY
inputs(4, 2, 1) = 1;
inputs(4, 4, 1) = 1;
labels(4, 1, 2) = 1;
%}
mse_log = net.trainOnline(inputs, labels, Ntimes, batch_size, num_iterations, 0);
%% can classify the C layer into A or B?

Ninput = 4; % alphabet size
Nhidden = 50; 
Ncontrol = 50; % really, 1 should be sufficient
Noutput = 2;
Ntimes = 2;

init_scale = 0.5;
num_iterations = 25000;
batch_size = 4;
num_examples = 4;
learning_rate = 0.75;

net = RNNmodel(Ninput, Nhidden, Noutput, Ncontrol, learning_rate, init_scale);

inputs = zeros(num_examples, Ninput, Ntimes);
labels = zeros(num_examples, Noutput, Ntimes);
% AX
inputs(1, 1, 1) = 1;
inputs(1, 3, 2) = 1;
labels(1, 1, 2) = 1;

% BX
inputs(2, 2, 1) = 1;
inputs(2, 3, 2) = 1;
labels(2, 2, 2) = 1;

% AY
inputs(3, 1, 1) = 1;
inputs(3, 4, 2) = 1;
labels(3, 2, 2) = 1;

% BY
inputs(4, 2, 1) = 1;
inputs(4, 4, 2) = 1;
labels(4, 1, 2) = 1;
%}
mse_log = net.trainOnline(inputs, labels, Ntimes, batch_size, num_iterations, 0);
%%
[a_hids, a_cons, a_outs, z_hids, z_cons, z_outs] = net.netValsSet(inputs, 2);
Ninput = Ncontrol; % alphabet size
Nhidden = 50; 
Noutput = 2;
Ntimes = 2;

control_activations = a_cons(1:4, :, 2);
input_set = zeros(4, Ncontrol, 2);
input_set(:, :, 2) = control_activations;
stimulus_given = zeros(4, 2, Ntimes);
stimulus_given(1, 1, 2) = 1;
stimulus_given(2, 2, 2) = 1;
stimulus_given(3, 1, 2) = 1;
stimulus_given(4, 2, 2) = 1;
anNet = RNNmodel(Ninput, Nhidden, Noutput, Ncontrol, learning_rate, init_scale);
anNet.trainOnline(input_set, stimulus_given, Ntimes, 4, num_iterations, 0);
%% test the wmRNN with AX-CPT

Ninput = 4; % alphabet size
Nhidden = 50; 
Ncontrol = 50; % really, 1 should be sufficient
Noutput = 2;
Ntimes = 3;

init_scale = 0.5;
num_iterations = 25000;
batch_size = 4;
num_examples = 4;
learning_rate = 0.5;

net = RNNmodel(Ninput, Nhidden, Noutput, Ncontrol, learning_rate, init_scale);

inputs = zeros(num_examples, Ninput, Ntimes);
labels = zeros(num_examples, Noutput, Ntimes);
% AX
inputs(1, 1, 1) = 1;
inputs(1, 3, 3) = 1;
labels(1, 1, 3) = 1;

% BX
inputs(2, 2, 1) = 1;
inputs(2, 3, 3) = 1;
labels(2, 2, 3) = 1;

% AY
inputs(3, 1, 1) = 1;
inputs(3, 4, 3) = 1;
labels(3, 2, 3) = 1;

% BY
inputs(4, 2, 1) = 1;
inputs(4, 4, 3) = 1;
labels(4, 1, 3) = 1;
%}
mse_log = net.trainOnline(inputs, labels, Ntimes, batch_size, num_iterations, 1);
%%
plot(mse_log);
%%
outs = net.predictSet(inputs, Ntimes);

figure(1);
subplot(1, 2, 1);
imagesc(outs(:, :, end));
colorbar;
caxis([0 1]);
subplot(1, 2, 2);
imagesc(labels(:, :, end));
colorbar;
caxis([0 1]);