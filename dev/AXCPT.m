% test the wmRNN with the AX-CPT task

Ninput = 4; % alphabet size
Nhidden = 10; 
Ncontrol = 10; % really, 1 should be sufficient
Noutput = 2;
Ntimes = 2;

init_scale = 0.5;
num_iterations = 12800;
batch_size = 4;
num_examples = 4;
learning_rate = 0.5;
test_size = 1000;

net = RNNmodel(Ninput, Nhidden, Noutput, Ncontrol, learning_rate, init_scale);
RNNnet.weights.w_CO = zeros(Noutput, Ncontrol);

%{
% create training data
inputs = zeros(num_examples, Ninput, Ntimes);
labels = zeros(num_examples, Noutput, Ntimes);
for i = 1:num_examples
    % create input
    index1 = randi([1, 2]); % first stimulus
    index2 = randi([3, 4]); % second stimulus
    inputs(i, index1, 1) = 1;
    inputs(i, index2, 2) = 1;
    
    % create label
    if index1 == 1 
        if index2 == 3
            labels(i, 1, 2) = 1; % point right if we have stimulus 2 preceded by stimulus 1
        else
            labels(i, 2, 2) = 1;
        end
    else
        if index2 == 3
            labels(i, 2, 2) = 1; % point right if we have stimulus 2 preceded by stimulus 1
        else
            labels(i, 1, 2) = 1;
        end
    end
end
%}

inputs = zeros(4, Ninput, Ntimes);
labels = zeros(4, Noutput, Ntimes);
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

% train the network
mse_log = net.trainOnline(inputs, labels, Ntimes, batch_size, num_iterations);
plot(mse_log);
%%
outs = net.predictSet(inputs, Ntimes);
figure(1);
subplot(1, 2, 1);
imagesc(outs);
colorbar;
caxis([0 1]);
subplot(1, 2, 2);
imagesc(labels(:, :, 2));
colorbar;
caxis([0 1]);

%{
% create test data
test_inputs = zeros(num_examples, Ninput, Ntimes);
test_labels = zeros(num_examples, Noutput, Ntimes);
for i = 1:test_size
    % create input
    index1 = randi([1, Ninput]); % first stimulus
    index2 = randi([1, Ninput]); % second stimulus
    test_inputs(i, index1, 1) = 1;
    test_inputs(i, index2, 2) = 1;
    
    % create label
    test_labels(i, 1, 1) = 1; % should always point left for first stimulus
    if index1 == 1 && index2 == 2
        test_labels(i, 2, 2) = 1; % point right if we have stimulus 2 preceded by stimulus 1
    else
        test_labels(i, 1, 2) = 1; % else point left again
    end
end

predictions = net.runSet(test_inputs, Ntimes);
%}