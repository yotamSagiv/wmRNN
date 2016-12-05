% test the wmRNN with the AX-CPT task

Ninput = 4; % alphabet size
Nhidden = 50; 
Ncontrol = 50; % really, 1 should be sufficient
Noutput = 2;
Ntimes = 3;

samples = 100;
init_scale = 0.5;
num_iterations = 60000;
batch_size = samples/2;
learning_rate = 0.1;
test_size = 1000;

net = RNNmodel(Ninput, Nhidden, Noutput, Ncontrol, learning_rate, init_scale);

inputs = zeros(samples, Ninput, Ntimes);
labels = zeros(samples, Noutput, Ntimes);

probabilities = [0.4 0.1 0.2 0.3];
%probabilities = [0.4 0.1 0.3 0.2];
%probabilities = [0.25 0.25 0.25 0.25];
cumProbabilities = cumsum(probabilities);

for i = 1:samples
    random = rand;
    if(random < cumProbabilities(1))
        inputs(i, 1, 1) = 1;
        inputs(i, 3, 3) = 1;
        labels(i, 1, 3) = 1;
    end
    if(random < cumProbabilities(2) && random > cumProbabilities(1))
        inputs(i, 1, 1) = 1;
        inputs(i, 4, 3) = 1;
        labels(i, 2, 3) = 1;
    end
    if(random < cumProbabilities(3) && random > cumProbabilities(2))
        inputs(i, 2, 1) = 1;
        inputs(i, 3, 3) = 1;
        labels(i, 2, 3) = 1;
    end
    if(random < cumProbabilities(4) && random > cumProbabilities(3))
        inputs(i, 2, 1) = 1;
        inputs(i, 4, 3) = 1;
        labels(i, 1, 3) = 1;
    end

end

% train the network
mse_log = net.trainOnline(inputs, labels, Ntimes, batch_size, num_iterations, 1);
%%
plot(mse_log);


%%

% AX
testSimple(1, 1, 1) = 1;
testSimple(1, 3, 3) = 1;
%labels(1, 1, 3) = 1;
% BX
testSimple(2, 2, 1) = 1;
testSimple(2, 3, 3) = 1;
%labels(2, 2, 3) = 1;
% AY
testSimple(3, 1, 1) = 1;
testSimple(3, 4, 3) = 1;
%labels(3, 2, 3) = 1;
% BY
testSimple(4, 2, 1) = 1;
testSimple(4, 4, 3) = 1;
%labels(4, 1, 3) = 1;

[a_hids_A, a_cons_A, a_outs, z_hids, z_cons, z_outs] = net.predictSetVerbose(testSimple([1 3],:,:), Ntimes);
[a_hids_B, a_cons_B, a_outs, z_hids, z_cons, z_outs] = net.predictSetVerbose(testSimple([2 4],:,:), Ntimes);

context_A_mean = mean(a_cons_A(:, :, end), 1);
hidden_A_mean = mean(a_hids_A(:, :, end), 1);

context_B_mean = mean(a_cons_B(:, :, end), 1);
hidden_B_mean = mean(a_hids_B(:, :, end), 1);

corr(transpose(context_A_mean), transpose(context_B_mean))
corr(transpose(hidden_A_mean), transpose(hidden_B_mean))
%  
% figure(1);
% subplot(1, 2, 1);
% imagesc(outs);
% colorbar;
% caxis([0 1]);
% subplot(1, 2, 2);
% imagesc(labels(:, :, 2));
% colorbar;
% caxis([0 1]);

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