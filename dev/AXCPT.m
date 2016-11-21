% test the wmRNN with the AX-CPT task

Ninput = 5; % alphabet size
Nhidden = 100; 
Ncontrol = 10; % really, 1 should be sufficient
Noutput = 2;
Ntimes = 2;

init_scale = 0.1;
num_iterations = 200;
batch_size = 10;
num_examples = 300;
learning_rate = 0.3;
test_size = 1000;

net = RNNmodel(Ninput, Nhidden, Noutput, Ncontrol, learning_rate, init_scale);

% create training data
inputs = zeros(num_examples, Ninput, Ntimes);
labels = zeros(num_examples, Noutput, Ntimes);
for i = 1:num_examples
    % create input
    index1 = randi([1, Ninput]); % first stimulus
    index2 = randi([1, Ninput]); % second stimulus
    inputs(i, index1, 1) = 1;
    inputs(i, index2, 2) = 1;
    
    % create label
    labels(i, 1, 1) = 1; % should always point left for first stimulus
    if index1 == 1 && index2 == 2
        labels(i, 2, 2) = 1; % point right if we have stimulus 2 preceded by stimulus 1
    else
        labels(i, 1, 2) = 1; % else point left again
    end
end

% train the network
net.trainOnline(inputs, labels, Ntimes, batch_size, num_iterations);

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

predictions = net.runSet(test_inputs, Ntimes)