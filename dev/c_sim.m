%% simulate the chunking task

% training properties
Ntimes = 3;
learning_rate = 0.5;
num_examples = 100;
batch_size = 50;
num_iterations = 25000;
init_scale = 0.5;

% simulation properties
NObjects = 4;
NPositions = 4;
NLoadObjects = 2;
NLoadPositions = 2;
NWorkingMemoryItems = (NObjects + NLoadObjects) * (NPositions + NLoadPositions);

% network properties
Ninput = NWorkingMemoryItems + NObjects + NLoadObjects;
Nhidden = 100;
Ncontrol = 100;
Noutput = NPositions + NLoadPositions;

% create the training data
inputs = zeros(num_examples, Ninput, Ntimes);
labels = zeros(num_examples, Noutput, Ntimes);

for i = 1:num_examples
    object_indices = 1:NPositions;
    % pick the location of A
    A_index = round(rand * (NPositions - 1)) + 1;
    
    % with 80% probability, B is after A
    if rand < 0.8
        B_index = mod(A_index + 1, NPositions);
        if B_index == 0
            B_index = NPositions;
        end
    else % otherwise, it is before A
        B_index = mod(A_index - 1, NPositions);
        if B_index == 0
            B_index = NPositions;
        end
    end
    
    % pick the locations of the remaining distractor objects
    distractor_indices = object_indices;
    distractor_indices([find(object_indices == A_index), find(object_indices == B_index)]) = [];
    distractor_indices = distractor_indices(randperm(length(distractor_indices))); % randomly permute the distractors
            
    object_indices = [A_index, B_index, distractor_indices];
    
    % pick the locations of the memory load objects
    load_object_indices = (NPositions + 1):(NPositions + NLoadPositions);
    load_object_indices = load_object_indices(randperm(length(load_object_indices)));
    for j = 1:length(load_object_indices)
        if rand < 0.5
            load_object_indices(j) = 0;
        end
    end

    object_indices = [object_indices, load_object_indices];
    
    % convert object indices to task layer representation
    for j = 1:length(object_indices)
        if object_indices(j) == 0
            continue
        end

        index = ((j - 1) * (NPositions + NLoadPositions)) + object_indices(j);
        inputs(i, index, 1) = 1;   
    end
    
    % pick an object to cue
    object_set = 1:(NObjects + NLoadObjects);
    object_set(find(object_indices == 0)) = []; % make sure that we can't cue null objects
    object_set(3:(NObjects - NLoadObjects)) = []; % make sure that we don't cue the distractors

    inputs(i, NWorkingMemoryItems + object_set(round(rand * (length(object_set) - 1)) + 1), 3) = 1;
    
    % set the corresponding label
    labels(i, object_indices(find(inputs(i, (NWorkingMemoryItems + 1):end, 2))), 3) = 1;
end

%% train
net = RNNmodel(Ninput, Nhidden, Noutput, Ncontrol, learning_rate, init_scale);
net.trainOnline(inputs, labels, Ntimes, batch_size, num_iterations, 1);
%% check performance on test data
test_set_size = 1000;

% create the test data
test_inputs = zeros(num_examples, Ninput, Ntimes);
test_labels = zeros(num_examples, Noutput, Ntimes);

for i = 1:test_set_size
    object_indices = 1:NPositions;
    % pick the location of A
    A_index = round(rand * (NPositions - 1)) + 1;
    
    % with 80% probability, B is after A
    if rand < 0.8
        B_index = mod(A_index + 1, NPositions);
        if B_index == 0
            B_index = NPositions;
        end
    else % otherwise, it is before A
        B_index = mod(A_index - 1, NPositions);
        if B_index == 0
            B_index = NPositions;
        end
    end
    
    % pick the locations of the remaining distractor objects
    distractor_indices = object_indices;
    distractor_indices([find(object_indices == A_index), find(object_indices == B_index)]) = [];
    distractor_indices = distractor_indices(randperm(length(distractor_indices))); % randomly permute the distractors
            
    object_indices = [A_index, B_index, distractor_indices];
    
    % pick the locations of the memory load objects
    load_object_indices = (NPositions + 1):(NPositions + NLoadPositions);
    load_object_indices = load_object_indices(randperm(length(load_object_indices)));
    for j = 1:length(load_object_indices)
        if rand < 0.5
            load_object_indices(j) = 0;
        end
    end

    object_indices = [object_indices, load_object_indices];
    
    % convert object indices to task layer representation
    for j = 1:length(object_indices)
        if object_indices(j) == 0
            continue
        end

        index = ((j - 1) * (NPositions + NLoadPositions)) + object_indices(j);
        test_inputs(i, index, 1) = 1;   
    end
    
    % pick an object to cue
    object_set = 1:(NObjects + NLoadObjects);
    object_set(find(object_indices == 0)) = []; % make sure that we can't cue null objects
    object_set(3:(NObjects - NLoadObjects)) = []; % make sure that we don't cue the distractors

    test_inputs(i, NWorkingMemoryItems + object_set(round(rand * (length(object_set) - 1)) + 1), 3) = 1;
    
    
    % set the corresponding label
    test_labels(i, object_indices(find(test_inputs(i, (NWorkingMemoryItems + 1):end, 2))), 3) = 1;
end

disp(net.costSet(test_inputs, Ntimes, test_labels));