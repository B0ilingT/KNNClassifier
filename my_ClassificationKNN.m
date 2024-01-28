classdef my_ClassificationKNN < handle
    %This class takes in training data and labels and uses them to create a
    %KNN object capable of generating predictions on inputted Test
    %Data
    properties
             
        X % training examples
        Y % training labels
        NumNeighbors % number of nearest neighbours to consider       
        Verbose % are we printing out debug as we go?
        ClassNames % each of the class labels in our problem
        ProbScore%stores the probability scores
        TestExamples%stores test examples
    end
    
    methods
        
        % constructor: implementing the fitting phase        
        function obj = my_ClassificationKNN(X, Y, NumNeighbors, Verbose)
            
            % set up our training data:
            obj.X = X;
            obj.Y = Y;
            % store the number of nearest neighbours we're using:
            obj.NumNeighbors = NumNeighbors;
            % are we printing out debug as we go?:
            obj.Verbose = Verbose;
            obj.ClassNames = unique(obj.Y);
            obj.ProbScore = [];          
        end        
        % the prediction phase:
        function [predictions, prob_scores] = predict(obj, test_examples)
            obj.TestExamples = test_examples;
            unique_labels = unique(obj.Y);            
            if(obj.Verbose == true)
                % Initialize output array
                predictions = categorical;
                % Calculate number of test examples and number of features
                num_test = size(test_examples, 1);
                prob_scores = zeros(num_test, length(unique(obj.Y)));
                % Calculate sum of squares for each row of test data
                nA = sum(test_examples.^2, 2);
                % Calculate sum of squares for each row of training data
                nB = sum(obj.X.^2, 2);
                % Initialize distance matrix
                D = zeros(num_test, size(obj.X, 1));
                % Calculate distance between each test example and each training example
                % Sort distances in ascending order
                for i = 1:num_test
                    for j = 1:size(obj.X, 1)
                        D(i, j) = sqrt(nA(i) + nB(j) - 2*test_examples(i, :)*obj.X(j, :)');
                    end
                    % Sort distances in ascending order
                    [~, ind] = sort(D(i,:));
                    % Get indices of the closest num_neighbors distances
                    ind_closest = ind(1:obj.NumNeighbors);
                    % Get labels of the closest num_neighbors points
                    labels_closest = obj.Y(ind_closest);
                    % Iterate over each unique label
                    for j = 1:length(unique_labels)
                        % Count the number of occurrences of the current label among the closest num_neighbors points
                        count = sum(labels_closest == unique_labels(j));
                        % Calculate the probability score as the count divided by the total number of closest num_neighbors points
                        prob_scores(i,j) = count/obj.NumNeighbors;
                    end
                    [~, idx_max] = max(prob_scores(i,:));
                    predictions(i,:) = unique_labels(idx_max);
                end
                obj.ProbScore = prob_scores;
            end
            if(obj.Verbose == false)
                % Initialize output array
                predictions = categorical;
                % Calculate number of test examples and number of features
                num_test = size(test_examples, 1);
                prob_scores = zeros(num_test, length(unique(obj.Y)));
                % Calculate sum of squares for each row of test data
                nA = sum(test_examples.^2, 2);
                % Calculate sum of squares for each row of training data
                nB = sum(obj.X.^2, 2);
                % Initialize distance matrix
                D = zeros(num_test, size(obj.X, 1));
                % Calculate distance between each test example and each training example
                % Sort distances in ascending order
                for i = 1:num_test
                    for j = 1:size(obj.X, 1)
                        D(i, j) = sqrt(nA(i) + nB(j) - 2*test_examples(i, :)*obj.X(j, :)');
                    end
                    % Sort distances in ascending order
                    [~, ind] = sort(D(i,:));
                    % Get indices of the closest num_neighbors distances
                    ind_closest = ind(1:obj.NumNeighbors);
                    % Get labels of the closest num_neighbors points
                    labels_closest = obj.Y(ind_closest);
                    % Iterate over each unique label
                    for j = 1:length(unique_labels)
                        % Count the number of occurrences of the current label among the closest num_neighbors points
                        count = sum(labels_closest == unique_labels(j));
                        % Calculate the probability score as the count divided by the total number of closest num_neighbors points
                        prob_scores(i,j) = count/obj.NumNeighbors;
                    end
                    [~, idx_max] = max(prob_scores(i,:));
                    predictions(i,:) = unique_labels(idx_max);
                end
                obj.ProbScore = prob_scores;
            end
        end    
    end
    
end
