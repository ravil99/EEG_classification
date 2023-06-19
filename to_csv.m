counter = 1;

for subject_num = 1:20

%subject_num = 7;

%%

matrixOfIneterest = cell2mat(EEG_data{1, subject_num}.Ambiguity);
a15 = find(matrixOfIneterest == 0.1500);
a25 = find(matrixOfIneterest == 0.2500);
a40 = find(matrixOfIneterest == 0.4000);
a45 = find(matrixOfIneterest == 0.4500);
a55 = find(matrixOfIneterest == 0.5500);
a60 = find(matrixOfIneterest == 0.6000);
a75 = find(matrixOfIneterest == 0.7500);
a85 = find(matrixOfIneterest == 0.8500);

%%

ambiguity_array = a85;

%%

for k = 1:length(ambiguity_array)

    ambiguity_index = ambiguity_array(k);
    % новое
    for i = 1:31
    fs = 250;
    data = EEG_data{1, subject_num}.trial{1, ambiguity_index};
    %theta_filtered = zeros(size(data));
    %theta_filtered(i,:) = bandpass(data(i,:), [4 8], fs);
    %alpha_filtered = zeros(size(data));
    %alpha_filtered(i,:) = bandpass(data(i,:), [8 14], fs);
    beta_filtered = zeros(size(data));
    beta_filtered(i,:) = bandpass(data(i,:), [14 30], fs);
    end
    file = beta_filtered;
    % конец нового
    %file = EEG_data{1, subject_num}.trial(1, ambiguity_index);
    csvwrite(sprintf('a85_%d.csv', counter), file);
    counter = counter + 1;

end

end