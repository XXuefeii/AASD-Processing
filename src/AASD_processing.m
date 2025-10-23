clear
clc
eeglab;

%% 2. Create each file individually and save them as a .set file
S_i = '1';

save_dir = 'D:\Research\Data\Mathlab Working Path';

if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

EEG = pop_loadcnt('D:\xxxx\S1.cnt', 'dataformat', 'auto');

EEG.data(end-3:end, :) = [];
EEG.nbchan = size(EEG.data, 1);

EEG.chanlocs(end-3:end) = [];

valid_triggers = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]; % 仅选择音频开始的标签
sample_rate = EEG.srate;
duration = 60; 

for j = 1:length(EEG.event)
    trigger = EEG.event(j).type;
    if ismember(trigger, valid_triggers)
        start_idx = EEG.event(j).latency;
        end_idx = min(start_idx + duration * sample_rate, size(EEG.data, 2));
        
        EEG_segment = EEG;
        EEG_segment.data = EEG.data(:, start_idx:end_idx);
        EEG_segment.pnts = size(EEG_segment.data, 2);
        EEG_segment.times = EEG.times(start_idx:end_idx);
        EEG_segment.chanlocs = EEG.chanlocs;
        
        epoch_events = EEG.event([EEG.event.latency] >= start_idx & [EEG.event.latency] <= end_idx);
        for k = 1:length(epoch_events)
            epoch_events(k).latency = epoch_events(k).latency - start_idx + 1;
        end
        EEG_segment.event = epoch_events;
        
        filename = sprintf('S%s_trial%03d_t%d.set', S_i, j, trigger);
        pop_saveset(EEG_segment, 'filename', filename, 'filepath', save_dir);
        
        mat_filename = sprintf('%sS%s_trial%03d_t%d.mat', save_dir, S_i, j, trigger);
        save(mat_filename, 'EEG_segment');
    end
end

%% 3. merge all the data together and filter 
sets=dir('D:\xxxxx\*trial*set');

for i =1:length(sets)
  sets(i).name
    i
    EEG = pop_loadset(sets(i).name);
    if i==1  
        merged = EEG ; 
    else
        merged = pop_mergeset(merged,EEG) ;
    end   
end
%filter the merged data
EEG=pop_eegfiltnew(merged, 0.1, 45)  
EEG = pop_saveset( EEG,'filename','merged_BP' );

%% 4. downsampling
EEG = pop_resample(EEG, 128);
EEG = pop_saveset( EEG,'filename','merged_BP_downsampling' );

%% 5. rereference the data to the average of the mastoid channels
EEG=pop_reref(EEG,{ 'M1' 'M2' })
EEG = pop_saveset( EEG,'filename','merged_BP_downsampling_chremoved_reref' );

%% 6.epoch the data
EEG=pop_epoch(EEG)
EEG = pop_saveset( EEG,'filename','merged_BP_downsampling_chremoved_reref_epoched' );

%% 7. run ICA
%EEG = pop_runica(EEG, 'extended',1);
%EEG = pop_saveset( EEG,'filename','merged_BP_downsampling_chremoved_reref_epoched_ICA' );

%% 
EEG = pop_saveset( EEG,'filename',['subject' S_i] );