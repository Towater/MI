# -- coding: utf-8 --
# @Time : 2023/9/6 15:24
# @Author : cyy
# @File : MIUtools.py
import mne
import numpy as np
from scipy import signal
import scipy
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
from scipy.signal import spectrogram, istft

#相比new_read_data_mat修复了不能读A04的问题
def all_read_data_mat(path):
    dataset = scipy.io.loadmat(path)
    trial_t = dataset['data'][0][3][0][0][1]  # 时间是一样的，取一次就行
    start_time = []
    for temp in trial_t:
        start_time.append(temp[0])
    event = 6 * 48
    channel = 22
    timepoint = 1000
    start_time = np.array(start_time)  # 用时间分割太麻烦了，采样率250，每次8秒
    total_data = np.zeros((event, channel, timepoint))
    total_label = []
    basic = timepoint  # 时间基点
    index = 0
    struct_num = len(dataset["data"][0])
    offset = 500 #偏移两秒后开始
    for i in range(0, struct_num):
        raw = dataset['data'][0][i][0][0]  # 从第四个struct开始，一共9个
        # 0-记录数据，1-开始时间，2-事件，3-采样率
        if len(raw[2]!=0):
            labels = raw[2]
            data = raw[0]
            for temp_labels in labels:
                total_label.append(temp_labels[0])
            for temp in start_time:
                total_data[index, :, :] = np.transpose(data[temp+offset :temp+ offset + basic, 0:22])
                index += 1

    #total_data = signal_filter_butter(total_data, 250, 7, 35)
    total_label = np.array(total_label)
    return total_data, total_label

# 提取BCI Competition IV 2a 的数据(mat版)
# 默认提取四类,二类(L,R)请用下面的for2
def new_read_data_mat(path):
    dataset = scipy.io.loadmat(path)
    timepoint = dataset['data'][0][3][0][0][1]  # 时间是一样的，取一次就行
    start_time = []
    for temp in timepoint:
        start_time.append(temp[0])
    event = 6 * 48
    channel = 22
    timepoint = 1000
    start_time = np.array(start_time)  # 用时间分割太麻烦了，采样率250，每次8秒
    total_data = np.zeros((event, channel, timepoint))
    total_label = []
    basic = timepoint  # 时间基点
    index = 0
    for i in range(3, 9):
        raw = dataset['data'][0][i][0][0]  # 从第四个struct开始，一共9个
        # 0-记录数据，1-开始时间，2-事件，3-采样率
        labels = raw[2]
        data = raw[0]
        for temp_labels in labels:
            total_label.append(temp_labels[0])
        for temp in start_time:
            total_data[index, :, :] = np.transpose(data[temp + 500:temp + 500 + basic, 0:22])
            index += 1

    total_label = np.array(total_label)
    X = len(total_data)  # 没什么意义，方便下面
    # total_data = total_data.T
    # total_data = total_data.reshape(X // basic, 22, basic)

    # for j in range(22):
    #     scaler = StandardScaler()
    #     scaler.fit(total_data[:, j, :])
    #     total_data[:, j, :] = scaler.transform(total_data[:, j, :])

    # shuffle_num = np.random.permutation(len(total_data))
    # total_data = total_data[shuffle_num, :, :]
    # total_label = total_label[shuffle_num]
    total_data = signal_filter_butter(total_data, 250, 7, 35)
    return total_data, total_label


# 相比原版增加数据扩充
def new_read_data_mat_sr(path):
    dataset = scipy.io.loadmat(path)
    timepoint = dataset['data'][0][3][0][0][1]  # 时间是一样的，取一次就行
    start_time = []
    for temp in timepoint:
        start_time.append(temp[0])
    event = 6 * 48
    channel = 22
    timepoint = 1000
    start_time = np.array(start_time)  # 用时间分割太麻烦了，采样率250，每次8秒
    total_data = np.zeros((event, channel, timepoint))
    total_label = []
    basic = timepoint  # 时间基点
    index = 0
    for i in range(3, 9):
        raw = dataset['data'][0][i][0][0]  # 从第四个struct开始，一共9个
        # 0-记录数据，1-开始时间，2-事件，3-采样率
        labels = raw[2]
        data = raw[0]
        for temp_labels in labels:
            total_label.append(temp_labels[0])
        for temp in start_time:
            total_data[index, :, :] = np.transpose(data[temp + 500:temp + 500 + basic, 0:22])
            index += 1

    total_label = np.array(total_label)
    X = len(total_data)  # 没什么意义，方便下面
    # total_data = total_data.T
    # total_data = total_data.reshape(X // basic, 22, basic)

    # for j in range(22):
    #     scaler = StandardScaler()
    #     scaler.fit(total_data[:, j, :])
    #     total_data[:, j, :] = scaler.transform(total_data[:, j, :])



    aug_data, aug_label = interaug(X, total_data, total_label)
    total_data = np.concatenate((total_data, aug_data), axis=0)
    total_label = np.concatenate((total_label, aug_label))
    shuffle_num = np.random.permutation(len(total_data))
    total_data = total_data[shuffle_num, :, :]
    total_label = total_label[shuffle_num]
    total_data = signal_filter_butter(total_data, 250, 7, 35)
    return total_data, total_label


def new_read_data_mat_for2(path):
    total_data = []
    data, labels = new_read_data_mat(path)
    # 1-左手，2-右手，3-脚，4-舌头
    left_index = np.where(labels == 1)
    right_index = np.where(labels == 2)
    total_hand_index = np.concatenate((left_index, right_index))
    for i in left_index:
        left_data = data[i]

    for k in right_index:
        right_data = data[k]

    total_data = [left_data, right_data]
    total_data = np.concatenate(total_data)
    print(total_data.shape)
    labels_left = np.ones(len(left_index[0]))
    labels_right = np.ones(len(right_index[0])) + 1
    total_labels = np.concatenate((labels_left, labels_right))
    # -------------新增洗牌-------------
    shuffle_num = np.random.permutation(len(total_data))
    total_data = total_data[shuffle_num, :, :]
    total_labels = total_labels[shuffle_num]
    return total_data, total_labels


def new_read_data_mat_for_same_labels(path):
    dataset = scipy.io.loadmat(path)
    timepoint = dataset['data'][0][3][0][0][1]  # 时间是一样的，取一次就行
    start_time = []
    for temp in timepoint:
        start_time.append(temp[0])
    event = 6 * 48
    channel = 22
    timepoint = 1000
    start_time = np.array(start_time)  # 用时间分割太麻烦了，采样率250，每次8秒
    total_data = np.zeros((event, channel, timepoint))
    total_label = []
    basic = timepoint  # 时间基点
    index = 0
    for i in range(3, 9):
        raw = dataset['data'][0][i][0][0]  # 从第四个struct开始，一共9个
        # 0-记录数据，1-开始时间，2-事件，3-采样率
        labels = raw[2]
        data = raw[0]
        for temp_labels in labels:
            total_label.append(temp_labels[0])
        for temp in start_time:
            total_data[index, :, :] = np.transpose(data[temp + 500:temp + 500 + basic, 0:22])
            index += 1

    total_label = np.array(total_label)

    same_datas = np.zeros((4, event // 4, channel, timepoint))
    same_labels = []
    a = b = c = d = 0
    for i,v in enumerate(total_label):
        if v == 1:
            same_datas[0, a, :, :] = total_data[i]
            a += 1
        if v == 2:
            same_datas[1, b, :, :] = total_data[i]
            b += 1
        if v == 3:
            same_datas[2, c, :, :] = total_data[i]
            c += 1
        if v == 4:
            same_datas[3, d, :, :] = total_data[i]
            d += 1

    num_artificial_trials = 36
    temp_datas,temp_labels = [],[]
    for temp in range(len(same_datas)):
        temp_datas.append(generate_artificial_trials_stft(same_datas[temp],num_artificial_trials=num_artificial_trials))
        temp_labels.append(np.full(num_artificial_trials,temp+1))
    same_datas = np.concatenate(temp_datas)
    same_labels = np.concatenate(temp_labels)

    return same_datas, same_labels

def generate_artificial_trials_stft(trials, sample_rate=250, window_length=0.25, num_artificial_trials=20):
    # STFT 参数
    window_size = int(window_length * sample_rate)  # 250ms
    hop_size = window_size // 2  # 50% overlap

    # 随机重组时间频率表示
    def random_recombination(trials):
        # 存储时间频率表示的列表
        time_freq_representations = []

        # 对每个 band-pass filtered trial 进行 STFT
        for trial in trials:
            time_freq = []
            for channel in trial:
                # Make length of channel multiple of window_size by zero-padding
                if len(channel) % window_size != 0:
                    pad_size = window_size - (len(channel) % window_size)
                    channel = np.pad(channel, (0, pad_size), 'constant', constant_values=0)

                _, _, tf = spectrogram(channel, fs=sample_rate, window='hamming', nperseg=window_size, noverlap=hop_size)
                time_freq.append(tf)
            time_freq_representations.append(time_freq)

        num_windows = len(time_freq_representations[0])  # 假设每个 trial 的时间窗口数相同
        num_trials = len(time_freq_representations)

        recombined_time_freq = []
        for window_index in range(num_windows):
            recombined_window = []
            for trial_index in range(num_trials):
                recombined_window.append(time_freq_representations[trial_index][window_index])

            np.random.shuffle(recombined_window)
            recombined_time_freq.append(np.concatenate(recombined_window, axis=0))

        return recombined_time_freq

    # 逆STFT参数
    inverse_window_size = window_size
    inverse_hop_size = hop_size

    # 逆STFT得到最终的人工试验
    def generate_artificial_trial(recombined_time_freq):
        artificial_trial = []
        for channel in recombined_time_freq:
            _, artificial_channel = istft(channel, fs=sample_rate, window='hamming', nperseg=inverse_window_size, noverlap=inverse_hop_size)

            # Pad zeros to match the length with 1000
            artificial_channel = np.pad(artificial_channel, (0, 1000 - len(artificial_channel)), 'constant')

            artificial_trial.append(artificial_channel)
        return np.array(artificial_trial)

    # 重复上述过程生成多个人工试验
    artificial_trials = []
    for _ in range(num_artificial_trials):
        recombined_time_freq = random_recombination(trials)
        artificial_trial = generate_artificial_trial(recombined_time_freq)
        artificial_trials.append(artificial_trial)

    return artificial_trials

#使用短时傅里叶变换完成数据增强
def new_read_data_mat_sr_stft(path):
    same_datas, same_labels=new_read_data_mat_for_same_labels(path)
    total_data,total_labels=new_read_data_mat(path)
    total_data = np.concatenate((total_data,same_datas))
    total_labels=np.concatenate((total_labels,same_labels))
    total_data = signal_filter_butter(total_data, 250, 7, 35)
    return total_data,total_labels


# --------------------------------------------------------------------------

ch_names = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3',
            'CP1', 'CPz', 'CP2', 'CP4', 'CP6']


def signal_filter_butter(data, frequency=256, lowpass=0.5, highpass=45):
    [b, a] = signal.butter(6, [lowpass / frequency * 2, highpass / frequency * 2], 'bandpass')
    Signal_pro = signal.filtfilt(b, a, data)
    return Signal_pro


def interaug(batch_size, timg, label):
    aug_data = []
    aug_label = []
    for cls4aug in range(4):
        cls_idx = np.where(label == cls4aug + 1)
        tmp_data = timg[cls_idx]
        tmp_label = label[cls_idx]
        tmp_aug_data = np.zeros((int(batch_size / 4), 22, 1000))
        print(tmp_aug_data.shape)
        for ri in range(int(batch_size / 4)):  # b = 72
            for rj in range(8):
                rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                tmp_aug_data[ri, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, rj * 125:(rj + 1) * 125]

        aug_data.append(tmp_aug_data)
        aug_label.append(tmp_label[:int(batch_size / 4)])
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]
    print("data", aug_data.shape)
    return aug_data, aug_label


# 提取我们自制的mat格式文件
# 不分左右手转置为[实验次数*通道*时间点]
def ParseDataTrain(mat_path):
    try:
        RawData = sio.loadmat(mat_path)
    except BaseException:
        print("出错！")

    fs = RawData['nfo'][0][0][0].item()
    x = RawData['data_received']
    # matlab从1开始，python从0
    channels = (RawData['nfo'][0][0][1][0]) - 1
    cnt = x[:, channels]

    hand_event = [0, 1]  # 0右手 1左手
    global __event_data
    __event_data = RawData['mark'][1]  # 获取左右手数据
    event_data = __event_data
    Right_Data = []
    Left_Data = []
    basic = 0
    time = [0, 4000]  # 默认取时间0-4秒的数据,最大4秒
    startp = time[0] / 1000 * fs;
    endp = time[1] / 1000 * fs
    timepoint = endp - startp
    Result_Right_Data = []
    Result_Left_Data = []
    Total_hand_data = []
    Result_hand_Data = []

    for i in range(0, len(event_data)):
        Total_hand_data = cnt[basic:(i + 1) * 1024, :]
        Total_hand_data = Total_hand_data[int(startp):int(endp), :]
        Result_hand_Data.append(Total_hand_data)
        basic = ((i + 1) * 1024)
    # Result_hand_Data = np.concatenate(Result_hand_Data)
    Total_hand_data = np.zeros((len(event_data), 21, int(timepoint)))

    for j in range(0, len(event_data)):
        Total_hand_data[j, :, :] = np.transpose(Result_hand_Data[j])

    # for j in range(21):
    #     scaler = StandardScaler()
    #     scaler.fit(Total_hand_data[:, j, :])
    #     Total_hand_data[:, j, :] = scaler.transform(Total_hand_data[:, j, :])

    # standardize
    # target_mean = np.mean(Total_hand_data)
    # target_std = np.std(Total_hand_data)
    # Total_hand_data = (Total_hand_data - target_mean) / target_std

    info = myinfo(total_hand_data=Total_hand_data, labels=event_data)  # 返回左右手数据以及标签
    return info


class myinfo:
    def __init__(self, left_hand_data='null', right_hand_data='null', total_hand_data='null', labels='null'):
        self.total_hand_data = total_hand_data
        self.right_hand_data = right_hand_data
        self.left_hand_data = left_hand_data
        self.labels = labels
