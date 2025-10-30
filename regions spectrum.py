import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.ticker import FuncFormatter
import scipy


def plot_spectrum(TimCoe, dt, mode_indices, sigLength, save_path):
    Fs = 1 / dt
    halflength = sigLength // 2
    f = Fs * np.arange(halflength + 1) / sigLength
    
    # 计算合成频谱 (保持与原始逻辑一致)
    summed_spectrum = np.zeros(halflength + 1)
    for idx in mode_indices:
        alpha = TimCoe[:, idx]
        Y = np.fft.fft(alpha, n=sigLength)
        summed_spectrum += np.abs(Y[:halflength + 1]) / len(alpha) * 2
    
    # 使用与 plot_combined_spectrum 相同的自动坐标轴逻辑
    def auto_axis_limits(x, y):
        y_threshold = np.max(y) * 0.2  # 振幅阈值
        
        # X轴范围计算
        cumulative_energy = np.cumsum(y**2)  # 能量累积
        total_energy = cumulative_energy[-1]
        valid_freqs = x[cumulative_energy <= total_energy*0.95]
        x_max = valid_freqs[-1] * 1.2 if len(valid_freqs) > 0 else x[-1]
        
        # Y轴范围计算
        y_max = np.percentile(y[y >= y_threshold], 99) * 1.5 if np.any(y >= y_threshold) else 1.0
        return (0, min(x_max, 50000)), (0, y_max)  # 遵守Nyquist限制

    xlim, ylim = auto_axis_limits(f, summed_spectrum)
    # 统一可视化样式
    plt.figure(figsize=(12, 6))
    plt.plot(f, summed_spectrum, 
            color='#d62728',  # 使用相同红色
            linewidth=2,
            label='Frequency Spectrum')
    
    # 添加主频标记
    main_freq = f[np.argmax(summed_spectrum)]
    plt.axvline(main_freq, 
               color='gray', 
               linestyle='--',
               label=f'Dominant Freq: {main_freq:.1f} Hz')
    
    # 统一坐标轴设置
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('Spectrum', fontsize=14)
     # 相同网格和图例样式
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    # 统一保存参数
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'spectrum.png'), dpi=200)
    plt.close()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f, 
        y=summed_spectrum,
        mode='lines',
        name='Spectrum',
        line=dict(color='red')
    ))
    fig.update_layout(
        title='Frequency Spectrum',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Amplitude',
        xaxis=dict(range=[0, xlim]),
    )
    fig.write_html(os.path.join(save_path, 'spectrum.html'))

def preprocess(Utx):

    m = Utx.shape[1] 
    N = Utx.shape[0]  
    U0x = np.mean(Utx, axis=0)
    Utx = Utx - U0x  
    U, S, phiU = np.linalg.svd(Utx, full_matrices=False)
    phiU = phiU.T
    An = U @ np.diag(S)  
    Ds = (S ** 2) / N    

    return U0x, An, phiU, Ds,U0x



def plot_coeff(TimCoe, dt, mode_indices, save_path, detrend_window_ratio=0.2):

    summed_coeff = np.sum(TimCoe[:, mode_indices], axis=1)
    t = dt * np.arange(len(summed_coeff)) 
    plt.figure()
    plt.plot(t, summed_coeff, 'purple', label='Summed Modes')
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:.1e}"))
    plt.savefig(os.path.join(save_path, 'coeff_plot.png'))  
    plt.close()

    # 生成去趋势图
    window_size = max(int(len(summed_coeff)*detrend_window_ratio), 11)
    window_size += 1 if window_size%2==0 else 0
    trend = scipy.signal.savgol_filter(summed_coeff, window_size, 3)
    residual = summed_coeff - trend
    
    plt.figure()
    plt.plot(t, residual, 'purple', label='Detrended')
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:.1e}"))
    plt.savefig(os.path.join(save_path, 'detrend_coeff_plot.png'))  
    plt.close()

    plt.figure(figsize=(10,6))
    plt.subplot(211)
    plt.plot(t, summed_coeff, 'b', alpha=0.5, label='Original')
    plt.plot(t, trend, 'r', label='Trend')
    plt.legend()
    
    plt.subplot(212)
    plt.plot(t, residual, 'g', label='Detrended')
    plt.legend()
    
    plt.savefig(os.path.join(save_path, 'coeff_with_detrend.png')) 
    plt.close()


def spectrum_main(file_directory, output_directory, n_snapshots, dt,sigLength):
    """
    读取图像文件,进行振动图和频谱图的绘制。
    
    参数:
    - file_directory: 输入图像文件的目录
    - output_directory: 输出结果保存的目录
    - n_snapshots: 读取的快照数量
    - dt: 快照时间间隔
    - sigLength: 傅里叶变换时的信号长度(大于快照张数,同时取2的次幂)
    """

    # 读取数据 
    UV = []
    filenames = sorted([f for f in os.listdir(file_directory) if f.endswith(('.jpeg', '.jpg', '.png', '.bmp', '.tiff'))]) 
    for j, filename in enumerate(filenames[:n_snapshots]):
        filepath = os.path.join(file_directory, filename)
        Data = imread(filepath)  
        print(f'正在读取文件: {filepath}')  
        if Data.ndim == 3:  # 如果是彩色图像
            GDATA = rgb2gray(Data)
        else:  # 如果是灰度图像
            GDATA = Data
        data = GDATA.astype(np.double)
        UV.append(data.flatten())  
    UV = np.array(UV)

    summed_mode_indices=[1]
    summed_coeff_mode_indices = [1,2,3]
    U0x, An, phiU, Ds,U0x = preprocess(UV)
    zero_based = [idx-1 for idx in summed_coeff_mode_indices]
    valid_indices = [idx for idx in zero_based if 0 <= idx < An.shape[1]]
    output_dir = os.path.join(output_directory, 'regions spectrum')
    os.makedirs(output_dir, exist_ok=True)
    plot_coeff(An, dt, valid_indices, output_dir)
    zero_based = [idx - 1 for idx in summed_mode_indices]
    valid_indices = [idx for idx in zero_based if 0 <= idx < 20]
    plot_spectrum(An, dt, valid_indices, sigLength, output_dir)
    print(f'频谱生成完成,结果保存在文件夹 {output_dir}')

# 示例调用
file_directory = save_path=r'D:\CODE\shock and leaf detections\result\shockwave_extract'  #输入的是叶片或者激波的二值化提取结果（extract文件夹）
output_directory = r"D:\CODE\shock and leaf detections\SHOCKRESULT"
spectrum_main(file_directory, output_directory, n_snapshots=20, dt=1/30,sigLength=1024)
