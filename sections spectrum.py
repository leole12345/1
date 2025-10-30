# 运行本代码默认先把轮廓提取的运行一遍
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt, hilbert,detrend
from scipy.signal import find_peaks
from scipy.signal.windows import hann
from scipy.interpolate import interp1d
import plotly.graph_objects as go

def load_images(folder_path):
    """加载文件夹内所有图像（灰度模式）"""
    img_list = []
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    files = sorted(os.listdir(folder_path))
    for f in files:
        if f.lower().endswith(valid_exts):
            img_path = os.path.join(folder_path, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_list.append(img)
            else:
                print(f"Warning: Failed to load {img_path}")
    return img_list


def create_time_surface(data):
    """将三维数据转换为时间-空间矩阵"""
    if data.ndim != 3:
        raise ValueError(f"数据维度错误，需要3D输入，实际得到 {data.shape}")
    return data.reshape(data.shape[0], -1)  # 形状 (Time, 50)

def preprocess_data(time_surface, fps=100):
    """数据预处理：降噪、带通滤波、归一化"""
    # 滑动平均降噪
    smoothed = savgol_filter(time_surface, window_length=5, polyorder=3, axis=0)
    
    # 调整带通范围至10-45Hz (远离Nyquist边界)
    low, high = 3/(0.5*fps), 50/(0.5*fps)
    b, a = butter(N=3, Wn=[low, high], btype='band')
    filtered = filtfilt(b, a, smoothed, axis=0) 
    # 归一化
    return (filtered - np.mean(filtered)) / np.std(filtered)


def extract_brightness_columns(img_list, x1=30, roi=(0, 470)):
    """
    提取指定X位置的亮度数据，并记录ROI起始Y坐标
    返回值: (亮度数据数组, roi起始Y)
    """
    columns_data = []
    for img in img_list:
        col_data = img[roi[0]:roi[1], x1].reshape(-1, 1)
        columns_data.append(col_data)
    return np.array(columns_data), roi[0]  

def get_amplitude_envelope(signal):
    """通过Hilbert变换获取振幅包络"""
    analytic = hilbert(signal)
    return np.abs(analytic)

def plot_vibration_curves(curves, fps=100):
    """绘制振动曲线及其包络"""
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(len(next(iter(curves.values())))) / fps
    
    for label, signal in curves.items():
        envelope = get_amplitude_envelope(signal)
        plt.plot(time_axis, envelope, label=f"{label} Envelope", lw=2)
        plt.plot(time_axis, signal, alpha=0.5, label=f"{label} Raw")
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Normalized Amplitude', fontsize=12)
    plt.title('Vibration Characteristics', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('vibration_curves.png', dpi=200)
    plt.show()

def detect_wave_crest(time_surface, height_threshold=0.5, min_distance=5):
    """
    检测每个时间步的波峰位置
    参数:
        time_surface: 时空云图 (Time, Y)
        height_threshold: 波峰最小高度（相对于归一化亮度）
        min_distance: 波峰间最小间距（像素）
    返回:
        crest_positions: 各时间步的波峰Y位置列表（可能存在缺失）
    """
    crest_positions = []
    for t in range(time_surface.shape[0]):
        profile = time_surface[t, :]
        peaks, _ = find_peaks(profile, height=height_threshold, distance=min_distance)
        if len(peaks) > 0:
            main_peak = peaks[np.argmax(profile[peaks])]  # 选择最显著的波峰
            crest_positions.append(main_peak)
        else:
            crest_positions.append(np.nan)  # 标记缺失值
    return np.array(crest_positions)



def plot_overlay(time_surface, curves, roi_start_y, fps=100):
    """在云图上叠加振动曲线"""
    plt.figure(figsize=(15, 8))
    
    # 绘制云图
    plt.imshow(time_surface.T, cmap='viridis', aspect='auto',
               extent=[0, time_surface.shape[0]/fps, 
                       roi_start_y, roi_start_y + time_surface.shape[1]],
               origin='lower')
    plt.colorbar(label='Normalized Brightness')
    
    # 叠加振动曲线
    time_axis = np.arange(time_surface.shape[0]) / fps
    for label, signal in curves.items():
        y_pos = int(label.split('=')[1])  # 解析真实Y坐标
        plt.plot(time_axis, y_pos + signal * 10,  # 缩放信号便于显示
                 'r', linewidth=1.5, label=label)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (Pixel)')
    plt.title('Vibration Curves Overlay on Cloud')
    plt.legend()
    plt.savefig('overlay.png', dpi=200)
    plt.show()

def interpolate_trajectory(crest_positions, method='cubic'):
    """插值填补缺失的波峰位置"""
    time = np.arange(len(crest_positions))
    valid = ~np.isnan(crest_positions)
    if np.sum(valid) < 2:
        return None
    f = interp1d(time[valid], crest_positions[valid], kind=method, fill_value="extrapolate")
    return f(time)

def plot_wave_trajectory(time_surface, trajectory, roi_start_y=0):
    """可视化云图与波轨迹（横轴为帧序号）"""
    plt.figure(figsize=(15, 8))
    
    # 绘制时空云图
    plt.imshow(time_surface.T, cmap='viridis', aspect='auto',
               extent=[0, time_surface.shape[0],  # 横轴范围改为帧数
                       roi_start_y, roi_start_y + time_surface.shape[1]],
               origin='lower')
    
    # 绘制波峰轨迹
    frame_indices = np.arange(len(trajectory))  # 帧序号代替时间
    plt.plot(frame_indices, roi_start_y + trajectory, 
             'r', linewidth=2, label='Wave Crest')
    
    plt.colorbar(label='Normalized Brightness')
    plt.xlabel('Frame Index', fontsize=12)  # 修改横轴标签
    plt.ylabel('Y Position (Pixel)', fontsize=12)
    plt.title('Wave Propagation Trajectory (Frame Index)', fontsize=14)
    plt.legend()
    plt.savefig('wave_trajectory.png', dpi=200)
    plt.show()

def visualize_result(time_surface):
    plt.figure(figsize=(12, 6))
    plt.imshow(time_surface.T, cmap='viridis', aspect='auto')
    plt.colorbar(label='Brightness Intensity')
    plt.xlabel('Time Step')
    plt.ylabel('Spatial Position (Y-axis)')
    plt.title('Temporal-Spatial Brightness')
    plt.savefig('time_surface.png', dpi=300, bbox_inches='tight')
    plt.show()




def optimize_trajectory(positions, fps=200, cutoff_freq=50):
    """轨迹优化（增加数据有效性检查）"""
    # 检查有效数据点
    valid = ~np.isnan(positions)
    if np.sum(valid) < 2:
        raise ValueError("有效数据点不足（需≥2），请检查二值化参数或数据质量")
    
    time = np.arange(len(positions))
    try:
        f = interp1d(time[valid], positions[valid], kind='linear', fill_value="extrapolate")
        filled = f(time)
    except Exception as e:
        print(f"插值失败：{str(e)}，使用最近有效值填充")
        filled = positions.copy()
        last_valid = np.where(valid)[0][0]
        filled[:] = filled[last_valid]  # 用最后一个有效值填充全段
    
    # 频域滤波
    n = len(filled)
    fft_data = np.fft.fft(filled)
    freqs = np.fft.fftfreq(n, 1/fps)
    
    # 低通滤波
    filtered_fft = fft_data.copy()
    filtered_fft[np.abs(freqs) > cutoff_freq] = 0
    smooth_traj = np.real(np.fft.ifft(filtered_fft))
    
    return smooth_traj

def detect_edge(time_surface):
    """检测每行的亮度梯度边缘（分界线）"""
    edge_positions = []
    for t in range(time_surface.shape[0]):
        profile = time_surface[t, :]
        
        # 计算梯度（寻找最大负梯度位置，即亮到暗的过渡）
        gradient = np.gradient(profile)
        edge_y = np.argmin(gradient)  # 负梯度最大处
        
        # 验证梯度强度（避免噪声误判）
        if gradient[edge_y] < -0.1:  # 阈值根据实际数据调整
            edge_positions.append(edge_y)
        else:
            edge_positions.append(np.nan)
    return np.array(edge_positions)

def preprocess_trajectory(trajectory):
    """准备用于频谱分析的轨迹信号"""
    # 去线性趋势
    detrended = detrend(trajectory)
    
    # # 去均值
    zero_mean = detrended - np.mean(detrended)
    return zero_mean

def plot_spectrogram(signal, fs=200, vmax=30):
    """绘制频谱图"""
    f, t, Sxx = signal.spectrogram(signal, fs=fs, 
                                  nperseg=256,  # 时间窗口长度
                                  noverlap=128, # 重叠样本数
                                  scaling='spectrum')
    
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, 10*np.log10(Sxx), 
                  shading='gouraud', 
                  cmap='inferno', 
                  vmax=vmax)
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Vibration Spectrogram')
    plt.ylim(0, 100)  # 聚焦在0-100Hz
    plt.tight_layout()
    plt.savefig('spectrogram.png', dpi=200)
    plt.show()


def plot_frequency_spectrum(signal, fs=200):
    """绘制傅里叶频谱"""
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, 1/fs)
    
    # 取正频率部分
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_abs = np.abs(fft_vals[pos_mask])
    
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, 20*np.log10(fft_abs/np.max(fft_abs)),  # 转换为dB
             linewidth=1.5)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude (dB)', fontsize=12)
    plt.title('Frequency Spectrum', fontsize=14)
    plt.grid(True)
    plt.xlim(0, 100)  # 聚焦在0-100Hz
    plt.tight_layout()
    plt.savefig('spectrum.png', dpi=200)
    plt.show()

def plt_time_domain(arr, fs=1, ylabel='Amp(mg)', title='原始数据时域图', img_save_path=None, x_vline=None, y_hline=None):
    """
    :fun: 绘制时域图模板
    :param arr: 输入一维数组数据
    :param fs: 采样频率
    :param ylabel: y轴标签
    :param title: 图标题
    :return: None
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    font = {'family': 'Times New Roman', 'size': '20', 'color': '0.5', 'weight': 'bold'}
    
    plt.figure(figsize=(12,4))
    length = len(arr)
    t = np.linspace(0, length/fs, length)
    plt.plot(t, arr, c='g')
    plt.xlabel('t(s)')
    plt.ylabel(ylabel)
    plt.title(title)
    if x_vline:
        plt.vlines(x=x_vline, ymin=np.min(arr), ymax=np.max(arr), linestyle='--', colors='r')
    if y_hline:
        plt.hlines(y=0.2, xmin=np.min(t), xmax=np.max(t), linestyle=':', colors='y')
    #===保存图片====#
    if img_save_path:
        plt.savefig(img_save_path, dpi=500, bbox_inches = 'tight')
    plt.show()

def plt_fft_img(arr, fs=None, ylabel='Amplitude', title='Frequency Spectrum', 
               img_save_path=None, vline=None, hline=None, 
               xlim=None, apply_window=True, remove_dc=True):
    """修改后的频谱分析函数（支持帧数频率）"""
    # 预处理步骤（同原函数）
    processed_signal = arr.copy()
    if remove_dc:
        processed_signal -= np.mean(processed_signal)
    x = np.arange(len(processed_signal))
    coeffs = np.polyfit(x, processed_signal, 3)
    trend = np.polyval(coeffs, x)
    processed_signal -= trend
    if apply_window:
        window = hann(len(processed_signal))
        processed_signal *= window
        window_correction = np.mean(window**2)
    else:
        window_correction = 1.0

    # 关键修改：当未提供fs时，使用帧数作为采样单位
    if fs is None:
        fs = 1  # 1 sample per frame
        freq_unit = 'Cycle/Frame' 
        show_cycle_length = True
    else:
        freq_unit = 'Hz'
        show_cycle_length = False

    # FFT计算
    n = len(processed_signal)
    fft_result = np.fft.fft(processed_signal)
    fft_freq = np.fft.fftfreq(n, d=1/fs)
    fft_amp = 2 * np.abs(fft_result) / (n * np.sqrt(window_correction))
    
    # 显示周期长度代替频率
    pos_mask = fft_freq > 0  # 仅显示正频率
    freqs = fft_freq[pos_mask]
    amps = fft_amp[pos_mask]
    
    if show_cycle_length:
        # 转换频率为周期长度（帧数）
        cycle_lengths = 1 / freqs
        valid = (cycle_lengths >= 2) & (cycle_lengths <= len(arr)//2)
        x_values = cycle_lengths[valid]
        y_values = amps[valid]
        xlabel = 'Cycle Length (Frames)'
    else:
        x_values = freqs
        y_values = amps
        xlabel = f'Frequency ({freq_unit})'

    # 绘图
    plt.figure(figsize=(12,4))
    plt.plot(x_values, y_values, 'b-', lw=1)
    plt.xlabel(xlabel)
    plt.ylabel('Amplitude')
    
    if vline and show_cycle_length:
        plt.axvline(x=10, color='r', linestyle='--', label='10-Frame Cycle')
        plt.legend()
    
    if img_save_path:
        plt.savefig(img_save_path, dpi=500, bbox_inches='tight')
    plt.show()

def process_single_region(images, x_pos, roi, fps):
    """处理单个区域并返回平滑后的轨迹"""
    # 提取亮度数据
    brightness_data, roi_start_y = extract_brightness_columns(images, x1=x_pos, roi=roi)
    if brightness_data.size == 0:
        return np.array([])  # 返回空数组表示无效数据
    
    # 构建时间表面
    time_surface = create_time_surface(brightness_data)
    
    # 二值化处理
    ret, img = cv2.threshold(time_surface, 5, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    binary_surface = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    # 检测边缘
    edge_positions = detect_edge(binary_surface)
    
    # 轨迹插值与平滑
    time = np.arange(len(edge_positions))
    valid = ~np.isnan(edge_positions)
    
    if np.sum(valid) < 2:  # 有效数据不足
        return np.array([])
    
    try:
        f = interp1d(time[valid], edge_positions[valid], kind='linear', fill_value="extrapolate")
        filled_traj = f(time)
        smoothed_traj = savgol_filter(filled_traj, window_length=13, polyorder=3)
    except Exception as e:
        print(f"Error processing X={x_pos}, ROI={roi}: {str(e)}")
        return np.array([])
    plt.figure(figsize=(15, 8))
    plt.imshow(img.T, cmap='viridis', aspect='auto')
    
    plt.plot(np.arange(len(smoothed_traj)), 
              smoothed_traj, 
             'r-', lw=1, label='Boundary Edge')
    
    plt.xlabel('Frame Index')
    plt.ylabel('Y Position (Pixel)')
    plt.title('Boundary Vibration Trajectory')
    plt.legend()
    plt.show()
    return smoothed_traj

def plot_multiple_time_domains(trajectories, labels, output_directory, fs=1, ylabel='Amplitude', title='Vibration Curves'):
    """绘制多个区域的振动时域曲线于同一张图"""
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(len(trajectories[0])) / fs
    
    # 使用默认颜色循环自动适配
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        plt.plot(time_axis, traj, 
                color=colors[i % len(colors)],  # 循环使用默认颜色
                label=label, 
                linewidth=1.5, 
                alpha=0.8)

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Normalized Amplitude', fontsize=12)
    plt.title('Multi-region Vibration Characteristics', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory,'multi_region_time_domain.png'), dpi=200)
    plt.show()

def plot_multiple_fft(trajectories, labels, output_directory,fs=1):
    """绘制多个区域的频谱图于同一张图"""
    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for traj, label, color in zip(trajectories, labels, colors):
        # 预处理信号
        processed = preprocess_trajectory(traj)
        
        # FFT计算
        n = len(processed)
        fft_vals = np.fft.fft(processed)
        freqs = np.fft.fftfreq(n, 1/fs)
        
        # 取正频率部分并归一化
        pos_mask = freqs > 0
        amp = np.abs(fft_vals[pos_mask]) * 2 / n  # 幅度校正
        freqs_pos = freqs[pos_mask]
        
        plt.plot(freqs_pos, amp, color=color, label=label, alpha=0.7)

    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('Multi-region Frequency Spectrum', fontsize=14)
    plt.xlim(0, 15)
    plt.ylim(0, 1)  # 根据实际数据调整
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory,'multi_region_spectrum.png', dpi=200))
    plt.show()

def plot_combined_spectrum(trajectories,output_directory,fs=1):
    """绘制综合频谱（自动缩放坐标轴）"""
    plt.figure(figsize=(12, 6))
    color = '#d62728'
    
    # 计算频谱数据
    all_amps = []
    max_len = max(len(traj) for traj in trajectories)
    
    for traj in trajectories:
        processed = preprocess_trajectory(traj)
        n = len(processed)
        
        padded = np.zeros(max_len)
        padded[:n] = processed
        
        fft_vals = np.fft.fft(padded)
        freqs = np.fft.fftfreq(max_len, 1/fs)
        
        pos_mask = freqs > 0
        amp = np.abs(fft_vals[pos_mask]) * 2 / max_len
        all_amps.append(amp)
    
    avg_amp = np.mean(all_amps, axis=0)
    freqs = freqs[pos_mask]

    # 自动计算坐标范围
    def auto_axis_limits(x, y):
        """智能坐标轴范围计算"""
        # 有效数据阈值：大于最大值的1%
        y_threshold = np.max(y) * 0.2
        
        # X轴范围：包含95%能量的频率范围
        cumulative_energy = np.cumsum(y)
        total_energy = cumulative_energy[-1]
        valid_freqs = x[cumulative_energy <= total_energy*0.95]
        x_max = valid_freqs[-1] * 1.2 if len(valid_freqs) > 0 else x[-1]
        
        # Y轴范围：基于数据分布的动态缩放
        y_max = np.percentile(y[y >= y_threshold], 99) * 1.5 if np.any(y >= y_threshold) else 1.0
        return (0, min(x_max, 50000)), (0, y_max)  # 限制最大X轴为100Hz

    xlim, ylim = auto_axis_limits(freqs, avg_amp)
    
    # 绘制曲线
    plt.plot(freqs, avg_amp, color=color, linewidth=2)
    
    # 标记主频
    main_freq = freqs[np.argmax(avg_amp)]
    plt.axvline(main_freq, color='gray', linestyle='--', 
               label=f'Dominant Freq: {main_freq:.1f} Hz')
    
    # 设置坐标轴
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Normalized Amplitude', fontsize=12)
    plt.title('domain average spectrum.png', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory,'domain average spectrum.png'), dpi=200)
    plt.show()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, 
        y=avg_amp,
        mode='lines',
        name='Spectrum',
        line=dict(color='red')
    ))
    fig.update_layout(
        title='Frequency Spectrum',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Amplitude',
        xaxis=dict(range=[0, 1.4]),
    )
    fig.write_html(os.path.join(output_directory, 'domain average spectrum.html'))

def main_analysis_pipeline(
    file_directory, 
    output_directory, 
    regions, 
    dt=1/30, 
    num_images=None  # 新增参数：控制读取的图像数量（None表示全部）
):
    # 结果容器
    results = {
        'trajectories': [],
        'labels': [],
        'figures': {
            'time_domain': None,
            'spectrum': None
        },
        'image_count': 0  # 记录实际处理的图像数量
    }
    fps = 1/dt

    # 1. 加载图像（支持读取前N张）
    try:
        images = load_images(file_directory)
        # 如果指定了num_images，截取前N张
        if num_images is not None and num_images > 0:
            images = images[:num_images]
        results['image_count'] = len(images)  # 更新实际处理的图像数量
        if not images:
            raise ValueError("未找到有效图像文件")
    except Exception as e:
        print(f"图像加载失败: {str(e)}")
        return results

    # 2. 处理所有区域（后续逻辑不变）
    for idx, (x, roi) in enumerate(regions, 1):
        print(f"\n处理区域 {idx}/{len(regions)}: X={x}, ROI={roi}")
        try:
            traj = process_single_region(images, x, roi, fps)
            if len(traj) > 0:
                results['trajectories'].append(traj)
                results['labels'].append(f"region{idx} (X={x})")
        except Exception as e:
            print(f"区域 {x} 处理异常: {str(e)}")
            continue

    # 3. 结果验证与可视化（后续逻辑不变）
    if len(results['trajectories']) > 0:
        # 时域图
        time_domain_path = f"vibration_time_domain.png"
        plot_multiple_time_domains(
            results['trajectories'], 
            results['labels'],
            output_directory,
            fs=fps
        )
        results['figures']['time_domain'] = time_domain_path
        
        # 频谱图
        spectrum_path = f"domain average spectrum.png"
        plot_combined_spectrum(
            results['trajectories'],
            output_directory,
            fs=fps
        )
        results['figures']['spectrum'] = spectrum_path
    else:
        print("警告：所有区域处理失败，无有效数据输出")
    
    return results

def poly_to_str(coeff):
    """优化后的多项式表达式生成"""
    terms = []
    for i, c in enumerate(coeff):
        power = len(coeff) - i - 1
        term = f"{c:.3f}x^{power}" if power > 1 else f"{c:.3f}x" if power == 1 else f"{c:.3f}"
        terms.append(term)
    return "y = " + " + ".join(terms).replace("x^0", "").replace("x^1", "x")


def calculate_blade_width(contour):
    """改进的叶片宽度计算（PCA方法）"""
    points = contour.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(points, mean=None)
    ortho_vec = eigenvectors[0][::-1] * [-1, 1]  # 获取垂直方向向量
    projections = np.dot(points - mean, ortho_vec)
    return np.ptp(projections)


def find_best_offset(args):
    """多线程偏移量计算单元"""
    offset, x_plot, y_fit, edge_map, distance_map, intersection_threshold = args
    curve = np.column_stack((x_plot, y_fit + offset)).astype(int)
    valid = (curve[:, 0] >= 0) & (curve[:, 0] < edge_map.shape[1]) & (curve[:, 1] >= 0) & (
            curve[:, 1] < edge_map.shape[0])
    valid_curve = curve[valid]
    intersections = edge_map[valid_curve[:, 1], valid_curve[:, 0]]
    intersection_count = np.count_nonzero(intersections)
    if intersection_count < intersection_threshold:
        return (offset, -np.inf)  # 如果交点数量小于阈值，返回负无穷
    return (offset, np.mean(distance_map[valid_curve[:, 1], valid_curve[:, 0]]))


def optimize_offset(y_fit, x_range, edge_map, offset_range, intersection_threshold):
    """优化后的偏移量搜索算法"""
    # 生成距离变换图
    distance_map = cv2.distanceTransform(255 - edge_map, cv2.DIST_L2, 3)

    # 多线程计算
    with ThreadPoolExecutor() as executor:
        args = [(o, x_range, y_fit, edge_map, distance_map, intersection_threshold) for o in offset_range]
        results = executor.map(find_best_offset, args)

    # 寻找最大平均距离
    best_offset = max(results, key=lambda x: x[1])[0]
    return best_offset


def draw_curves_on_image(image, x_values, min_area=2000, intersection_threshold=10):
    """主处理流程"""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 边缘检测优化
    edges = cv2.Canny(enhanced, 80, 150)
    kernel = np.ones((3, 3), np.uint8)
    edge_map = cv2.dilate(edges, kernel, iterations=2)

    # 改进的轮廓检测
    contours, _ = cv2.findContours(edge_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    if not filtered_contours:
        return image

    # 选择面积最大的轮廓
    largest_contour = max(filtered_contours, key=cv2.contourArea)

    # 计算平均叶片宽度
    width = calculate_blade_width(largest_contour)

    # 曲线拟合与绘制
    points = largest_contour.squeeze()
    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)

    # 多项式拟合
    coeff = np.polyfit(x, y, 3)
    x_plot = np.linspace(x.min(), x.max(), 200)
    y_fit = np.polyval(coeff, x_plot)

    # 动态偏移范围
    upper_range = np.arange(-width * 0.6, -width * 0.1, 0.5)
    lower_range = np.arange(width * 0.1, width * 0.6, 0.5)

    # 优化偏移量计算
    yellow_upper = optimize_offset(y_fit, x_plot, edge_map, upper_range, intersection_threshold)
    yellow_lower = optimize_offset(y_fit, x_plot, edge_map, lower_range, intersection_threshold)

    # 绘制曲线
    red_curve = np.column_stack((x_plot, y_fit)).astype(np.int32)
    yellow_upper_curve = np.column_stack((x_plot, y_fit + yellow_upper)).astype(np.int32)
    yellow_lower_curve = np.column_stack((x_plot, y_fit + yellow_lower)).astype(np.int32)

    cv2.polylines(image, [red_curve], False, (0, 0, 255), 2)
    cv2.polylines(image, [yellow_upper_curve], False, (0, 255, 255), 2)
    cv2.polylines(image, [yellow_lower_curve], False, (0, 255, 255), 2)

    # 计算并标注多个x值对应的曲线值
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0  # 增大字体大小
    color_yellow = (255, 0, 255)
    color_green = (0, 255, 0)
    thickness = 1
    regions = []
    for x_val in x_values:
        yellow_upper_y = int(np.polyval(coeff, x_val) + yellow_upper + 25)
        yellow_lower_y = int(np.polyval(coeff, x_val) + yellow_lower + 25)

        # 计算上黄色曲线值的区间
        upper_range_min = yellow_upper_y - 20
        upper_range_max = yellow_upper_y + 15

        cv2.putText(image, f"{yellow_upper_y}", (x_val, int(yellow_upper_y)), font, font_scale, color_yellow, thickness)
        print(f"当x = {x_val}时，上黄色曲线的值: {yellow_upper_y}")
        print(f"当x = {x_val}时，上黄色曲线值的区间: [{upper_range_min}, {upper_range_max}]")
        regions.append( (x_val, (upper_range_min, upper_range_max)) )
        # 绘制绿色垂直线
        cv2.line(image, (x_val, 0), (x_val, image.shape[0]), color_green, 2)
        # 在垂直线下方标注 x 值，距离底部 20 像素
        cv2.putText(image, f"x={x_val}", (x_val + 5, image.shape[0] - 20), font, font_scale, color_green, thickness)

    return image,regions

def process_images_in_folder(image, output_folder, x_values):
    """批量处理函数"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  
    output_path = os.path.join(output_folder, f"domain_regions.png")
    result,regions= draw_curves_on_image(image, x_values)
    cv2.imwrite(output_path, result)
    return regions

def sections_spectrum(input_folder,output_folder,data_path,dt,num_images):

    data = np.load(data_path, allow_pickle=True)
    divided_ranges = data["divided_ranges"]
    leaf_contours_dict = data['leaf_contours'].item()  # 转换为字典
    # 获取第一张图片的叶片轮廓
    first_image_key = next(iter(leaf_contours_dict.keys()))
    leaf_contours = leaf_contours_dict[first_image_key]
    image_files = [filename for filename in os.listdir(input_folder) if
                    filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
    first_image_file = image_files[0]
    input_path = os.path.join(input_folder, first_image_file)
    image = cv2.imread(input_path)
    leafimage = np.full_like(image, 255)  # 全白背景
    mask = np.zeros_like(image)
    if len(leaf_contours) > 0:
        contours_reshaped = [cnt.reshape((-1, 1, 2)) for cnt in leaf_contours]
        cv2.fillPoly(mask, contours_reshaped, color=(255, 255, 255))
        leafimage[mask == 255] = image[mask == 255]
    output_dir = os.path.join(output_folder, 'sections spectrum')
    os.makedirs(output_dir, exist_ok=True)
    regions = process_images_in_folder(leafimage, output_dir,  divided_ranges)
    result = main_analysis_pipeline(
            file_directory=input_folder,
            output_directory = output_dir,
            regions=regions,
            dt = dt,
            num_images=500
        ) 
    print("\n分析完成")


input_folder = r'D:\CODE\Frequency extract\yepian-origin'  #输入的是原始图片
data_path =r'D:\CODE\Frequency extract\result\all_data.npz'   #需要输入检测功能输出的npz文件
output_folder = r'D:\CODE\Frequency extract\result'
dt = 1/30
num_images = 20
sections_spectrum(input_folder,output_folder,data_path,dt,num_images)