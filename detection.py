import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import generate_binary_structure, iterate_structure
import os
from line_profiler import LineProfiler


def directional_weighted_sobel(img, expected_angle, tolerance=50, power=2):
    img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grad_x = cv2.Sobel(img_8u, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_8u, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度方向和幅值
    grad_angle = np.arctan2(grad_x, grad_y) * 180 / np.pi  # [-180, 180]
    grad_angle = np.where(grad_angle < 0, grad_angle + 180, grad_angle)  # [0, 180)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 计算方向权重
    angle_diff = np.abs(grad_angle - expected_angle)
    angle_diff = np.minimum(angle_diff, 180 - angle_diff)  
    weight = np.cos(np.deg2rad(angle_diff / tolerance * 90)) ** power
    weight[angle_diff > tolerance] = 0  

    # 应用方向加权并归一化
    weighted_grad = grad_mag * weight
    weighted_grad = cv2.normalize(weighted_grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return weighted_grad, grad_mag.astype(np.uint8), grad_angle


def apply_sobel_otsu_roi(img, expected_angle, min_final_area):
    """对单张图像执行：Sobel梯度 → Otsu阈值 → ROI掩膜"""
    ret, thresh = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    weighted_grad, raw_grad, grad_angle = directional_weighted_sobel(thresh, expected_angle)

    ret, thresh1 = cv2.threshold(weighted_grad, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  
    closed = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    smoothed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel, iterations=1)
    skeleton = skeletonize(smoothed // 255).astype(np.uint8) * 255
    pruned_skel = halcon_style_pruning(skeleton, max_length=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pruned_skel, connectivity=8)
    min_length = 10  


    filtered_skel = np.zeros_like(pruned_skel)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_length:
            filtered_skel[labels == i] = 255
    contours, _ = cv2.findContours(filtered_skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(
        contour_img,
        contours,
        -1,  
        color=(0, 255, 0),  
        thickness=1  
    )

    mask = np.zeros_like(thresh)  
    rect_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    h, w = mask.shape[:2]
    temp_mask = np.zeros((h, w), dtype=np.uint8)

    for cnt in contours:
  
        (cx, cy), (orig_w, orig_h), angle = cv2.minAreaRect(cnt)
        new_w = orig_w * 1
        new_h = orig_h * 1.6
        expanded_rect = ((cx, cy), (new_w, new_h), angle)
        expanded_box = cv2.boxPoints(expanded_rect).astype(np.intp)
  
        cv2.fillPoly(temp_mask, [expanded_box], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 根据重叠程度调整核大小
    merged_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel)
    merged_contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    epsilon_ratio = 0.005
    for cnt in merged_contours:
        # 面积过滤
        if cv2.contourArea(cnt) < min_final_area:
            continue

        # 多边形逼近
        epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)

        cv2.drawContours(rect_img, [approx], 0, (255, 0, 0), 2)
        cv2.fillPoly(mask, [approx], 255)

    result = cv2.bitwise_and(thresh, mask)  # 此时矩形外为白色，矩形内保留原值
    result = cv2.bitwise_or(result, ~mask)

    contours1 = safe_contour_detection(result)
    min_length = 10  # 最小轮廓长度
    min_area = 10  # 最小轮廓面积
    min_aspect_ratio = 0  # 最小宽高比
    border_margin = 0

    filtered_contours = []
    img_height, img_width = result.shape[:2]
    for cnt in contours1:
        length = cv2.arcLength(cnt, closed=False)
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / max(h, 1e-5)  
        is_full_border = (x == 0 and y == 0
                          and w == img_width
                          and h == img_height)

   
        is_thin_border = (w == img_width and h == img_height
                          and cv2.contourArea(cnt) == img_width * img_height
                          and cv2.contourArea(cnt[:-1]) == 0)
        if (length >= min_length) and (area >= min_area) and (aspect_ratio >= min_aspect_ratio) and (
                aspect_ratio >= min_aspect_ratio) and not (is_full_border or is_thin_border):
            filtered_contours.append(cnt)

    return thresh, weighted_grad, closed, filtered_skel, result, filtered_contours


def find_endpoints(skel):
    """改进的端点检测算法"""
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    conv = cv2.filter2D(skel, -1, kernel)
    return np.argwhere((conv >= 10) & (conv <= 11))  # Halcon端点检测策略


def safe_contour_detection(image, border_size=300):
    bordered = cv2.copyMakeBorder(
        image,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=255
    )


    contours, _ = cv2.findContours(
        bordered,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )


    valid_contours = []
    offset = np.array([[-border_size, -border_size]])  

    for cnt in contours:
        adjusted_cnt = cnt + offset
        x, y, w, h = cv2.boundingRect(adjusted_cnt)
        if x + w <= 0 or y + h <= 0:
            continue
        adjusted_cnt[:, :, 0] = np.clip(adjusted_cnt[:, :, 0], 0, image.shape[1] - 1)
        adjusted_cnt[:, :, 1] = np.clip(adjusted_cnt[:, :, 1], 0, image.shape[0] - 1)

        valid_contours.append(adjusted_cnt)

    return valid_contours


def visualize_curves(image, segments):
    """
    安全坐标转换：防止溢出
    """

    disp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()

    for seg in segments:
        try:
            if isinstance(seg, tuple):

                x1, y1, x2, y2 = map(int, seg)
                cv2.line(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:

                pts = np.array(seg, dtype=np.int64).reshape(-1, 1, 2) 
                pts = np.clip(pts, 0, [image.shape[1] - 1, image.shape[0] - 1])  
                cv2.polylines(disp, [pts], False, (255, 0, 0), 1)
        except OverflowError:
            print(f"坐标值溢出已跳过: {seg[:2]}...")

    return disp


def halcon_style_pruning(skeleton, max_length=10):
    skel = (skeleton > 127).astype(np.uint8)
    struct = generate_binary_structure(2, 2)  

    # Halcon算法
    for _ in range(max_length):

        endpoints = find_endpoints(skel)
        if len(endpoints) == 0:
            break
        endpoint_mask = np.zeros_like(skel)
        for y, x in endpoints:
            endpoint_mask[y, x] = 1
        dilated = iterate_structure(struct, 2).astype(np.uint8)
        endpoint_dilated = cv2.dilate(endpoint_mask, dilated)
        to_prune = cv2.bitwise_and(skel, endpoint_dilated)
        skel = cv2.subtract(skel, to_prune)

    return skel * 255


def visualize_processing_steps(images, titles, processed_results, output_path, original_filename):
    """兼容单图像的可视化函数"""
    num_images = len(images)

    plt.figure(figsize=(15, 5 * num_images))  
    for i in range(num_images):

        plt.subplot(num_images, 6, 6 * i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Original", fontsize=10)
        plt.axis('off')
        steps = [
            ("Binary", processed_results[i][0], 'gray'),
            ("Gradient", processed_results[i][1], 'gray'),
            ("Enhance", processed_results[i][2], 'gray'),
            ("Extract", processed_results[i][3], 'gray'),
            ("Bounds", processed_results[i][4], 'gray')
        ]

        for j in range(5):
            plt.subplot(num_images, 6, 6 * i + j + 2)
            plt.imshow(steps[j][1], cmap=steps[j][2])
            plt.title(f"{steps[j][0]}", fontsize=10)
            plt.axis('off')
    save_name = f"{original_filename}_processing_steps.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, save_name), dpi=300, bbox_inches='tight')
    plt.close()  
    # plt.show()


def preprocess_image(image_path, clahe):
    """图像预处理流程：去噪 → 背景校正 → CLAHE增强"""

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    denoised = cv2.fastNlMeansDenoising(
        img,
        h=20,  
        templateWindowSize=3, 
        searchWindowSize=3  
    )

    if clahe:
        clahe1 = cv2.createCLAHE(
            clipLimit=1.3,  
            tileGridSize=(3, 3)  
        )
        denoised = clahe1.apply(denoised)

    
    return img, denoised


# 叶片轮廓提取
def extract_contours(binary_image):

    h, w = binary_image.shape
    contours = []
    visited = np.zeros((h, w), dtype=np.int32)
    NBD = 1  
    for i in range(h):
        for j in range(w):
            if visited[i, j] == 0 and binary_image[i, j] != 0:
                contour = []
                stack = [(i, j)]
                visited[i, j] = NBD

                while stack:
                    y, x = stack.pop()
                    contour.append((x, y)) 

                    neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
                    for ny, nx in neighbors:
                        if 0 <= ny < h and 0 <= nx < w:
                            if visited[ny, nx] == 0 and binary_image[ny, nx] != 0:
                                visited[ny, nx] = NBD
                                stack.append((ny, nx))

                contours.append(np.array(contour))
                NBD += 1
    return contours


def process_image(image, output_path):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(image)
    blurred = cv2.GaussianBlur(contrast_enhanced, (7, 7), 0)
    denoised = cv2.medianBlur(blurred, 3)

    _, binary = cv2.threshold(image, 4, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    contours = extract_contours(closing)

    min_area = 1000
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    result = np.full_like(image, 255) 
    mask = np.zeros_like(image)

    if len(filtered_contours) > 0:
        contours_reshaped = [cnt.reshape((-1, 1, 2)) for cnt in filtered_contours]
        cv2.fillPoly(mask, contours_reshaped, color=255)
        result[mask == 255] = image[mask == 255]

    cv2.imwrite(output_path, result)
    return filtered_contours


def subtract_shockwave_from_leaf(leaf_x_range, shockwave_ranges):
    """
    从叶片轮廓范围中减去激波范围
    """
    new_leaf_ranges = [leaf_x_range]
    for shock_range in shockwave_ranges:
        start, end = shock_range
        temp_ranges = []
        for leaf_start, leaf_end in new_leaf_ranges:
            if end < leaf_start or start > leaf_end:
                # 激波范围与叶片范围无交集
                temp_ranges.append((leaf_start, leaf_end))
            elif start <= leaf_start and end >= leaf_end:
                # 激波范围完全覆盖叶片范围
                continue
            elif start <= leaf_start and end < leaf_end:
                # 激波范围从左侧覆盖部分叶片范围
                temp_ranges.append((end, leaf_end))
            elif start > leaf_start and end >= leaf_end:
                # 激波范围从右侧覆盖部分叶片范围
                temp_ranges.append((leaf_start, start))
            elif start > leaf_start and end < leaf_end:
                # 激波范围在叶片范围内部
                temp_ranges.append((leaf_start, start))
                temp_ranges.append((end, leaf_end))
        new_leaf_ranges = temp_ranges
    return new_leaf_ranges


def shockwave_leaf_detection(input_path, output_path, angle, boundsarea, shockwave=False, leaf_x_range=None, is_first_image=False):

    original_filename = os.path.splitext(os.path.basename(input_path))[0]
    original, denoised = preprocess_image(input_path, clahe=True)
    if shockwave:
        result_dir = os.path.join(output_path, 'shockwave_result')
        process_dir = os.path.join(output_path, 'process')
        extract_dir = os.path.join(output_path, 'shockwave_extract')
        # white_bg_dir = os.path.join(output_path, 'white_bg')  

        os.makedirs(result_dir, exist_ok=True)
        # os.makedirs(process_dir, exist_ok=True)
        os.makedirs(extract_dir, exist_ok=True)
        # os.makedirs(white_bg_dir, exist_ok=True)

        processed_results = []
        thresh, weighted_grad, closed, filtered_skel, extract, filtered_contours = apply_sobel_otsu_roi(denoised, angle,
                                                                                                        boundsarea)
        result_filename = f"{original_filename}_shockwave_extract.png"
        cv2.imwrite(os.path.join(extract_dir, result_filename), extract)
        processed_results.append((thresh, weighted_grad, closed, filtered_skel, extract))
        # 保存处理步骤到process目录
        # visualize_processing_steps(
        #     images=[original],
        #     titles=["Processed Result"],
        #     processed_results=processed_results,
        #     output_path=process_dir,  # 修改输出路径
        #     original_filename=original_filename
        # )

        contour_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(
            contour_img,
            filtered_contours,
            -1,  
            color=(0, 255, 0),  
            thickness=1  
        )
        contour_save_name = f"{original_filename}_shockwave_result.png"
        cv2.imwrite(os.path.join(result_dir, contour_save_name), contour_img)
        white_bg_img = original.copy()
        mask = np.zeros_like(original)
        cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
        white_bg_img[mask == 255] = 255
        # cv2.imwrite(os.path.join(white_bg_dir, white_bg_filename), white_bg_img)
        # white_bg_image_path = os.path.join(white_bg_dir, white_bg_filename)      
    else:
        white_bg_img = original
        filtered_contours = None

    leaf_contour_dir = os.path.join(output_path, 'leaf_extract')
    os.makedirs(leaf_contour_dir, exist_ok=True)
    white_bg_filename = f"{original_filename}_leaf_extract.png"
    leaf_contour_output_path = os.path.join(leaf_contour_dir, white_bg_filename)
    leaf_contours= process_image(white_bg_img, leaf_contour_output_path)
    contour_img1 = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    leafresult_dir = os.path.join(output_path, 'leaf_result')
    os.makedirs(leafresult_dir, exist_ok=True)
    cv2.drawContours(
            contour_img1,
            leaf_contours,
            -1,  
            color=(0, 0, 255),  
            thickness=1  
        )
    contour_save_name = f"{original_filename}_leaf_result.png"
    cv2.imwrite(os.path.join(leafresult_dir, contour_save_name), contour_img1)
    all_x_coords = []
    for cnt in leaf_contours:
        if cnt.ndim == 2:
            all_x_coords.extend(cnt[:, 0])
        elif cnt.ndim == 3:
            all_x_coords.extend(cnt[:, 0, 0])
    if all_x_coords:
        leaf_x_range = (min(all_x_coords), max(all_x_coords))
    else:
        leaf_x_range = None
    # if is_first_image:
    #     print(f"叶片轮廓的 x 范围: {leaf_x_range}")

    if shockwave:
        # 打印激波的 x 范围
        shockwave_x_ranges = []
        for cnt in filtered_contours:
            if cnt.ndim == 2:
                x_coords = cnt[:, 0]
            elif cnt.ndim == 3:
                x_coords = cnt[:, 0, 0]
            shockwave_x_range = (min(x_coords), max(x_coords))
            shockwave_x_ranges.append(shockwave_x_range)
    else:
        shockwave_x_ranges = None
    return filtered_contours, leaf_contours,leaf_x_range, shockwave_x_ranges

def batch_detection(
    input_folder, 
    output_folder, 
    angle, 
    boundsarea, 
    shockwave=False,
    num_images=None  
):
    os.makedirs(output_folder, exist_ok=True)

    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    file_list = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith(supported_ext)],
        key=lambda x: x.lower()  
    )

    if num_images is not None and num_images > 0:
        file_list = file_list[:num_images]

    leaf_x_range = None
    all_shockwave_ranges = []
    
    # 初始化存储所有图片数据的字典
    filtered_data = {}  
    leaf_data = {}      

    for i, filename in enumerate(file_list):
        try:
            input_path = os.path.join(input_folder, filename)
            filtered_contours, leaf_contours, leaf_x_range, shockwave_x_ranges = shockwave_leaf_detection(
                input_path=input_path,
                output_path=output_folder,
                angle=angle,
                boundsarea=boundsarea,
                shockwave=shockwave,
                is_first_image=(i == 0)
            )
            if shockwave:
                all_shockwave_ranges.extend(shockwave_x_ranges)

            # 将轮廓数据存入字典 (文件名作为键)
            base_name = os.path.splitext(filename)[0]
            if filtered_contours is not None:
                filtered_data[base_name] = [np.array(cnt) for cnt in filtered_contours]
            if leaf_contours is not None:
                leaf_data[base_name] = [np.array(cnt) for cnt in leaf_contours]

            print(f"处理 {filename} 完成 ({i+1}/{len(file_list)})")  # 添加处理进度提示

        except Exception as e:
            print(f"处理 {filename} 出错: {str(e)}")
            continue

    # 计算切面值
    divided_ranges = None
    if leaf_x_range:
        if shockwave and all_shockwave_ranges:
            new_leaf_x_ranges = subtract_shockwave_from_leaf(leaf_x_range, all_shockwave_ranges)
        else:
            new_leaf_x_ranges = [leaf_x_range]
        
        divided_ranges = []
        for start, end in new_leaf_x_ranges:
            interval = (end - start) / 3
            divided_ranges.extend([round(start + interval), round(start + 2 * interval)])
        # print(f"三等分后的切面值: {divided_ranges}")

    # 保存为 NPZ 文件（包含所有数据）
    np.savez_compressed(
        os.path.join(output_folder, "all_data.npz"),
        filtered_contours=filtered_data,  
        leaf_contours=leaf_data,          
        divided_ranges=np.array(divided_ranges, dtype=np.int32) if divided_ranges else np.array([]),
        processed_images_count=len(file_list)  
    )

    return divided_ranges

if __name__ == "__main__":
    input_path = r"D:\CODE\shock and leaf detections\yepian-origin"
    output_path = r"D:\CODE\shock and leaf detections\result"
    angle=110
    boundsarea=4000
    shockwave=True
    num_images = 50
    batch_detection(
                input_folder=input_path,
                output_folder=output_path,
                angle=angle,
                boundsarea=boundsarea,
                shockwave = shockwave,
                num_images=num_images
            )